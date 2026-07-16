import unittest
from datetime import datetime, timedelta, timezone

from app.ingestion import IncrementalIngestionService
from app.models import ListSyncState, XPost
from app.providers import (
    MockXIngestionProvider,
    build_mock_ingestion_posts,
)
from app.storage import InMemoryPostStore
from app.x_api_provider import XFetchPage


NOW = datetime(2026, 7, 16, 9, 0, tzinfo=timezone.utc)


def clone_post(post: XPost, *, post_id: str, source_list_id: str | None = None) -> XPost:
    payload = post.to_dict()
    payload["id"] = post_id
    payload["conversation_id"] = post_id
    payload["source_list_id"] = source_list_id or post.source_list_id
    payload["raw_payload"] = {**payload["raw_payload"], "id": post_id}
    return XPost.from_dict(payload)


class FailingCommitStore(InMemoryPostStore):
    def commit_list_batch(self, posts, list_id, state, **kwargs):
        raise OSError("injected entity write failure")


class CheckpointOnceStore(InMemoryPostStore):
    """Simulate entity durability followed by one checkpoint write failure."""

    def __init__(self):
        super().__init__()
        self.fail_once = True

    def commit_list_batch(self, posts, list_id, state, **kwargs):
        if self.fail_once:
            self.fail_once = False
            self.save_posts(posts)
            raise OSError("injected checkpoint replacement failure")
        return super().commit_list_batch(posts, list_id, state, **kwargs)


class PartialResponseProvider(MockXIngestionProvider):
    async def fetch_page(self, *args, **kwargs):
        page = await super().fetch_page(*args, **kwargs)
        page.raw_errors = [{"title": "one post unavailable"}]
        return page


class ConcurrentCommitOnceStore(InMemoryPostStore):
    """Inject another writer immediately before the first guarded commit."""

    def __init__(self):
        super().__init__()
        self.inject_once = True

    def commit_list_batch(self, posts, list_id, state, **kwargs):
        if self.inject_once:
            self.inject_once = False
            super().commit_list_batch(posts, list_id, state)
        return super().commit_list_batch(posts, list_id, state, **kwargs)


class RepeatingTokenProvider(MockXIngestionProvider):
    async def fetch_page(self, *args, **kwargs):
        if kwargs.get("pagination_token") == "repeated-token":
            kwargs["pagination_token"] = "1"
        page = await super().fetch_page(*args, **kwargs)
        page.next_token = "repeated-token"
        return page


class IncrementalIngestionTests(unittest.IsolatedAsyncioTestCase):
    def service(
        self,
        provider,
        store,
        *,
        page_size=2,
        max_pages=20,
        initial_fetch_limit=100,
        progress=None,
    ):
        return IncrementalIngestionService(
            provider=provider,
            store=store,
            page_size=page_size,
            max_pages=max_pages,
            initial_fetch_limit=initial_fetch_limit,
            now_factory=lambda: NOW,
            progress_callback=progress,
        )

    async def test_first_run_saves_canonical_posts_and_independent_checkpoints(self):
        provider = MockXIngestionProvider()
        store = InMemoryPostStore()

        result = await self.service(provider, store).ingest(
            ["mock://mixed", "mock://secondary"], run_id="first", max_tweets=100
        )

        self.assertEqual(result.status, "success")
        self.assertEqual(result.run.new_post_count, 6)
        self.assertEqual(len(store.read_posts()), 6)
        self.assertTrue(store.has_membership("9005", "mock://mixed"))
        self.assertTrue(store.has_membership("9005", "mock://secondary"))
        self.assertEqual(store.get_sync_state("mock://mixed").newest_post_id, "9005")
        self.assertEqual(store.get_sync_state("mock://secondary").newest_post_id, "9005")
        self.assertEqual(store.get_run("first").status, "success")

    async def test_final_run_clears_ingestion_owner_and_preserves_final_heartbeat(self):
        store = InMemoryPostStore()
        tick = 0

        def next_now():
            nonlocal tick
            value = NOW + timedelta(seconds=tick)
            tick += 1
            return value

        service = IncrementalIngestionService(
            provider=MockXIngestionProvider(),
            store=store,
            now_factory=next_now,
            owner_id_factory=lambda: "active-owner",
        )

        result = await service.ingest(["mock://mixed"], run_id="owner-finalized")

        stored = store.get_run("owner-finalized")
        self.assertIsNone(result.run.ingestion_owner_id)
        self.assertIsNone(stored.ingestion_owner_id)
        self.assertIsNotNone(stored.finished_at)
        self.assertEqual(stored.finished_at, stored.ingestion_heartbeat_at)
        self.assertGreater(stored.ingestion_heartbeat_at, stored.started_at)

    async def test_second_identical_run_is_idempotent_and_no_new_posts(self):
        store = InMemoryPostStore()
        first_provider = MockXIngestionProvider()
        await self.service(first_provider, store).ingest(
            ["mock://mixed"], run_id="one", max_tweets=100
        )
        second_provider = MockXIngestionProvider()

        result = await self.service(second_provider, store).ingest(
            ["mock://mixed"], run_id="two", max_tweets=100
        )

        self.assertEqual(result.status, "no_new_posts")
        self.assertEqual(result.run.new_post_count, 0)
        self.assertEqual(len(store.read_posts()), 5)
        self.assertEqual(len(second_provider.fetch_calls), 1)
        self.assertEqual(result.list_results[0].stop_reason, "checkpoint_reached")

    async def test_followup_only_commits_posts_newer_than_checkpoint(self):
        store = InMemoryPostStore()
        base = build_mock_ingestion_posts()["mock://mixed"]
        await self.service(MockXIngestionProvider({"mock://mixed": base}), store).ingest(
            ["mock://mixed"], run_id="baseline"
        )
        newer = clone_post(base[0], post_id="9006")

        result = await self.service(
            MockXIngestionProvider({"mock://mixed": [newer, *base]}), store
        ).ingest(["mock://mixed"], run_id="increment")

        self.assertEqual([post.id for post in result.new_posts], ["9006"])
        self.assertEqual(store.get_sync_state("mock://mixed").newest_post_id, "9006")
        self.assertEqual(len(store.read_posts()), 6)

    async def test_deleted_checkpoint_is_crossed_by_older_numeric_id(self):
        store = InMemoryPostStore()
        store.save_sync_state(
            ListSyncState(list_id="mock://mixed", newest_post_id="9004", last_status="success")
        )
        base = build_mock_ingestion_posts()["mock://mixed"]
        values = [base[0], base[2]]  # 9005 then 9003; checkpoint 9004 is absent.

        result = await self.service(
            MockXIngestionProvider({"mock://mixed": values}), store
        ).ingest(["mock://mixed"], run_id="deleted-checkpoint")

        self.assertEqual([post.id for post in result.new_posts], ["9005"])
        self.assertEqual(result.list_results[0].stop_reason, "checkpoint_reached")
        self.assertEqual(store.get_sync_state("mock://mixed").newest_post_id, "9005")

    async def test_second_page_failure_does_not_save_or_advance_checkpoint(self):
        store = InMemoryPostStore()
        store.save_sync_state(
            ListSyncState(list_id="mock://mixed", newest_post_id="8999", last_status="success")
        )
        provider = MockXIngestionProvider(
            page_failures={("mock://mixed", 2): TimeoutError("page two timeout")}
        )

        result = await self.service(provider, store).ingest(
            ["mock://mixed"], run_id="page-two-failure"
        )

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.list_results[0].page_count, 1)
        self.assertEqual(store.get_sync_state("mock://mixed").newest_post_id, "8999")
        self.assertEqual(store.read_posts(), [])

    async def test_max_tweets_before_checkpoint_is_incomplete(self):
        store = InMemoryPostStore()
        store.save_sync_state(
            ListSyncState(list_id="mock://mixed", newest_post_id="8990", last_status="success")
        )

        result = await self.service(MockXIngestionProvider(), store).ingest(
            ["mock://mixed"], run_id="window", max_tweets=2
        )

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.list_results[0].stop_reason, "max_tweets_before_checkpoint")
        self.assertEqual(store.get_sync_state("mock://mixed").newest_post_id, "8990")
        self.assertEqual(result.run.fetched_count, 2)

    async def test_timeline_end_before_checkpoint_is_incomplete(self):
        store = InMemoryPostStore()
        store.save_sync_state(
            ListSyncState(list_id="mock://mixed", newest_post_id="8990", last_status="success")
        )
        posts = build_mock_ingestion_posts()["mock://mixed"][:2]

        result = await self.service(
            MockXIngestionProvider({"mock://mixed": posts}), store
        ).ingest(["mock://mixed"], run_id="timeline-gap", max_tweets=100)

        self.assertEqual(result.status, "failed")
        self.assertEqual(
            result.list_results[0].stop_reason,
            "timeline_ended_before_checkpoint",
        )
        self.assertEqual(store.get_sync_state("mock://mixed").newest_post_id, "8990")
        self.assertEqual(store.read_posts(), [])

    async def test_max_pages_before_checkpoint_is_incomplete(self):
        store = InMemoryPostStore()
        store.save_sync_state(
            ListSyncState(list_id="mock://mixed", newest_post_id="8990", last_status="success")
        )

        result = await self.service(
            MockXIngestionProvider(), store, page_size=1, max_pages=1
        ).ingest(["mock://mixed"], run_id="page-cap", max_tweets=100)

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.list_results[0].stop_reason, "max_pages_before_checkpoint")
        self.assertEqual(store.get_sync_state("mock://mixed").newest_post_id, "8990")
        self.assertEqual(store.read_posts(), [])

    async def test_repeated_pagination_token_does_not_advance(self):
        store = InMemoryPostStore()
        store.save_sync_state(
            ListSyncState(list_id="mock://mixed", newest_post_id="8990", last_status="success")
        )

        result = await self.service(
            RepeatingTokenProvider(), store, page_size=1
        ).ingest(["mock://mixed"], run_id="repeated-token", max_tweets=100)

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.list_results[0].stop_reason, "repeated_pagination_token")
        self.assertEqual(store.get_sync_state("mock://mixed").newest_post_id, "8990")
        self.assertEqual(store.read_posts(), [])

    async def test_initial_limit_creates_intentional_baseline(self):
        store = InMemoryPostStore()
        provider = MockXIngestionProvider()

        result = await self.service(
            provider, store, page_size=2, initial_fetch_limit=2
        ).ingest(["mock://mixed"], run_id="baseline")

        self.assertEqual(result.status, "success")
        self.assertEqual(result.run.new_post_count, 2)
        self.assertEqual(result.list_results[0].stop_reason, "initial_limit")
        self.assertFalse(result.list_results[0].pagination_complete)
        self.assertEqual(store.get_sync_state("mock://mixed").newest_post_id, "9005")

    async def test_initial_limit_at_last_allowed_page_commits_baseline(self):
        store = InMemoryPostStore()

        result = await self.service(
            MockXIngestionProvider(),
            store,
            page_size=2,
            max_pages=1,
            initial_fetch_limit=2,
        ).ingest(["mock://mixed"], run_id="exact-page-boundary")

        self.assertEqual(result.status, "success")
        self.assertEqual(result.run.new_post_count, 2)
        self.assertEqual(result.list_results[0].stop_reason, "initial_limit")
        self.assertFalse(result.list_results[0].pagination_complete)
        self.assertEqual(store.get_sync_state("mock://mixed").newest_post_id, "9005")

    async def test_first_sync_page_cap_is_an_intentional_baseline(self):
        store = InMemoryPostStore()

        result = await self.service(
            MockXIngestionProvider(),
            store,
            page_size=1,
            max_pages=2,
            initial_fetch_limit=5,
        ).ingest(["mock://mixed"], run_id="initial-page-cap")

        self.assertEqual(result.status, "success")
        self.assertEqual(result.run.new_post_count, 2)
        self.assertEqual(result.list_results[0].stop_reason, "initial_page_limit")
        self.assertFalse(result.list_results[0].pagination_complete)

    async def test_partial_and_all_list_failures_have_distinct_run_status(self):
        partial_store = InMemoryPostStore()
        partial = await self.service(
            MockXIngestionProvider(page_failures={("mock://empty", 1): RuntimeError("500")}),
            partial_store,
        ).ingest(["mock://mixed", "mock://empty"], run_id="partial")
        self.assertEqual(partial.status, "partial_success")
        self.assertEqual(partial.run.successful_lists, ["mock://mixed"])
        self.assertEqual(partial.run.failed_lists, ["mock://empty"])

        failed_store = InMemoryPostStore()
        failed = await self.service(
            MockXIngestionProvider(
                page_failures={
                    ("mock://mixed", 1): RuntimeError("500"),
                    ("mock://empty", 1): TimeoutError("timeout"),
                }
            ),
            failed_store,
        ).ingest(["mock://mixed", "mock://empty"], run_id="failed")
        self.assertEqual(failed.status, "failed")
        self.assertEqual(failed.run.successful_lists, [])

    async def test_cross_list_duplicate_is_one_entity_with_two_memberships(self):
        store = InMemoryPostStore()
        result = await self.service(MockXIngestionProvider(), store).ingest(
            ["mock://mixed", "mock://secondary"], run_id="cross-list"
        )

        self.assertEqual(result.run.new_post_count, 6)
        self.assertGreaterEqual(result.run.duplicate_count, 1)
        shared = [post for post in store.read_posts() if post.id == "9005"]
        self.assertEqual(len(shared), 1)
        memberships = store.read_memberships()
        self.assertIn({"post_id": "9005", "list_id": "mock://mixed"}, memberships)
        self.assertIn({"post_id": "9005", "list_id": "mock://secondary"}, memberships)

    async def test_concurrent_checkpoint_change_retries_without_duplicate_output(self):
        store = ConcurrentCommitOnceStore()

        result = await self.service(MockXIngestionProvider(), store).ingest(
            ["mock://mixed"], run_id="concurrent"
        )

        self.assertEqual(result.status, "no_new_posts")
        self.assertEqual(result.run.new_post_count, 0)
        self.assertEqual(len(store.read_posts()), 5)
        self.assertEqual(store.get_sync_state("mock://mixed").newest_post_id, "9005")

    async def test_in_batch_duplicates_and_empty_list_are_idempotent(self):
        store = InMemoryPostStore()
        duplicates = await self.service(MockXIngestionProvider(), store).ingest(
            ["mock://duplicates"], run_id="duplicates"
        )
        empty = await self.service(MockXIngestionProvider(), store).ingest(
            ["mock://empty"], run_id="empty"
        )

        self.assertEqual(duplicates.run.new_post_count, 1)
        self.assertGreaterEqual(duplicates.run.duplicate_count, 1)
        self.assertEqual(len(store.read_posts("mock://duplicates")), 1)
        self.assertEqual(empty.status, "no_new_posts")
        self.assertEqual(store.get_sync_state("mock://empty").last_status, "no_new_posts")

    async def test_partial_200_response_does_not_advance(self):
        store = InMemoryPostStore()
        store.save_sync_state(
            ListSyncState(list_id="mock://mixed", newest_post_id="8999", last_status="success")
        )

        result = await self.service(PartialResponseProvider(), store).ingest(
            ["mock://mixed"], run_id="partial-response"
        )

        self.assertEqual(result.list_results[0].status, "incomplete")
        self.assertEqual(result.list_results[0].stop_reason, "partial_api_response")
        self.assertEqual(store.get_sync_state("mock://mixed").newest_post_id, "8999")

    async def test_invalid_non_numeric_id_uses_membership_fallback(self):
        base = build_mock_ingestion_posts()["mock://mixed"][0]
        invalid = clone_post(base, post_id="not-a-snowflake")
        store = InMemoryPostStore()
        first_provider = MockXIngestionProvider({"mock://mixed": [invalid]})
        first = await self.service(first_provider, store).ingest(
            ["mock://mixed"], run_id="invalid-one"
        )
        self.assertEqual(first.status, "success")
        self.assertEqual(store.get_sync_state("mock://mixed").newest_post_id, "not-a-snowflake")

        second = await self.service(
            MockXIngestionProvider({"mock://mixed": [invalid]}), store
        ).ingest(["mock://mixed"], run_id="invalid-two")
        self.assertEqual(second.status, "no_new_posts")

    async def test_storage_failure_preserves_checkpoint(self):
        store = FailingCommitStore()
        store.save_sync_state(
            ListSyncState(list_id="mock://mixed", newest_post_id="8999", last_status="success")
        )

        result = await self.service(MockXIngestionProvider(), store).ingest(
            ["mock://mixed"], run_id="storage-failure"
        )

        self.assertEqual(result.status, "failed")
        state = store.get_sync_state("mock://mixed")
        self.assertEqual(state.newest_post_id, "8999")
        self.assertEqual(state.last_status, "failed")

    async def test_checkpoint_write_failure_recovers_from_durable_membership(self):
        store = CheckpointOnceStore()
        first = await self.service(MockXIngestionProvider(), store).ingest(
            ["mock://mixed"], run_id="checkpoint-failure"
        )
        self.assertEqual(first.status, "failed")
        self.assertIsNone(store.get_sync_state("mock://mixed").newest_post_id)
        self.assertEqual(len(store.read_posts()), 5)

        second = await self.service(MockXIngestionProvider(), store).ingest(
            ["mock://mixed"], run_id="checkpoint-recovery"
        )

        self.assertEqual(second.status, "no_new_posts")
        self.assertEqual(store.get_sync_state("mock://mixed").newest_post_id, "9005")
        self.assertEqual(len(store.read_posts()), 5)

    async def test_progress_contains_page_and_count_fields(self):
        events = []
        result = await self.service(
            MockXIngestionProvider(), InMemoryPostStore(), progress=events.append
        ).ingest(["mock://mixed"], run_id="progress")

        self.assertTrue(result.success)
        self.assertGreaterEqual(len(events), 1)
        self.assertEqual(events[0]["run_id"], "progress")
        self.assertEqual(events[0]["list_id"], "mock://mixed")
        self.assertIn("page_number", events[0])
        self.assertIn("duplicate_count", events[0])

    async def test_mock_provider_exposes_failure_and_missing_field_scenarios(self):
        provider = MockXIngestionProvider()
        for list_id in ("mock://rate-limit", "mock://server-error", "mock://timeout"):
            with self.assertRaises(Exception):
                await provider.fetch_page(list_id)

        first = await provider.fetch_page(
            "mock://second-page-failure", max_results=2, page_number=1
        )
        self.assertEqual(len(first.posts), 2)
        self.assertTrue(first.next_token)
        with self.assertRaisesRegex(RuntimeError, "second page"):
            await provider.fetch_page(
                "mock://second-page-failure",
                pagination_token=first.next_token,
                max_results=2,
                page_number=2,
            )

        duplicates = await provider.fetch_page("mock://duplicates", max_results=1)
        self.assertEqual(len(duplicates.posts), 1)
        self.assertEqual(duplicates.next_token, "1")
        missing = await provider.fetch_page("mock://missing-fields")
        self.assertIsNone(missing.posts[0].author)
        self.assertEqual(missing.posts[0].metrics.likes, 0)
        invalid = await provider.fetch_page("mock://invalid-id")
        self.assertEqual(invalid.posts[0].id, "invalid-id")


if __name__ == "__main__":
    unittest.main()
