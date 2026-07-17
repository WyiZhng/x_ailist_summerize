from __future__ import annotations

import json
import os
import shutil
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from app.ingestion import IncrementalIngestionService
from app.storage import InMemoryPostStore
from app.x_list_summarizer import SessionVerificationState, XListFetcher


TEST_AUTH_TOKEN = "TEST_AUTH_TOKEN_DO_NOT_USE"
TEST_CT0 = "TEST_CT0_DO_NOT_USE"


class FakeBatch(list):
    def __init__(self, values=(), *, next_cursor=None):
        super().__init__(values)
        self.next_cursor = next_cursor


class FakeTweet:
    def __init__(self, tweet_id: str, processed: list[str] | None = None) -> None:
        self.id = tweet_id
        self.user = SimpleNamespace(screen_name=f"author_{tweet_id}")
        self._processed = processed
        self._legacy = {
            "entities": {"urls": []},
            "extended_entities": {"media": []},
        }

    @property
    def text(self) -> str:
        if self._processed is not None:
            self._processed.append(self.id)
        return f"post {self.id}"


class FakeTwikitClient:
    def __init__(self, *, batches=(), user_error: Exception | None = None) -> None:
        self.batches = list(batches)
        self.user_error = user_error
        self.user_calls = 0
        self.timeline_calls: list[dict[str, object]] = []
        self.list_calls = 0
        self.loaded_cookie_path: str | None = None
        self._base_headers = {}

    def load_cookies(self, path: str) -> None:
        self.loaded_cookie_path = path

    def get_cookies(self) -> dict[str, str]:
        return {"auth_token": TEST_AUTH_TOKEN, "ct0": TEST_CT0}

    async def user(self):
        self.user_calls += 1
        if self.user_error is not None:
            raise self.user_error
        return SimpleNamespace(screen_name="test-user")

    async def get_list(self, _list_id: str):
        self.list_calls += 1
        return SimpleNamespace(
            name="AI Coding / Agent",
            member_count=12,
            user=None,
            creator=None,
        )

    async def get_list_tweets(self, list_id: str, *, count: int, cursor=None):
        self.timeline_calls.append(
            {"list_id": list_id, "count": count, "cursor": cursor}
        )
        if not self.batches:
            return FakeBatch()
        value = self.batches.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    async def get(self, *_args, **_kwargs):
        return {}, None


class TwikitFetcherTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temporary = Path.cwd() / "tests" / f".tmp-twikit-{uuid.uuid4().hex}"
        self.temporary.mkdir(parents=True)
        self.cookies_path = self.temporary / "cookies.json"
        self.cookies_path.write_text(
            json.dumps({"auth_token": TEST_AUTH_TOKEN, "ct0": TEST_CT0}),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.temporary, ignore_errors=True)

    def fetcher(self, client: FakeTwikitClient) -> XListFetcher:
        fetcher = XListFetcher(cookies_path=self.cookies_path)
        fetcher.client = client
        return fetcher

    def test_https_proxy_is_passed_explicitly_to_twikit(self) -> None:
        with mock.patch.dict(os.environ, {"HTTPS_PROXY": "http://127.0.0.1:7897"}), \
             mock.patch("app.x_list_summarizer.Client") as client_class:
            XListFetcher(cookies_path=self.cookies_path)

        client_class.assert_called_once_with(
            "en-US", proxy="http://127.0.0.1:7897"
        )


class TwikitSessionVerificationTests(TwikitFetcherTestCase):
    async def test_successful_preflight_is_valid(self) -> None:
        client = FakeTwikitClient()
        fetcher = self.fetcher(client)

        with self.assertLogs("app.x_list_summarizer", level="INFO") as captured:
            success, message = await fetcher.login()

        self.assertTrue(success)
        self.assertIn("@test-user", message)
        self.assertEqual(fetcher.session_verification_state, SessionVerificationState.VALID)
        self.assertIn("session_verification=valid", "\n".join(captured.output))

    async def test_unauthorized_preflight_is_invalid_and_stops(self) -> None:
        client = FakeTwikitClient(user_error=RuntimeError("status: 401 unauthorized"))
        fetcher = self.fetcher(client)

        success, message = await fetcher.login()

        self.assertFalse(success)
        self.assertIn("401", message)
        self.assertEqual(fetcher.session_verification_state, SessionVerificationState.INVALID)
        self.assertEqual(client.timeline_calls, [])

    async def test_cloudflare_403_is_inconclusive_and_allows_timeline(self) -> None:
        client = FakeTwikitClient(
            user_error=RuntimeError(
                "status: 403 Attention Required! Sorry, you have been blocked by Cloudflare cf-ray"
            ),
            batches=[FakeBatch([FakeTweet("1")])],
        )
        fetcher = self.fetcher(client)

        with self.assertLogs("app.x_list_summarizer", level="INFO") as captured:
            success, message = await fetcher.login()
            posts = await fetcher.fetch_list_tweets("123", max_tweets=1)

        logs = "\n".join(captured.output)
        self.assertTrue(success)
        self.assertIn("Cloudflare", message)
        self.assertEqual(
            fetcher.session_verification_state,
            SessionVerificationState.INCONCLUSIVE,
        )
        self.assertEqual(len(posts), 1)
        self.assertIn("session_verification=inconclusive", logs)
        self.assertIn("timeline_fetch=success", logs)

    async def test_unrecognized_403_is_not_ignored(self) -> None:
        client = FakeTwikitClient(user_error=RuntimeError("status: 403 forbidden"))
        fetcher = self.fetcher(client)

        success, message = await fetcher.login()

        self.assertFalse(success)
        self.assertIn("forbidden", message.lower())
        self.assertEqual(fetcher.session_verification_state, SessionVerificationState.INVALID)

    async def test_rate_limited_preflight_does_not_retry(self) -> None:
        client = FakeTwikitClient(user_error=RuntimeError("status: 429 rate limit"))
        fetcher = self.fetcher(client)

        success, message = await fetcher.verify_session(retries=5)

        self.assertFalse(success)
        self.assertIn("429", message)
        self.assertEqual(client.user_calls, 1)
        self.assertEqual(
            fetcher.session_verification_state,
            SessionVerificationState.INCONCLUSIVE,
        )

    async def test_inconclusive_preflight_then_timeline_401_fails(self) -> None:
        client = FakeTwikitClient(
            user_error=RuntimeError("403 Cloudflare Attention Required"),
            batches=[RuntimeError("status: 401 unauthorized")],
        )
        fetcher = self.fetcher(client)

        success, _ = await fetcher.login()
        self.assertTrue(success)
        with self.assertLogs("app.x_list_summarizer", level="WARNING") as captured:
            with self.assertRaisesRegex(Exception, "unauthorized"):
                await fetcher.fetch_list_tweets("123", max_tweets=1)
        self.assertIn("timeline_fetch=unauthorized", "\n".join(captured.output))

    async def test_inconclusive_preflight_then_timeline_403_fails(self) -> None:
        client = FakeTwikitClient(
            user_error=RuntimeError("403 Cloudflare Attention Required"),
            batches=[RuntimeError("status: 403 forbidden")],
        )
        fetcher = self.fetcher(client)

        success, _ = await fetcher.login()
        self.assertTrue(success)
        with self.assertLogs("app.x_list_summarizer", level="WARNING") as captured:
            with self.assertRaisesRegex(Exception, "forbidden"):
                await fetcher.fetch_list_tweets("123", max_tweets=1)
        self.assertIn("timeline_fetch=forbidden", "\n".join(captured.output))

    async def test_session_logs_and_messages_redact_cookie_values(self) -> None:
        client = FakeTwikitClient(
            user_error=RuntimeError(
                f"403 Cloudflare auth_token={TEST_AUTH_TOKEN} ct0={TEST_CT0}"
            )
        )
        fetcher = self.fetcher(client)

        with self.assertLogs("app.x_list_summarizer", level="INFO") as captured:
            _success, message = await fetcher.login()

        combined = message + "\n" + "\n".join(captured.output)
        self.assertNotIn(TEST_AUTH_TOKEN, combined)
        self.assertNotIn(TEST_CT0, combined)


class TwikitTweetLimitTests(TwikitFetcherTestCase):
    async def test_oversized_batch_processes_only_one_for_limit_one(self) -> None:
        processed: list[str] = []
        batch = FakeBatch(
            [FakeTweet(str(index), processed) for index in range(78)],
            next_cursor="unused-next-page",
        )
        client = FakeTwikitClient(batches=[batch])
        fetcher = self.fetcher(client)

        posts = await fetcher.fetch_list_tweets("123", max_tweets=1)

        self.assertEqual(len(posts), 1)
        self.assertEqual(processed, ["0"])
        self.assertEqual(len(client.timeline_calls), 1)
        self.assertEqual(client.timeline_calls[0]["count"], 1)
        self.assertEqual(fetcher.last_pagination_count, 0)

    async def test_oversized_batch_respects_limit_five(self) -> None:
        processed: list[str] = []
        client = FakeTwikitClient(
            batches=[FakeBatch([FakeTweet(str(index), processed) for index in range(78)])]
        )
        fetcher = self.fetcher(client)

        posts = await fetcher.fetch_list_tweets("123", max_tweets=5)

        self.assertEqual(len(posts), 5)
        self.assertEqual(processed, ["0", "1", "2", "3", "4"])
        self.assertEqual(len(client.timeline_calls), 1)

    async def test_short_first_page_requests_second_page(self) -> None:
        client = FakeTwikitClient(
            batches=[
                FakeBatch([FakeTweet("1"), FakeTweet("2")], next_cursor="page-2"),
                FakeBatch([FakeTweet("3"), FakeTweet("4"), FakeTweet("5")]),
            ]
        )
        fetcher = self.fetcher(client)

        posts = await fetcher.fetch_list_tweets("123", max_tweets=5)

        self.assertEqual(len(posts), 5)
        self.assertEqual(len(client.timeline_calls), 2)
        self.assertEqual(client.timeline_calls[1]["cursor"], "page-2")
        self.assertEqual(client.timeline_calls[1]["count"], 3)
        self.assertEqual(fetcher.last_pagination_count, 1)

    async def test_full_first_page_does_not_request_second_page(self) -> None:
        client = FakeTwikitClient(
            batches=[
                FakeBatch(
                    [FakeTweet(str(index)) for index in range(10)],
                    next_cursor="unused-page-2",
                ),
                FakeBatch([FakeTweet("never-requested")]),
            ]
        )
        fetcher = self.fetcher(client)

        posts = await fetcher.fetch_list_tweets("123", max_tweets=5)

        self.assertEqual(len(posts), 5)
        self.assertEqual(len(client.timeline_calls), 1)

    async def test_zero_and_negative_limits_do_not_request_x(self) -> None:
        client = FakeTwikitClient(batches=[FakeBatch([FakeTweet("1")])])
        fetcher = self.fetcher(client)

        self.assertEqual(await fetcher.fetch_list_tweets("123", max_tweets=0), [])
        self.assertEqual(await fetcher.fetch_list_tweets("123", max_tweets=-1), [])
        self.assertEqual(client.list_calls, 0)
        self.assertEqual(client.timeline_calls, [])

    async def test_duplicate_items_still_cannot_exceed_limit(self) -> None:
        duplicate = FakeTweet("same")
        client = FakeTwikitClient(
            batches=[FakeBatch([duplicate, duplicate, duplicate], next_cursor="unused")]
        )
        fetcher = self.fetcher(client)

        posts = await fetcher.fetch_list_tweets("123", max_tweets=2)

        self.assertEqual([post["id"] for post in posts], ["same", "same"])
        self.assertEqual(len(client.timeline_calls), 1)

    async def test_each_list_call_has_its_own_limit(self) -> None:
        client = FakeTwikitClient(
            batches=[
                FakeBatch([FakeTweet("1"), FakeTweet("2"), FakeTweet("3")]),
                FakeBatch([FakeTweet("4"), FakeTweet("5"), FakeTweet("6")]),
            ]
        )
        fetcher = self.fetcher(client)

        first = await fetcher.fetch_list_tweets("111", max_tweets=2)
        second = await fetcher.fetch_list_tweets("222", max_tweets=2)

        self.assertEqual(len(first), 2)
        self.assertEqual(len(second), 2)
        self.assertEqual([call["count"] for call in client.timeline_calls], [2, 2])

    async def test_page_normalization_stops_before_processing_oversized_batch(self) -> None:
        processed: list[str] = []
        batch = FakeBatch(
            [FakeTweet(str(index), processed) for index in range(78, 0, -1)],
            next_cursor="next-page",
        )
        client = FakeTwikitClient(batches=[batch])
        fetcher = self.fetcher(client)
        fetcher._session_ready = True

        page = await fetcher.fetch_page("123", max_results=5)

        self.assertEqual(5, len(page.posts))
        self.assertEqual(["78", "77", "76", "75", "74"], processed)
        self.assertEqual(1, len(client.timeline_calls))
        self.assertEqual("next-page", page.next_token)

    async def test_two_incremental_runs_checkpoint_and_deduplicate_twikit_posts(self) -> None:
        first_batch = FakeBatch(
            [FakeTweet("103"), FakeTweet("102"), FakeTweet("101")]
        )
        second_batch = FakeBatch(
            [FakeTweet("103"), FakeTweet("102"), FakeTweet("101")]
        )
        fetcher = self.fetcher(FakeTwikitClient(batches=[first_batch, second_batch]))
        fetcher._session_ready = True
        store = InMemoryPostStore()
        service = IncrementalIngestionService(
            provider=fetcher,
            store=store,
            page_size=20,
            initial_fetch_limit=20,
        )

        first = await service.ingest(["123"], run_id="twikit-first", max_tweets=20)
        checkpoint = store.get_sync_state("123")
        second = await service.ingest(["123"], run_id="twikit-second", max_tweets=20)
        checkpoint_after = store.get_sync_state("123")

        self.assertEqual("success", first.status)
        self.assertEqual(3, first.run.new_post_count)
        self.assertEqual("103", checkpoint.newest_post_id)
        self.assertEqual("no_new_posts", second.status)
        self.assertEqual(0, second.run.new_post_count)
        self.assertEqual("103", checkpoint_after.newest_post_id)
        self.assertEqual(3, len(store.read_posts()))


if __name__ == "__main__":
    unittest.main()
