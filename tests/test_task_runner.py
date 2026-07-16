from __future__ import annotations

import asyncio
import contextlib
import io
import json
import shutil
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from app.providers import (
    MockReportGenerator,
    MockSummaryProvider,
    MockXIngestionProvider,
    MockXProvider,
    ReportGenerator,
    SummaryProvider,
    XProvider,
)
from app.storage import InMemoryPostStore
from app.task_runner import (
    DigestTaskRequest,
    DigestTaskRunner,
    TaskProgress,
    legacy_aggregate_by_links,
    main,
)


@contextlib.contextmanager
def workspace_temporary_directory():
    """Use normal workspace permissions instead of Windows tempfile's 0700 ACL."""

    parent = Path(__file__).resolve().parent / ".task_runner_tmp"
    parent.mkdir(parents=True, exist_ok=True)
    directory = parent / uuid.uuid4().hex
    directory.mkdir()
    try:
        yield directory
    finally:
        shutil.rmtree(directory)
        try:
            parent.rmdir()
        except OSError:
            pass


class DigestTaskRunnerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temporary = workspace_temporary_directory()
        self.output_dir = self.temporary.__enter__()
        self.addCleanup(self.temporary.__exit__, None, None, None)

    async def test_mock_provider_covers_required_content_shapes(self) -> None:
        provider = MockXProvider()
        self.assertIsInstance(provider, XProvider)
        self.assertIsInstance(MockSummaryProvider(), SummaryProvider)
        self.assertIsInstance(MockReportGenerator(), ReportGenerator)

        tweets = await provider.fetch_list_tweets("mock://mixed")
        languages = {tweet.get("language") for tweet in tweets}
        self.assertEqual({"zh", "en"}, languages)
        self.assertTrue(any(tweet.get("links") for tweet in tweets))
        self.assertTrue(any(tweet.get("is_retweet") for tweet in tweets))
        self.assertTrue(any(tweet.get("is_quote") for tweet in tweets))
        self.assertTrue(any(not tweet.get("links") for tweet in tweets))
        self.assertTrue(any(tweet.get("hostile_html") for tweet in tweets))
        ids = [tweet["id"] for tweet in tweets]
        self.assertLess(len(set(ids)), len(ids), "fixture must contain an intentional duplicate")

    async def test_success_preserves_legacy_aggregation_and_writes_safe_mock_report(self) -> None:
        events: list[TaskProgress] = []
        x_provider = MockXProvider()
        summary_provider = MockSummaryProvider()
        report_generator = MockReportGenerator()
        runner = DigestTaskRunner(
            x_provider=x_provider,
            summary_provider=summary_provider,
            report_generator=report_generator,
            progress_callback=events.append,
        )
        request = DigestTaskRequest.from_values(
            ["mock://mixed"],
            output_dir=self.output_dir,
            ai_model="mock-bilingual",
        )

        result = await runner.run(request)

        self.assertTrue(result.success)
        self.assertEqual("succeeded", result.status)
        self.assertEqual(7, result.tweet_count)
        self.assertEqual(4, result.link_count)
        self.assertEqual(1, len(result.aggregated["no_links"]))
        article = "https://example.com/research/agent-systems"
        groups = dict(result.aggregated["by_link"])
        self.assertEqual(3, len(groups[article]), "phase 0 must not silently change duplicate semantics")
        self.assertTrue(any(tweet.get("is_retweet") for tweet in groups[article]))
        self.assertIn("中文", json.dumps(result.tweets, ensure_ascii=False))
        self.assertIsNotNone(result.report_path)
        self.assertTrue(result.report_path.exists())
        self.assertIsNotNone(result.history_path)
        self.assertTrue(result.history_path.exists())

        rendered = result.report_path.read_text(encoding="utf-8")
        self.assertNotIn("<script>window.__mock_xss", rendered)
        self.assertIn("&lt;script&gt;window.__mock_xss", rendered)
        self.assertIn("中文", rendered)

        history = json.loads(result.history_path.read_text(encoding="utf-8"))
        self.assertEqual("succeeded", history[result.report_path.name]["status"])
        self.assertEqual(7, history[result.report_path.name]["tweets"])

        percentages = [event.percent for event in events]
        self.assertEqual(sorted(percentages), percentages)
        self.assertEqual(0, percentages[0])
        self.assertEqual(100, percentages[-1])
        self.assertEqual("complete", events[-1].phase)

    async def test_partial_list_failure_still_generates_report_and_history(self) -> None:
        events: list[TaskProgress] = []
        runner = DigestTaskRunner(
            x_provider=MockXProvider(fail_lists=["mock://broken"], latency=0.001),
            summary_provider=MockSummaryProvider(),
            report_generator=MockReportGenerator(),
            progress_callback=events.append,
            max_concurrency=2,
        )

        result = await runner.run(
            DigestTaskRequest.from_values(
                ["mock://mixed", "mock://broken"],
                output_dir=self.output_dir,
            )
        )

        self.assertTrue(result.success)
        self.assertTrue(result.partial)
        self.assertEqual("partial", result.status)
        self.assertEqual(("mock://mixed",), result.completed_lists)
        self.assertEqual(1, len(result.list_failures))
        self.assertEqual("mock://broken", result.list_failures[0].list_url)
        self.assertIn("Mock list failure", result.list_failures[0].error)
        self.assertTrue(result.report_path and result.report_path.exists())
        self.assertTrue(result.history_path and result.history_path.exists())
        history = json.loads(result.history_path.read_text(encoding="utf-8"))
        entry = history[result.report_path.name]
        self.assertEqual("partial", entry["status"])
        self.assertEqual("mock://broken", entry["failed_lists"][0]["list_url"])

        fetch_events = [event for event in events if event.phase == "fetching"]
        self.assertTrue(any(event.completed_lists == 1 for event in fetch_events))
        self.assertTrue(any(event.completed_lists == 2 for event in fetch_events))
        self.assertEqual(1, max(event.failed_lists for event in fetch_events))

    async def test_injected_fetch_delay_policy_is_forwarded_without_slowing_tests(self) -> None:
        class DelayRecorder(MockXProvider):
            def __init__(self) -> None:
                super().__init__()
                self.delays: list[float] = []

            async def fetch_list_tweets(
                self,
                list_url_or_id: str,
                max_tweets: int = 100,
                delay: float = 0,
            ) -> list[dict[str, Any]]:
                self.delays.append(delay)
                return await super().fetch_list_tweets(
                    list_url_or_id,
                    max_tweets=max_tweets,
                    delay=0,
                )

        provider = DelayRecorder()
        runner = DigestTaskRunner(
            x_provider=provider,
            summary_provider=MockSummaryProvider(),
            report_generator=MockReportGenerator(),
            fetch_delay_factory=lambda index: index * 0.5,
        )
        result = await runner.run(
            DigestTaskRequest.from_values(
                ["mock://mixed", "mock://secondary"],
                output_dir=self.output_dir,
            )
        )

        self.assertTrue(result.success)
        self.assertEqual([0.0, 0.5], provider.delays)

    async def test_all_list_failures_return_failed_dataclass(self) -> None:
        events: list[TaskProgress] = []
        runner = DigestTaskRunner(
            x_provider=MockXProvider(fail_lists=["mock://broken"]),
            summary_provider=MockSummaryProvider(),
            report_generator=MockReportGenerator(),
            progress_callback=events.append,
        )

        result = await runner.run(
            DigestTaskRequest.from_values(["mock://broken"], output_dir=self.output_dir)
        )

        self.assertFalse(result.success)
        self.assertEqual("failed", result.status)
        self.assertEqual(1, len(result.list_failures))
        self.assertIn("Mock list failure", result.error or "")
        self.assertIsNone(result.report_path)
        self.assertIsNone(result.history_path)
        self.assertEqual("failed", events[-1].phase)

    async def test_run_id_is_propagated_and_provider_secrets_are_redacted(self) -> None:
        secret = "provider-secret-token-123456789"
        events: list[TaskProgress] = []

        class LeakyProvider(MockXProvider):
            async def fetch_list_tweets(
                self,
                list_url_or_id: str,
                max_tweets: int = 100,
                delay: float = 0,
            ) -> list[dict[str, Any]]:
                raise RuntimeError(f"Authorization: Bearer {secret}")

        runner = DigestTaskRunner(
            x_provider=LeakyProvider(),
            summary_provider=MockSummaryProvider(),
            report_generator=MockReportGenerator(),
            progress_callback=events.append,
            run_id_factory=lambda: "run-safe-123",
            sensitive_values=(secret,),
        )
        with self.assertLogs("app.task_runner", level="ERROR") as captured:
            result = await runner.run(
                DigestTaskRequest.from_values(["mock://mixed"], output_dir=self.output_dir)
            )

        self.assertFalse(result.success)
        self.assertEqual(result.run_id, "run-safe-123")
        self.assertTrue(events)
        self.assertTrue(all(event.run_id == "run-safe-123" for event in events))
        self.assertNotIn(secret, result.error or "")
        self.assertNotIn(secret, result.list_failures[0].error)
        self.assertNotIn(secret, "\n".join(captured.output))
        self.assertIn("[REDACTED]", result.error or "")

    async def test_llm_exception_and_legacy_error_string_fail_cleanly(self) -> None:
        for return_error_string in (False, True):
            with self.subTest(return_error_string=return_error_string):
                subdir = self.output_dir / str(return_error_string)
                runner = DigestTaskRunner(
                    x_provider=MockXProvider(),
                    summary_provider=MockSummaryProvider(
                        fail=True,
                        return_error_string=return_error_string,
                    ),
                    report_generator=MockReportGenerator(),
                )
                result = await runner.run(
                    DigestTaskRequest.from_values(["mock://mixed"], output_dir=subdir)
                )
                self.assertFalse(result.success)
                self.assertEqual("failed", result.status)
                self.assertIn("mock", (result.error or "").lower())
                self.assertIsNone(result.report_path)
                self.assertFalse((subdir / "history.json").exists())

    async def test_no_link_digest_is_valid(self) -> None:
        runner = DigestTaskRunner(
            x_provider=MockXProvider(),
            summary_provider=MockSummaryProvider(),
            report_generator=MockReportGenerator(),
        )
        result = await runner.run(
            DigestTaskRequest.from_values(["mock://no-links"], output_dir=self.output_dir)
        )
        self.assertTrue(result.success)
        self.assertEqual(0, result.link_count)
        self.assertEqual(1, len(result.aggregated["no_links"]))
        self.assertEqual("No shared links in this mock digest.", result.summary)

    async def test_legacy_fetcher_supplies_aggregator_and_report_generator(self) -> None:
        class LegacyFetcher:
            def __init__(self) -> None:
                self.aggregate_called = False
                self.report_called = False
                self.list_info = {
                    "name": "Legacy",
                    "owner": "legacy-owner",
                    "member_count": 1,
                }

            async def login(self) -> tuple[bool, str]:
                return True, "OK"

            async def fetch_list_tweets(
                self,
                list_url_or_id: str,
                max_tweets: int = 100,
                delay: float = 0,
            ) -> list[dict[str, Any]]:
                return [
                    {
                        "id": "legacy-1",
                        "text": "legacy",
                        "author": "legacy",
                        "links": [],
                        "likes": 0,
                        "retweets": 0,
                        "replies": 0,
                        "quotes": 0,
                        "bookmarks": 0,
                    }
                ]

            def aggregate_by_links(self, tweets: list[dict[str, Any]]) -> dict[str, Any]:
                self.aggregate_called = True
                return {"by_link": [], "no_links": tweets, "legacy_marker": True}

            def generate_html_report(
                self,
                aggregated: dict[str, Any],
                ai_summary: str,
                output_path: Path,
                tweet_count: int = 0,
                ai_model: str = "",
            ) -> None:
                self.report_called = True
                Path(output_path).write_text("legacy report", encoding="utf-8")

        class LegacySummary:
            def summarize(self, aggregated_data: dict[str, Any]) -> str:
                self.aggregated = aggregated_data
                return "legacy summary"

        fetcher = LegacyFetcher()
        summary = LegacySummary()
        runner = DigestTaskRunner(fetcher=fetcher, summary_provider=summary)
        result = await runner.run(
            DigestTaskRequest.from_values(["legacy-list"], output_dir=self.output_dir)
        )

        self.assertTrue(result.success)
        self.assertTrue(fetcher.aggregate_called)
        self.assertTrue(fetcher.report_called)
        self.assertTrue(result.aggregated["legacy_marker"])
        self.assertIs(summary.aggregated, result.aggregated)

    async def test_factories_and_async_progress_callback_are_supported(self) -> None:
        events: list[TaskProgress] = []
        x_provider = MockXProvider()
        summary_provider = MockSummaryProvider()
        report_generator = MockReportGenerator()
        report_factory_dependency: list[Any] = []

        async def progress(event: TaskProgress) -> None:
            await asyncio.sleep(0)
            events.append(event)

        def report_factory(provider: Any) -> MockReportGenerator:
            report_factory_dependency.append(provider)
            return report_generator

        runner = DigestTaskRunner(
            fetcher_factory=lambda: x_provider,
            summary_provider_factory=lambda: summary_provider,
            report_generator_factory=report_factory,
            progress_callback=progress,
        )
        result = await runner.run(
            list_urls=["mock://mixed"],
            output_dir=self.output_dir,
        )

        self.assertTrue(result.success)
        self.assertEqual([x_provider], report_factory_dependency)
        self.assertTrue(events)
        self.assertEqual(100, events[-1].percent)

    async def test_in_process_concurrent_history_updates_are_not_lost(self) -> None:
        first = DigestTaskRunner(
            x_provider=MockXProvider(),
            summary_provider=MockSummaryProvider(),
            report_generator=MockReportGenerator(),
        )
        second = DigestTaskRunner(
            x_provider=MockXProvider(),
            summary_provider=MockSummaryProvider(),
            report_generator=MockReportGenerator(),
        )

        first_result, second_result = await asyncio.gather(
            first.run(
                DigestTaskRequest.from_values(
                    ["mock://mixed"],
                    output_dir=self.output_dir,
                    report_filename="first.html",
                )
            ),
            second.run(
                DigestTaskRequest.from_values(
                    ["mock://secondary"],
                    output_dir=self.output_dir,
                    report_filename="second.html",
                )
            ),
        )

        self.assertTrue(first_result.success)
        self.assertTrue(second_result.success)
        history = json.loads((self.output_dir / "history.json").read_text(encoding="utf-8"))
        self.assertEqual({"first.html", "second.html"}, set(history))

    async def test_corrupt_history_is_rebuilt_without_orphaning_report(self) -> None:
        history_path = self.output_dir / "history.json"
        history_path.write_text('{"broken": ', encoding="utf-8")
        runner = DigestTaskRunner(
            x_provider=MockXProvider(),
            summary_provider=MockSummaryProvider(),
            report_generator=MockReportGenerator(),
        )

        with self.assertLogs("app.task_runner", level="WARNING"):
            result = await runner.run(
                DigestTaskRequest.from_values(
                    ["mock://mixed"],
                    output_dir=self.output_dir,
                )
            )

        self.assertTrue(result.success, result.error)
        self.assertTrue(result.report_path and result.report_path.exists())
        rebuilt = json.loads(history_path.read_text(encoding="utf-8"))
        self.assertIn(result.report_path.name, rebuilt)

    async def test_existing_report_is_not_overwritten(self) -> None:
        existing = self.output_dir / "existing.html"
        existing.write_text("keep-me", encoding="utf-8")
        runner = DigestTaskRunner(
            x_provider=MockXProvider(),
            summary_provider=MockSummaryProvider(),
            report_generator=MockReportGenerator(),
        )

        result = await runner.run(
            DigestTaskRequest.from_values(
                ["mock://mixed"],
                output_dir=self.output_dir,
                report_filename=existing.name,
            )
        )

        self.assertFalse(result.success)
        self.assertIn("Refusing to overwrite", result.error or "")
        self.assertEqual(existing.read_text(encoding="utf-8"), "keep-me")

    def test_request_rejects_report_path_traversal(self) -> None:
        for filename in ("../outside.html", "report with spaces.html", "report.htm"):
            with self.subTest(filename=filename):
                with self.assertRaises(ValueError):
                    DigestTaskRequest.from_values(
                        ["mock://mixed"],
                        output_dir=self.output_dir,
                        report_filename=filename,
                    )

    def test_request_and_runner_reject_non_integer_limits(self) -> None:
        for invalid in (True, 1.5, "10", 0, -1):
            with self.subTest(max_tweets=invalid):
                with self.assertRaises(ValueError):
                    DigestTaskRequest.from_values(
                        ["mock://mixed"],
                        output_dir=self.output_dir,
                        max_tweets=invalid,
                    )

        for invalid in (True, 1.5, "4", 0, -1):
            with self.subTest(max_concurrency=invalid):
                with self.assertRaises(ValueError):
                    DigestTaskRunner(
                        x_provider=MockXProvider(),
                        summary_provider=MockSummaryProvider(),
                        report_generator=MockReportGenerator(),
                        max_concurrency=invalid,
                    )

    def test_request_rejects_reserved_or_unserializable_history_metadata(self) -> None:
        with self.assertRaisesRegex(ValueError, "reserved fields"):
            DigestTaskRequest.from_values(
                ["mock://mixed"],
                output_dir=self.output_dir,
                history_metadata={"status": "forged"},
            )
        with self.assertRaisesRegex(ValueError, "JSON-serializable"):
            DigestTaskRequest.from_values(
                ["mock://mixed"],
                output_dir=self.output_dir,
                history_metadata={"custom": object()},
            )

    def test_fallback_ranking_keeps_existing_author_cap_order(self) -> None:
        def tweet(tweet_id: str, author: str, link: str, likes: int) -> dict[str, Any]:
            return {
                "id": tweet_id,
                "text": tweet_id,
                "author": author,
                "links": [link],
                "likes": likes,
                "retweets": 0,
                "replies": 0,
                "quotes": 0,
                "bookmarks": 0,
            }

        tweets = [
            tweet("1", "solo", "https://one.example", 30),
            tweet("2", "solo", "https://two.example", 20),
            tweet("3", "solo", "https://three.example", 10),
            tweet("4", "a", "https://community.example", 1),
            tweet("5", "b", "https://community.example", 1),
        ]
        links = [link for link, _ in legacy_aggregate_by_links(tweets)["by_link"]]
        self.assertEqual(
            [
                "https://one.example",
                "https://two.example",
                "https://community.example",
                "https://three.example",
            ],
            links,
        )

    async def test_incremental_posts_enter_existing_summary_report_and_history(self) -> None:
        store = InMemoryPostStore()
        summary_provider = MockSummaryProvider()
        report_generator = MockReportGenerator()
        runner = DigestTaskRunner(
            x_provider=MockXIngestionProvider(),
            summary_provider=summary_provider,
            report_generator=report_generator,
            post_store=store,
        )
        request = DigestTaskRequest.from_values(
            ["mock://mixed"],
            output_dir=self.output_dir,
            data_dir=self.output_dir / "data",
            incremental=True,
            page_size=2,
        )

        result = await runner.run(request)

        self.assertEqual(result.status, "succeeded")
        self.assertEqual(result.new_post_count, 5)
        self.assertEqual(result.tweet_count, 5)
        self.assertEqual(len(summary_provider.calls), 1)
        self.assertEqual(len(report_generator.calls), 1)
        self.assertTrue(result.report_path.is_file())
        self.assertTrue(result.history_path.is_file())
        self.assertEqual(store.get_run(result.run_id).report_status, "succeeded")

    async def test_incremental_partial_and_all_failed_statuses_are_persisted(self) -> None:
        partial_store = InMemoryPostStore()
        partial_runner = DigestTaskRunner(
            x_provider=MockXIngestionProvider(
                page_failures={
                    ("mock://empty", 1): RuntimeError("injected list failure")
                }
            ),
            summary_provider=MockSummaryProvider(),
            report_generator=MockReportGenerator(),
            post_store=partial_store,
        )
        partial = await partial_runner.run(
            DigestTaskRequest.from_values(
                ["mock://mixed", "mock://empty"],
                output_dir=self.output_dir / "partial",
                incremental=True,
            )
        )

        self.assertEqual(partial.status, "partial")
        self.assertEqual(partial.ingestion_status, "partial_success")
        self.assertTrue(partial.report_path and partial.report_path.is_file())
        self.assertEqual(partial_store.get_run(partial.run_id).status, "partial_success")

        failed_store = InMemoryPostStore()
        failed_runner = DigestTaskRunner(
            x_provider=MockXIngestionProvider(
                page_failures={
                    ("mock://empty", 1): RuntimeError("injected list failure")
                }
            ),
            summary_provider=MockSummaryProvider(),
            report_generator=MockReportGenerator(),
            post_store=failed_store,
        )
        failed = await failed_runner.run(
            DigestTaskRequest.from_values(
                ["mock://empty"],
                output_dir=self.output_dir / "failed",
                incremental=True,
            )
        )

        self.assertEqual(failed.status, "failed")
        self.assertEqual(failed.ingestion_status, "failed")
        self.assertIsNone(failed.report_path)
        self.assertEqual(failed_store.get_run(failed.run_id).status, "failed")

    async def test_incremental_progress_and_canonical_provider_skip_legacy_login(self) -> None:
        class CanonicalProvider(MockXIngestionProvider):
            async def login(self):
                raise AssertionError("canonical provider must not use a legacy login probe")

        events: list[TaskProgress] = []
        runner = DigestTaskRunner(
            x_provider=CanonicalProvider(),
            summary_provider=MockSummaryProvider(),
            report_generator=MockReportGenerator(),
            progress_callback=events.append,
            post_store=InMemoryPostStore(),
        )

        result = await runner.run(
            DigestTaskRequest.from_values(
                ["mock://mixed"],
                output_dir=self.output_dir,
                incremental=True,
                page_size=2,
            )
        )

        self.assertTrue(result.success)
        ingestion_events = [event for event in events if event.phase == "ingesting"]
        self.assertGreaterEqual(len(ingestion_events), 1)
        self.assertTrue(any("page" in event.message for event in ingestion_events))
        self.assertTrue(any("new" in event.message for event in ingestion_events))

    async def test_no_new_posts_skips_llm_report_and_history(self) -> None:
        store = InMemoryPostStore()
        first = DigestTaskRunner(
            x_provider=MockXIngestionProvider(),
            summary_provider=MockSummaryProvider(),
            report_generator=MockReportGenerator(),
            post_store=store,
        )
        request = DigestTaskRequest.from_values(
            ["mock://mixed"], output_dir=self.output_dir, incremental=True
        )
        first_result = await first.run(request)
        history_before = first_result.history_path.read_bytes()
        summary_provider = MockSummaryProvider()
        report_generator = MockReportGenerator()
        second = DigestTaskRunner(
            x_provider=MockXIngestionProvider(),
            summary_provider=summary_provider,
            report_generator=report_generator,
            post_store=store,
        )

        result = await second.run(request)

        self.assertEqual(result.status, "no_new_posts")
        self.assertTrue(result.success)
        self.assertEqual(result.new_post_count, 0)
        self.assertEqual(summary_provider.calls, [])
        self.assertEqual(report_generator.calls, [])
        self.assertIsNone(result.report_path)
        self.assertEqual(first_result.history_path.read_bytes(), history_before)

    async def test_report_failure_keeps_checkpoint_and_resumes_without_x_call(self) -> None:
        class FailingReport:
            def generate_html_report(self, *_args, **_kwargs):
                raise RuntimeError("injected report failure")

        store = InMemoryPostStore()
        provider = MockXIngestionProvider()
        failing = DigestTaskRunner(
            x_provider=provider,
            summary_provider=MockSummaryProvider(),
            report_generator=FailingReport(),
            post_store=store,
        )
        request = DigestTaskRequest.from_values(
            ["mock://mixed"], output_dir=self.output_dir, incremental=True
        )

        failed = await failing.run(request)

        self.assertEqual(failed.status, "failed")
        self.assertEqual(store.get_sync_state("mock://mixed").newest_post_id, "9005")
        stored_run = store.get_run(failed.run_id)
        self.assertEqual(stored_run.status, "success")
        self.assertEqual(stored_run.report_status, "failed")
        x_calls = len(provider.fetch_calls)

        retry = DigestTaskRunner(
            summary_provider=MockSummaryProvider(),
            report_generator=MockReportGenerator(),
            post_store=store,
        )
        retry_request = DigestTaskRequest.from_values(
            ["mock://mixed"],
            output_dir=self.output_dir,
            incremental=True,
            resume_ingestion_run_id=failed.run_id,
        )
        retried = await retry.run(retry_request)

        self.assertEqual(retried.status, "succeeded")
        self.assertTrue(retried.report_path.is_file())
        self.assertEqual(len(provider.fetch_calls), x_calls)
        self.assertEqual(store.get_run(failed.run_id).report_status, "succeeded")

        repeated = await retry.run(retry_request)
        self.assertEqual(repeated.status, "failed")
        self.assertIn("already complete", repeated.error or "")
        self.assertEqual(store.get_run(failed.run_id).report_status, "succeeded")

    async def test_only_one_process_can_claim_the_same_report(self) -> None:
        store = InMemoryPostStore()
        fetched = await DigestTaskRunner(
            x_provider=MockXIngestionProvider(),
            post_store=store,
        ).run(
            DigestTaskRequest.from_values(
                ["mock://mixed"],
                output_dir=self.output_dir,
                incremental=True,
                fetch_only=True,
            )
        )
        started = asyncio.Event()
        release = asyncio.Event()

        class BlockingSummary:
            async def summarize(self, _aggregated):
                started.set()
                await release.wait()
                return "claimed summary"

        request = DigestTaskRequest.from_values(
            ["mock://mixed"],
            output_dir=self.output_dir,
            incremental=True,
            resume_ingestion_run_id=fetched.run_id,
        )
        first_runner = DigestTaskRunner(
            summary_provider=BlockingSummary(),
            report_generator=MockReportGenerator(),
            post_store=store,
        )
        second_runner = DigestTaskRunner(
            summary_provider=MockSummaryProvider(),
            report_generator=MockReportGenerator(),
            post_store=store,
        )

        first_task = asyncio.create_task(first_runner.run(request))
        await asyncio.wait_for(started.wait(), timeout=2)
        second = await second_runner.run(request)
        self.assertEqual(second.status, "failed")
        self.assertIn("in progress", second.error or "")
        self.assertEqual(store.get_run(fetched.run_id).report_status, "generating")

        release.set()
        first = await first_task
        self.assertEqual(first.status, "succeeded")
        self.assertEqual(store.get_run(fetched.run_id).report_status, "succeeded")

    async def test_replaced_report_owner_cannot_publish_history_or_success(self) -> None:
        store = InMemoryPostStore()
        fetched = await DigestTaskRunner(
            x_provider=MockXIngestionProvider(),
            post_store=store,
        ).run(
            DigestTaskRequest.from_values(
                ["mock://mixed"],
                output_dir=self.output_dir,
                incremental=True,
                fetch_only=True,
            )
        )
        old_started = asyncio.Event()
        release_old = asyncio.Event()
        new_started = asyncio.Event()

        class OldBlockingSummary:
            async def summarize(self, _aggregated):
                old_started.set()
                await release_old.wait()
                return "old owner summary"

        class NewBlockingSummary:
            async def summarize(self, _aggregated):
                new_started.set()
                await asyncio.Event().wait()

        request = DigestTaskRequest.from_values(
            ["mock://mixed"],
            output_dir=self.output_dir,
            incremental=True,
            resume_ingestion_run_id=fetched.run_id,
        )
        stale_clock = datetime.now(timezone.utc) - timedelta(hours=2)
        old_task = asyncio.create_task(
            DigestTaskRunner(
                summary_provider=OldBlockingSummary(),
                report_generator=MockReportGenerator(),
                post_store=store,
                now_factory=lambda: stale_clock,
            ).run(request)
        )
        await asyncio.wait_for(old_started.wait(), timeout=2)
        old_claim = store.get_run(fetched.run_id).report_claim_id
        self.assertTrue(old_claim)

        new_task = asyncio.create_task(
            DigestTaskRunner(
                summary_provider=NewBlockingSummary(),
                report_generator=MockReportGenerator(),
                post_store=store,
            ).run(request)
        )
        try:
            await asyncio.wait_for(new_started.wait(), timeout=2)
            new_claim = store.get_run(fetched.run_id).report_claim_id
            self.assertTrue(new_claim)
            self.assertNotEqual(old_claim, new_claim)

            release_old.set()
            old_result = await asyncio.wait_for(old_task, timeout=2)

            current = store.get_run(fetched.run_id)
            self.assertEqual(old_result.status, "failed")
            self.assertEqual(current.report_status, "generating")
            self.assertEqual(current.report_claim_id, new_claim)
            self.assertFalse((self.output_dir / "history.json").exists())
        finally:
            if not old_task.done():
                old_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await old_task
            if not new_task.done():
                new_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await new_task

    async def test_fixed_report_takeover_preserves_new_owner_artifacts(self) -> None:
        store = InMemoryPostStore()
        fetched = await DigestTaskRunner(
            x_provider=MockXIngestionProvider(),
            post_store=store,
        ).run(
            DigestTaskRequest.from_values(
                ["mock://mixed"],
                output_dir=self.output_dir,
                incremental=True,
                fetch_only=True,
            )
        )
        old_generator_started = asyncio.Event()
        release_old_generator = asyncio.Event()

        class BlockingOldReport:
            async def generate_html_report(
                self, _aggregated, _summary, output_path, **_kwargs
            ):
                old_generator_started.set()
                await release_old_generator.wait()
                path = Path(output_path)
                path.write_text("<html>OLD OWNER</html>", encoding="utf-8")
                return path

        class NewOwnerReport:
            def generate_html_report(
                self, _aggregated, _summary, output_path, **_kwargs
            ):
                path = Path(output_path)
                path.write_text("<html>NEW OWNER</html>", encoding="utf-8")
                return path

        request = DigestTaskRequest.from_values(
            ["mock://mixed"],
            output_dir=self.output_dir,
            incremental=True,
            resume_ingestion_run_id=fetched.run_id,
            report_filename="shared-report.html",
        )
        stale_clock = datetime.now(timezone.utc) - timedelta(hours=2)
        old_task = asyncio.create_task(
            DigestTaskRunner(
                summary_provider=MockSummaryProvider(),
                report_generator=BlockingOldReport(),
                post_store=store,
                now_factory=lambda: stale_clock,
            ).run(request)
        )
        await asyncio.wait_for(old_generator_started.wait(), timeout=2)

        try:
            new_result = await asyncio.wait_for(
                DigestTaskRunner(
                    summary_provider=MockSummaryProvider(),
                    report_generator=NewOwnerReport(),
                    post_store=store,
                ).run(request),
                timeout=2,
            )
            self.assertEqual(new_result.status, "succeeded")
            self.assertEqual(
                (self.output_dir / "shared-report.html").read_text(encoding="utf-8"),
                "<html>NEW OWNER</html>",
            )
        finally:
            release_old_generator.set()

        old_result = await asyncio.wait_for(old_task, timeout=2)
        final_report = self.output_dir / "shared-report.html"
        history = json.loads(
            (self.output_dir / "history.json").read_text(encoding="utf-8")
        )
        durable = store.get_run(fetched.run_id)

        self.assertEqual(old_result.status, "failed")
        self.assertEqual(final_report.read_text(encoding="utf-8"), "<html>NEW OWNER</html>")
        self.assertEqual(history[final_report.name]["status"], "succeeded")
        self.assertEqual(history[final_report.name]["run_id"], fetched.run_id)
        self.assertEqual(durable.report_status, "succeeded")
        self.assertIsNone(durable.report_claim_id)
        self.assertEqual(list(self.output_dir.glob(".report-*.tmp.html")), [])

    async def test_cancelled_report_releases_claim_and_can_retry(self) -> None:
        store = InMemoryPostStore()
        fetched = await DigestTaskRunner(
            x_provider=MockXIngestionProvider(),
            post_store=store,
        ).run(
            DigestTaskRequest.from_values(
                ["mock://mixed"],
                output_dir=self.output_dir,
                incremental=True,
                fetch_only=True,
            )
        )
        started = asyncio.Event()

        class BlockingSummary:
            async def summarize(self, _aggregated):
                started.set()
                await asyncio.Event().wait()

        request = DigestTaskRequest.from_values(
            ["mock://mixed"],
            output_dir=self.output_dir,
            incremental=True,
            resume_ingestion_run_id=fetched.run_id,
        )
        task = asyncio.create_task(
            DigestTaskRunner(
                summary_provider=BlockingSummary(),
                report_generator=MockReportGenerator(),
                post_store=store,
            ).run(request)
        )
        await asyncio.wait_for(started.wait(), timeout=2)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        cancelled = store.get_run(fetched.run_id)
        self.assertEqual(cancelled.report_status, "failed")
        self.assertIsNone(cancelled.report_claim_id)
        self.assertIsNone(cancelled.report_claimed_at)

        retried = await DigestTaskRunner(
            summary_provider=MockSummaryProvider(),
            report_generator=MockReportGenerator(),
            post_store=store,
        ).run(request)
        self.assertEqual(retried.status, "succeeded")

    async def test_cancelled_complete_progress_preserves_committed_artifacts(self) -> None:
        store = InMemoryPostStore()
        run_id = "cancel-commit"

        async def cancel_on_complete(event: TaskProgress) -> None:
            if event.phase == "complete":
                raise asyncio.CancelledError

        runner = DigestTaskRunner(
            x_provider=MockXIngestionProvider(),
            summary_provider=MockSummaryProvider(),
            report_generator=MockReportGenerator(),
            progress_callback=cancel_on_complete,
            post_store=store,
            run_id_factory=lambda: run_id,
        )
        request = DigestTaskRequest.from_values(
            ["mock://mixed"],
            output_dir=self.output_dir,
            incremental=True,
            report_filename="committed-before-cancel.html",
        )

        with self.assertRaises(asyncio.CancelledError):
            await runner.run(request)

        report_path = self.output_dir / "committed-before-cancel.html"
        history_path = self.output_dir / "history.json"
        durable = store.get_run(run_id)
        history = json.loads(history_path.read_text(encoding="utf-8"))

        self.assertEqual(durable.status, "success")
        self.assertEqual(durable.report_status, "succeeded")
        self.assertIsNone(durable.report_claim_id)
        self.assertTrue(report_path.is_file())
        self.assertIn(report_path.name, history)
        self.assertEqual(history[report_path.name]["run_id"], run_id)
        self.assertEqual(list(self.output_dir.glob(".report-*.tmp.html")), [])

    async def test_stale_report_claim_is_reclaimed_with_a_new_owner(self) -> None:
        store = InMemoryPostStore()
        fetched = await DigestTaskRunner(
            x_provider=MockXIngestionProvider(),
            post_store=store,
        ).run(
            DigestTaskRequest.from_values(
                ["mock://mixed"],
                output_dir=self.output_dir,
                incremental=True,
                fetch_only=True,
            )
        )
        stale = store.get_run(fetched.run_id)
        stale.report_status = "generating"
        stale.report_claim_id = "crashed-owner"
        stale.report_claimed_at = datetime.now(timezone.utc) - timedelta(hours=2)
        store.update_run(stale)

        resumed = await DigestTaskRunner(
            summary_provider=MockSummaryProvider(),
            report_generator=MockReportGenerator(),
            post_store=store,
        ).run(
            DigestTaskRequest.from_values(
                ["mock://mixed"],
                output_dir=self.output_dir,
                incremental=True,
                resume_ingestion_run_id=fetched.run_id,
            )
        )

        final = store.get_run(fetched.run_id)
        self.assertEqual(resumed.status, "succeeded")
        self.assertEqual(final.report_status, "succeeded")
        self.assertIsNone(final.report_claim_id)
        self.assertIsNone(final.report_claimed_at)

    async def test_summary_failure_keeps_checkpoint_and_resumes_without_x_call(self) -> None:
        store = InMemoryPostStore()
        provider = MockXIngestionProvider()
        failing = DigestTaskRunner(
            x_provider=provider,
            summary_provider=MockSummaryProvider(fail=True),
            report_generator=MockReportGenerator(),
            post_store=store,
        )
        request = DigestTaskRequest.from_values(
            ["mock://mixed"], output_dir=self.output_dir, incremental=True
        )

        failed = await failing.run(request)

        self.assertEqual(failed.status, "failed")
        self.assertEqual(store.get_sync_state("mock://mixed").newest_post_id, "9005")
        self.assertEqual(store.get_run(failed.run_id).report_status, "failed")
        x_calls = len(provider.fetch_calls)

        retried = await DigestTaskRunner(
            summary_provider=MockSummaryProvider(),
            report_generator=MockReportGenerator(),
            post_store=store,
        ).run(
            DigestTaskRequest.from_values(
                ["mock://mixed"],
                output_dir=self.output_dir,
                incremental=True,
                resume_ingestion_run_id=failed.run_id,
            )
        )

        self.assertEqual(retried.status, "succeeded")
        self.assertEqual(len(provider.fetch_calls), x_calls)
        self.assertEqual(store.get_run(failed.run_id).report_status, "succeeded")

    async def test_failed_durable_run_can_resume_and_is_finalized(self) -> None:
        store = InMemoryPostStore()
        fetched = await DigestTaskRunner(
            x_provider=MockXIngestionProvider(),
            post_store=store,
        ).run(
            DigestTaskRequest.from_values(
                ["mock://mixed"],
                output_dir=self.output_dir,
                incremental=True,
                fetch_only=True,
            )
        )
        interrupted = store.get_run(fetched.run_id)
        interrupted.status = "failed"
        interrupted.finished_at = None
        interrupted.report_status = "not_started"
        store.update_run(interrupted)
        newer_state = store.get_sync_state("mock://mixed")
        newer_state.last_run_id = "a-newer-no-new-run"
        store.save_sync_state(newer_state)

        resumed = await DigestTaskRunner(
            summary_provider=MockSummaryProvider(),
            report_generator=MockReportGenerator(),
            post_store=store,
        ).run(
            DigestTaskRequest.from_values(
                ["mock://mixed"],
                output_dir=self.output_dir,
                incremental=True,
                resume_ingestion_run_id=fetched.run_id,
            )
        )

        finalized = store.get_run(fetched.run_id)
        self.assertEqual(resumed.status, "succeeded")
        self.assertEqual(finalized.status, "success")
        self.assertIsNotNone(finalized.finished_at)
        self.assertEqual(finalized.report_status, "succeeded")

    async def test_fresh_running_ingestion_run_cannot_be_resumed(self) -> None:
        store = InMemoryPostStore()
        fetched = await DigestTaskRunner(
            x_provider=MockXIngestionProvider(),
            post_store=store,
        ).run(
            DigestTaskRequest.from_values(
                ["mock://mixed"],
                output_dir=self.output_dir,
                incremental=True,
                fetch_only=True,
            )
        )
        active = store.get_run(fetched.run_id)
        active.status = "running"
        active.finished_at = None
        active.ingestion_owner_id = "active-ingestion-owner"
        active.ingestion_heartbeat_at = datetime.now(timezone.utc)
        active.report_status = "not_started"
        store.update_run(active)

        summary_provider = MockSummaryProvider()
        report_generator = MockReportGenerator()

        result = await DigestTaskRunner(
            summary_provider=summary_provider,
            report_generator=report_generator,
            post_store=store,
        ).run(
            DigestTaskRequest.from_values(
                ["mock://mixed"],
                output_dir=self.output_dir,
                incremental=True,
                resume_ingestion_run_id=fetched.run_id,
            )
        )

        self.assertEqual(result.status, "failed")
        self.assertIn("completed ingestion run", result.error or "")
        current = store.get_run(fetched.run_id)
        self.assertEqual(current.status, "running")
        self.assertEqual(current.ingestion_owner_id, "active-ingestion-owner")
        self.assertEqual(current.report_status, "not_started")
        self.assertEqual(summary_provider.calls, [])
        self.assertEqual(report_generator.calls, [])
        self.assertFalse((self.output_dir / "history.json").exists())

    async def test_stale_running_ingestion_resumes_stored_posts_and_completes_report(self) -> None:
        store = InMemoryPostStore()
        fetched = await DigestTaskRunner(
            x_provider=MockXIngestionProvider(),
            post_store=store,
        ).run(
            DigestTaskRequest.from_values(
                ["mock://mixed"],
                output_dir=self.output_dir,
                incremental=True,
                fetch_only=True,
            )
        )
        stale = store.get_run(fetched.run_id)
        stale.status = "running"
        stale.finished_at = None
        stale.ingestion_owner_id = "crashed-ingestion-owner"
        stale.ingestion_heartbeat_at = datetime.now(timezone.utc) - timedelta(hours=2)
        stale.report_status = "not_started"
        store.update_run(stale)
        summary_provider = MockSummaryProvider()
        report_generator = MockReportGenerator()

        result = await DigestTaskRunner(
            summary_provider=summary_provider,
            report_generator=report_generator,
            post_store=store,
        ).run(
            DigestTaskRequest.from_values(
                ["mock://mixed"],
                output_dir=self.output_dir,
                incremental=True,
                resume_ingestion_run_id=fetched.run_id,
            )
        )

        finalized = store.get_run(fetched.run_id)
        self.assertEqual(result.status, "succeeded")
        self.assertEqual(result.new_post_count, 5)
        self.assertTrue(result.report_path and result.report_path.is_file())
        self.assertTrue(result.history_path and result.history_path.is_file())
        self.assertEqual(len(summary_provider.calls), 1)
        self.assertEqual(len(report_generator.calls), 1)
        self.assertEqual(finalized.status, "success")
        self.assertIsNotNone(finalized.finished_at)
        self.assertIsNone(finalized.ingestion_owner_id)
        self.assertEqual(finalized.report_status, "succeeded")
        self.assertTrue(
            any(
                item.get("kind") == "stale_ingestion_recovered"
                for item in finalized.errors
            )
        )

    async def test_fetch_only_persists_without_resolving_llm_or_report(self) -> None:
        store = InMemoryPostStore()
        runner = DigestTaskRunner(
            x_provider=MockXIngestionProvider(),
            post_store=store,
        )
        request = DigestTaskRequest.from_values(
            ["mock://mixed"],
            output_dir=self.output_dir,
            incremental=True,
            fetch_only=True,
        )

        result = await runner.run(request)

        self.assertEqual(result.status, "succeeded")
        self.assertEqual(result.new_post_count, 5)
        self.assertIsNone(result.report_path)
        self.assertEqual(store.get_run(result.run_id).report_status, "skipped_fetch_only")


class OfflineCliTests(unittest.TestCase):
    def test_python_module_mock_entry_behavior(self) -> None:
        with workspace_temporary_directory() as directory:
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = main(["--mock", "--output-dir", str(directory)])
            self.assertEqual(0, exit_code)
            output = stdout.getvalue()
            self.assertIn("Digest complete", output)
            self.assertIn('"status": "succeeded"', output)
            reports = list(Path(directory).glob("summary_*.html"))
            self.assertEqual(1, len(reports))
            self.assertTrue((Path(directory) / "history.json").exists())

    def test_incremental_fetch_only_cli_is_idempotent(self) -> None:
        with workspace_temporary_directory() as directory:
            data_dir = directory / "data"
            arguments = [
                "--mock",
                "--fetch-only",
                "--list",
                "mock://mixed",
                "--data-dir",
                str(data_dir),
                "--output-dir",
                str(directory / "output"),
                "--page-size",
                "2",
            ]
            first_stdout = io.StringIO()
            with contextlib.redirect_stdout(first_stdout):
                first_exit = main(arguments)
            second_stdout = io.StringIO()
            with contextlib.redirect_stdout(second_stdout):
                second_exit = main(arguments)

            self.assertEqual(first_exit, 0)
            self.assertEqual(second_exit, 0)
            self.assertIn('"new_post_count": 5', first_stdout.getvalue())
            self.assertIn('"status": "no_new_posts"', second_stdout.getvalue())
            self.assertEqual(list((directory / "output").glob("*.html")), [])


if __name__ == "__main__":
    unittest.main()
