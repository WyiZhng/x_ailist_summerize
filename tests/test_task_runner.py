from __future__ import annotations

import asyncio
import contextlib
import io
import json
import shutil
import unittest
import uuid
from pathlib import Path
from typing import Any

from app.providers import (
    MockReportGenerator,
    MockSummaryProvider,
    MockXProvider,
    ReportGenerator,
    SummaryProvider,
    XProvider,
)
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


if __name__ == "__main__":
    unittest.main()
