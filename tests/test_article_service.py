from __future__ import annotations

import json
import shutil
import unittest
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import httpx

from app.article_fetcher import ArticleFetchConfig, ArticleFetcher
from app.article_models import Article, article_id_for_url
from app.article_service import ArticleService, ArticleServiceConfig, main
from app.article_storage import InMemoryArticleStore
from app.models import ListSyncState, XAuthor, XPost
from app.network_security import NetworkSecurityValidator
from app.providers import MockReportGenerator, MockSummaryProvider, MockXIngestionProvider
from app.storage import FilePostStore, InMemoryPostStore
from app.task_runner import DigestTaskRequest, DigestTaskRunner


PUBLIC_IP = "93.184.216.34"
HTML = (
    "<html><head><title>Article</title></head><body><article>"
    "<p>This external article contains useful durable body text for the test suite.</p>"
    "</article></body></html>"
)


def post(post_id: str, *urls: str) -> XPost:
    author = XAuthor(id="author-1", username="tester")
    return XPost(
        id=post_id,
        text="stored post",
        author_id=author.id,
        author=author,
        created_at=datetime(2026, 7, 16, tzinfo=timezone.utc),
        language="en",
        conversation_id=post_id,
        source_list_id="list-1",
        urls=list(urls),
    )


class ArticleServiceTests(unittest.IsolatedAsyncioTestCase):
    async def service(self, handler, *, store=None, failure_retry_hours=0):
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        self.addAsyncCleanup(client.aclose)
        fetcher = ArticleFetcher(
            config=ArticleFetchConfig(retry_attempts=0),
            client=client,
            network_security=NetworkSecurityValidator(
                resolver=lambda _host, _port: [PUBLIC_IP]
            ),
        )
        article_store = store or InMemoryArticleStore()
        return ArticleService(
            store=article_store,
            fetcher=fetcher,
            config=ArticleServiceConfig(failure_retry_hours=failure_retry_hours),
        ), article_store

    async def test_reads_external_urls_and_excludes_x_media_and_shortlinks(self) -> None:
        requests = []

        def handler(request):
            requests.append(str(request.url))
            return httpx.Response(200, headers={"Content-Type": "text/html"}, text=HTML)

        service, store = await self.service(handler)
        result = await service.process_posts(
            [
                post(
                    "p1",
                    "https://example.com/story",
                    "https://x.com/user/status/1",
                    "https://t.co/short",
                    "https://pbs.twimg.com/media/test.jpg",
                    "file:///etc/passwd",
                )
            ]
        )
        self.assertEqual(1, result.candidate_url_count)
        self.assertEqual(1, result.success_count)
        self.assertEqual(1, len(requests))
        self.assertEqual(1, len(store.read_links()))

    async def test_no_external_links_skips_fetching(self) -> None:
        calls = []
        service, _ = await self.service(
            lambda request: calls.append(request) or httpx.Response(500)
        )
        result = await service.process_posts([post("p1", "https://x.com/a/status/1")])
        self.assertEqual(0, result.request_count)
        self.assertEqual([], calls)

    async def test_tracking_variants_fetch_once_and_link_both_posts(self) -> None:
        calls = []

        def handler(request):
            calls.append(request)
            return httpx.Response(200, headers={"Content-Type": "text/html"}, text=HTML)

        service, store = await self.service(handler)
        result = await service.process_posts(
            [
                post("p1", "https://example.com/story?utm_source=x"),
                post("p2", "https://example.com/story?utm_campaign=y"),
            ]
        )
        self.assertEqual(1, result.normalized_url_count)
        self.assertEqual(1, result.request_count)
        self.assertEqual(1, len(store.read_articles()))
        self.assertEqual(2, len(store.read_links()))

    async def test_one_article_failure_does_not_block_other_articles(self) -> None:
        def handler(request):
            if request.url.path == "/blocked":
                return httpx.Response(403)
            return httpx.Response(200, headers={"Content-Type": "text/html"}, text=HTML)

        service, store = await self.service(handler)
        result = await service.process_posts(
            [
                post("p1", "https://example.com/blocked"),
                post("p2", "https://example.com/good"),
            ]
        )
        self.assertEqual(1, result.success_count)
        self.assertEqual(1, result.failed_count)
        self.assertEqual(2, len(store.read_articles()))
        self.assertEqual(2, len(store.read_attempts()))

    async def test_repeat_run_uses_ttl_cache_without_new_request(self) -> None:
        calls = 0

        def handler(_request):
            nonlocal calls
            calls += 1
            return httpx.Response(200, headers={"Content-Type": "text/html"}, text=HTML)

        service, store = await self.service(handler)
        posts = [post("p1", "https://example.com/story")]
        first = await service.process_posts(posts)
        second = await service.process_posts(posts)
        self.assertEqual(1, first.request_count)
        self.assertEqual(0, second.request_count)
        self.assertEqual(1, second.cache_hit_count)
        self.assertEqual(1, calls)
        self.assertEqual(1, len(store.read_links()))

    async def test_retry_failed_only_processes_existing_failures(self) -> None:
        store = InMemoryArticleStore()
        now = datetime.now(timezone.utc)
        failed_url = "https://example.com/failed"
        good_url = "https://example.com/good"
        store.save_article(
            Article(
                id=article_id_for_url(failed_url),
                normalized_url=failed_url,
                original_urls=[failed_url],
                domain="example.com",
                fetched_at=now,
                checked_at=now,
                status="failed",
                error_code="timeout",
            )
        )
        store.save_article(
            Article(
                id=article_id_for_url(good_url),
                normalized_url=good_url,
                original_urls=[good_url],
                domain="example.com",
                content_text="Existing successful article body text for caching.",
                fetched_at=now,
                checked_at=now,
                status="fetched",
            )
        )
        paths = []

        def handler(request):
            paths.append(request.url.path)
            return httpx.Response(200, headers={"Content-Type": "text/html"}, text=HTML)

        service, _ = await self.service(handler, store=store)
        result = await service.process_posts(
            [post("p1", failed_url), post("p2", good_url)], retry_failed=True
        )
        self.assertEqual(["/failed"], paths)
        self.assertEqual(1, result.success_count)

    async def test_failure_backoff_skips_immediate_default_retry(self) -> None:
        store = InMemoryArticleStore()
        now = datetime.now(timezone.utc)
        url = "https://example.com/failed"
        store.save_article(
            Article(
                id=article_id_for_url(url),
                normalized_url=url,
                original_urls=[url],
                domain="example.com",
                fetched_at=now,
                checked_at=now,
                status="failed",
                error_code="http_403",
            )
        )
        calls = []
        service, _ = await self.service(
            lambda request: calls.append(request) or httpx.Response(500),
            store=store,
            failure_retry_hours=6,
        )
        result = await service.process_posts([post("p1", url)])
        self.assertEqual(0, result.request_count)
        self.assertEqual([], calls)

    async def test_article_processing_does_not_modify_post_checkpoint(self) -> None:
        post_store = InMemoryPostStore()
        state = ListSyncState(
            list_id="list-1",
            newest_post_id="900",
            last_success_at=datetime.now(timezone.utc),
        )
        post_store.save_sync_state(state)
        service, _ = await self.service(
            lambda _request: httpx.Response(403)
        )
        await service.process_posts([post("p1", "https://example.com/blocked")])
        self.assertEqual("900", post_store.get_sync_state("list-1").newest_post_id)

    async def test_max_articles_per_run_is_enforced(self) -> None:
        calls = []
        service, _ = await self.service(
            lambda request: calls.append(request) or httpx.Response(
                200, headers={"Content-Type": "text/html"}, text=HTML
            )
        )
        values = [post(str(index), f"https://example.com/story/{index}") for index in range(5)]
        result = await service.process_posts(values, max_articles=3)
        self.assertEqual(3, result.request_count)
        self.assertEqual(3, len(calls))

    async def test_structured_logs_never_include_full_url_or_query_secret(self) -> None:
        secret = "TEST_SECRET_QUERY_DO_NOT_LOG"
        service, _ = await self.service(
            lambda _request: httpx.Response(
                200, headers={"Content-Type": "text/html"}, text=HTML
            )
        )
        with self.assertLogs("app.article_service", level="INFO") as captured:
            result = await service.process_posts(
                [post("p1", f"https://example.com/private-path?token={secret}")],
                run_id="TEST-RUN",
            )
        logs = "\n".join(captured.output)
        self.assertEqual(1, result.success_count)
        self.assertNotIn(secret, logs)
        self.assertNotIn("private-path", logs)
        self.assertNotIn("https://", logs)
        self.assertIn("domain=example.com", logs)


class ArticleRunnerBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_article_failure_after_checkpoint_does_not_fail_digest(self) -> None:
        class FailingArticles:
            calls = 0

            async def process_posts(self, posts, *, run_id=None):
                self.calls += 1
                raise RuntimeError("TEST article failure")

        articles = FailingArticles()
        summary = MockSummaryProvider()
        store = InMemoryPostStore()
        runner = DigestTaskRunner(
            x_provider=MockXIngestionProvider(),
            summary_provider=summary,
            report_generator=MockReportGenerator(),
            post_store=store,
            article_service=articles,
        )
        directory = Path.cwd() / "tests" / f".tmp-runner-articles-{uuid.uuid4().hex}"
        directory.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, directory, True)

        result = await runner.run(
            DigestTaskRequest.from_values(
                ["mock://mixed"], output_dir=directory, incremental=True
            )
        )
        self.assertTrue(result.success)
        self.assertEqual(1, articles.calls)
        self.assertGreater(result.new_post_count, 0)
        self.assertIsNotNone(store.get_sync_state("mock://mixed"))
        self.assertEqual(1, len(summary.calls))


class ArticleCliTests(unittest.TestCase):
    def test_mock_cli_reads_stored_posts_without_x_or_llm(self) -> None:
        directory = Path.cwd() / "tests" / f".tmp-cli-articles-{uuid.uuid4().hex}"
        directory.mkdir(parents=True)
        try:
            FilePostStore(directory).save_posts(
                [post("p1", "https://example.com/story")]
            )
            with mock.patch("builtins.print") as captured:
                code = main(
                    [
                        "--from-stored-posts",
                        "--mock",
                        "--data-dir",
                        str(directory),
                        "--max-articles",
                        "1",
                    ]
                )
            self.assertEqual(0, code)
            payload = json.loads(captured.call_args.args[0])
            self.assertEqual(1, payload["request_count"])
            self.assertEqual(1, payload["success_count"])
        finally:
            shutil.rmtree(directory, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
