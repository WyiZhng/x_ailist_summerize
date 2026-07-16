from __future__ import annotations

import shutil
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from app.article_models import (
    Article,
    ArticleFetchAttempt,
    PostArticleLink,
    article_id_for_url,
    stable_sha256,
)
from app.article_storage import (
    ArticleStorageCorruptionError,
    ArticleStorageError,
    FileArticleStore,
    InMemoryArticleStore,
)
from app.models import utc_now


def make_article(
    url: str = "https://example.com/story",
    *,
    content: str = "Durable article body text with enough useful information.",
    status: str = "fetched",
    checked_at: datetime | None = None,
) -> Article:
    return Article(
        id=article_id_for_url(url),
        normalized_url=url,
        original_urls=[url],
        domain="example.com",
        title="Stored article",
        content_text=content,
        content_hash=stable_sha256(content),
        word_count=8,
        fetched_at=checked_at or utc_now(),
        checked_at=checked_at or utc_now(),
        status=status,
    )


class FileArticleStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = Path.cwd() / "tests" / f".tmp-articles-{uuid.uuid4().hex}"
        self.directory.mkdir(parents=True)
        self.store = FileArticleStore(self.directory)

    def tearDown(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_new_article_metadata_and_content_are_saved_separately(self) -> None:
        article = make_article()
        self.store.save_article(article)

        loaded = self.store.get_by_normalized_url(article.normalized_url)
        self.assertEqual(article.content_text, loaded.content_text)
        self.assertEqual(article.content_hash, loaded.content_hash)
        metadata = self.store.articles_path.read_text(encoding="utf-8")
        self.assertNotIn(article.content_text, metadata)
        self.assertTrue((self.store.content_dir / f"{article.content_hash}.txt").is_file())

    def test_same_normalized_url_is_deduplicated_and_original_urls_merge(self) -> None:
        article = make_article()
        self.store.save_article(article)
        second = make_article()
        second.original_urls = ["https://example.com/story?utm_source=x"]
        self.store.save_article(second)

        values = self.store.read_articles()
        self.assertEqual(1, len(values))
        self.assertEqual(2, len(values[0].original_urls))

    def test_multiple_posts_can_link_to_one_article_idempotently(self) -> None:
        article = make_article()
        self.store.save_article(article)
        first = PostArticleLink("post-1", article.id, article.original_urls[0])
        second = PostArticleLink("post-2", article.id, article.original_urls[0])
        self.store.link_post_article(first)
        self.store.link_post_article(first)
        self.store.link_post_article(second)
        self.assertEqual(2, len(self.store.read_links()))

    def test_same_content_hash_reuses_one_body_file_for_different_urls(self) -> None:
        content = "Identical article body content shared by mirrored URLs."
        self.store.save_article(make_article("https://example.com/a", content=content))
        self.store.save_article(make_article("https://mirror.example.com/a", content=content))
        self.assertEqual(1, len(list(self.store.content_dir.glob("*.txt"))))
        self.assertEqual(2, len(self.store.read_articles()))

    def test_fetch_attempts_are_auditable_and_idempotent(self) -> None:
        now = utc_now()
        attempt = ArticleFetchAttempt(
            id="attempt-test-1",
            original_url="https://example.com/story",
            started_at=now,
            finished_at=now,
            status="failed",
            error_code="http_500",
        )
        self.store.save_fetch_attempt(attempt)
        self.store.save_fetch_attempt(attempt)
        self.assertEqual(1, len(self.store.read_attempts()))
        self.assertEqual("http_500", self.store.read_attempts()[0].error_code)

    def test_ttl_cache_uses_checked_at_and_success_status(self) -> None:
        now = datetime(2026, 7, 16, tzinfo=timezone.utc)
        fresh = make_article(checked_at=now - timedelta(hours=10))
        self.store.save_article(fresh)
        self.assertFalse(self.store.should_fetch(fresh.normalized_url, now))
        self.assertTrue(
            self.store.should_fetch(fresh.normalized_url, now + timedelta(hours=200))
        )
        failed = make_article("https://example.com/failed", status="failed")
        self.store.save_article(failed)
        self.assertTrue(self.store.should_fetch(failed.normalized_url, now))

    def test_missing_or_tampered_content_file_is_explicit_corruption(self) -> None:
        article = make_article()
        self.store.save_article(article)
        content_path = self.store.content_dir / f"{article.content_hash}.txt"
        content_path.write_text("tampered", encoding="utf-8")
        with self.assertRaises(ArticleStorageCorruptionError):
            self.store.read_articles()

    def test_malformed_jsonl_is_not_silently_overwritten(self) -> None:
        self.store.articles_path.parent.mkdir(parents=True, exist_ok=True)
        self.store.articles_path.write_text("{broken json\n", encoding="utf-8")
        with self.assertRaises(ArticleStorageCorruptionError):
            self.store.save_article(make_article())
        self.assertEqual("{broken json\n", self.store.articles_path.read_text(encoding="utf-8"))

    def test_metadata_write_failure_preserves_previous_article_file(self) -> None:
        first = make_article()
        self.store.save_article(first)
        before = self.store.articles_path.read_bytes()
        original_atomic = __import__(
            "app.article_storage", fromlist=["_atomic_write"]
        )._atomic_write

        def fail_metadata(path, text):
            if path == self.store.articles_path:
                raise OSError("TEST injected metadata failure")
            return original_atomic(path, text)

        updated = make_article(content="Updated content that must not replace metadata.")
        with mock.patch("app.article_storage._atomic_write", side_effect=fail_metadata):
            with self.assertRaises(OSError):
                self.store.save_article(updated)
        self.assertEqual(before, self.store.articles_path.read_bytes())
        self.assertEqual(first.content_text, self.store.read_articles()[0].content_text)

    def test_relation_to_missing_article_is_rejected(self) -> None:
        with self.assertRaises(ArticleStorageError):
            self.store.link_post_article(
                PostArticleLink("post-1", "missing-article", "https://example.com")
            )

    def test_dangling_relation_in_damaged_file_is_reported_as_corruption(self) -> None:
        self.store.links_path.parent.mkdir(parents=True, exist_ok=True)
        self.store.links_path.write_text(
            '{"post_id":"p","article_id":"missing","original_url":"https://example.com",'
            '"discovered_at":"2026-07-16T00:00:00+00:00"}\n',
            encoding="utf-8",
        )
        with self.assertRaises(ArticleStorageCorruptionError):
            self.store.read_links()

    def test_failed_refresh_can_preserve_old_successful_content(self) -> None:
        article = make_article()
        self.store.save_article(article)
        failed = article.clone()
        failed.status = "failed"
        failed.error_code = "timeout"
        failed.error_message = "Refresh timed out"
        self.store.save_article(failed)
        loaded = self.store.get_by_normalized_url(article.normalized_url)
        self.assertEqual("failed", loaded.status)
        self.assertEqual(article.content_text, loaded.content_text)


class StoreParityTests(unittest.TestCase):
    def test_in_memory_store_matches_core_dedup_link_and_cache_behavior(self) -> None:
        store = InMemoryArticleStore()
        article = make_article()
        store.save_article(article)
        store.save_article(article)
        store.link_post_article(PostArticleLink("post-1", article.id, article.original_urls[0]))
        store.link_post_article(PostArticleLink("post-1", article.id, article.original_urls[0]))
        self.assertEqual(1, len(store.read_articles()))
        self.assertEqual(1, len(store.read_links()))
        self.assertFalse(store.should_fetch(article.normalized_url, utc_now()))


if __name__ == "__main__":
    unittest.main()
