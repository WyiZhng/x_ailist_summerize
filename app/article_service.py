"""Independent article enrichment for already-persisted X posts."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

import httpx

from .article_fetcher import ArticleFetchConfig, ArticleFetcher
from .article_models import Article, PostArticleLink
from .article_storage import ArticleStorageError, ArticleStore, FileArticleStore
from .config import load_config
from .models import XPost, utc_now
from .network_security import NetworkSecurityValidator
from .storage import FilePostStore
from .url_normalizer import UrlNormalizationError, UrlNormalizer


logger = logging.getLogger(__name__)

_NON_ARTICLE_DOMAINS = frozenset(
    {
        "x.com",
        "www.x.com",
        "twitter.com",
        "www.twitter.com",
        "t.co",
        "pic.twitter.com",
        "pbs.twimg.com",
        "video.twimg.com",
    }
)


@dataclass(frozen=True, slots=True)
class ArticleServiceConfig:
    enabled: bool = True
    max_articles_per_run: int = 20
    cache_ttl_hours: int = 168
    failure_retry_hours: int = 6

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "ArticleServiceConfig":
        value = value or {}
        return cls(
            enabled=bool(value.get("enabled", True)),
            max_articles_per_run=max(1, int(value.get("max_articles_per_run", 20))),
            cache_ttl_hours=max(0, int(value.get("cache_ttl_hours", 168))),
            failure_retry_hours=max(0, int(value.get("failure_retry_hours", 6))),
        )


@dataclass(slots=True)
class ArticleProcessingResult:
    candidate_url_count: int = 0
    normalized_url_count: int = 0
    cache_hit_count: int = 0
    request_count: int = 0
    success_count: int = 0
    failed_count: int = 0
    blocked_count: int = 0
    unsupported_count: int = 0
    content_char_count: int = 0
    relation_count: int = 0
    processed_articles: list[Article] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["processed_articles"] = [item.to_dict() for item in self.processed_articles]
        return value


class ArticleService:
    """Fetch/cache article entities without touching X checkpoints or the LLM."""

    def __init__(
        self,
        *,
        store: ArticleStore,
        fetcher: ArticleFetcher,
        config: ArticleServiceConfig | Mapping[str, Any] | None = None,
        normalizer: UrlNormalizer | None = None,
        now_factory=utc_now,
    ) -> None:
        self.store = store
        self.fetcher = fetcher
        self.config = (
            config
            if isinstance(config, ArticleServiceConfig)
            else ArticleServiceConfig.from_mapping(config)
        )
        self.normalizer = normalizer or UrlNormalizer()
        self._now = now_factory

    async def aclose(self) -> None:
        await self.fetcher.aclose()

    @staticmethod
    def _is_candidate(url: str) -> bool:
        try:
            parsed = urlsplit(url)
        except ValueError:
            return False
        hostname = (parsed.hostname or "").rstrip(".").lower()
        return bool(
            parsed.scheme.lower() in {"http", "https"}
            and hostname
            and hostname not in _NON_ARTICLE_DOMAINS
            and not hostname.endswith(".twimg.com")
            and not hostname.endswith(".x.com")
            and not hostname.endswith(".twitter.com")
        )

    def _retry_allowed(self, article: Article, now: datetime) -> bool:
        return article.checked_at + timedelta(hours=self.config.failure_retry_hours) <= now

    async def process_posts(
        self,
        posts: Sequence[XPost],
        *,
        run_id: str | None = None,
        max_articles: int | None = None,
        retry_failed: bool = False,
    ) -> ArticleProcessingResult:
        result = ArticleProcessingResult()
        if not self.config.enabled:
            return result
        grouped: dict[str, list[tuple[XPost, str]]] = {}
        for post in posts:
            for original_url in post.urls:
                if not self._is_candidate(original_url):
                    continue
                result.candidate_url_count += 1
                try:
                    normalized = self.normalizer.normalize(original_url)
                except UrlNormalizationError:
                    fetch_result = await self.fetcher.fetch(
                        original_url, run_id=run_id
                    )
                    for attempt in fetch_result.attempts:
                        self.store.save_fetch_attempt(attempt)
                    result.blocked_count += int(
                        bool(
                            fetch_result.attempts
                            and fetch_result.attempts[-1].status == "blocked"
                        )
                    )
                    result.failed_count += int(
                        not fetch_result.attempts
                        or fetch_result.attempts[-1].status != "blocked"
                    )
                    continue
                grouped.setdefault(normalized.normalized_url, []).append((post, original_url))
        result.normalized_url_count = len(grouped)
        limit = max_articles or self.config.max_articles_per_run
        now = self._now()

        for normalized_url, references in list(grouped.items())[:limit]:
            first_original = references[0][1]
            existing = self.store.get_by_normalized_url(normalized_url)
            if existing is None:
                existing = self.store.get_by_original_url(first_original)

            if retry_failed:
                if existing is None or existing.status in {"fetched", "unchanged"}:
                    continue
            elif existing is not None:
                if existing.status in {"fetched", "unchanged"} and not self.store.should_fetch(
                    existing.normalized_url, now
                ):
                    result.cache_hit_count += 1
                    for post, original_url in references:
                        self.store.link_post_article(
                            PostArticleLink(post.id, existing.id, original_url, now)
                        )
                        result.relation_count += 1
                    logger.info(
                        "article_cache run_id=%s post_id=%s article_id=%s domain=%s "
                        "fetch_status=%s http_status=%s redirect_count=0 response_bytes=0 "
                        "extractor=%s word_count=%s cache_hit=true retry_attempt=0 duration_ms=0",
                        run_id,
                        references[0][0].id,
                        existing.id,
                        existing.domain,
                        existing.status,
                        existing.http_status,
                        existing.extractor,
                        existing.word_count,
                    )
                    continue
                if existing.status not in {"fetched", "unchanged"} and not self._retry_allowed(
                    existing, now
                ):
                    continue

            fetch_result = await self.fetcher.fetch(
                first_original,
                existing=existing,
                run_id=run_id,
            )
            result.request_count += fetch_result.request_count
            for attempt in fetch_result.attempts:
                self.store.save_fetch_attempt(attempt)
            article = fetch_result.article
            last_attempt = fetch_result.attempts[-1] if fetch_result.attempts else None
            logger.info(
                "article_result run_id=%s post_id=%s article_id=%s domain=%s "
                "fetch_status=%s http_status=%s redirect_count=%s response_bytes=%s "
                "extractor=%s word_count=%s cache_hit=false retry_attempt=%s duration_ms=%s",
                run_id,
                references[0][0].id,
                article.id if article else None,
                article.domain if article else (urlsplit(normalized_url).hostname or "invalid"),
                article.status if article else "failed",
                article.http_status if article else None,
                fetch_result.redirect_count,
                last_attempt.response_bytes if last_attempt else 0,
                article.extractor if article else None,
                article.word_count if article else None,
                last_attempt.retry_attempt if last_attempt else 0,
                last_attempt.duration_ms if last_attempt else 0,
            )
            if article is None:
                result.failed_count += 1
                continue
            try:
                self.store.save_article(article)
                for post, original_url in references:
                    self.store.link_post_article(
                        PostArticleLink(post.id, article.id, original_url, now)
                    )
                    result.relation_count += 1
                    logger.info(
                        "article_link run_id=%s post_id=%s article_id=%s domain=%s fetch_status=%s",
                        run_id,
                        post.id,
                        article.id,
                        article.domain,
                        article.status,
                    )
            except ArticleStorageError:
                result.failed_count += 1
                logger.error(
                    "article_storage run_id=%s article_id=%s domain=%s fetch_status=storage_error",
                    run_id,
                    article.id,
                    article.domain,
                )
                continue
            result.processed_articles.append(article)
            if article.status in {"fetched", "unchanged"}:
                result.success_count += 1
                result.content_char_count += len(article.content_text or "")
            elif article.status == "blocked":
                result.blocked_count += 1
            elif article.status == "unsupported":
                result.unsupported_count += 1
            else:
                result.failed_count += 1
        return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract articles from stored X posts")
    parser.add_argument("--from-stored-posts", action="store_true", default=True)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--max-articles", type=int)
    return parser


async def _run_cli(args: argparse.Namespace) -> ArticleProcessingResult:
    config = load_config()
    article_config = config.get("articles", {})
    data_dir = args.data_dir or Path(config.get("storage", {}).get("data_dir", "data"))
    posts = FilePostStore(data_dir).read_posts()
    store = FileArticleStore(
        data_dir,
        cache_ttl_hours=int(article_config.get("cache_ttl_hours", 168)),
    )
    client = None
    validator = None
    if args.mock:
        html = (
            "<html lang='en'><head><title>Mock article</title></head>"
            "<body><article><p>This deterministic mock article contains enough "
            "body text for offline extraction and storage validation.</p></article></body></html>"
        )
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200, headers={"Content-Type": "text/html; charset=utf-8"}, text=html
                )
            )
        )
        validator = NetworkSecurityValidator(
            resolver=lambda _host, _port: ["93.184.216.34"]
        )
    fetcher = ArticleFetcher(
        config=ArticleFetchConfig.from_mapping(article_config),
        client=client,
        network_security=validator,
    )
    service = ArticleService(
        store=store,
        fetcher=fetcher,
        config=ArticleServiceConfig.from_mapping(article_config),
    )
    try:
        return await service.process_posts(
            posts,
            run_id="articles-only",
            max_articles=args.max_articles,
            retry_failed=args.retry_failed,
        )
    finally:
        if client is None:
            await service.aclose()
        else:
            await client.aclose()


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.max_articles is not None and args.max_articles < 1:
        raise SystemExit("--max-articles must be positive")
    result = asyncio.run(_run_cli(args))
    safe = result.to_dict()
    safe.pop("processed_articles", None)
    print(json.dumps(safe, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ArticleProcessingResult",
    "ArticleService",
    "ArticleServiceConfig",
    "main",
]
