"""Secure, bounded HTTP fetching for untrusted external articles."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from email.message import Message
from email.utils import parsedate_to_datetime
from typing import Any, Awaitable, Callable, Mapping
from urllib.parse import urljoin

import httpx

from .article_extractor import extract_article
from .article_models import (
    Article,
    ArticleFetchAttempt,
    NormalizedUrl,
    article_id_for_url,
)
from .models import utc_now
from .network_security import (
    NetworkSecurityError,
    NetworkSecurityValidator,
    safe_url_log_fields,
)
from .url_normalizer import UrlNormalizationError, UrlNormalizer


logger = logging.getLogger(__name__)

Sleep = Callable[[float], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class ArticleFetchConfig:
    timeout_seconds: float = 15.0
    max_redirects: int = 5
    max_response_bytes: int = 5_242_880
    max_article_chars: int = 50_000
    retry_attempts: int = 2
    user_agent: str = "x-ai-daily/1.0"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "ArticleFetchConfig":
        value = value or {}
        return cls(
            timeout_seconds=float(value.get("timeout_seconds", 15)),
            max_redirects=int(value.get("max_redirects", 5)),
            max_response_bytes=int(value.get("max_response_bytes", 5_242_880)),
            max_article_chars=int(value.get("max_article_chars", 50_000)),
            retry_attempts=int(value.get("retry_attempts", 2)),
            user_agent=str(value.get("user_agent") or "x-ai-daily/1.0"),
        )

    def __post_init__(self) -> None:
        if not 0 < self.timeout_seconds <= 120:
            raise ValueError("timeout_seconds must be between 0 and 120")
        if not 0 <= self.max_redirects <= 10:
            raise ValueError("max_redirects must be between 0 and 10")
        if not 1 <= self.max_response_bytes <= 50 * 1024 * 1024:
            raise ValueError("max_response_bytes is outside the safe range")
        if not 1 <= self.max_article_chars <= 1_000_000:
            raise ValueError("max_article_chars is outside the safe range")
        if not 0 <= self.retry_attempts <= 5:
            raise ValueError("retry_attempts must be between 0 and 5")


@dataclass(slots=True)
class ArticleFetchResult:
    article: Article | None
    normalized_url: NormalizedUrl | None
    attempts: list[ArticleFetchAttempt] = field(default_factory=list)
    request_count: int = 0
    redirect_count: int = 0

    @property
    def success(self) -> bool:
        return bool(self.article and self.article.status in {"fetched", "unchanged"})


class ArticleFetcher:
    """Fetch one article without cookies, browser state, or automatic redirects."""

    _REDIRECTS = frozenset({301, 302, 303, 307, 308})
    _HTML_TYPES = frozenset({"text/html", "application/xhtml+xml"})

    def __init__(
        self,
        *,
        config: ArticleFetchConfig | Mapping[str, Any] | None = None,
        client: httpx.AsyncClient | None = None,
        normalizer: UrlNormalizer | None = None,
        network_security: NetworkSecurityValidator | None = None,
        sleep: Sleep = asyncio.sleep,
        now_factory: Callable[[], datetime] = utc_now,
    ) -> None:
        self.config = (
            config
            if isinstance(config, ArticleFetchConfig)
            else ArticleFetchConfig.from_mapping(config)
        )
        self.normalizer = normalizer or UrlNormalizer()
        self.network_security = network_security or NetworkSecurityValidator()
        self._sleep = sleep
        self._now = now_factory
        self._owns_client = client is None
        self.client = client or httpx.AsyncClient(
            follow_redirects=False,
            timeout=httpx.Timeout(self.config.timeout_seconds),
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    @staticmethod
    def _content_type(value: str | None) -> tuple[str, str | None]:
        if not value:
            return "", None
        message = Message()
        message["content-type"] = value
        return message.get_content_type().lower(), message.get_param("charset")

    def _retry_after(self, response: httpx.Response) -> float:
        value = response.headers.get("Retry-After", "").strip()
        try:
            return min(60.0, max(0.0, float(value)))
        except ValueError:
            try:
                target = parsedate_to_datetime(value)
                if target.tzinfo is None:
                    return 0.0
                return min(60.0, max(0.0, (target - self._now()).total_seconds()))
            except (TypeError, ValueError, OverflowError):
                return 0.0

    def _attempt(
        self,
        *,
        original_url: str,
        started_at: datetime,
        status: str,
        article_id: str | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
        http_status: int | None = None,
        redirect_count: int = 0,
        response_bytes: int = 0,
        retry_attempt: int = 0,
    ) -> ArticleFetchAttempt:
        finished = self._now()
        duration = max(0, int((finished - started_at).total_seconds() * 1000))
        return ArticleFetchAttempt(
            id=f"attempt_{uuid.uuid4().hex}",
            article_id=article_id,
            original_url=original_url,
            started_at=started_at,
            finished_at=finished,
            status=status,
            error_code=error_code,
            error_message=(error_message[:500] if error_message else None),
            http_status=http_status,
            redirect_count=redirect_count,
            response_bytes=response_bytes,
            retry_attempt=retry_attempt,
            duration_ms=duration,
        )

    def _failed_article(
        self,
        normalized: NormalizedUrl,
        existing: Article | None,
        *,
        status: str,
        error_code: str,
        error_message: str,
        http_status: int | None = None,
        checked_at: datetime | None = None,
    ) -> Article:
        article = existing.clone() if existing else Article(
            id=article_id_for_url(normalized.normalized_url),
            normalized_url=normalized.normalized_url,
            original_urls=[normalized.original_url],
            resolved_url=normalized.resolved_url,
            canonical_url=normalized.canonical_url,
            domain=normalized.domain,
        )
        article.original_urls = list(
            dict.fromkeys([*article.original_urls, normalized.original_url])
        )
        article.status = status
        article.checked_at = checked_at or self._now()
        article.http_status = http_status
        article.error_code = error_code
        article.error_message = error_message[:500]
        return article

    async def fetch(
        self,
        original_url: str,
        *,
        existing: Article | None = None,
        run_id: str | None = None,
    ) -> ArticleFetchResult:
        attempts: list[ArticleFetchAttempt] = []
        request_count = 0
        redirects = 0
        try:
            initial = self.normalizer.normalize(original_url)
        except UrlNormalizationError as exc:
            started = self._now()
            attempts.append(
                self._attempt(
                    original_url=original_url,
                    started_at=started,
                    status="blocked" if exc.code == "blocked_private_address" else "failed",
                    error_code=exc.code,
                    error_message=str(exc),
                )
            )
            return ArticleFetchResult(None, None, attempts)

        current_url = initial.normalized_url
        conditional_headers: dict[str, str] = {}
        if existing and existing.etag:
            conditional_headers["If-None-Match"] = existing.etag
        if existing and existing.last_modified:
            conditional_headers["If-Modified-Since"] = existing.last_modified
        headers = {
            "User-Agent": self.config.user_agent,
            "Accept": "text/html,application/xhtml+xml;q=0.9",
            "Accept-Encoding": "gzip, deflate",
            **conditional_headers,
        }
        total_timeout = self.config.timeout_seconds * (
            self.config.max_redirects + self.config.retry_attempts + 2
        )

        try:
            async with asyncio.timeout(total_timeout):
                retry_attempt = 0
                while True:
                    started = self._now()
                    try:
                        await self.network_security.validate_url(current_url)
                    except NetworkSecurityError as exc:
                        code = "redirect_blocked" if redirects else exc.code
                        blocked = code in {
                            "blocked_private_address",
                            "redirect_blocked",
                            "invalid_url",
                        }
                        failure_status = "blocked" if blocked else "failed"
                        attempt = self._attempt(
                            original_url=original_url,
                            started_at=started,
                            status=failure_status,
                            error_code=code,
                            error_message=str(exc),
                            redirect_count=redirects,
                            retry_attempt=retry_attempt,
                        )
                        attempts.append(attempt)
                        article = self._failed_article(
                            initial,
                            existing,
                            status=failure_status,
                            error_code=code,
                            error_message=str(exc),
                        )
                        return ArticleFetchResult(article, initial, attempts, request_count, redirects)

                    try:
                        async with self.client.stream(
                            "GET",
                            current_url,
                            headers=headers,
                            follow_redirects=False,
                            timeout=self.config.timeout_seconds,
                        ) as response:
                            request_count += 1
                            status_code = response.status_code
                            if status_code in self._REDIRECTS:
                                location = response.headers.get("Location")
                                if not location:
                                    raise httpx.ProtocolError("redirect response has no Location")
                                redirects += 1
                                attempts.append(
                                    self._attempt(
                                        original_url=original_url,
                                        started_at=started,
                                        status="redirect",
                                        http_status=status_code,
                                        redirect_count=redirects,
                                        retry_attempt=retry_attempt,
                                    )
                                )
                                if redirects > self.config.max_redirects:
                                    article = self._failed_article(
                                        initial,
                                        existing,
                                        status="failed",
                                        error_code="too_many_redirects",
                                        error_message="Maximum redirect count exceeded",
                                        http_status=status_code,
                                    )
                                    return ArticleFetchResult(
                                        article, initial, attempts, request_count, redirects
                                    )
                                current_url = urljoin(current_url, location)
                                continue

                            if status_code == 304 and existing is not None:
                                article = existing.clone()
                                article.status = "unchanged"
                                article.checked_at = self._now()
                                article.http_status = 304
                                article.error_code = None
                                article.error_message = None
                                attempts.append(
                                    self._attempt(
                                        original_url=original_url,
                                        started_at=started,
                                        status="unchanged",
                                        article_id=article.id,
                                        http_status=304,
                                        redirect_count=redirects,
                                        retry_attempt=retry_attempt,
                                    )
                                )
                                return ArticleFetchResult(
                                    article, initial, attempts, request_count, redirects
                                )

                            error_code = None
                            retryable = False
                            if status_code in {401, 403, 404, 429}:
                                error_code = f"http_{status_code}"
                                retryable = status_code == 429
                            elif 500 <= status_code <= 599:
                                error_code = "http_5xx"
                                retryable = True
                            elif status_code >= 400:
                                error_code = f"http_{status_code}"
                            if error_code:
                                attempts.append(
                                    self._attempt(
                                        original_url=original_url,
                                        started_at=started,
                                        status="failed",
                                        error_code=error_code,
                                        error_message=f"HTTP request failed with status {status_code}",
                                        http_status=status_code,
                                        redirect_count=redirects,
                                        retry_attempt=retry_attempt,
                                    )
                                )
                                if retryable and retry_attempt < self.config.retry_attempts:
                                    retry_attempt += 1
                                    if status_code == 429:
                                        await self._sleep(self._retry_after(response))
                                    continue
                                article = self._failed_article(
                                    initial,
                                    existing,
                                    status="failed",
                                    error_code=error_code,
                                    error_message=f"HTTP request failed with status {status_code}",
                                    http_status=status_code,
                                )
                                return ArticleFetchResult(
                                    article, initial, attempts, request_count, redirects
                                )

                            content_type, charset = self._content_type(
                                response.headers.get("Content-Type")
                            )
                            if content_type not in self._HTML_TYPES:
                                attempts.append(
                                    self._attempt(
                                        original_url=original_url,
                                        started_at=started,
                                        status="unsupported",
                                        error_code="unsupported_content_type",
                                        error_message="Response is not an HTML document",
                                        http_status=status_code,
                                        redirect_count=redirects,
                                        retry_attempt=retry_attempt,
                                    )
                                )
                                article = self._failed_article(
                                    initial,
                                    existing,
                                    status="unsupported",
                                    error_code="unsupported_content_type",
                                    error_message="Response is not an HTML document",
                                    http_status=status_code,
                                )
                                article.content_type = content_type or None
                                return ArticleFetchResult(
                                    article, initial, attempts, request_count, redirects
                                )

                            body = bytearray()
                            async for chunk in response.aiter_bytes():
                                body.extend(chunk)
                                if len(body) > self.config.max_response_bytes:
                                    attempts.append(
                                        self._attempt(
                                            original_url=original_url,
                                            started_at=started,
                                            status="too_large",
                                            error_code="response_too_large",
                                            error_message="Decoded response exceeded the byte limit",
                                            http_status=status_code,
                                            redirect_count=redirects,
                                            response_bytes=len(body),
                                            retry_attempt=retry_attempt,
                                        )
                                    )
                                    article = self._failed_article(
                                        initial,
                                        existing,
                                        status="too_large",
                                        error_code="response_too_large",
                                        error_message="Decoded response exceeded the byte limit",
                                        http_status=status_code,
                                    )
                                    article.content_type = content_type
                                    return ArticleFetchResult(
                                        article, initial, attempts, request_count, redirects
                                    )
                            try:
                                html = bytes(body).decode(charset or "utf-8", errors="strict")
                            except (LookupError, UnicodeDecodeError):
                                attempts.append(
                                    self._attempt(
                                        original_url=original_url,
                                        started_at=started,
                                        status="failed",
                                        error_code="decode_error",
                                        error_message="HTML response could not be decoded",
                                        http_status=status_code,
                                        redirect_count=redirects,
                                        response_bytes=len(body),
                                        retry_attempt=retry_attempt,
                                    )
                                )
                                article = self._failed_article(
                                    initial,
                                    existing,
                                    status="failed",
                                    error_code="decode_error",
                                    error_message="HTML response could not be decoded",
                                    http_status=status_code,
                                )
                                article.content_type = content_type
                                return ArticleFetchResult(
                                    article, initial, attempts, request_count, redirects
                                )

                            extracted = extract_article(
                                html, max_article_chars=self.config.max_article_chars
                            )
                            declared_canonical = (
                                urljoin(current_url, extracted.canonical_urls[0])
                                if len(extracted.canonical_urls) == 1
                                else None
                            )
                            safe_canonical = None
                            if declared_canonical:
                                try:
                                    await self.network_security.validate_url(declared_canonical)
                                except NetworkSecurityError:
                                    safe_canonical = None
                                else:
                                    safe_canonical = declared_canonical
                            final_normalized = self.normalizer.normalize(
                                original_url,
                                resolved_url=current_url,
                                canonical_url=safe_canonical,
                            )
                            final_normalized.canonical_url = declared_canonical
                            article_id = article_id_for_url(final_normalized.normalized_url)
                            original_urls = [original_url]
                            if existing:
                                original_urls = [*existing.original_urls, original_url]
                            article_status = "fetched" if extracted.content_text else "empty"
                            error = None if extracted.content_text else "extraction_empty"
                            article = Article(
                                id=article_id,
                                normalized_url=final_normalized.normalized_url,
                                original_urls=list(dict.fromkeys(original_urls)),
                                resolved_url=current_url,
                                canonical_url=declared_canonical,
                                domain=final_normalized.domain,
                                title=extracted.title,
                                author=extracted.author,
                                published_at=extracted.published_at,
                                site_name=extracted.site_name,
                                language=extracted.language,
                                excerpt=extracted.excerpt,
                                content_text=extracted.content_text,
                                content_hash=extracted.content_hash,
                                word_count=extracted.word_count,
                                fetched_at=self._now(),
                                checked_at=self._now(),
                                status=article_status,
                                http_status=status_code,
                                content_type=content_type,
                                etag=response.headers.get("ETag"),
                                last_modified=response.headers.get("Last-Modified"),
                                extractor=extracted.extractor,
                                error_code=error,
                                error_message=(
                                    "No article body could be extracted" if error else None
                                ),
                                content_truncated=extracted.content_truncated,
                            )
                            attempts.append(
                                self._attempt(
                                    original_url=original_url,
                                    started_at=started,
                                    status=article_status,
                                    article_id=article.id,
                                    error_code=error,
                                    error_message=article.error_message,
                                    http_status=status_code,
                                    redirect_count=redirects,
                                    response_bytes=len(body),
                                    retry_attempt=retry_attempt,
                                )
                            )
                            fields = safe_url_log_fields(final_normalized.normalized_url)
                            logger.info(
                                "article_fetch run_id=%s article_id=%s domain=%s path_hash=%s "
                                "fetch_status=%s http_status=%s redirect_count=%s response_bytes=%s "
                                "extractor=%s word_count=%s cache_hit=false retry_attempt=%s duration_ms=%s",
                                run_id,
                                article.id,
                                fields["domain"],
                                fields["path_hash"],
                                article.status,
                                article.http_status,
                                redirects,
                                len(body),
                                article.extractor,
                                article.word_count,
                                retry_attempt,
                                attempts[-1].duration_ms,
                            )
                            return ArticleFetchResult(
                                article,
                                final_normalized,
                                attempts,
                                request_count,
                                redirects,
                            )
                    except httpx.TimeoutException:
                        code = "timeout"
                    except httpx.RequestError:
                        code = "connection_error"
                    if retry_attempt < self.config.retry_attempts:
                        attempts.append(
                            self._attempt(
                                original_url=original_url,
                                started_at=started,
                                status="failed",
                                error_code=code,
                                error_message="Article request failed before a response",
                                redirect_count=redirects,
                                retry_attempt=retry_attempt,
                            )
                        )
                        retry_attempt += 1
                        continue
                    attempts.append(
                        self._attempt(
                            original_url=original_url,
                            started_at=started,
                            status="failed",
                            error_code=code,
                            error_message="Article request failed before a response",
                            redirect_count=redirects,
                            retry_attempt=retry_attempt,
                        )
                    )
                    article = self._failed_article(
                        initial,
                        existing,
                        status="failed",
                        error_code=code,
                        error_message="Article request failed before a response",
                    )
                    return ArticleFetchResult(article, initial, attempts, request_count, redirects)
        except TimeoutError:
            attempts.append(
                self._attempt(
                    original_url=original_url,
                    started_at=self._now(),
                    status="failed",
                    error_code="timeout",
                    error_message="Total article fetch timeout exceeded",
                    redirect_count=redirects,
                )
            )
            article = self._failed_article(
                initial,
                existing,
                status="failed",
                error_code="timeout",
                error_message="Total article fetch timeout exceeded",
            )
            return ArticleFetchResult(article, initial, attempts, request_count, redirects)


__all__ = [
    "ArticleFetchConfig",
    "ArticleFetchResult",
    "ArticleFetcher",
]
