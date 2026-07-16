"""Storage-friendly models for untrusted external article content."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping

from .models import datetime_to_json, ensure_aware_datetime, parse_datetime, utc_now


ARTICLE_STATUSES = frozenset(
    {
        "pending",
        "fetched",
        "unchanged",
        "failed",
        "blocked",
        "unsupported",
        "empty",
        "too_large",
    }
)


def stable_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def article_id_for_url(normalized_url: str) -> str:
    return f"article_{stable_sha256(normalized_url)}"


@dataclass(slots=True)
class NormalizedUrl:
    original_url: str
    resolved_url: str | None
    canonical_url: str | None
    normalized_url: str
    domain: str
    url_hash: str
    removed_tracking_params: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_url": self.original_url,
            "resolved_url": self.resolved_url,
            "canonical_url": self.canonical_url,
            "normalized_url": self.normalized_url,
            "domain": self.domain,
            "url_hash": self.url_hash,
            "removed_tracking_params": list(self.removed_tracking_params),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "NormalizedUrl":
        return cls(
            original_url=str(value.get("original_url") or ""),
            resolved_url=(str(value["resolved_url"]) if value.get("resolved_url") else None),
            canonical_url=(str(value["canonical_url"]) if value.get("canonical_url") else None),
            normalized_url=str(value.get("normalized_url") or ""),
            domain=str(value.get("domain") or ""),
            url_hash=str(value.get("url_hash") or ""),
            removed_tracking_params=[
                str(item)
                for item in value.get("removed_tracking_params", [])
                if isinstance(item, str)
            ],
        )


@dataclass(slots=True)
class Article:
    id: str
    normalized_url: str
    original_urls: list[str]
    resolved_url: str | None = None
    canonical_url: str | None = None
    domain: str = ""
    title: str | None = None
    author: str | None = None
    published_at: datetime | None = None
    site_name: str | None = None
    language: str | None = None
    excerpt: str | None = None
    content_text: str | None = None
    content_hash: str | None = None
    word_count: int | None = None
    fetched_at: datetime = field(default_factory=utc_now)
    checked_at: datetime = field(default_factory=utc_now)
    status: str = "pending"
    http_status: int | None = None
    content_type: str | None = None
    etag: str | None = None
    last_modified: str | None = None
    extractor: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    content_truncated: bool = False
    content_is_untrusted: bool = True

    def __post_init__(self) -> None:
        self.id = str(self.id)
        self.normalized_url = str(self.normalized_url)
        self.original_urls = list(
            dict.fromkeys(str(item) for item in self.original_urls if str(item))
        )
        self.published_at = parse_datetime(self.published_at, field_name="published_at")
        self.fetched_at = ensure_aware_datetime(self.fetched_at, field_name="fetched_at")
        self.checked_at = ensure_aware_datetime(self.checked_at, field_name="checked_at")
        if self.status not in ARTICLE_STATUSES:
            raise ValueError(f"unsupported article status: {self.status}")
        if self.word_count is not None:
            self.word_count = max(0, int(self.word_count))
        if self.http_status is not None:
            self.http_status = int(self.http_status)

    def to_dict(self, *, include_content: bool = True) -> dict[str, Any]:
        value = {
            "id": self.id,
            "normalized_url": self.normalized_url,
            "original_urls": list(self.original_urls),
            "resolved_url": self.resolved_url,
            "canonical_url": self.canonical_url,
            "domain": self.domain,
            "title": self.title,
            "author": self.author,
            "published_at": datetime_to_json(self.published_at),
            "site_name": self.site_name,
            "language": self.language,
            "excerpt": self.excerpt,
            "content_hash": self.content_hash,
            "word_count": self.word_count,
            "fetched_at": datetime_to_json(self.fetched_at),
            "checked_at": datetime_to_json(self.checked_at),
            "status": self.status,
            "http_status": self.http_status,
            "content_type": self.content_type,
            "etag": self.etag,
            "last_modified": self.last_modified,
            "extractor": self.extractor,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "content_truncated": self.content_truncated,
            "content_is_untrusted": self.content_is_untrusted,
        }
        if include_content:
            value["content_text"] = self.content_text
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Article":
        fetched_at = parse_datetime(value.get("fetched_at"), field_name="fetched_at")
        checked_at = parse_datetime(value.get("checked_at"), field_name="checked_at")
        return cls(
            id=str(value.get("id") or ""),
            normalized_url=str(value.get("normalized_url") or ""),
            original_urls=[
                str(item)
                for item in value.get("original_urls", [])
                if isinstance(item, str)
            ],
            resolved_url=value.get("resolved_url"),
            canonical_url=value.get("canonical_url"),
            domain=str(value.get("domain") or ""),
            title=value.get("title"),
            author=value.get("author"),
            published_at=parse_datetime(value.get("published_at"), field_name="published_at"),
            site_name=value.get("site_name"),
            language=value.get("language"),
            excerpt=value.get("excerpt"),
            content_text=value.get("content_text"),
            content_hash=value.get("content_hash"),
            word_count=value.get("word_count"),
            fetched_at=fetched_at or utc_now(),
            checked_at=checked_at or fetched_at or utc_now(),
            status=str(value.get("status") or "pending"),
            http_status=value.get("http_status"),
            content_type=value.get("content_type"),
            etag=value.get("etag"),
            last_modified=value.get("last_modified"),
            extractor=value.get("extractor"),
            error_code=value.get("error_code"),
            error_message=value.get("error_message"),
            content_truncated=bool(value.get("content_truncated", False)),
            content_is_untrusted=bool(value.get("content_is_untrusted", True)),
        )

    def clone(self) -> "Article":
        return Article.from_dict(copy.deepcopy(self.to_dict()))


@dataclass(slots=True)
class PostArticleLink:
    post_id: str
    article_id: str
    original_url: str
    discovered_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        self.post_id = str(self.post_id)
        self.article_id = str(self.article_id)
        self.original_url = str(self.original_url)
        self.discovered_at = ensure_aware_datetime(
            self.discovered_at, field_name="discovered_at"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "post_id": self.post_id,
            "article_id": self.article_id,
            "original_url": self.original_url,
            "discovered_at": datetime_to_json(self.discovered_at),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PostArticleLink":
        discovered = parse_datetime(value.get("discovered_at"), field_name="discovered_at")
        return cls(
            post_id=str(value.get("post_id") or ""),
            article_id=str(value.get("article_id") or ""),
            original_url=str(value.get("original_url") or ""),
            discovered_at=discovered or utc_now(),
        )


@dataclass(slots=True)
class ArticleFetchAttempt:
    id: str
    original_url: str
    started_at: datetime
    finished_at: datetime
    status: str
    article_id: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    http_status: int | None = None
    redirect_count: int = 0
    response_bytes: int = 0
    retry_attempt: int = 0
    duration_ms: int = 0

    def __post_init__(self) -> None:
        self.started_at = ensure_aware_datetime(self.started_at, field_name="started_at")
        self.finished_at = ensure_aware_datetime(self.finished_at, field_name="finished_at")
        self.redirect_count = max(0, int(self.redirect_count))
        self.response_bytes = max(0, int(self.response_bytes))
        self.retry_attempt = max(0, int(self.retry_attempt))
        self.duration_ms = max(0, int(self.duration_ms))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "article_id": self.article_id,
            "original_url": self.original_url,
            "started_at": datetime_to_json(self.started_at),
            "finished_at": datetime_to_json(self.finished_at),
            "status": self.status,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "http_status": self.http_status,
            "redirect_count": self.redirect_count,
            "response_bytes": self.response_bytes,
            "retry_attempt": self.retry_attempt,
            "duration_ms": self.duration_ms,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ArticleFetchAttempt":
        started = parse_datetime(value.get("started_at"), field_name="started_at")
        finished = parse_datetime(value.get("finished_at"), field_name="finished_at")
        return cls(
            id=str(value.get("id") or ""),
            article_id=(str(value["article_id"]) if value.get("article_id") else None),
            original_url=str(value.get("original_url") or ""),
            started_at=started or utc_now(),
            finished_at=finished or started or utc_now(),
            status=str(value.get("status") or "failed"),
            error_code=value.get("error_code"),
            error_message=value.get("error_message"),
            http_status=value.get("http_status"),
            redirect_count=value.get("redirect_count", 0),
            response_bytes=value.get("response_bytes", 0),
            retry_attempt=value.get("retry_attempt", 0),
            duration_ms=value.get("duration_ms", 0),
        )


__all__ = [
    "ARTICLE_STATUSES",
    "Article",
    "ArticleFetchAttempt",
    "NormalizedUrl",
    "PostArticleLink",
    "article_id_for_url",
    "stable_sha256",
]
