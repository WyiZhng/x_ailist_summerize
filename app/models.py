"""Canonical, storage-friendly models for X ingestion.

The project deliberately uses dataclasses instead of adding a validation
dependency.  Every model has an explicit JSON representation and all temporal
values are either ``None`` or timezone-aware.  API payloads are copied before
they are retained so later normalization cannot mutate the evidence received
from X.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Mapping


JsonObject = dict[str, Any]


def utc_now() -> datetime:
    """Return an aware UTC timestamp (kept as a function for test injection)."""

    return datetime.now(timezone.utc)


def ensure_aware_datetime(value: datetime, *, field_name: str = "datetime") -> datetime:
    """Reject naive datetimes instead of silently assigning an assumed zone."""

    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must include timezone information")
    return value


def parse_datetime(value: Any, *, field_name: str = "datetime") -> datetime | None:
    """Parse RFC 3339/ISO 8601 (and legacy RFC 2822) without losing timezone."""

    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return ensure_aware_datetime(value, field_name=field_name)
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be an ISO timestamp, datetime, or None")

    text = value.strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00" if text.endswith("Z") else text)
    except ValueError:
        try:
            parsed = parsedate_to_datetime(text)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} is not a supported timestamp") from exc
    return ensure_aware_datetime(parsed, field_name=field_name)


def datetime_to_json(value: datetime | None) -> str | None:
    if value is None:
        return None
    return ensure_aware_datetime(value).isoformat()


def _string(value: Any, default: str = "") -> str:
    return value if isinstance(value, str) else default


def _id(value: Any) -> str:
    return "" if value is None else str(value)


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _optional_id(value: Any) -> str | None:
    normalized = _id(value)
    return normalized or None


def _count(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return max(0, number)


def _optional_count(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return max(0, number)


@dataclass(slots=True)
class XAuthor:
    id: str
    username: str = ""
    name: str | None = None
    profile_image_url: str | None = None

    def __post_init__(self) -> None:
        self.id = _id(self.id)
        self.username = _string(self.username)
        self.name = _optional_string(self.name)
        self.profile_image_url = _optional_string(self.profile_image_url)

    def to_dict(self) -> JsonObject:
        return {
            "id": self.id,
            "username": self.username,
            "name": self.name,
            "profile_image_url": self.profile_image_url,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "XAuthor":
        return cls(
            id=value.get("id"),
            username=_string(value.get("username")),
            name=_optional_string(value.get("name")),
            profile_image_url=_optional_string(value.get("profile_image_url")),
        )


@dataclass(slots=True)
class XPostMetrics:
    likes: int = 0
    retweets: int = 0
    replies: int = 0
    quotes: int = 0
    bookmarks: int = 0
    impressions: int | None = None

    def __post_init__(self) -> None:
        self.likes = _count(self.likes)
        self.retweets = _count(self.retweets)
        self.replies = _count(self.replies)
        self.quotes = _count(self.quotes)
        self.bookmarks = _count(self.bookmarks)
        self.impressions = _optional_count(self.impressions)

    def to_dict(self) -> JsonObject:
        return {
            "likes": self.likes,
            "retweets": self.retweets,
            "replies": self.replies,
            "quotes": self.quotes,
            "bookmarks": self.bookmarks,
            "impressions": self.impressions,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | None) -> "XPostMetrics":
        value = value or {}
        return cls(
            likes=value.get("likes", 0),
            retweets=value.get("retweets", 0),
            replies=value.get("replies", 0),
            quotes=value.get("quotes", 0),
            bookmarks=value.get("bookmarks", 0),
            impressions=value.get("impressions"),
        )


@dataclass(slots=True)
class XPostReference:
    relation_type: str
    referenced_post_id: str

    def __post_init__(self) -> None:
        # Unknown future API relation types are intentionally retained.
        self.relation_type = _string(self.relation_type, "unknown") or "unknown"
        self.referenced_post_id = _id(self.referenced_post_id)

    def to_dict(self) -> JsonObject:
        return {
            "relation_type": self.relation_type,
            "referenced_post_id": self.referenced_post_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "XPostReference":
        return cls(
            relation_type=_string(value.get("relation_type"), "unknown"),
            referenced_post_id=value.get("referenced_post_id"),
        )


@dataclass(slots=True)
class XPost:
    id: str
    text: str
    author_id: str
    author: XAuthor | None
    created_at: datetime | None
    language: str | None
    conversation_id: str | None
    source_list_id: str
    metrics: XPostMetrics = field(default_factory=XPostMetrics)
    references: list[XPostReference] = field(default_factory=list)
    urls: list[str] = field(default_factory=list)
    media: list[JsonObject] = field(default_factory=list)
    raw_payload: JsonObject = field(default_factory=dict)
    fetched_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        self.id = _id(self.id)
        self.text = _string(self.text)
        self.author_id = _id(self.author_id)
        if isinstance(self.author, Mapping):
            self.author = XAuthor.from_dict(self.author)
        elif self.author is not None and not isinstance(self.author, XAuthor):
            self.author = None
        self.created_at = parse_datetime(self.created_at, field_name="created_at")
        self.language = _optional_string(self.language)
        self.conversation_id = _optional_string(self.conversation_id)
        self.source_list_id = _id(self.source_list_id)
        if not isinstance(self.metrics, XPostMetrics):
            self.metrics = XPostMetrics.from_dict(self.metrics)
        self.references = [
            item if isinstance(item, XPostReference) else XPostReference.from_dict(item)
            for item in self.references
        ]
        self.urls = [item for item in self.urls if isinstance(item, str) and item]
        self.media = [copy.deepcopy(item) for item in self.media if isinstance(item, dict)]
        self.raw_payload = copy.deepcopy(self.raw_payload) if isinstance(self.raw_payload, dict) else {}
        self.fetched_at = ensure_aware_datetime(self.fetched_at, field_name="fetched_at")

    def to_dict(self) -> JsonObject:
        return {
            "id": self.id,
            "text": self.text,
            "author_id": self.author_id,
            "author": self.author.to_dict() if self.author else None,
            "created_at": datetime_to_json(self.created_at),
            "language": self.language,
            "conversation_id": self.conversation_id,
            "source_list_id": self.source_list_id,
            "metrics": self.metrics.to_dict(),
            "references": [reference.to_dict() for reference in self.references],
            "urls": list(self.urls),
            "media": copy.deepcopy(self.media),
            "raw_payload": copy.deepcopy(self.raw_payload),
            "fetched_at": datetime_to_json(self.fetched_at),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "XPost":
        author = value.get("author")
        fetched_at = parse_datetime(value.get("fetched_at"), field_name="fetched_at")
        return cls(
            id=value.get("id"),
            text=_string(value.get("text")),
            author_id=value.get("author_id"),
            author=XAuthor.from_dict(author) if isinstance(author, Mapping) else None,
            created_at=parse_datetime(value.get("created_at"), field_name="created_at"),
            language=_optional_string(value.get("language")),
            conversation_id=_optional_string(value.get("conversation_id")),
            source_list_id=value.get("source_list_id"),
            metrics=XPostMetrics.from_dict(
                value.get("metrics") if isinstance(value.get("metrics"), Mapping) else None
            ),
            references=[
                XPostReference.from_dict(item)
                for item in value.get("references", [])
                if isinstance(item, Mapping)
            ],
            urls=list(value.get("urls", [])) if isinstance(value.get("urls"), list) else [],
            media=list(value.get("media", [])) if isinstance(value.get("media"), list) else [],
            raw_payload=(
                dict(value.get("raw_payload", {}))
                if isinstance(value.get("raw_payload"), Mapping)
                else {}
            ),
            fetched_at=fetched_at or utc_now(),
        )

    def to_legacy_tweet(self) -> JsonObject:
        """Adapt the canonical model to the phase-0 report pipeline."""

        return {
            "id": self.id,
            "text": self.text,
            "author": self.author.username if self.author else "unknown",
            "author_id": self.author_id,
            "created_at": datetime_to_json(self.created_at),
            "language": self.language,
            "conversation_id": self.conversation_id,
            "links": list(self.urls),
            "media": copy.deepcopy(self.media),
            "card": None,
            "likes": self.metrics.likes,
            "retweets": self.metrics.retweets,
            "replies": self.metrics.replies,
            "quotes": self.metrics.quotes,
            "bookmarks": self.metrics.bookmarks,
            "impressions": self.metrics.impressions,
            "references": [reference.to_dict() for reference in self.references],
            "raw_payload": copy.deepcopy(self.raw_payload),
        }


@dataclass(slots=True)
class ListSyncState:
    list_id: str
    newest_post_id: str | None = None
    newest_post_created_at: datetime | None = None
    last_attempt_at: datetime | None = None
    last_success_at: datetime | None = None
    last_run_id: str | None = None
    last_status: str = "never"
    last_error: str | None = None

    def __post_init__(self) -> None:
        self.list_id = _id(self.list_id)
        self.newest_post_id = _optional_id(self.newest_post_id)
        self.newest_post_created_at = parse_datetime(
            self.newest_post_created_at, field_name="newest_post_created_at"
        )
        self.last_attempt_at = parse_datetime(self.last_attempt_at, field_name="last_attempt_at")
        self.last_success_at = parse_datetime(self.last_success_at, field_name="last_success_at")
        self.last_run_id = _optional_id(self.last_run_id)
        self.last_status = _string(self.last_status, "never") or "never"
        self.last_error = _optional_string(self.last_error)

    def to_dict(self) -> JsonObject:
        return {
            "list_id": self.list_id,
            "newest_post_id": self.newest_post_id,
            "newest_post_created_at": datetime_to_json(self.newest_post_created_at),
            "last_attempt_at": datetime_to_json(self.last_attempt_at),
            "last_success_at": datetime_to_json(self.last_success_at),
            "last_run_id": self.last_run_id,
            "last_status": self.last_status,
            "last_error": self.last_error,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ListSyncState":
        return cls(**{key: value.get(key) for key in cls.__dataclass_fields__})


@dataclass(slots=True)
class IngestionRun:
    run_id: str
    started_at: datetime
    finished_at: datetime | None = None
    status: str = "running"
    ingestion_owner_id: str | None = None
    ingestion_heartbeat_at: datetime | None = None
    requested_lists: list[str] = field(default_factory=list)
    successful_lists: list[str] = field(default_factory=list)
    failed_lists: list[str] = field(default_factory=list)
    fetched_count: int = 0
    new_post_count: int = 0
    duplicate_count: int = 0
    new_post_ids: list[str] = field(default_factory=list)
    errors: list[JsonObject] = field(default_factory=list)
    report_status: str = "not_started"
    report_error: str | None = None
    report_claim_id: str | None = None
    report_claimed_at: datetime | None = None

    VALID_STATUSES = frozenset(
        {"running", "success", "partial_success", "failed", "no_new_posts"}
    )

    def __post_init__(self) -> None:
        self.run_id = _id(self.run_id)
        self.started_at = ensure_aware_datetime(self.started_at, field_name="started_at")
        self.finished_at = parse_datetime(self.finished_at, field_name="finished_at")
        if self.status not in self.VALID_STATUSES:
            raise ValueError(f"Unsupported ingestion status: {self.status}")
        self.ingestion_owner_id = _optional_id(self.ingestion_owner_id)
        self.ingestion_heartbeat_at = parse_datetime(
            self.ingestion_heartbeat_at, field_name="ingestion_heartbeat_at"
        )
        self.requested_lists = [str(item) for item in self.requested_lists]
        self.successful_lists = [str(item) for item in self.successful_lists]
        self.failed_lists = [str(item) for item in self.failed_lists]
        self.fetched_count = _count(self.fetched_count)
        self.new_post_count = _count(self.new_post_count)
        self.duplicate_count = _count(self.duplicate_count)
        self.new_post_ids = [str(item) for item in self.new_post_ids]
        self.errors = [copy.deepcopy(item) for item in self.errors if isinstance(item, dict)]
        self.report_status = _string(self.report_status, "not_started") or "not_started"
        self.report_error = _optional_string(self.report_error)
        self.report_claim_id = _optional_id(self.report_claim_id)
        self.report_claimed_at = parse_datetime(
            self.report_claimed_at, field_name="report_claimed_at"
        )

    def to_dict(self) -> JsonObject:
        return {
            "run_id": self.run_id,
            "started_at": datetime_to_json(self.started_at),
            "finished_at": datetime_to_json(self.finished_at),
            "status": self.status,
            "ingestion_owner_id": self.ingestion_owner_id,
            "ingestion_heartbeat_at": datetime_to_json(self.ingestion_heartbeat_at),
            "requested_lists": list(self.requested_lists),
            "successful_lists": list(self.successful_lists),
            "failed_lists": list(self.failed_lists),
            "fetched_count": self.fetched_count,
            "new_post_count": self.new_post_count,
            "duplicate_count": self.duplicate_count,
            "new_post_ids": list(self.new_post_ids),
            "errors": copy.deepcopy(self.errors),
            "report_status": self.report_status,
            "report_error": self.report_error,
            "report_claim_id": self.report_claim_id,
            "report_claimed_at": datetime_to_json(self.report_claimed_at),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "IngestionRun":
        started_at = parse_datetime(value.get("started_at"), field_name="started_at")
        if started_at is None:
            raise ValueError("started_at is required")
        return cls(
            run_id=str(value.get("run_id", "")),
            started_at=started_at,
            finished_at=parse_datetime(value.get("finished_at"), field_name="finished_at"),
            status=_string(value.get("status"), "running"),
            ingestion_owner_id=_optional_id(value.get("ingestion_owner_id")),
            ingestion_heartbeat_at=parse_datetime(
                value.get("ingestion_heartbeat_at"),
                field_name="ingestion_heartbeat_at",
            ),
            requested_lists=list(value.get("requested_lists", [])),
            successful_lists=list(value.get("successful_lists", [])),
            failed_lists=list(value.get("failed_lists", [])),
            fetched_count=value.get("fetched_count", 0),
            new_post_count=value.get("new_post_count", 0),
            duplicate_count=value.get("duplicate_count", 0),
            new_post_ids=list(value.get("new_post_ids", [])),
            errors=list(value.get("errors", [])),
            report_status=_string(value.get("report_status"), "not_started"),
            report_error=_optional_string(value.get("report_error")),
            report_claim_id=_optional_id(value.get("report_claim_id")),
            report_claimed_at=parse_datetime(
                value.get("report_claimed_at"), field_name="report_claimed_at"
            ),
        )


__all__ = [
    "IngestionRun",
    "JsonObject",
    "ListSyncState",
    "XAuthor",
    "XPost",
    "XPostMetrics",
    "XPostReference",
    "datetime_to_json",
    "ensure_aware_datetime",
    "parse_datetime",
    "utc_now",
]
