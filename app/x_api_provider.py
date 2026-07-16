"""Official X API v2 provider with bounded retries and canonical models.

The List Posts endpoint currently documents pagination but not ``since_id``.
Consequently ``supports_since_id`` defaults to ``False`` and incremental
collection falls back to walking pages until the stored post ID is reached.
Callers may explicitly enable the capability for a compatible API contract.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Awaitable, Callable, Mapping, Sequence

import httpx

from .models import XAuthor, XPost, XPostMetrics, XPostReference, parse_datetime, utc_now


logger = logging.getLogger(__name__)

Sleep = Callable[[float], Awaitable[None]]

TWEET_FIELDS = (
    "id,text,author_id,created_at,lang,conversation_id,public_metrics,"
    "entities,attachments,referenced_tweets,in_reply_to_user_id,note_tweet"
)
EXPANSIONS = (
    "author_id,attachments.media_keys,referenced_tweets.id,"
    "referenced_tweets.id.author_id,"
    "referenced_tweets.id.attachments.media_keys"
)
USER_FIELDS = "id,username,name,profile_image_url"
MEDIA_FIELDS = (
    "media_key,type,url,preview_image_url,variants,alt_text,width,height,duration_ms"
)
API_RECENT_POST_LIMIT = 800


def _header_int(headers: Mapping[str, str], name: str) -> int | None:
    raw = headers.get(name)
    if raw is None:
        # Plain test mappings may not be case-insensitive like httpx.Headers.
        raw = next((value for key, value in headers.items() if key.lower() == name.lower()), None)
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError, OverflowError):
        return None


@dataclass(frozen=True, slots=True)
class RateLimitInfo:
    limit: int | None = None
    remaining: int | None = None
    reset_at: int | None = None
    retry_after: float | None = None

    @classmethod
    def from_headers(cls, headers: Mapping[str, str]) -> "RateLimitInfo":
        retry_after_raw = headers.get("retry-after")
        if retry_after_raw is None:
            retry_after_raw = next(
                (value for key, value in headers.items() if key.lower() == "retry-after"), None
            )
        try:
            retry_after = max(0.0, float(retry_after_raw)) if retry_after_raw else None
        except (TypeError, ValueError, OverflowError):
            retry_after = None
        return cls(
            limit=_header_int(headers, "x-rate-limit-limit"),
            remaining=_header_int(headers, "x-rate-limit-remaining"),
            reset_at=_header_int(headers, "x-rate-limit-reset"),
            retry_after=retry_after,
        )

    def to_dict(self) -> dict[str, int | float | None]:
        return {
            "limit": self.limit,
            "remaining": self.remaining,
            "reset_at": self.reset_at,
            "retry_after": self.retry_after,
        }


class XApiProviderError(RuntimeError):
    """Sanitized, machine-readable provider failure."""

    def __init__(
        self,
        message: str,
        *,
        kind: str,
        status_code: int | None = None,
        retryable: bool = False,
        attempts: int = 1,
        rate_limit: RateLimitInfo | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.status_code = status_code
        self.retryable = retryable
        self.attempts = attempts
        self.rate_limit = rate_limit

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "message": str(self),
            "status_code": self.status_code,
            "retryable": self.retryable,
            "attempts": self.attempts,
            "rate_limit": self.rate_limit.to_dict() if self.rate_limit else None,
        }


@dataclass(slots=True)
class XFetchPage:
    list_id: str
    posts: list[XPost] = field(default_factory=list)
    next_token: str | None = None
    rate_limit: RateLimitInfo | None = None
    raw_errors: list[dict[str, Any]] = field(default_factory=list)
    page_number: int = 1
    requested_since_id: str | None = None
    since_id_sent: bool = False

    @property
    def fetched_count(self) -> int:
        return len(self.posts)


@dataclass(slots=True)
class XFetchResult:
    list_id: str
    posts: list[XPost] = field(default_factory=list)
    pages: list[XFetchPage] = field(default_factory=list)
    complete: bool = True
    stop_reason: str = "exhausted"
    next_token: str | None = None
    rate_limit: RateLimitInfo | None = None
    raw_errors: list[dict[str, Any]] = field(default_factory=list)

    @property
    def fetched_count(self) -> int:
        return len(self.posts)

    @property
    def page_count(self) -> int:
        return len(self.pages)


def compare_post_ids(left: str, right: str) -> int | None:
    """Compare Snowflake IDs, degrading to equality for malformed IDs.

    Returns ``1`` when ``left`` is newer, ``0`` when equal, ``-1`` when older,
    and ``None`` when ordering is unknowable.  Unknown ordering is never treated
    as old because doing so could silently lose a post.
    """

    left_text, right_text = str(left), str(right)
    if left_text == right_text:
        return 0
    # Only canonical ASCII Snowflakes are orderable.  ``str.isdigit()`` also
    # accepts values such as superscript digits, while leading-zero IDs can be
    # distinct strings with the same integer value.  Both cases must fall back
    # to exact membership checks instead of creating a false checkpoint hit.
    snowflake = re.compile(r"[1-9][0-9]{0,18}\Z")
    if snowflake.fullmatch(left_text) and snowflake.fullmatch(right_text):
        return 1 if int(left_text) > int(right_text) else -1
    return None


def _list_id(value: str) -> str:
    text = str(value).strip()
    if re.fullmatch(r"[0-9]{1,19}", text):
        return text
    match = re.fullmatch(
        r"(?:https?://)?(?:www\.)?(?:x|twitter)\.com/"
        r"(?:i/)?lists/([0-9]{1,19})(?:[/?#].*)?",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1)
    raise ValueError("X List ID must contain 1 to 19 decimal digits")


def _mapping_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [copy.deepcopy(item) for item in value if isinstance(item, dict)]


class OfficialXApiProvider:
    """Fetch and normalize public List posts using app-only Bearer auth."""

    API_BASE = "https://api.x.com/2"

    def __init__(
        self,
        bearer_token: str,
        *,
        client: httpx.AsyncClient | None = None,
        base_url: str = API_BASE,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_base_delay: float = 0.5,
        max_retry_delay: float = 30.0,
        sleep: Sleep = asyncio.sleep,
        supports_since_id: bool = False,
        fetched_at_factory: Callable[[], datetime] = utc_now,
    ) -> None:
        token = (bearer_token or "").strip()
        if not token:
            raise ValueError("Bearer Token is required")
        if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
            raise ValueError("max_retries must be a non-negative integer")
        if retry_base_delay < 0 or max_retry_delay < 0:
            raise ValueError("retry delays must be non-negative")
        self._bearer_token = token
        self._client = client
        self._owns_client = client is None
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_base_delay = float(retry_base_delay)
        self.max_retry_delay = float(max_retry_delay)
        self._sleep = sleep
        self.supports_since_id = bool(supports_since_id)
        self._fetched_at_factory = fetched_at_factory
        self.last_rate_limit: RateLimitInfo | None = None
        self.last_fetch_result: XFetchResult | None = None
        self.list_info: dict[str, Any] = {
            "name": "X List Summary",
            "list_names": [],
            "owner": "Unknown",
            "owner_name": "Unknown",
            "member_count": 0,
            "profile_image_url": None,
        }

    @property
    def bearer_token_configured(self) -> bool:
        return bool(self._bearer_token)

    async def __aenter__(self) -> "OfficialXApiProvider":
        return self

    async def __aexit__(self, *_args: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    def _retry_delay(self, attempt: int, rate_limit: RateLimitInfo | None) -> float:
        candidates = [self.retry_base_delay * (2**attempt)]
        if rate_limit:
            if rate_limit.retry_after is not None:
                candidates.append(rate_limit.retry_after)
            elif rate_limit.reset_at is not None:
                candidates.append(max(0.0, rate_limit.reset_at - time.time()))
        return min(self.max_retry_delay, max(candidates))

    async def _request_json(
        self,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        run_id: str | None = None,
        list_id: str | None = None,
        page_number: int | None = None,
    ) -> tuple[dict[str, Any], RateLimitInfo, int]:
        url = f"{self.base_url}{path}"
        headers = {
            "Authorization": f"Bearer {self._bearer_token}",
            "User-Agent": "x-list-summarizer/1.0",
        }
        client = self._get_client()

        for attempt in range(self.max_retries + 1):
            response: httpx.Response | None = None
            try:
                response = await client.get(url, headers=headers, params=dict(params or {}), timeout=self.timeout)
            except httpx.TimeoutException:
                kind = "timeout"
            except httpx.RequestError:
                kind = "network"
            else:
                rate_limit = RateLimitInfo.from_headers(response.headers)
                self.last_rate_limit = rate_limit
                status = response.status_code
                if status == 429 or 500 <= status <= 599:
                    kind = "rate_limited" if status == 429 else "server_error"
                elif status == 401:
                    raise XApiProviderError(
                        "X API authentication failed (401)",
                        kind="unauthorized",
                        status_code=status,
                        attempts=attempt + 1,
                        rate_limit=rate_limit,
                    )
                elif status == 403:
                    raise XApiProviderError(
                        "X API access is forbidden for this List (403)",
                        kind="forbidden",
                        status_code=status,
                        attempts=attempt + 1,
                        rate_limit=rate_limit,
                    )
                elif status == 404:
                    raise XApiProviderError(
                        "X List was not found (404)",
                        kind="not_found",
                        status_code=status,
                        attempts=attempt + 1,
                        rate_limit=rate_limit,
                    )
                elif status >= 400:
                    raise XApiProviderError(
                        f"X API request failed ({status})",
                        kind="client_error",
                        status_code=status,
                        attempts=attempt + 1,
                        rate_limit=rate_limit,
                    )
                else:
                    try:
                        payload = response.json()
                    except (ValueError, TypeError):
                        raise XApiProviderError(
                            "X API returned malformed JSON",
                            kind="malformed_response",
                            status_code=status,
                            attempts=attempt + 1,
                            rate_limit=rate_limit,
                        ) from None
                    if not isinstance(payload, dict):
                        raise XApiProviderError(
                            "X API returned an unexpected response shape",
                            kind="malformed_response",
                            status_code=status,
                            attempts=attempt + 1,
                            rate_limit=rate_limit,
                        )
                    return payload, rate_limit, attempt

            retryable = kind in {"timeout", "network", "rate_limited", "server_error"}
            status_code = response.status_code if response is not None else None
            rate_limit = (
                RateLimitInfo.from_headers(response.headers) if response is not None else None
            )
            if attempt >= self.max_retries:
                messages = {
                    "timeout": "X API request timed out after bounded retries",
                    "network": "X API network request failed after bounded retries",
                    "rate_limited": "X API rate limit persisted after bounded retries",
                    "server_error": "X API server error persisted after bounded retries",
                }
                raise XApiProviderError(
                    messages[kind],
                    kind=kind,
                    status_code=status_code,
                    retryable=retryable,
                    attempts=attempt + 1,
                    rate_limit=rate_limit,
                ) from None

            delay = self._retry_delay(attempt, rate_limit)
            logger.warning(
                "x_api_retry run_id=%s list_id=%s page_number=%s retry_attempt=%s status=%s "
                "rate_limit_remaining=%s kind=%s",
                run_id or "-",
                list_id or "-",
                page_number or "-",
                attempt + 1,
                status_code or "network_error",
                rate_limit.remaining if rate_limit else None,
                kind,
            )
            await self._sleep(delay)

        raise AssertionError("unreachable")

    def normalize_post(
        self,
        payload: Mapping[str, Any],
        *,
        source_list_id: str,
        users_by_id: Mapping[str, Mapping[str, Any]],
        media_by_key: Mapping[str, Mapping[str, Any]],
        fetched_at: datetime | None = None,
    ) -> XPost:
        """Map one documented v2 Post object, safely degrading missing fields."""

        post_id = "" if payload.get("id") is None else str(payload.get("id"))
        author_id = (
            "" if payload.get("author_id") is None else str(payload.get("author_id"))
        )
        author_payload = users_by_id.get(author_id)
        author = XAuthor.from_dict(author_payload) if author_payload else None

        metrics_payload = payload.get("public_metrics")
        metrics_payload = metrics_payload if isinstance(metrics_payload, Mapping) else {}
        metrics = XPostMetrics(
            likes=metrics_payload.get("like_count", 0),
            retweets=metrics_payload.get("retweet_count", 0),
            replies=metrics_payload.get("reply_count", 0),
            quotes=metrics_payload.get("quote_count", 0),
            bookmarks=metrics_payload.get("bookmark_count", 0),
            impressions=metrics_payload.get("impression_count"),
        )

        references: list[XPostReference] = []
        raw_references = payload.get("referenced_tweets")
        if isinstance(raw_references, list):
            for raw_reference in raw_references:
                if not isinstance(raw_reference, Mapping) or raw_reference.get("id") in (None, ""):
                    continue
                references.append(
                    XPostReference(
                        relation_type=str(raw_reference.get("type") or "unknown"),
                        referenced_post_id=str(raw_reference["id"]),
                    )
                )

        urls: list[str] = []
        entities = payload.get("entities")
        note_tweet = payload.get("note_tweet")
        note_entities = note_tweet.get("entities") if isinstance(note_tweet, Mapping) else None
        for entity_group in (entities, note_entities):
            raw_urls = entity_group.get("urls") if isinstance(entity_group, Mapping) else None
            if isinstance(raw_urls, list):
                for item in raw_urls:
                    if not isinstance(item, Mapping):
                        continue
                    candidate = (
                        item.get("unwound_url")
                        or item.get("expanded_url")
                        or item.get("url")
                    )
                    if isinstance(candidate, str) and candidate and candidate not in urls:
                        urls.append(candidate)

        media: list[dict[str, Any]] = []
        attachments = payload.get("attachments")
        media_keys = attachments.get("media_keys") if isinstance(attachments, Mapping) else None
        if isinstance(media_keys, list):
            for key in media_keys:
                included = media_by_key.get(str(key))
                if included:
                    media.append(copy.deepcopy(dict(included)))

        created_at: datetime | None
        try:
            created_at = parse_datetime(payload.get("created_at"), field_name="created_at")
        except (TypeError, ValueError):
            # A malformed optional timestamp must not discard the whole page.
            created_at = None

        note_text = note_tweet.get("text") if isinstance(note_tweet, Mapping) else None
        return XPost(
            id=post_id,
            text=(
                note_text
                if isinstance(note_text, str)
                else payload.get("text") if isinstance(payload.get("text"), str) else ""
            ),
            author_id=author_id,
            author=author,
            created_at=created_at,
            language=payload.get("lang") if isinstance(payload.get("lang"), str) else None,
            conversation_id=(
                str(payload.get("conversation_id"))
                if payload.get("conversation_id") not in (None, "")
                else None
            ),
            source_list_id=source_list_id,
            metrics=metrics,
            references=references,
            urls=urls,
            media=media,
            raw_payload=copy.deepcopy(dict(payload)),
            fetched_at=fetched_at or self._fetched_at_factory(),
        )

    async def fetch_page(
        self,
        list_id: str,
        *,
        pagination_token: str | None = None,
        max_results: int = 100,
        since_id: str | None = None,
        page_number: int = 1,
        run_id: str | None = None,
    ) -> XFetchPage:
        normalized_list_id = _list_id(list_id)
        if isinstance(max_results, bool) or not isinstance(max_results, int) or not 1 <= max_results <= 100:
            raise ValueError("max_results must be an integer between 1 and 100")
        if isinstance(page_number, bool) or not isinstance(page_number, int) or page_number < 1:
            raise ValueError("page_number must be a positive integer")

        params: dict[str, Any] = {
            "max_results": max_results,
            "tweet.fields": TWEET_FIELDS,
            "expansions": EXPANSIONS,
            "user.fields": USER_FIELDS,
            "media.fields": MEDIA_FIELDS,
        }
        if pagination_token:
            params["pagination_token"] = str(pagination_token)
        since_id_sent = bool(since_id and self.supports_since_id)
        if since_id_sent:
            params["since_id"] = str(since_id)

        payload, rate_limit, retry_attempt = await self._request_json(
            f"/lists/{normalized_list_id}/tweets",
            params=params,
            run_id=run_id,
            list_id=normalized_list_id,
            page_number=page_number,
        )
        raw_errors = _mapping_list(payload.get("errors"))
        raw_data = payload.get("data", [])
        if raw_data is None:
            raw_data = []
        if not isinstance(raw_data, list):
            raise XApiProviderError(
                "X API returned an unexpected data shape",
                kind="malformed_response",
                status_code=200,
                rate_limit=rate_limit,
            )
        if any(not isinstance(item, Mapping) for item in raw_data):
            raise XApiProviderError(
                "X API returned a non-object item in data",
                kind="malformed_response",
                status_code=200,
                rate_limit=rate_limit,
            )
        if raw_errors and not raw_data and "data" not in payload:
            raise XApiProviderError(
                "X API returned an application error",
                kind="api_error",
                status_code=200,
                rate_limit=rate_limit,
            )

        includes = payload.get("includes")
        includes = includes if isinstance(includes, Mapping) else {}
        users_by_id = {
            str(item.get("id")): item
            for item in includes.get("users", [])
            if isinstance(item, Mapping) and item.get("id") not in (None, "")
        }
        media_by_key = {
            str(item.get("media_key")): item
            for item in includes.get("media", [])
            if isinstance(item, Mapping) and item.get("media_key") not in (None, "")
        }
        fetched_at = self._fetched_at_factory()
        posts = [
            self.normalize_post(
                item,
                source_list_id=normalized_list_id,
                users_by_id=users_by_id,
                media_by_key=media_by_key,
                fetched_at=fetched_at,
            )
            for item in raw_data
        ]
        meta = payload.get("meta")
        meta = meta if isinstance(meta, Mapping) else {}
        next_token = meta.get("next_token")
        next_token = str(next_token) if next_token not in (None, "") else None

        logger.info(
            "x_api_page run_id=%s list_id=%s page_number=%s fetched_count=%s "
            "retry_attempt=%s rate_limit_remaining=%s status=success",
            run_id or "-",
            normalized_list_id,
            page_number,
            len(posts),
            retry_attempt,
            rate_limit.remaining,
        )
        return XFetchPage(
            list_id=normalized_list_id,
            posts=posts,
            next_token=next_token,
            rate_limit=rate_limit,
            raw_errors=raw_errors,
            page_number=page_number,
            requested_since_id=str(since_id) if since_id else None,
            since_id_sent=since_id_sent,
        )

    async def fetch_list_posts(
        self,
        list_id: str,
        *,
        max_tweets: int = 100,
        page_size: int = 100,
        max_pages: int = 20,
        since_id: str | None = None,
        pagination_token: str | None = None,
        run_id: str | None = None,
    ) -> XFetchResult:
        normalized_list_id = _list_id(list_id)
        if isinstance(max_tweets, bool) or not isinstance(max_tweets, int) or max_tweets < 1:
            raise ValueError("max_tweets must be a positive integer")
        if isinstance(page_size, bool) or not isinstance(page_size, int) or not 1 <= page_size <= 100:
            raise ValueError("page_size must be an integer between 1 and 100")
        if isinstance(max_pages, bool) or not isinstance(max_pages, int) or max_pages < 1:
            raise ValueError("max_pages must be a positive integer")

        posts: list[XPost] = []
        pages: list[XFetchPage] = []
        raw_errors: list[dict[str, Any]] = []
        seen_post_ids: set[str] = set()
        seen_tokens: set[str] = {str(pagination_token)} if pagination_token else set()
        next_token = str(pagination_token) if pagination_token else None
        stop_reason = "exhausted"
        complete = True

        for page_number in range(1, max_pages + 1):
            remaining = max_tweets - len(posts)
            requested_page_size = min(page_size, remaining)
            page = await self.fetch_page(
                normalized_list_id,
                pagination_token=next_token,
                max_results=requested_page_size,
                since_id=since_id,
                page_number=page_number,
                run_id=run_id,
            )
            pages.append(page)
            raw_errors.extend(page.raw_errors)
            reached_checkpoint = False

            for post in page.posts:
                if since_id:
                    ordering = compare_post_ids(post.id, str(since_id))
                    if ordering in {-1, 0}:
                        reached_checkpoint = True
                        break
                if post.id in seen_post_ids:
                    continue
                seen_post_ids.add(post.id)
                posts.append(post)
                if len(posts) >= max_tweets:
                    break

            if reached_checkpoint:
                stop_reason = "since_id_reached"
                next_token = page.next_token
                complete = True
                break

            next_token = page.next_token
            if len(posts) >= max_tweets:
                if next_token:
                    stop_reason = "max_tweets"
                    complete = False
                else:
                    stop_reason = "exhausted"
                    complete = True
                break
            if not next_token:
                if (
                    since_id
                    and not self.supports_since_id
                    and sum(item.fetched_count for item in pages) >= API_RECENT_POST_LIMIT
                ):
                    stop_reason = "recent_post_limit"
                    complete = False
                else:
                    stop_reason = "empty" if not posts else "exhausted"
                    complete = True
                break
            if next_token in seen_tokens:
                raise XApiProviderError(
                    "X API repeated a pagination token",
                    kind="pagination_error",
                    retryable=False,
                    rate_limit=page.rate_limit,
                )
            seen_tokens.add(next_token)
        else:
            stop_reason = "max_pages"
            complete = not bool(next_token)

        if raw_errors:
            complete = False
            stop_reason = "partial_api_response"

        result = XFetchResult(
            list_id=normalized_list_id,
            posts=posts,
            pages=pages,
            complete=complete,
            stop_reason=stop_reason,
            next_token=next_token,
            rate_limit=pages[-1].rate_limit if pages else None,
            raw_errors=raw_errors,
        )
        self.last_fetch_result = result
        logger.info(
            "x_api_complete run_id=%s list_id=%s page_number=%s fetched_count=%s "
            "stop_reason=%s rate_limit_remaining=%s status=success",
            run_id or "-",
            normalized_list_id,
            result.page_count,
            result.fetched_count,
            result.stop_reason,
            result.rate_limit.remaining if result.rate_limit else None,
        )
        return result

    async def fetch_list_tweets(
        self,
        list_url_or_id: str,
        max_tweets: int = 100,
        delay: float = 0,
    ) -> list[dict[str, Any]]:
        """Legacy phase-0 adapter implementing ``app.providers.XProvider``."""

        if delay > 0:
            await self._sleep(delay)
        result = await self.fetch_list_posts(list_url_or_id, max_tweets=max_tweets)
        return [post.to_legacy_tweet() for post in result.posts]


# Concise alias for callers that prefer the existing naming convention.
XApiProvider = OfficialXApiProvider


__all__ = [
    "EXPANSIONS",
    "API_RECENT_POST_LIMIT",
    "MEDIA_FIELDS",
    "OfficialXApiProvider",
    "RateLimitInfo",
    "TWEET_FIELDS",
    "USER_FIELDS",
    "XApiProvider",
    "XApiProviderError",
    "XFetchPage",
    "XFetchResult",
    "compare_post_ids",
]
