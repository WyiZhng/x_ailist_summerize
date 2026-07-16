"""Provider contracts and deterministic offline providers.

The protocols intentionally mirror the existing ``XListFetcher`` /
``XApiFetcher`` and ``LLMProvider`` methods.  Existing objects therefore work
without inheriting from a new base class, while tests can inject small fakes.
"""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Mapping, Protocol, Sequence, runtime_checkable

from .models import XAuthor, XPost, XPostMetrics, XPostReference
from .security import escape_html_text
from .x_api_provider import RateLimitInfo, XApiProviderError, XFetchPage


Tweet = dict[str, Any]
AggregatedDigest = dict[str, Any]
SummaryResult = str | Awaitable[str]
ReportResult = str | Path | None | Awaitable[str | Path | None]


@runtime_checkable
class XProvider(Protocol):
    """Minimum asynchronous X source used by :class:`DigestTaskRunner`.

    ``XListFetcher`` and ``XApiFetcher`` already implement this protocol.
    ``list_info`` is deliberately not required because a custom provider may
    not have list metadata; the runner uses it opportunistically when present.
    """

    async def fetch_list_tweets(
        self,
        list_url_or_id: str,
        max_tweets: int = 100,
        delay: float = 0,
    ) -> list[Tweet]:
        """Return normalized tweets for one list."""


@runtime_checkable
class SummaryProvider(Protocol):
    """Summary contract implemented by the existing ``LLMProvider``."""

    def summarize(self, aggregated_data: AggregatedDigest) -> SummaryResult:
        """Return summary text, synchronously or asynchronously."""


@runtime_checkable
class ReportGenerator(Protocol):
    """Report contract implemented by the existing X fetcher classes."""

    def generate_html_report(
        self,
        aggregated: AggregatedDigest,
        ai_summary: str,
        output_path: str | Path,
        tweet_count: int = 0,
        ai_model: str = "",
    ) -> ReportResult:
        """Write a report to ``output_path``."""


def _tweet(
    tweet_id: str,
    text: str,
    author: str,
    *,
    links: Sequence[str] = (),
    likes: int = 0,
    retweets: int = 0,
    replies: int = 0,
    quotes: int = 0,
    bookmarks: int = 0,
    **markers: Any,
) -> Tweet:
    """Build a tweet with the fields expected by the legacy aggregator."""

    value: Tweet = {
        "id": tweet_id,
        "text": text,
        "author": author,
        "links": list(links),
        "media": [],
        "card": None,
        "likes": likes,
        "retweets": retweets,
        "replies": replies,
        "quotes": quotes,
        "bookmarks": bookmarks,
    }
    value.update(markers)
    return value


def build_mock_lists() -> dict[str, list[Tweet]]:
    """Return deterministic fixtures covering common and hostile inputs.

    The duplicate is intentional.  The phase-0 runner preserves the existing
    application's aggregation semantics and does not silently introduce a new
    global tweet-deduplication rule.
    """

    article = "https://example.com/research/agent-systems"
    chinese_article = "https://example.cn/posts/人工智能"
    quoted_article = "https://news.example.org/quoted-story"
    hostile_article = "https://evil.example/html-test"

    english = _tweet(
        "100",
        "A practical English post about agent systems.",
        "alice",
        links=(article,),
        likes=20,
        retweets=3,
        replies=2,
        bookmarks=4,
        language="en",
        is_retweet=False,
        is_quote=False,
    )
    duplicate = copy.deepcopy(english)
    duplicate["duplicate_of"] = "100"

    return {
        "mock://mixed": [
            english,
            _tweet(
                "101",
                "中文内容：这是一条关于人工智能趋势的讨论。",
                "小明",
                links=(chinese_article,),
                likes=12,
                replies=1,
                language="zh",
                is_retweet=False,
                is_quote=False,
            ),
            _tweet(
                "102",
                "RT @alice: A practical English post about agent systems.",
                "bob",
                links=(article,),
                likes=3,
                retweets=7,
                language="en",
                is_retweet=True,
                is_quote=False,
                source_tweet_id="100",
            ),
            _tweet(
                "103",
                "Quoting this analysis because the evidence is useful.",
                "carol",
                links=(quoted_article,),
                likes=9,
                quotes=2,
                language="en",
                is_retweet=False,
                is_quote=True,
                quoted_tweet_id="90",
            ),
            duplicate,
            _tweet(
                "104",
                "A standalone observation with no external link.",
                "dave",
                likes=2,
                language="en",
                is_retweet=False,
                is_quote=False,
            ),
            _tweet(
                "105",
                '<script>window.__mock_xss = "unsafe"</script><img src=x onerror=alert(1)>',
                "mallory",
                links=(hostile_article,),
                likes=1,
                language="en",
                is_retweet=False,
                is_quote=False,
                hostile_html=True,
            ),
        ],
        "mock://secondary": [
            _tweet(
                "200",
                "A second list also links to the agent systems article.",
                "erin",
                links=(article,),
                likes=8,
                retweets=1,
                language="en",
                is_retweet=False,
                is_quote=False,
            )
        ],
        "mock://no-links": [
            _tweet(
                "300",
                "只有文本，没有链接。",
                "无链接用户",
                likes=1,
                language="zh",
                is_retweet=False,
                is_quote=False,
            )
        ],
    }


class MockXProvider:
    """Deterministic, dependency-free X provider for tests and demos."""

    def __init__(
        self,
        lists: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
        *,
        fail_lists: Sequence[str] = (),
        login_ok: bool = True,
        latency: float = 0,
    ) -> None:
        self._lists = copy.deepcopy(dict(lists or build_mock_lists()))
        self.fail_lists = set(fail_lists)
        self.login_ok = login_ok
        self.latency = max(0.0, latency)
        self.fetch_calls: list[str] = []
        self.list_info: dict[str, Any] = {
            "name": "Mock X Digest",
            "list_names": [],
            "owner": "mock-owner",
            "owner_name": "Mock Owner",
            "member_count": 7,
            "profile_image_url": None,
        }

    async def login(self) -> tuple[bool, str]:
        return (self.login_ok, "Mock login OK" if self.login_ok else "Mock login failed")

    async def fetch_list_tweets(
        self,
        list_url_or_id: str,
        max_tweets: int = 100,
        delay: float = 0,
    ) -> list[Tweet]:
        import asyncio

        self.fetch_calls.append(list_url_or_id)
        wait = max(0.0, delay) + self.latency
        if wait:
            await asyncio.sleep(wait)
        if list_url_or_id in self.fail_lists:
            raise RuntimeError(f"Mock list failure: {list_url_or_id}")
        if list_url_or_id not in self._lists:
            raise KeyError(f"Unknown mock list: {list_url_or_id}")
        self.list_info["list_names"].append(list_url_or_id.removeprefix("mock://"))
        return copy.deepcopy(list(self._lists[list_url_or_id])[:max_tweets])


def _mock_post(
    post_id: str,
    text: str,
    author_id: str,
    username: str,
    source_list_id: str,
    *,
    urls: Sequence[str] = (),
    references: Sequence[tuple[str, str]] = (),
    metrics: Mapping[str, int] | None = None,
) -> XPost:
    """Build a canonical deterministic post for incremental tests and demos."""

    counts = dict(metrics or {})
    raw = {
        "id": post_id,
        "text": text,
        "author_id": author_id,
        "created_at": "2026-01-01T08:00:00Z",
        "lang": "zh" if any("\u4e00" <= char <= "\u9fff" for char in text) else "en",
        "public_metrics": copy.deepcopy(counts),
        "entities": {
            "urls": [{"expanded_url": item, "url": item} for item in urls]
        },
        "referenced_tweets": [
            {"type": relation, "id": target} for relation, target in references
        ],
    }
    return XPost(
        id=post_id,
        text=text,
        author_id=author_id,
        author=XAuthor(id=author_id, username=username, name=username.title()),
        created_at=datetime(2026, 1, 1, 8, 0, tzinfo=timezone.utc),
        language=raw["lang"],
        conversation_id=post_id,
        source_list_id=source_list_id,
        metrics=XPostMetrics(
            likes=counts.get("like_count", 0),
            retweets=counts.get("retweet_count", 0),
            replies=counts.get("reply_count", 0),
            quotes=counts.get("quote_count", 0),
            bookmarks=counts.get("bookmark_count", 0),
            impressions=counts.get("impression_count"),
        ),
        references=[
            XPostReference(relation_type=relation, referenced_post_id=target)
            for relation, target in references
        ],
        urls=list(urls),
        raw_payload=raw,
        fetched_at=datetime(2026, 1, 1, 8, 1, tzinfo=timezone.utc),
    )


def build_mock_ingestion_posts() -> dict[str, list[XPost]]:
    """Return newest-first canonical posts with cross-List duplication."""

    mixed = "mock://mixed"
    secondary = "mock://secondary"
    shared = _mock_post(
        "9005",
        "A practical update on agent systems https://example.com/agents",
        "u1",
        "alice",
        mixed,
        urls=("https://example.com/agents",),
        metrics={"like_count": 20, "retweet_count": 3, "reply_count": 2},
    )
    invalid = _mock_post(
        "invalid-id",
        "An intentionally malformed post ID for fallback testing.",
        "u7",
        "invalid",
        "mock://invalid-id",
    )
    missing_fields = XPost.from_dict(
        {
            **shared.to_dict(),
            "id": "8900",
            "author_id": "missing-author",
            "author": None,
            "source_list_id": "mock://missing-fields",
            "metrics": {},
        }
    )
    duplicate_one = XPost.from_dict(
        {**shared.to_dict(), "source_list_id": "mock://duplicates"}
    )
    return {
        mixed: [
            shared,
            _mock_post(
                "9004",
                "中文模型进展 https://example.cn/model",
                "u2",
                "xiaoming",
                mixed,
                urls=("https://example.cn/model",),
                references=(("quoted", "8800"),),
                metrics={"like_count": 12, "quote_count": 2},
            ),
            _mock_post(
                "9003",
                "Retweeting an earlier research note.",
                "u3",
                "bob",
                mixed,
                references=(("retweeted", "8700"),),
            ),
            _mock_post(
                "9002",
                "Replying with a useful implementation detail.",
                "u4",
                "carol",
                mixed,
                references=(("replied_to", "8600"),),
            ),
            _mock_post(
                "9001",
                '<script>window.__mock_ingestion_xss = true</script>',
                "u5",
                "mallory",
                mixed,
                urls=("https://example.org/security",),
            ),
        ],
        secondary: [
            XPost.from_dict({**shared.to_dict(), "source_list_id": secondary}),
            _mock_post(
                "8999",
                "A second List contains a distinct product update.",
                "u6",
                "erin",
                secondary,
                urls=("https://example.net/product",),
            ),
        ],
        "mock://empty": [],
        "mock://duplicates": [duplicate_one, XPost.from_dict(duplicate_one.to_dict())],
        "mock://invalid-id": [invalid],
        "mock://missing-fields": [missing_fields],
        "mock://second-page-failure": [
            clone
            for clone in (
                _mock_post(
                    str(8805 - index),
                    f"Paged mock post {index}",
                    f"paged-{index}",
                    f"paged{index}",
                    "mock://second-page-failure",
                )
                for index in range(5)
            )
        ],
    }


class MockXIngestionProvider:
    """Deterministic page-level provider with injectable per-page failures."""

    def __init__(
        self,
        posts_by_list: Mapping[str, Sequence[XPost]] | None = None,
        *,
        page_failures: Mapping[tuple[str, int], BaseException] | None = None,
    ) -> None:
        source = posts_by_list or build_mock_ingestion_posts()
        self._posts = {
            str(list_id): [XPost.from_dict(post.to_dict()) for post in posts]
            for list_id, posts in source.items()
        }
        default_failures: dict[tuple[str, int], BaseException] = {
            ("mock://rate-limit", 1): XApiProviderError(
                "Mock X API rate limit",
                kind="rate_limited",
                status_code=429,
                retryable=True,
            ),
            ("mock://server-error", 1): XApiProviderError(
                "Mock X API server error",
                kind="server_error",
                status_code=500,
                retryable=True,
            ),
            ("mock://timeout", 1): TimeoutError("Mock X API timeout"),
            ("mock://second-page-failure", 2): RuntimeError(
                "Mock second page failure"
            ),
        }
        default_failures.update(page_failures or {})
        self.page_failures = default_failures
        self.fetch_calls: list[dict[str, Any]] = []

    async def fetch_page(
        self,
        list_id: str,
        *,
        pagination_token: str | None = None,
        max_results: int = 100,
        page_number: int = 1,
        run_id: str | None = None,
    ) -> XFetchPage:
        if isinstance(max_results, bool) or not isinstance(max_results, int) or not 1 <= max_results <= 100:
            raise ValueError("max_results must be between 1 and 100")
        self.fetch_calls.append(
            {
                "list_id": list_id,
                "pagination_token": pagination_token,
                "max_results": max_results,
                "page_number": page_number,
                "run_id": run_id,
            }
        )
        failure = self.page_failures.get((list_id, page_number))
        if failure is not None:
            raise failure
        if list_id not in self._posts:
            raise KeyError(f"Unknown mock ingestion list: {list_id}")
        try:
            offset = int(pagination_token) if pagination_token else 0
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid mock pagination token") from exc
        values = self._posts[list_id]
        selected = values[offset : offset + max_results]
        next_offset = offset + len(selected)
        next_token = str(next_offset) if next_offset < len(values) else None
        return XFetchPage(
            list_id=list_id,
            posts=[XPost.from_dict(post.to_dict()) for post in selected],
            next_token=next_token,
            rate_limit=RateLimitInfo(limit=900, remaining=899, reset_at=2_000_000_000),
            page_number=page_number,
        )


class MockSummaryProvider:
    """Deterministic summary provider with configurable failure behavior."""

    def __init__(self, *, fail: bool = False, return_error_string: bool = False) -> None:
        self.fail = fail
        self.return_error_string = return_error_string
        self.calls: list[AggregatedDigest] = []

    def summarize(self, aggregated_data: AggregatedDigest) -> str:
        self.calls.append(aggregated_data)
        if self.fail:
            if self.return_error_string:
                return "Error with mock LLM: requested failure"
            raise RuntimeError("Mock LLM failure")

        lines: list[str] = []
        for link, tweets in aggregated_data.get("by_link", []):
            languages = {tweet.get("language") for tweet in tweets}
            language_note = "中英混合" if {"zh", "en"}.issubset(languages) else "mock insight"
            lines.append(f"{link} :: {language_note}; {len(tweets)} mention(s).")
        return "\n".join(lines) if lines else "No shared links in this mock digest."


class MockReportGenerator:
    """Small safe report writer used by the offline CLI and tests.

    It is not intended to replace the production report.  Escaping hostile mock
    values here ensures the test fixture itself cannot create executable HTML.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def generate_html_report(
        self,
        aggregated: AggregatedDigest,
        ai_summary: str,
        output_path: str | Path,
        tweet_count: int = 0,
        ai_model: str = "",
    ) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "tweet_count": tweet_count,
            "ai_model": ai_model,
            "summary": ai_summary,
            "aggregated": aggregated,
        }
        self.calls.append(payload)
        rendered = escape_html_text(json.dumps(payload, ensure_ascii=False, default=str))
        path.write_text(
            "<!doctype html><meta charset=\"utf-8\">"
            "<title>Mock X Digest</title><h1>Mock X Digest</h1>"
            f"<pre id=\"mock-payload\">{rendered}</pre>",
            encoding="utf-8",
        )


__all__ = [
    "AggregatedDigest",
    "MockReportGenerator",
    "MockSummaryProvider",
    "MockXIngestionProvider",
    "MockXProvider",
    "ReportGenerator",
    "SummaryProvider",
    "Tweet",
    "XProvider",
    "build_mock_ingestion_posts",
    "build_mock_lists",
]
