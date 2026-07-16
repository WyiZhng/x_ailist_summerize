"""Provider contracts and deterministic offline providers.

The protocols intentionally mirror the existing ``XListFetcher`` /
``XApiFetcher`` and ``LLMProvider`` methods.  Existing objects therefore work
without inheriting from a new base class, while tests can inject small fakes.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Awaitable, Mapping, Protocol, Sequence, runtime_checkable

from .security import escape_html_text


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
    "MockXProvider",
    "ReportGenerator",
    "SummaryProvider",
    "Tweet",
    "XProvider",
    "build_mock_lists",
]
