from __future__ import annotations

import contextlib
import io
import importlib
import shutil
import sys
import types
import unittest
import uuid
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from unittest import mock

from app.providers import MockSummaryProvider, MockXProvider
from app.task_runner import DigestTaskRequest, DigestTaskRunner


def _module(name: str, **attributes: Any) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


class _OfflineClient:
    """Import-only SDK stand-in; no test may perform real I/O."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass


_OFFLINE_DEPENDENCIES = {
    "twikit": _module("twikit", Client=_OfflineClient),
    "httpx": _module("httpx", AsyncClient=_OfflineClient, HTTPError=OSError),
    "requests": _module("requests"),
    "anthropic": _module("anthropic", Anthropic=_OfflineClient),
    "openai": _module("openai", OpenAI=_OfflineClient),
}


# These tests exercise pure methods only.  Loading the production modules with
# import-only SDK stand-ins keeps collection deterministic on a clean machine
# without installing providers or supplying credentials.
with mock.patch.dict(sys.modules, _OFFLINE_DEPENDENCIES):
    x_module = importlib.import_module("app.x_list_summarizer")
    llm_module = importlib.import_module("app.llm_providers")

XListFetcher = x_module.XListFetcher
XApiFetcher = x_module.XApiFetcher
LLMProvider = llm_module.LLMProvider


def _fetcher(**list_info: Any) -> Any:
    value = XListFetcher.__new__(XListFetcher)
    value.list_info = {
        "name": "Test List",
        "list_names": [],
        "owner": "owner",
        "owner_name": "Owner",
        "member_count": 1,
        "profile_image_url": None,
    }
    value.list_info.update(list_info)
    return value


def _tweet(
    tweet_id: str,
    author: str,
    *,
    links: list[str] | None = None,
    text: str = "text",
    likes: int = 0,
    retweets: int = 0,
    replies: int = 0,
    quotes: int = 0,
    bookmarks: int = 0,
    **extra: Any,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "id": tweet_id,
        "author": author,
        "text": text,
        "links": list(links or []),
        "media": [],
        "card": None,
        "likes": likes,
        "retweets": retweets,
        "replies": replies,
        "quotes": quotes,
        "bookmarks": bookmarks,
    }
    value.update(extra)
    return value


@contextlib.contextmanager
def workspace_temporary_directory():
    parent = Path(__file__).resolve().parent / ".core-function-tmp"
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


class XListPureFunctionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fetcher = _fetcher()

    def test_extract_list_id_accepts_numeric_and_canonical_x_urls(self) -> None:
        cases = {
            "123456789": "123456789",
            "https://x.com/i/lists/111?ref=home": "111",
            "https://x.com/openai/lists/222": "222",
            "https://twitter.com/i/lists/333/": "333",
            "not-a-list-id": "not-a-list-id",
        }
        for value, expected in cases.items():
            with self.subTest(value=value):
                self.assertEqual(expected, self.fetcher.extract_list_id(value))

    def test_extract_owner_from_canonical_user_list_urls(self) -> None:
        cases = {
            "https://x.com/alice/lists/123": "alice",
            "https://twitter.com/bob_42/lists/research": "bob_42",
            "https://example.com/alice/lists/123": None,
            "https://x.com/alice/status/123": None,
        }
        for value, expected in cases.items():
            with self.subTest(value=value):
                self.assertEqual(expected, self.fetcher.extract_owner_from_url(value))

    def test_aggregate_by_links_uses_engagement_diversity_and_author_cap_order(self) -> None:
        tweets = [
            _tweet("1", "solo", links=["https://one.example"], likes=30),
            _tweet("2", "solo", links=["https://two.example"], likes=20),
            _tweet("3", "solo", links=["https://three.example"], likes=10),
            _tweet("4", "alice", links=["https://community.example"], likes=1),
            _tweet("5", "bob", links=["https://community.example"], likes=1),
            _tweet("6", "plain", text="没有链接 / no link"),
        ]

        aggregated = self.fetcher.aggregate_by_links(tweets)
        ordered_links = [link for link, _ in aggregated["by_link"]]

        self.assertEqual(
            [
                "https://one.example",
                "https://two.example",
                "https://community.example",
                "https://three.example",
            ],
            ordered_links,
        )
        self.assertEqual([tweets[-1]], aggregated["no_links"])
        self.assertEqual(2, len(dict(aggregated["by_link"])["https://community.example"]))

    def test_aggregate_preserves_duplicate_and_multi_link_inputs(self) -> None:
        duplicate = _tweet(
            "duplicate",
            "same-author",
            links=["https://a.example", "https://b.example"],
            likes=2,
        )
        aggregated = self.fetcher.aggregate_by_links([duplicate, duplicate])
        groups = dict(aggregated["by_link"])

        self.assertEqual(2, len(groups["https://a.example"]))
        self.assertEqual(2, len(groups["https://b.example"]))
        self.assertIs(groups["https://a.example"][0], duplicate)
        self.assertEqual([], aggregated["no_links"])

    def test_parse_ai_insights_indexes_exact_casefolded_domain_and_base_domain(self) -> None:
        summary = "\n".join(
            [
                "1. https://Blog.Example.com/path :: First insight :: keeps delimiter",
                "example.net :: Domain-only insight",
                "malformed output without delimiter",
                "https://empty.example :: ",
            ]
        )

        insights = self.fetcher._parse_ai_insights(summary)

        exact = "https://Blog.Example.com/path"
        self.assertEqual("First insight :: keeps delimiter", insights[exact])
        self.assertEqual("First insight :: keeps delimiter", insights[exact.lower()])
        self.assertEqual("First insight :: keeps delimiter", insights["blog.example.com"])
        self.assertEqual("First insight :: keeps delimiter", insights["example.com"])
        self.assertEqual("Domain-only insight", insights["example.net"])
        self.assertNotIn("malformed output without delimiter", insights)
        self.assertNotIn("https://empty.example", insights)

    def test_parse_ai_insights_empty_inputs_are_stable(self) -> None:
        self.assertEqual({}, self.fetcher._parse_ai_insights(None))
        self.assertEqual({}, self.fetcher._parse_ai_insights(""))
        self.assertEqual({}, self.fetcher._parse_ai_insights("just prose"))


class RealFetcherFailureContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_x_api_complete_list_failure_reaches_runner_partial_status(self) -> None:
        fetcher = XApiFetcher.__new__(XApiFetcher)
        bearer = "bearer-test-secret-123456789"
        fetcher.bearer_token = bearer
        fetcher.client = _OfflineClient()
        fetcher.list_owner_pref = None
        fetcher.list_info = {
            "name": "X List Summary",
            "list_names": [],
            "owner": "Unknown",
            "owner_name": "Unknown",
            "member_count": 0,
            "profile_image_url": None,
        }

        async def fake_get(path: str, **_kwargs: Any) -> dict[str, Any]:
            if path == "/users/by/username/x":
                return {"data": {"id": "x"}}
            if "/lists/bad" in path:
                raise OSError(f"offline network failure Authorization: Bearer {bearer}")
            if path == "/lists/good":
                return {
                    "data": {"name": "Good", "member_count": 1},
                    "includes": {"users": [{"username": "owner", "name": "Owner"}]},
                }
            if path == "/lists/good/tweets":
                return {
                    "data": [
                        {
                            "id": "1",
                            "text": "A real normalized API tweet",
                            "author_id": "u1",
                            "entities": {
                                "urls": [{"expanded_url": "https://example.com/article"}]
                            },
                            "public_metrics": {
                                "like_count": 1,
                                "retweet_count": 0,
                                "reply_count": 0,
                                "quote_count": 0,
                                "bookmark_count": 0,
                            },
                        }
                    ],
                    "includes": {"users": [{"id": "u1", "username": "good"}]},
                    "meta": {},
                }
            raise AssertionError(f"Unexpected API path: {path}")

        fetcher._get = fake_get
        with workspace_temporary_directory() as directory:
            runner = DigestTaskRunner(
                fetcher=fetcher,
                summary_provider=MockSummaryProvider(),
                sensitive_values=(bearer,),
            )
            with contextlib.redirect_stdout(io.StringIO()):
                result = await runner.run(
                    DigestTaskRequest.from_values(
                        ["good", "bad"],
                        max_tweets=5,
                        output_dir=directory,
                    )
                )

            self.assertTrue(result.success, result.error)
            self.assertTrue(result.partial)
            self.assertEqual(("good",), result.completed_lists)
            self.assertEqual("bad", result.list_failures[0].list_url)
            self.assertNotIn(bearer, result.list_failures[0].error)
            self.assertTrue(result.report_path and result.report_path.exists())


class PromptBuilderTests(unittest.TestCase):
    def provider(self, name: str = "openai") -> Any:
        value = LLMProvider.__new__(LLMProvider)
        value.provider = name
        value.config = {}
        return value

    def test_prompt_sorts_by_share_count_limits_samples_and_preserves_bilingual_text(self) -> None:
        links = [
            ("https://one.example", [_tweet("1", "one", text="one")]),
            (
                "https://three.example",
                [
                    _tweet("31", "甲", text="中文第一行\n第二行"),
                    _tweet("32", "two", text="English context"),
                    _tweet("33", "three", text="third"),
                ],
            ),
            (
                "https://four.example",
                [
                    _tweet("41", "a", text="sample-a"),
                    _tweet("42", "b", text="sample-b"),
                    _tweet("43", "c", text="sample-c"),
                    _tweet("44", "d", text="sample-d-must-not-appear"),
                ],
            ),
        ]
        no_links = [_tweet(str(index), f"plain-{index}", text=f"plain-{index}") for index in range(7)]

        prompt = self.provider()._build_prompt({"by_link": links, "no_links": no_links})

        self.assertLess(prompt.index("[https://four.example]"), prompt.index("[https://three.example]"))
        self.assertLess(prompt.index("[https://three.example]"), prompt.index("[https://one.example]"))
        self.assertIn("中文第一行 第二行", prompt)
        self.assertIn("English context", prompt)
        self.assertNotIn("sample-d-must-not-appear", prompt)
        self.assertIn("plain-4", prompt)
        self.assertNotIn("plain-5", prompt)
        self.assertIn("Output EXACTLY ONE LINE per link", prompt)

    def test_prompt_limits_to_twenty_links_and_truncates_url_and_tweet_text(self) -> None:
        long_url = "https://example.com/" + ("a" * 120)
        long_text = "T" * 250 + "MUST_NOT_APPEAR"
        links = [(long_url, [_tweet("long", "author", text=long_text)])]
        links.extend(
            (f"https://example.com/{index}", [_tweet(str(index), "author", text="x")])
            for index in range(25)
        )

        prompt = self.provider()._build_prompt({"by_link": links, "no_links": []})

        self.assertIn(f"[{long_url[:80]}]", prompt)
        self.assertNotIn(long_url, prompt)
        self.assertIn("T" * 200, prompt)
        self.assertNotIn("MUST_NOT_APPEAR", prompt)
        self.assertEqual(20, prompt.count(" tweets\n"))

    def test_provider_specific_prompt_size_guards_are_offline_and_deterministic(self) -> None:
        huge_author = "author-" + ("x" * 2000)
        tweets = [_tweet(str(index), huge_author, text="context") for index in range(3)]
        links = [(f"https://example.com/{index}", tweets) for index in range(20)]
        data = {"by_link": links, "no_links": []}

        with self.assertLogs(llm_module.logger, level="WARNING") as captured:
            default_prompt = self.provider("openai")._build_prompt(data)
            groq_prompt = self.provider("groq")._build_prompt(data)

        self.assertTrue(default_prompt.endswith("[TRUNCATED DUE TO SIZE]"))
        self.assertTrue(groq_prompt.endswith("[TRUNCATED FOR GROQ LIMIT]"))
        self.assertLess(len(groq_prompt), len(default_prompt))
        self.assertTrue(any("truncating" in message for message in captured.output))
        self.assertTrue(any("trimming to 15K" in message for message in captured.output))


class _AttributeCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.attributes: list[tuple[str, str, str | None]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.attributes.extend((tag, name, value) for name, value in attrs)


class RealReportSecurityTests(unittest.IsolatedAsyncioTestCase):
    async def test_task_runner_real_report_escapes_hostile_content_and_blocks_unsafe_urls(self) -> None:
        marker = "PWNED_STAGE0_7f3a"
        hostile_html = f'<img src=x onerror="{marker}"><script>{marker}</script>'
        safe_link = "https://example.com/article?a=1&b=2"
        hostile_tweet = _tweet(
            "1\" onmouseover=\"PWNED_STAGE0_7f3a",
            f'attacker\" onmouseover=\"{marker}',
            links=[safe_link],
            text=hostile_html,
            likes=-4,
            retweets=0,
            card={
                "title": hostile_html,
                "description": hostile_html,
                "image": f'javascript:alert("{marker}")',
            },
            media=[
                {
                    "type": "photo",
                    "url": f'javascript:alert("{marker}")',
                }
            ],
        )
        standalone = _tweet(
            "2",
            "独立用户",
            text=f"中文 English {hostile_html}",
        )
        unsafe_link_tweet = _tweet(
            "3",
            "unsafe-link",
            links=[f'javascript:alert("{marker}")'],
            text="unsafe URL must render only as text",
        )
        x_provider = MockXProvider(
            lists={"mock://hostile": [hostile_tweet, standalone, unsafe_link_tweet]}
        )

        report_generator = _fetcher(
            list_names=[hostile_html],
            owner_name=hostile_html,
            owner=hostile_html,
            member_count=hostile_html,
            profile_image_url=f'javascript:alert("{marker}")',
        )

        class HostileSummaryProvider:
            def summarize(self, aggregated_data: dict[str, Any]) -> str:
                return f"{safe_link} :: {hostile_html}"

        with workspace_temporary_directory() as directory:
            runner = DigestTaskRunner(
                x_provider=x_provider,
                summary_provider=HostileSummaryProvider(),
                report_generator=report_generator,
                aggregator=report_generator.aggregate_by_links,
            )
            result = await runner.run(
                DigestTaskRequest.from_values(
                    ["mock://hostile"],
                    output_dir=directory,
                    ai_model=hostile_html,
                )
            )

            self.assertTrue(result.success, result.error)
            rendered = result.report_path.read_text(encoding="utf-8")

        lowered = rendered.lower()
        self.assertNotIn(f'<script>{marker.lower()}</script>', lowered)
        self.assertNotIn("<img src=x", lowered)
        self.assertNotIn(f'onerror="{marker.lower()}"', lowered)
        self.assertNotIn('href="javascript:', lowered)
        self.assertNotIn('src="javascript:', lowered)
        self.assertIn("&lt;script&gt;", rendered)
        self.assertIn("中文 English", rendered)
        self.assertIn("https://example.com/article?a=1&amp;b=2", rendered)
        self.assertIn("javascript:alert", rendered, "unsafe URL is retained only as escaped text")

        collector = _AttributeCollector()
        collector.feed(rendered)
        self.assertFalse(
            any(name.lower() in {"onerror", "onmouseover"} for _, name, _ in collector.attributes)
        )
        for tag, name, value in collector.attributes:
            if name.lower() in {"href", "src"} and value:
                with self.subTest(tag=tag, name=name, value=value):
                    self.assertFalse(value.lower().startswith(("javascript:", "vbscript:", "data:text")))


if __name__ == "__main__":
    unittest.main()
