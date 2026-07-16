from __future__ import annotations

import logging
import unittest
from datetime import datetime, timezone
from typing import Any

import httpx

from app.models import IngestionRun, ListSyncState, XPost, parse_datetime
from app.x_api_provider import (
    EXPANSIONS,
    OfficialXApiProvider,
    XApiProviderError,
    compare_post_ids,
)


def post(post_id: str, **values: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "id": post_id,
        "text": f"post-{post_id}",
        "author_id": "u1",
        "created_at": "2026-07-16T08:30:00.000Z",
    }
    result.update(values)
    return result


class QueueTransport(httpx.AsyncBaseTransport):
    def __init__(self, items: list[httpx.Response | Exception]) -> None:
        self.items = list(items)
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if not self.items:
            raise AssertionError("Unexpected HTTP request")
        item = self.items.pop(0)
        if isinstance(item, Exception):
            raise item
        item.request = request
        return item


def response(
    status: int = 200,
    *,
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    return httpx.Response(status, json=payload or {}, headers=headers)


class ModelTests(unittest.TestCase):
    def test_models_round_trip_timezone_aware_values(self) -> None:
        stamp = parse_datetime("2026-07-16T08:30:00+08:00")
        self.assertIsNotNone(stamp)
        state = ListSyncState(
            list_id="123",
            newest_post_id="999",
            newest_post_created_at=stamp,
            last_attempt_at=stamp,
            last_success_at=stamp,
            last_status="success",
        )
        rebuilt_state = ListSyncState.from_dict(state.to_dict())
        self.assertEqual(stamp, rebuilt_state.newest_post_created_at)
        self.assertIsNotNone(rebuilt_state.newest_post_created_at.utcoffset())

        run = IngestionRun(
            run_id="run-1",
            started_at=stamp,
            status="partial_success",
            requested_lists=["123", "456"],
            successful_lists=["123"],
            failed_lists=["456"],
            new_post_ids=["999"],
            report_status="failed",
            report_error="offline test",
        )
        self.assertEqual(run.to_dict(), IngestionRun.from_dict(run.to_dict()).to_dict())

    def test_models_reject_naive_timestamps(self) -> None:
        with self.assertRaisesRegex(ValueError, "timezone"):
            ListSyncState(list_id="123", last_attempt_at=datetime(2026, 7, 16))

    def test_snowflake_comparison_degrades_for_non_numeric_ids(self) -> None:
        self.assertEqual(1, compare_post_ids("101", "100"))
        self.assertEqual(-1, compare_post_ids("99", "100"))
        self.assertEqual(0, compare_post_ids("weird", "weird"))
        self.assertIsNone(compare_post_ids("new-weird", "old-weird"))
        self.assertIsNone(compare_post_ids("001", "1"))
        self.assertIsNone(compare_post_ids("²", "1"))


class OfficialXApiProviderTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.clients: list[httpx.AsyncClient] = []

    async def asyncTearDown(self) -> None:
        for client in self.clients:
            await client.aclose()

    def provider(
        self,
        items: list[httpx.Response | Exception],
        **options: Any,
    ) -> tuple[OfficialXApiProvider, QueueTransport]:
        transport = QueueTransport(items)
        client = httpx.AsyncClient(transport=transport)
        self.clients.append(client)
        return (
            OfficialXApiProvider(
                "TEST_BEARER_TOKEN_NOT_REAL",
                client=client,
                retry_base_delay=0,
                max_retry_delay=0,
                **options,
            ),
            transport,
        )

    async def test_list_id_rejects_unicode_digit_lookalikes(self) -> None:
        provider, transport = self.provider([])

        with self.assertRaisesRegex(ValueError, "List ID"):
            await provider.fetch_page("１２３")
        with self.assertRaisesRegex(ValueError, "List ID"):
            await provider.fetch_page("https://x.com/i/lists/１２３")
        with self.assertRaisesRegex(ValueError, "List ID"):
            await provider.fetch_page("https://example.com/x.com/i/lists/123")

        self.assertEqual([], transport.requests)

    async def test_official_response_maps_author_metrics_urls_media_and_references(self) -> None:
        raw_post = post(
            "900",
            text="short text",
            note_tweet={
                "text": "complete long-form post",
                "entities": {
                    "urls": [
                        {"expanded_url": "https://example.com/note-only-link"}
                    ]
                },
            },
            lang="en",
            conversation_id="850",
            public_metrics={
                "like_count": 10,
                "retweet_count": 3,
                "reply_count": 2,
                "quote_count": 1,
                "bookmark_count": 4,
                "impression_count": 500,
            },
            entities={
                "urls": [
                    {
                        "url": "https://t.co/a",
                        "expanded_url": "https://example.com/article",
                    },
                    {"unwound_url": "https://example.com/article"},
                ]
            },
            attachments={"media_keys": ["3_media"]},
            referenced_tweets=[
                {"type": "retweeted", "id": "700"},
                {"type": "quoted", "id": "701"},
                {"type": "replied_to", "id": "702"},
                {"type": "future_relation", "id": "703"},
            ],
        )
        payload = {
            "data": [raw_post],
            "includes": {
                "users": [
                    {
                        "id": "u1",
                        "username": "alice",
                        "name": "Alice",
                        "profile_image_url": "https://img.example/alice.jpg",
                    }
                ],
                "media": [
                    {
                        "media_key": "3_media",
                        "type": "photo",
                        "url": "https://img.example/photo.jpg",
                        "alt_text": "diagram",
                    }
                ],
            },
            "meta": {"result_count": 1},
        }
        provider, transport = self.provider(
            [
                response(
                    payload=payload,
                    headers={
                        "x-rate-limit-limit": "900",
                        "x-rate-limit-remaining": "899",
                        "x-rate-limit-reset": "1999999999",
                    },
                )
            ]
        )

        page = await provider.fetch_page("https://x.com/i/lists/123", max_results=10)

        self.assertEqual(1, page.fetched_count)
        normalized = page.posts[0]
        self.assertIsInstance(normalized, XPost)
        self.assertEqual("complete long-form post", normalized.text)
        self.assertEqual("alice", normalized.author.username)
        self.assertEqual("Alice", normalized.author.name)
        self.assertEqual(10, normalized.metrics.likes)
        self.assertEqual(3, normalized.metrics.retweets)
        self.assertEqual(500, normalized.metrics.impressions)
        self.assertEqual(
            ["https://example.com/article", "https://example.com/note-only-link"],
            normalized.urls,
        )
        self.assertEqual("3_media", normalized.media[0]["media_key"])
        self.assertEqual(
            ["retweeted", "quoted", "replied_to", "future_relation"],
            [item.relation_type for item in normalized.references],
        )
        self.assertEqual(raw_post, normalized.raw_payload)
        self.assertIsNot(raw_post, normalized.raw_payload)
        self.assertIsNotNone(normalized.created_at.utcoffset())
        self.assertEqual(899, page.rate_limit.remaining)

        request = transport.requests[0]
        self.assertEqual("/2/lists/123/tweets", request.url.path)
        self.assertEqual("10", request.url.params["max_results"])
        self.assertEqual(EXPANSIONS, request.url.params["expansions"])
        self.assertIn("public_metrics", request.url.params["tweet.fields"])
        self.assertIn("name", request.url.params["user.fields"])
        self.assertNotIn("TEST_BEARER_TOKEN_NOT_REAL", str(request.url))

    async def test_missing_optional_fields_degrade_without_losing_raw_post(self) -> None:
        raw_post = {
            "id": "901",
            "created_at": "not-a-time",
            "referenced_tweets": [
                {"type": "future_relation", "id": "800"},
                {"type": "quoted"},
                "invalid",
            ],
        }
        provider, _ = self.provider([response(payload={"data": [raw_post], "meta": {}})])

        page = await provider.fetch_page("123")
        normalized = page.posts[0]

        self.assertEqual("", normalized.text)
        self.assertEqual("", normalized.author_id)
        self.assertIsNone(normalized.author)
        self.assertIsNone(normalized.created_at)
        self.assertEqual(0, normalized.metrics.likes)
        self.assertEqual(["future_relation"], [item.relation_type for item in normalized.references])
        self.assertEqual(raw_post, normalized.raw_payload)

    async def test_null_post_id_stays_invalid_instead_of_becoming_literal_none(self) -> None:
        provider, _ = self.provider(
            [response(payload={"data": [{"id": None, "text": "invalid"}], "meta": {}})]
        )
        page = await provider.fetch_page("123")
        self.assertEqual("", page.posts[0].id)

    async def test_non_object_data_item_fails_the_entire_page(self) -> None:
        provider, _ = self.provider(
            [response(payload={"data": [post("902"), "damaged-item"], "meta": {}})]
        )

        with self.assertRaises(XApiProviderError) as caught:
            await provider.fetch_page("123")

        self.assertEqual("malformed_response", caught.exception.kind)
        self.assertEqual(200, caught.exception.status_code)

    async def test_pagination_and_max_tweets_are_exact(self) -> None:
        provider, transport = self.provider(
            [
                response(
                    payload={
                        "data": [post("106"), post("105"), post("104")],
                        "meta": {"next_token": "page-2"},
                    }
                ),
                # Deliberately return too many items: the provider still truncates exactly.
                response(
                    payload={
                        "data": [post("103"), post("102"), post("101")],
                        "meta": {"next_token": "page-3"},
                    }
                ),
            ]
        )

        result = await provider.fetch_list_posts("123", max_tweets=4, page_size=3)

        self.assertEqual(["106", "105", "104", "103"], [item.id for item in result.posts])
        self.assertEqual(2, result.page_count)
        self.assertFalse(result.complete)
        self.assertEqual("max_tweets", result.stop_reason)
        self.assertEqual("page-2", transport.requests[1].url.params["pagination_token"])
        self.assertEqual("1", transport.requests[1].url.params["max_results"])

    async def test_duplicate_posts_within_pages_are_returned_once(self) -> None:
        provider, _ = self.provider(
            [
                response(
                    payload={
                        "data": [post("105"), post("105")],
                        "meta": {"next_token": "next"},
                    }
                ),
                response(payload={"data": [post("104"), post("105")], "meta": {}}),
            ]
        )
        result = await provider.fetch_list_posts("123", max_tweets=10)
        self.assertEqual(["105", "104"], [item.id for item in result.posts])

    async def test_local_since_id_stops_pagination_and_is_not_sent_by_default(self) -> None:
        provider, transport = self.provider(
            [
                response(
                    payload={
                        "data": [post("105"), post("104")],
                        "meta": {"next_token": "older"},
                    }
                ),
                response(
                    payload={
                        "data": [post("103"), post("102")],
                        "meta": {"next_token": "unused"},
                    }
                ),
            ]
        )

        result = await provider.fetch_list_posts("123", max_tweets=10, since_id="103")

        self.assertEqual(["105", "104"], [item.id for item in result.posts])
        self.assertTrue(result.complete)
        self.assertEqual("since_id_reached", result.stop_reason)
        self.assertTrue(all("since_id" not in request.url.params for request in transport.requests))
        self.assertFalse(result.pages[0].since_id_sent)

    async def test_since_id_is_only_sent_when_capability_is_explicit(self) -> None:
        provider, transport = self.provider(
            [response(payload={"data": [post("105")], "meta": {}})],
            supports_since_id=True,
        )

        result = await provider.fetch_list_posts("123", max_tweets=10, since_id="103")

        self.assertEqual("103", transport.requests[0].url.params["since_id"])
        self.assertTrue(result.pages[0].since_id_sent)
        self.assertTrue(result.complete)

    async def test_empty_list_is_a_successful_distinct_result(self) -> None:
        provider, _ = self.provider(
            [
                response(
                    payload={"meta": {"result_count": 0}},
                    headers={"x-rate-limit-remaining": "88"},
                )
            ]
        )
        result = await provider.fetch_list_posts("123")
        self.assertEqual([], result.posts)
        self.assertTrue(result.complete)
        self.assertEqual("empty", result.stop_reason)
        self.assertEqual(88, result.rate_limit.remaining)

    async def test_rate_limit_server_error_and_timeout_use_bounded_retries(self) -> None:
        cases = [
            (
                [response(429, headers={"retry-after": "0"}), response(payload={"meta": {}})],
                2,
            ),
            (
                [response(500), response(503), response(payload={"meta": {}})],
                3,
            ),
            (
                [httpx.ReadTimeout("offline timeout"), response(payload={"meta": {}})],
                2,
            ),
        ]
        for items, expected_requests in cases:
            with self.subTest(expected_requests=expected_requests):
                sleeps: list[float] = []

                async def record_sleep(delay: float) -> None:
                    sleeps.append(delay)

                provider, transport = self.provider(items, sleep=record_sleep)
                page = await provider.fetch_page("123")
                self.assertEqual([], page.posts)
                self.assertEqual(expected_requests, len(transport.requests))
                self.assertEqual(expected_requests - 1, len(sleeps))

    async def test_retry_exhaustion_is_structured_and_token_never_enters_logs(self) -> None:
        secret = "TEST_BEARER_TOKEN_NOT_REAL"
        transport = QueueTransport([response(429), response(429), response(429)])
        client = httpx.AsyncClient(transport=transport)
        self.clients.append(client)
        provider = OfficialXApiProvider(
            secret,
            client=client,
            max_retries=2,
            retry_base_delay=0,
            max_retry_delay=0,
        )

        with self.assertLogs("app.x_api_provider", level=logging.WARNING) as captured:
            with self.assertRaises(XApiProviderError) as context:
                await provider.fetch_page("123", run_id="safe-run")

        error = context.exception
        self.assertEqual("rate_limited", error.kind)
        self.assertEqual(429, error.status_code)
        self.assertEqual(3, error.attempts)
        self.assertTrue(error.retryable)
        self.assertNotIn(secret, str(error))
        self.assertNotIn(secret, "\n".join(captured.output))

    async def test_auth_permission_not_found_and_application_errors_are_distinct(self) -> None:
        cases = [
            (response(401), "unauthorized"),
            (response(403), "forbidden"),
            (response(404), "not_found"),
            (
                response(
                    payload={
                        "errors": [
                            {"title": "offline fixture", "detail": "TEST ONLY", "status": 400}
                        ]
                    }
                ),
                "api_error",
            ),
        ]
        for item, expected_kind in cases:
            with self.subTest(expected_kind=expected_kind):
                provider, transport = self.provider([item])
                with self.assertRaises(XApiProviderError) as context:
                    await provider.fetch_page("123")
                self.assertEqual(expected_kind, context.exception.kind)
                self.assertEqual(1, len(transport.requests))

    async def test_repeated_pagination_token_is_rejected(self) -> None:
        provider, _ = self.provider(
            [
                response(payload={"data": [post("2")], "meta": {"next_token": "repeat"}}),
                response(payload={"data": [post("1")], "meta": {"next_token": "repeat"}}),
            ]
        )
        with self.assertRaises(XApiProviderError) as context:
            await provider.fetch_list_posts("123", max_tweets=10)
        self.assertEqual("pagination_error", context.exception.kind)

    async def test_legacy_adapter_preserves_canonical_evidence(self) -> None:
        raw = post(
            "100",
            lang="zh",
            conversation_id="90",
            referenced_tweets=[{"type": "quoted", "id": "80"}],
            entities={"urls": [{"expanded_url": "https://example.com"}]},
        )
        provider, _ = self.provider([response(payload={"data": [raw], "meta": {}})])

        tweets = await provider.fetch_list_tweets("123", max_tweets=1)

        self.assertEqual("100", tweets[0]["id"])
        self.assertEqual("https://example.com", tweets[0]["links"][0])
        self.assertEqual("90", tweets[0]["conversation_id"])
        self.assertEqual("quoted", tweets[0]["references"][0]["relation_type"])
        self.assertEqual(raw, tweets[0]["raw_payload"])


if __name__ == "__main__":
    unittest.main()
