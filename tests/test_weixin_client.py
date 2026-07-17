from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

from app.weixin_client import (
    OFFICIAL_API_BASE_URL,
    WeixinAuthenticationError,
    WeixinClient,
    WeixinClientError,
)
from app.weixin_models import WeixinCredentials
from app.weixin_state import WeixinStateStore


TEST_TOKEN_DO_NOT_USE = "TEST_ONLY_WEIXIN_TOKEN_DO_NOT_USE"
TEST_CONTEXT_DO_NOT_USE = "TEST_ONLY_CONTEXT_TOKEN_DO_NOT_USE"
TEST_USER_DO_NOT_USE = "TEST_ONLY_USER_ID_DO_NOT_USE"


def credentials(*, base_url: str = OFFICIAL_API_BASE_URL) -> WeixinCredentials:
    now = datetime.now(timezone.utc)
    return WeixinCredentials(
        token=TEST_TOKEN_DO_NOT_USE,
        account_id="TEST_ONLY_ACCOUNT_DO_NOT_USE",
        base_url=base_url,
        created_at=now,
        updated_at=now,
    )


def client_for(
    tmp_path: Path, transport: httpx.MockTransport, *, retries: int = 1
) -> WeixinClient:
    output = tmp_path / "output"
    output.mkdir(exist_ok=True)
    return WeixinClient(
        credentials(),
        state_store=WeixinStateStore(tmp_path / "data" / "weixin"),
        report_root=output,
        retry_attempts=retries,
        maximum_backoff_seconds=1,
        http_client=httpx.AsyncClient(transport=transport),
    )


@pytest.mark.asyncio
async def test_get_updates_normalizes_cursor_context_and_timezone(
    tmp_path: Path,
) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == f"Bearer {TEST_TOKEN_DO_NOT_USE}"
        return httpx.Response(
            200,
            json={
                "ret": 0,
                "get_updates_buf": "TEST_ONLY_NEXT_BUF",
                "msgs": [
                    {
                        "message_id": 123,
                        "from_user_id": TEST_USER_DO_NOT_USE,
                        "create_time_ms": 1_700_000_000_000,
                        "message_type": 1,
                        "context_token": TEST_CONTEXT_DO_NOT_USE,
                        "item_list": [{"type": 1, "text_item": {"text": "日报"}}],
                    }
                ],
            },
        )

    client = client_for(tmp_path, httpx.MockTransport(respond))
    batch = await client.get_updates("TEST_ONLY_OLD_BUF")
    await client.http.aclose()

    assert batch.get_updates_buf == "TEST_ONLY_NEXT_BUF"
    assert batch.messages[0].message_id == "123"
    assert batch.messages[0].context_token == TEST_CONTEXT_DO_NOT_USE
    assert batch.messages[0].received_at.tzinfo is not None


@pytest.mark.asyncio
async def test_authentication_error_is_not_retried_forever(tmp_path: Path) -> None:
    calls = 0

    def respond(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"ret": -14, "errcode": -14})

    client = client_for(tmp_path, httpx.MockTransport(respond), retries=3)
    with pytest.raises(WeixinAuthenticationError, match="re-login"):
        await client.get_updates(None)
    await client.http.aclose()
    assert calls == 1


@pytest.mark.asyncio
async def test_network_timeout_has_bounded_retry(tmp_path: Path) -> None:
    calls = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ReadTimeout("TEST_ONLY_TIMEOUT", request=request)
        return httpx.Response(200, json={"ret": 0, "msgs": [], "get_updates_buf": "b"})

    client = client_for(tmp_path, httpx.MockTransport(respond), retries=2)
    batch = await client.get_updates(None)
    await client.http.aclose()
    assert calls == 2
    assert batch.messages == ()


@pytest.mark.asyncio
async def test_missing_context_token_never_sends(tmp_path: Path) -> None:
    calls = 0

    def respond(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"ret": 0})

    client = client_for(tmp_path, httpx.MockTransport(respond))
    result = await client.send_text(TEST_USER_DO_NOT_USE, "", "help")
    await client.http.aclose()
    assert not result.success
    assert result.error_code == "missing_context_token"
    assert calls == 0


@pytest.mark.asyncio
async def test_send_file_encrypts_uploads_and_cleans_temporary_file(
    tmp_path: Path,
) -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/getuploadurl"):
            return httpx.Response(
                200, json={"ret": 0, "upload_param": "TEST_ONLY_UPLOAD"}
            )
        if request.url.path.endswith("/upload"):
            return httpx.Response(
                200, headers={"x-encrypted-param": "TEST_ONLY_DOWNLOAD"}
            )
        if request.url.path.endswith("/sendmessage"):
            body = json.loads(request.content)
            file_item = body["msg"]["item_list"][0]["file_item"]
            assert file_item["file_name"] == "AI-Report.html"
            assert file_item["media"]["aes_key"]
            return httpx.Response(200, json={"ret": 0})
        raise AssertionError(request.url)

    client = client_for(tmp_path, httpx.MockTransport(respond))
    report = tmp_path / "output" / "summary_valid.html"
    report.write_text("<html>TEST ONLY</html>", encoding="utf-8")
    result = await client.send_file(
        TEST_USER_DO_NOT_USE,
        TEST_CONTEXT_DO_NOT_USE,
        report,
        "AI-Report.html",
    )
    await client.http.aclose()

    assert result.success
    assert len(requests) == 3
    assert not list(client.state_store.temporary_dir.glob("*"))


@pytest.mark.asyncio
async def test_send_file_blocks_paths_outside_output_and_symlinks(
    tmp_path: Path,
) -> None:
    client = client_for(
        tmp_path,
        httpx.MockTransport(lambda _request: httpx.Response(500)),
    )
    outside = tmp_path / "config.json"
    outside.write_text("TEST ONLY", encoding="utf-8")
    with pytest.raises(WeixinClientError, match="outside"):
        await client.send_file(
            TEST_USER_DO_NOT_USE,
            TEST_CONTEXT_DO_NOT_USE,
            outside,
            "config.html",
        )

    html_outside = tmp_path / "summary_outside.html"
    html_outside.write_text("TEST ONLY", encoding="utf-8")
    link = tmp_path / "output" / "summary_link.html"
    try:
        link.symlink_to(html_outside)
    except (OSError, NotImplementedError):
        await client.http.aclose()
        return
    with pytest.raises(WeixinClientError):
        await client.send_file(
            TEST_USER_DO_NOT_USE,
            TEST_CONTEXT_DO_NOT_USE,
            link,
            "report.html",
        )
    await client.http.aclose()


def test_untrusted_api_base_url_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(WeixinClientError, match="Untrusted"):
        WeixinClient(
            credentials(base_url="https://attacker.example"),
            state_store=WeixinStateStore(tmp_path / "data"),
            report_root=tmp_path,
        )


def test_full_upload_url_is_restricted_to_official_cdn(tmp_path: Path) -> None:
    client = client_for(
        tmp_path,
        httpx.MockTransport(lambda _request: httpx.Response(200)),
    )
    trusted = client._resolve_upload_url(
        {
            "upload_full_url": (
                "https://novac2c.cdn.weixin.qq.com/c2c/upload?"
                "encrypted_query_param=TEST_ONLY&filekey=TEST_ONLY"
            )
        },
        file_key="TEST_ONLY",
    )
    assert trusted.startswith("https://novac2c.cdn.weixin.qq.com/c2c/upload?")
    with pytest.raises(WeixinClientError, match="Untrusted"):
        client._resolve_upload_url(
            {"upload_full_url": "https://attacker.example/c2c/upload"},
            file_key="TEST_ONLY",
        )
