from __future__ import annotations

import logging
from pathlib import Path

import httpx
import pytest

from app.weixin_auth import WeixinAuthClient, WeixinAuthError, run_login
from app.weixin_models import WeixinQRSession
from app.weixin_state import WeixinStateStore


TEST_TOKEN_DO_NOT_USE = "TEST_ONLY_AUTH_TOKEN_DO_NOT_USE"


class FakeAuthClient:
    def __init__(self, statuses: list[dict[str, object]]) -> None:
        self.statuses = statuses

    async def fetch_qr_code(self) -> WeixinQRSession:
        return WeixinQRSession("TEST_ONLY_SESSION", "TEST_ONLY_QR_CONTENT")

    async def poll_status(self, _session_id: str) -> dict[str, object]:
        return self.statuses.pop(0)


@pytest.mark.asyncio
async def test_successful_qr_login_saves_private_credentials_and_cleans_qr(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    store = WeixinStateStore(tmp_path / "weixin")
    shown: list[Path] = []
    client = FakeAuthClient(
        [
            {"status": "scaned"},
            {
                "status": "confirmed",
                "bot_token": TEST_TOKEN_DO_NOT_USE,
                "ilink_bot_id": "TEST_ONLY_BOT_ID",
                "baseurl": "https://ilinkai.weixin.qq.com",
            },
        ]
    )
    with caplog.at_level(logging.DEBUG):
        success = await run_login(
            store,
            auth_client=client,  # type: ignore[arg-type]
            poll_interval_seconds=0,
            on_qr_ready=lambda path: shown.append(path),
        )

    assert success
    assert shown and not shown[0].exists()
    assert store.load_credentials().token == TEST_TOKEN_DO_NOT_USE  # type: ignore[union-attr]
    assert TEST_TOKEN_DO_NOT_USE not in caplog.text
    assert not list(store.temporary_dir.glob("*"))


@pytest.mark.asyncio
async def test_qr_expiry_and_timeout_clean_temporary_file(tmp_path: Path) -> None:
    for statuses, timeout in (([{"status": "expired"}], 5), ([], 0)):
        store = WeixinStateStore(tmp_path / f"case-{timeout}")
        client = FakeAuthClient(statuses)
        with pytest.raises(WeixinAuthError):
            await run_login(
                store,
                auth_client=client,  # type: ignore[arg-type]
                timeout_seconds=timeout,
                poll_interval_seconds=0,
            )
        assert not list(store.temporary_dir.glob("*"))


@pytest.mark.asyncio
async def test_auth_network_timeout_is_sanitized() -> None:
    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("TEST_ONLY_RAW_TIMEOUT", request=request)

    http = httpx.AsyncClient(transport=httpx.MockTransport(timeout))
    client = WeixinAuthClient(http_client=http, request_timeout_seconds=1)
    with pytest.raises(WeixinAuthError, match="Unable to obtain"):
        await client.fetch_qr_code()
    await http.aclose()


@pytest.mark.asyncio
async def test_untrusted_qr_confirmed_base_url_is_rejected(tmp_path: Path) -> None:
    store = WeixinStateStore(tmp_path)
    client = FakeAuthClient(
        [
            {
                "status": "confirmed",
                "bot_token": TEST_TOKEN_DO_NOT_USE,
                "ilink_bot_id": "TEST_ONLY_BOT_ID",
                "baseurl": "https://attacker.example",
            }
        ]
    )
    with pytest.raises(Exception, match="Untrusted"):
        await run_login(store, auth_client=client, poll_interval_seconds=0)  # type: ignore[arg-type]
    assert not store.credentials_path.exists()
