from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.config import normalize_config
from app.weixin_client import WeixinAuthenticationError
from app.weixin_commands import ReportLocator, WeixinCommandHandler
from app.weixin_models import (
    WeixinCredentials,
    WeixinInboundMessage,
    WeixinSendResult,
    WeixinUpdateBatch,
)
from app.weixin_service import WeixinService
from app.weixin_state import WeixinStateStore


TEST_CONTEXT_DO_NOT_USE = "TEST_ONLY_CONTEXT_DO_NOT_USE"


class FakeClient:
    def __init__(self, *, send_success: bool = True) -> None:
        self.send_success = send_success
        self.text_calls: list[tuple[str, str, str]] = []
        self.file_calls: list[Path] = []
        self.batches: list[WeixinUpdateBatch] = []

    async def get_updates(self, _buf: str | None) -> WeixinUpdateBatch:
        return self.batches.pop(0)

    async def send_typing(self, *_args) -> WeixinSendResult:
        return WeixinSendResult(True)

    async def send_text(self, user: str, context: str, text: str) -> WeixinSendResult:
        self.text_calls.append((user, context, text))
        return WeixinSendResult(self.send_success)

    async def send_file(self, _user: str, _context: str, path: Path, _name: str):
        self.file_calls.append(path)
        return WeixinSendResult(self.send_success)


def message(message_id: str, user: str = "TEST_ONLY_USER_A") -> WeixinInboundMessage:
    return WeixinInboundMessage(
        message_id=message_id,
        user_id=user,
        message_type="text",
        text="帮助",
        context_token=f"{TEST_CONTEXT_DO_NOT_USE}_{user}",
        received_at=datetime.now(timezone.utc),
        raw_payload={},
    )


def service(
    tmp_path: Path, client: FakeClient
) -> tuple[WeixinService, WeixinStateStore]:
    output = tmp_path / "output"
    output.mkdir(exist_ok=True)
    store = WeixinStateStore(tmp_path / "data" / "weixin")
    handler = WeixinCommandHandler(
        report_locator=ReportLocator(output, (".html",)),
        config=normalize_config({}),
        cookies_path=tmp_path / "cookies.json",
    )
    return (
        WeixinService(
            client=client,  # type: ignore[arg-type]
            state_store=store,
            command_handler=handler,
            maximum_backoff_seconds=1,
        ),
        store,
    )


@pytest.mark.asyncio
async def test_batch_commits_cursor_after_success_and_restart_deduplicates(
    tmp_path: Path,
) -> None:
    client = FakeClient()
    bot, store = service(tmp_path, client)
    batch = WeixinUpdateBatch((message("TEST_ONLY_MESSAGE_1"),), "TEST_ONLY_BUF_1")

    assert await bot.process_batch(batch)
    assert store.load_sync_state().get_updates_buf == "TEST_ONLY_BUF_1"
    assert len(client.text_calls) == 1
    assert await bot.process_batch(batch)
    assert len(client.text_calls) == 1


@pytest.mark.asyncio
async def test_failed_message_does_not_advance_cursor_or_stop_other_messages(
    tmp_path: Path,
) -> None:
    client = FakeClient(send_success=False)
    bot, store = service(tmp_path, client)
    batch = WeixinUpdateBatch(
        (message("TEST_ONLY_FAIL_1"), message("TEST_ONLY_FAIL_2")),
        "MUST_NOT_COMMIT",
    )

    assert not await bot.process_batch(batch)
    assert store.load_sync_state().get_updates_buf is None
    assert len(client.text_calls) == 2


@pytest.mark.asyncio
async def test_context_tokens_remain_bound_to_each_user(tmp_path: Path) -> None:
    client = FakeClient()
    bot, store = service(tmp_path, client)
    await bot.process_batch(
        WeixinUpdateBatch(
            (message("A", "TEST_ONLY_USER_A"), message("B", "TEST_ONLY_USER_B")),
            "BUF",
        )
    )
    assert client.text_calls[0][1].endswith("TEST_ONLY_USER_A")
    assert client.text_calls[1][1].endswith("TEST_ONLY_USER_B")
    users = store.users_path.read_text(encoding="utf-8")
    assert "TEST_ONLY_CONTEXT_DO_NOT_USE_TEST_ONLY_USER_A" in users
    assert "TEST_ONLY_CONTEXT_DO_NOT_USE_TEST_ONLY_USER_B" in users


@pytest.mark.asyncio
async def test_authentication_failure_stops_and_marks_credentials_expired(
    tmp_path: Path,
) -> None:
    class ExpiredClient(FakeClient):
        async def get_updates(self, _buf: str | None) -> WeixinUpdateBatch:
            raise WeixinAuthenticationError("TEST_ONLY_EXPIRED")

    client = ExpiredClient()
    bot, store = service(tmp_path, client)
    now = datetime.now(timezone.utc)
    store.save_credentials(
        WeixinCredentials(
            token="TEST_ONLY_TOKEN_DO_NOT_USE",
            account_id="TEST_ONLY_ACCOUNT",
            base_url="https://ilinkai.weixin.qq.com",
            created_at=now,
            updated_at=now,
        )
    )
    with pytest.raises(WeixinAuthenticationError):
        await bot.run()
    assert store.load_credentials().status == "expired"  # type: ignore[union-attr]
