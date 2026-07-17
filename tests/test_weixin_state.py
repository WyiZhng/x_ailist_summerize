from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.weixin_models import WeixinCredentials, WeixinSyncState
from app.weixin_state import WeixinStateError, WeixinStateStore


TEST_TOKEN_DO_NOT_USE = "TEST_ONLY_BOT_TOKEN_DO_NOT_USE"


def credentials() -> WeixinCredentials:
    now = datetime.now(timezone.utc)
    return WeixinCredentials(
        token=TEST_TOKEN_DO_NOT_USE,
        account_id="TEST_ONLY_ACCOUNT_DO_NOT_USE",
        base_url="https://ilinkai.weixin.qq.com",
        created_at=now,
        updated_at=now,
    )


def test_credentials_and_sync_state_round_trip_atomically(tmp_path: Path) -> None:
    store = WeixinStateStore(tmp_path / "weixin")
    expected_credentials = credentials()
    store.save_credentials(expected_credentials)
    state = WeixinSyncState("TEST_ONLY_BUF", datetime.now(timezone.utc), "msg-1")
    store.save_sync_state(state)

    assert store.load_credentials() == expected_credentials
    assert store.load_sync_state() == state
    assert not list(store.data_dir.glob("*.tmp"))
    if os.name != "nt":
        assert store.credentials_path.stat().st_mode & 0o777 == 0o600


def test_corrupt_state_is_not_silently_overwritten(tmp_path: Path) -> None:
    store = WeixinStateStore(tmp_path)
    store.credentials_path.write_text("{damaged", encoding="utf-8")

    with pytest.raises(WeixinStateError, match="unreadable"):
        store.load_credentials()
    assert store.credentials_path.read_text(encoding="utf-8") == "{damaged"


def test_context_tokens_are_isolated_and_can_be_invalidated(tmp_path: Path) -> None:
    store = WeixinStateStore(tmp_path)
    now = datetime.now(timezone.utc)
    store.save_user_context("TEST_ONLY_USER_A", "TEST_ONLY_CONTEXT_A", "a", now)
    store.save_user_context("TEST_ONLY_USER_B", "TEST_ONLY_CONTEXT_B", "b", now)
    store.invalidate_user_context("TEST_ONLY_USER_A")

    users = json.loads(store.users_path.read_text(encoding="utf-8"))
    assert users[store.user_key("TEST_ONLY_USER_A")]["valid"] is False
    assert users[store.user_key("TEST_ONLY_USER_B")]["valid"] is True
    assert (
        users[store.user_key("TEST_ONLY_USER_B")]["context_token"]
        == "TEST_ONLY_CONTEXT_B"
    )


def test_processed_message_ids_are_stable_and_deduplicated(tmp_path: Path) -> None:
    store = WeixinStateStore(tmp_path)
    assert not store.is_processed("TEST_ONLY_MESSAGE")
    store.mark_processed("TEST_ONLY_MESSAGE", "help")
    assert store.is_processed("TEST_ONLY_MESSAGE")
    assert store.masked_message_id("TEST_ONLY_MESSAGE") == store.masked_message_id(
        "TEST_ONLY_MESSAGE"
    )
