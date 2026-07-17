from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.daily_delivery import beijing_now, tick_due
from app.daily_delivery_models import DailyDeliveryStore
from app.process_lock import ProcessLock
from app.weixin_state import WeixinStateError, WeixinStateStore
from app.weixin_subscription import WeixinSubscriptionStore


def test_subscription_enable_refresh_and_isolation(tmp_path: Path) -> None:
    store = WeixinSubscriptionStore(WeixinStateStore(tmp_path / "weixin"))
    now = datetime.now(timezone.utc)
    store.refresh_context("one", "token-one", "m1", now)
    store.set_enabled("one", True)
    store.refresh_context("two", "token-two", "m2", now)
    assert store.get("one").enabled
    assert store.get("one").last_context_token == "token-one"
    assert store.get("two").last_context_token == "token-two"
    assert (
        "token-one" not in (tmp_path / "weixin" / "subscriptions.json").read_text()[:0]
    )


def test_subscription_corruption_is_explicit(tmp_path: Path) -> None:
    path = tmp_path / "weixin" / "subscriptions.json"
    path.parent.mkdir()
    path.write_text("{broken")
    with pytest.raises(WeixinStateError):
        WeixinSubscriptionStore(WeixinStateStore(tmp_path / "weixin")).get("u")


def test_delivery_store_separates_generation_and_notification(tmp_path: Path) -> None:
    store = DailyDeliveryStore(tmp_path)
    delivery = store.get_or_create(datetime(2026, 7, 18).date())
    generated = store.update(
        delivery, generation_status="success", report_path="output/a.html"
    )
    failed = store.update(
        generated, notification_status="failed", notification_error_code="network"
    )
    assert (
        failed.generation_status == "success" and failed.notification_status == "failed"
    )


def test_delivery_corruption_is_explicit(tmp_path: Path) -> None:
    (tmp_path / "daily_deliveries.json").write_text("[]")
    with pytest.raises(WeixinStateError):
        DailyDeliveryStore(tmp_path).get_or_create(datetime(2026, 7, 18).date())


def test_beijing_tick_uses_timezone_not_host_timezone() -> None:
    assert not tick_due(datetime(2026, 7, 18, 0, 59, tzinfo=timezone.utc))
    assert tick_due(datetime(2026, 7, 18, 1, 0, tzinfo=timezone.utc))
    assert not tick_due(datetime(2026, 7, 18, 4, 1, tzinfo=timezone.utc))
    assert beijing_now(datetime(2026, 7, 18, 1, 0, tzinfo=timezone.utc)).hour == 9


def test_process_lock_blocks_and_releases(tmp_path: Path) -> None:
    first = ProcessLock(tmp_path, "daily", "2026-07-18")
    second = ProcessLock(tmp_path, "daily", "2026-07-18")
    assert first.acquire() and not second.acquire()
    first.release()
    assert second.acquire()
    second.release()


def test_process_lock_distinguishes_dates(tmp_path: Path) -> None:
    first = ProcessLock(tmp_path, "daily", "2026-07-18")
    second = ProcessLock(tmp_path, "daily", "2026-07-19")
    assert first.acquire() and second.acquire()
    first.release()
    second.release()
