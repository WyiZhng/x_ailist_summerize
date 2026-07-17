from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.daily_delivery_models import DailyDeliveryStore
from app.weixin_models import WeixinSendResult
from app.weixin_push import WeixinPushService
from app.weixin_state import WeixinStateStore
from app.weixin_subscription import WeixinSubscriptionStore


class FakeClient:
    def __init__(self, *, file_success: bool = True) -> None:
        self.file_success = file_success
        self.text_calls = 0
        self.file_calls = 0

    async def send_text(self, *_args):
        self.text_calls += 1
        return WeixinSendResult(True)

    async def send_file(self, *_args):
        self.file_calls += 1
        return WeixinSendResult(
            self.file_success,
            error_code="file_failed" if not self.file_success else None,
        )


@pytest.mark.asyncio
async def test_push_success_is_idempotent_and_records_age(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "summary_a.html").write_text("report")
    state = WeixinStateStore(tmp_path / "weixin")
    subscriptions = WeixinSubscriptionStore(state)
    subscriptions.refresh_context("u", "context", "m", datetime.now(timezone.utc))
    subscription = subscriptions.set_enabled("u", True)
    deliveries = DailyDeliveryStore(tmp_path / "delivery")
    delivery = deliveries.update(
        deliveries.get_or_create(datetime.now().date()),
        generation_status="success",
        report_path="output/summary_a.html",
    )
    client = FakeClient()
    service = WeixinPushService(
        client=client,
        subscriptions=subscriptions,
        deliveries=deliveries,
        report_root=output,
    )
    result = await service.send_daily_digest(subscription, delivery)
    assert (
        result.success
        and result.token_age_seconds is not None
        and client.file_calls == 1
    )
    assert (
        await service.send_daily_digest(
            subscription, deliveries.get(delivery.delivery_date)
        )
    ).status == "already_sent"


@pytest.mark.asyncio
async def test_push_file_failure_does_not_fail_generation(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "summary_a.html").write_text("report")
    subscriptions = WeixinSubscriptionStore(WeixinStateStore(tmp_path / "weixin"))
    subscriptions.refresh_context("u", "context", "m", datetime.now(timezone.utc))
    subscription = subscriptions.set_enabled("u", True)
    deliveries = DailyDeliveryStore(tmp_path / "delivery")
    delivery = deliveries.update(
        deliveries.get_or_create(datetime.now().date()),
        generation_status="success",
        report_path="output/summary_a.html",
    )
    result = await WeixinPushService(
        client=FakeClient(file_success=False),
        subscriptions=subscriptions,
        deliveries=deliveries,
        report_root=output,
    ).send_daily_digest(subscription, delivery)
    saved = deliveries.get(delivery.delivery_date)
    assert (
        not result.success
        and result.status == "partial_failure"
        and saved.generation_status == "success"
    )
