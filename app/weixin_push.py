"""Proactive, bounded Weixin delivery using only each user's saved context."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
from typing import Any

from .config import load_config
from .daily_delivery_models import DailyDeliveryStore
from .weixin_client import WeixinAuthenticationError, WeixinClient
from .weixin_commands import ReportLocator
from .weixin_models import DailyDelivery, PushResult, WeixinSubscription
from .weixin_state import WeixinStateStore
from .weixin_subscription import WeixinSubscriptionStore

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class WeixinPushService:
    """Send a summary and its approved report file with daily idempotency."""

    def __init__(
        self,
        *,
        client: WeixinClient,
        subscriptions: WeixinSubscriptionStore,
        deliveries: DailyDeliveryStore,
        report_root: Path,
    ) -> None:
        self.client, self.subscriptions, self.deliveries, self.report_root = (
            client,
            subscriptions,
            deliveries,
            report_root,
        )

    async def send_daily_digest(
        self,
        subscription: WeixinSubscription,
        delivery: DailyDelivery,
        *,
        force: bool = False,
    ) -> PushResult:
        if not subscription.enabled:
            return PushResult(False, "subscription_disabled")
        if subscription.context_status == "expired":
            return PushResult(False, "token_expired")
        if (
            not subscription.last_context_token
            or not subscription.last_context_received_at
        ):
            return PushResult(False, "token_missing")
        if delivery.notification_status == "sent" and not force:
            return PushResult(True, "already_sent")
        if not delivery.report_path:
            return PushResult(False, "report_missing")
        try:
            report = (self.report_root.parent / delivery.report_path).resolve(
                strict=True
            )
            report.relative_to(self.report_root.resolve())
        except (OSError, ValueError):
            return PushResult(False, "report_invalid")
        age = max(
            0,
            int(
                (
                    datetime.now(timezone.utc) - subscription.last_context_received_at
                ).total_seconds()
            ),
        )
        text = f"AI 每日情报｜{delivery.delivery_date:%Y-%m-%d}\n\n今日报告已生成\n生成时间：{(delivery.report_generated_at or datetime.now(timezone.utc)):%H:%M}\n\n完整 HTML 报告已附在下一条消息中。"
        self.deliveries.update(
            delivery,
            notification_status="sending",
            notification_attempts=delivery.notification_attempts + 1,
            notification_last_attempt_at=datetime.now(timezone.utc),
        )
        try:
            sent_text = await self.client.send_text(
                subscription.user_id, subscription.last_context_token, text
            )
            if not sent_text.success:
                return await self._failed(
                    subscription,
                    delivery,
                    "failed",
                    sent_text.error_code or "text_failed",
                    age,
                )
            sent_file = await self.client.send_file(
                subscription.user_id,
                subscription.last_context_token,
                report,
                f"AI日报-{delivery.delivery_date}.html",
            )
            if not sent_file.success:
                return await self._failed(
                    subscription,
                    delivery,
                    "partial_failure",
                    sent_file.error_code or "file_failed",
                    age,
                )
        except WeixinAuthenticationError:
            return await self._failed(
                subscription, delivery, "token_expired", "authentication_expired", age
            )
        saved = self.deliveries.update(
            delivery,
            notification_status="sent",
            notification_sent_at=datetime.now(timezone.utc),
            notification_error_code=None,
        )
        self.subscriptions.update_push(subscription, status="sent")
        self.deliveries.record_attempt(
            delivery=saved, status="sent", error_code=None, token_age_seconds=age
        )
        return PushResult(True, "sent", token_age_seconds=age)

    async def _failed(
        self,
        subscription: WeixinSubscription,
        delivery: DailyDelivery,
        status: str,
        error: str,
        age: int,
    ) -> PushResult:
        saved = self.deliveries.update(
            delivery, notification_status=status, notification_error_code=error
        )
        self.subscriptions.update_push(subscription, status=status, error_code=error)
        self.deliveries.record_attempt(
            delivery=saved, status=status, error_code=error, token_age_seconds=age
        )
        return PushResult(False, status, error, age)


def build_push_service(config: dict[str, Any]) -> WeixinPushService:
    weixin = config["weixin"]
    root = PROJECT_ROOT
    state = WeixinStateStore(root / weixin["data_dir"])
    credentials = state.load_credentials()
    if credentials is None:
        raise RuntimeError("Weixin credentials are missing")
    client = WeixinClient(
        credentials,
        state_store=state,
        report_root=root / "output",
        request_timeout_seconds=weixin["request_timeout_seconds"],
        poll_timeout_seconds=weixin["poll_timeout_seconds"],
        retry_attempts=weixin["retry_attempts"],
        maximum_backoff_seconds=weixin["maximum_backoff_seconds"],
    )
    return WeixinPushService(
        client=client,
        subscriptions=WeixinSubscriptionStore(state),
        deliveries=DailyDeliveryStore(root / "data" / "delivery"),
        report_root=root / "output",
    )


async def _main_async(args: argparse.Namespace) -> int:
    service = build_push_service(load_config(PROJECT_ROOT / "config.json"))
    try:
        subscriptions = service.subscriptions.all()
        if args.status:
            available = sum(
                item.context_status == "available" and bool(item.last_context_token)
                for item in subscriptions
            )
            expired = sum(item.context_status == "expired" for item in subscriptions)
            print(
                f"订阅用户数量：{sum(item.enabled for item in subscriptions)}\n可用 Token 数量：{available}\n失效 Token 数量：{expired}"
            )
            return 0
        active = [
            item
            for item in subscriptions
            if item.enabled and item.context_status == "available"
        ]
        if not active:
            print("没有可用的订阅会话凭据。")
            return 1
        subscription = active[-1]
        if args.send_test:
            result = await service.client.send_text(
                subscription.user_id,
                subscription.last_context_token or "",
                "微信主动推送测试成功。\n当前会话凭据仍可使用。",
            )
            service.subscriptions.update_push(
                subscription,
                status="sent" if result.success else "failed",
                error_code=result.error_code,
            )
            print("主动测试发送成功。" if result.success else "主动测试发送失败。")
            return 0 if result.success else 1
        delivery = service.deliveries.latest()
        if delivery is None:
            report = ReportLocator(service.report_root, (".html",)).latest()
            if report is None:
                print("尚未找到可发送的日报。")
                return 1
            delivery = service.deliveries.get_or_create(
                report.generated_at.astimezone(ZoneInfo("Asia/Shanghai")).date()
            )
            delivery = service.deliveries.update(
                delivery,
                generation_status="success",
                report_path=report.path.relative_to(PROJECT_ROOT).as_posix(),
                report_generated_at=report.generated_at,
                report_run_id=str(report.metadata.get("run_id") or "") or None,
            )
        if args.force:
            print("警告：--force 会显式重发已成功日报。")
        result = await service.send_daily_digest(
            subscription, delivery, force=args.force
        )
        print("日报发送成功。" if result.success else f"日报未发送：{result.status}")
        return 0 if result.success else 1
    finally:
        await service.client.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Weixin proactive push")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--send-test", action="store_true")
    group.add_argument("--send-latest", action="store_true")
    group.add_argument("--status", action="store_true")
    parser.add_argument("--force", action="store_true")
    return asyncio.run(_main_async(parser.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
