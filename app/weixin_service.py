"""Long-poll service connecting private Weixin commands to local reports."""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import load_config
from .weixin_client import (
    WeixinAuthenticationError,
    WeixinClient,
    WeixinClientError,
)
from .weixin_commands import ReportLocator, WeixinCommandHandler
from .weixin_models import WeixinInboundMessage, WeixinSyncState, WeixinUpdateBatch
from .weixin_state import WeixinStateError, WeixinStateStore
from .weixin_subscription import WeixinSubscriptionStore


logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class WeixinService:
    """Process batches transactionally and keep long polling cancellable."""

    def __init__(
        self,
        *,
        client: WeixinClient,
        state_store: WeixinStateStore,
        command_handler: WeixinCommandHandler,
        retry_attempts: int = 3,
        maximum_backoff_seconds: int = 120,
        subscription_store: WeixinSubscriptionStore | None = None,
    ) -> None:
        self.client = client
        self.state_store = state_store
        self.command_handler = command_handler
        self.retry_attempts = max(1, retry_attempts)
        self.maximum_backoff_seconds = max(1, maximum_backoff_seconds)
        self.stop_event = asyncio.Event()
        self.subscription_store = subscription_store

    def stop(self) -> None:
        """Request graceful shutdown after the current operation."""
        self.stop_event.set()

    async def _process_message(self, message: WeixinInboundMessage) -> bool:
        masked_id = self.state_store.masked_message_id(message.message_id)
        if self.state_store.is_processed(message.message_id):
            logger.info("weixin_message status=duplicate id=%s", masked_id)
            return True
        if not message.context_token:
            logger.warning("weixin_message status=missing_context id=%s", masked_id)
            self.state_store.mark_processed(message.message_id, "unreplyable")
            return True

        self.state_store.save_user_context(
            message.user_id,
            message.context_token,
            message.message_id,
            message.received_at,
        )
        if self.subscription_store:
            self.subscription_store.refresh_context(
                message.user_id,
                message.context_token,
                message.message_id,
                message.received_at,
            )
        result = self.command_handler.handle(message.text, user_id=message.user_id)
        if result.command == "retry_delivery" and self.subscription_store:
            from .daily_delivery_models import DailyDeliveryStore
            from .weixin_push import WeixinPushService

            deliveries = DailyDeliveryStore(PROJECT_ROOT / "data" / "delivery")
            delivery = deliveries.latest()
            subscription = self.subscription_store.get(message.user_id)
            if delivery is None or delivery.generation_status != "success":
                result = result.__class__(
                    "retry_delivery", "尚未找到可补发的已生成日报。"
                )
            elif delivery.notification_status == "sent":
                result = result.__class__(
                    "retry_delivery", "该日报已经成功发送，无需重复补发。"
                )
            elif subscription is None:
                result = result.__class__(
                    "retry_delivery", "当前会话凭据不可用，请稍后重试。"
                )
            else:
                outcome = await WeixinPushService(
                    client=self.client,
                    subscriptions=self.subscription_store,
                    deliveries=deliveries,
                    report_root=PROJECT_ROOT / "output",
                ).send_daily_digest(subscription, delivery)
                result = result.__class__(
                    "retry_delivery",
                    "日报补发成功。"
                    if outcome.success
                    else "日报补发失败，等待下次会话凭据刷新后重试。",
                )
        await self.client.send_typing(message.user_id, message.context_token)

        if result.file_path is not None:
            sent = await self.client.send_file(
                message.user_id,
                message.context_token,
                result.file_path,
                result.display_name or result.file_path.name,
            )
            self.state_store.record_send(
                message.message_id, result.command, sent.success
            )
            if not sent.success:
                fallback = await self.client.send_text(
                    message.user_id,
                    message.context_token,
                    "HTML 日报发送失败，请稍后重试。",
                )
                logger.warning(
                    "weixin_command command=report status=failed id=%s", masked_id
                )
                return fallback.success
            # File delivery is the idempotency boundary. Do not resend the file
            # if the optional confirmation text later fails.
            self.state_store.mark_processed(message.message_id, result.command)
            await self.client.send_text(
                message.user_id, message.context_token, "HTML 日报已发送。"
            )
            logger.info("weixin_command command=report status=success id=%s", masked_id)
            return True

        sent = await self.client.send_text(
            message.user_id, message.context_token, result.text
        )
        self.state_store.record_send(message.message_id, result.command, sent.success)
        if not sent.success:
            logger.warning(
                "weixin_command command=%s status=failed id=%s",
                result.command,
                masked_id,
            )
            return False
        self.state_store.mark_processed(message.message_id, result.command)
        logger.info(
            "weixin_command command=%s status=success id=%s",
            result.command,
            masked_id,
        )
        return True

    async def process_batch(self, batch: WeixinUpdateBatch) -> bool:
        """Process all messages, committing the cursor only if none failed."""
        batch_ok = True
        last_message_id: str | None = None
        for message in batch.messages:
            try:
                processed = await self._process_message(message)
            except WeixinAuthenticationError:
                raise
            except Exception as exc:
                logger.error(
                    "weixin_message status=error type=%s id=%s",
                    type(exc).__name__,
                    self.state_store.masked_message_id(message.message_id),
                )
                processed = False
            batch_ok = batch_ok and processed
            if processed:
                last_message_id = message.message_id
        if batch_ok:
            previous = self.state_store.load_sync_state()
            self.state_store.save_sync_state(
                WeixinSyncState(
                    get_updates_buf=batch.get_updates_buf,
                    updated_at=datetime.now(timezone.utc),
                    last_message_id=last_message_id or previous.last_message_id,
                )
            )
        return batch_ok

    async def run(self) -> None:
        """Run until stopped, with bounded exponential backoff."""
        state = self.state_store.load_sync_state()
        failures = 0
        logger.info("weixin_service status=started")
        while not self.stop_event.is_set():
            try:
                batch = await self.client.get_updates(state.get_updates_buf)
                if await self.process_batch(batch):
                    state = self.state_store.load_sync_state()
                    failures = 0
                else:
                    failures += 1
            except WeixinAuthenticationError:
                self.state_store.mark_credentials_expired()
                logger.error("weixin_service status=authentication_expired")
                raise
            except (WeixinClientError, WeixinStateError) as exc:
                failures += 1
                logger.warning(
                    "weixin_poll status=retry type=%s attempt=%d",
                    type(exc).__name__,
                    failures,
                )
            if failures:
                delay = min(2 ** min(failures - 1, 10), self.maximum_backoff_seconds)
                try:
                    await asyncio.wait_for(self.stop_event.wait(), timeout=delay)
                except TimeoutError:
                    pass
        logger.info("weixin_service status=stopped")


def _resolve_project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def build_service(config: dict[str, Any]) -> WeixinService:
    """Build the service from unified non-secret config and private state."""
    weixin_config = config["weixin"]
    state_store = WeixinStateStore(_resolve_project_path(weixin_config["data_dir"]))
    credentials = state_store.load_credentials()
    if credentials is None or credentials.status != "active":
        raise WeixinAuthenticationError(
            "Weixin credentials are missing or expired; run python -m app.weixin_auth"
        )
    output_dir = PROJECT_ROOT / "output"
    locator = ReportLocator(
        output_dir,
        tuple(weixin_config.get("allowed_report_extensions", [".html"])),
    )
    handler = WeixinCommandHandler(
        report_locator=locator,
        config=config,
        cookies_path=PROJECT_ROOT / "browser_session" / "cookies.json",
        subscription_store=WeixinSubscriptionStore(state_store),
    )
    client = WeixinClient(
        credentials,
        state_store=state_store,
        report_root=output_dir,
        request_timeout_seconds=weixin_config["request_timeout_seconds"],
        poll_timeout_seconds=weixin_config["poll_timeout_seconds"],
        retry_attempts=weixin_config["retry_attempts"],
        maximum_backoff_seconds=weixin_config["maximum_backoff_seconds"],
    )
    return WeixinService(
        client=client,
        state_store=state_store,
        command_handler=handler,
        retry_attempts=weixin_config["retry_attempts"],
        maximum_backoff_seconds=weixin_config["maximum_backoff_seconds"],
        subscription_store=WeixinSubscriptionStore(state_store),
    )


async def _run_service(service: WeixinService) -> None:
    loop = asyncio.get_running_loop()
    for signal_name in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signal_name, service.stop)
        except NotImplementedError:
            pass
    try:
        await service.run()
    finally:
        await service.client.close()


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the Weixin private-chat service."""
    parser = argparse.ArgumentParser(description="Run Weixin ClawBot service")
    parser.parse_args(argv)
    config = load_config(PROJECT_ROOT / "config.json")
    if not config["weixin"]["enabled"]:
        print("Weixin integration is disabled in config.json.")
        return 1
    try:
        service = build_service(config)
        asyncio.run(_run_service(service))
    except KeyboardInterrupt:
        return 130
    except (WeixinAuthenticationError, WeixinStateError) as exc:
        logger.error("Weixin service stopped (%s)", type(exc).__name__)
        print(str(exc))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
