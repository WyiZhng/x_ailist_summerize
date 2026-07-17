"""Beijing-time daily generation and push orchestration."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from .config import activate_project_environment, load_config
from .daily_delivery_models import DailyDeliveryStore
from .process_lock import ProcessLock
from .task_runner import DigestTaskRequest, DigestTaskRunner
from .weixin_push import PROJECT_ROOT, build_push_service
from .x_list_summarizer import XApiFetcher, XListFetcher
from .llm_providers import LLMProvider
from .storage import FilePostStore

BEIJING = ZoneInfo("Asia/Shanghai")


def beijing_now(now: datetime | None = None) -> datetime:
    return (now or datetime.now(tz=BEIJING)).astimezone(BEIJING)


def tick_due(
    now: datetime, *, hour: int = 9, minute: int = 0, deadline_hour: int = 12
) -> bool:
    local = beijing_now(now)
    return (local.hour, local.minute) >= (hour, minute) and (
        local.hour,
        local.minute,
    ) <= (deadline_hour, 0)


def _build_runner(config: dict) -> DigestTaskRunner:
    twitter = config["twitter"]
    root = PROJECT_ROOT
    fetcher = (
        XApiFetcher(
            bearer_token=twitter.get("api_bearer_token", ""),
            list_owner=twitter.get("list_owner"),
        )
        if twitter.get("fetch_method") == "api"
        else XListFetcher(
            cookies_path=root / "browser_session" / "cookies.json",
            list_owner=twitter.get("list_owner"),
            proxy=twitter.get("proxy") or None,
        )
    )
    data_dir = root / config["storage"]["data_dir"]
    return DigestTaskRunner(
        fetcher=fetcher,
        summary_provider=LLMProvider(config),
        post_store=FilePostStore(data_dir),
    )


class DailyDeliveryService:
    """Keep generation independent from notification and reuse existing runner."""

    def __init__(
        self,
        config: dict,
        *,
        store: DailyDeliveryStore | None = None,
        runner_factory=_build_runner,
        push_factory=build_push_service,
    ) -> None:
        self.config, self.settings = config, config["daily_delivery"]
        self.store = store or DailyDeliveryStore(PROJECT_ROOT / "data" / "delivery")
        self.runner_factory, self.push_factory = runner_factory, push_factory

    async def run_for_date(
        self, day, *, force_run: bool = False, retry_only: bool = False
    ) -> str:
        delivery = self.store.get_or_create(day, self.settings["timezone"])
        if delivery.notification_status == "sent" and not force_run:
            return "already_sent"
        if delivery.generation_status != "success" and not retry_only:
            delivery = self.store.update(delivery, generation_status="running")
            try:
                twitter = self.config["twitter"]
                runner = self.runner_factory(self.config)
                request = DigestTaskRequest.from_values(
                    twitter["list_urls"],
                    max_tweets=twitter["max_tweets"],
                    output_dir=PROJECT_ROOT / "output",
                    incremental=bool(twitter.get("incremental_sync", True)),
                    data_dir=PROJECT_ROOT / self.config["storage"]["data_dir"],
                    page_size=twitter["page_size"],
                    max_pages=twitter["max_pages"],
                    initial_fetch_limit=twitter["initial_fetch_limit"],
                )
                result = await runner.run(request)
                if not result.success or not result.report_path:
                    self.store.update(delivery, generation_status="failed")
                    return "generation_failed"
                relative = (
                    result.report_path.resolve().relative_to(PROJECT_ROOT).as_posix()
                )
                delivery = self.store.update(
                    delivery,
                    generation_status="success",
                    report_run_id=result.run_id,
                    report_path=relative,
                    report_generated_at=(
                        result.finished_at.replace(tzinfo=timezone.utc)
                        if result.finished_at.tzinfo is None
                        else result.finished_at
                    ),
                )
            except Exception:
                self.store.update(delivery, generation_status="failed")
                return "generation_failed"
        if delivery.generation_status != "success":
            return "generation_not_available"
        if not self.settings["push_enabled"]:
            self.store.update(delivery, notification_status="subscription_disabled")
            return "push_disabled"
        push = self.push_factory(self.config)
        try:
            subscriptions = [item for item in push.subscriptions.all() if item.enabled]
            if not subscriptions:
                self.store.update(delivery, notification_status="subscription_disabled")
                return "no_subscription"
            outcomes = [
                await push.send_daily_digest(item, delivery) for item in subscriptions
            ]
            return (
                "sent"
                if any(item.success for item in outcomes)
                else outcomes[-1].status
            )
        finally:
            await push.client.close()

    async def tick(self, now: datetime | None = None) -> str:
        current = beijing_now(now)
        if not self.settings["enabled"]:
            return "disabled"
        if not tick_due(
            current,
            hour=self.settings["scheduled_hour"],
            minute=self.settings["scheduled_minute"],
            deadline_hour=self.settings["catchup_deadline_hour"],
        ):
            return "outside_window"
        lock = ProcessLock(
            PROJECT_ROOT / "data" / "runtime" / "locks",
            "daily",
            current.date().isoformat(),
        )
        if not lock.acquire():
            return "locked"
        try:
            return await self.run_for_date(current.date())
        finally:
            lock.release()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Daily report delivery")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-now", action="store_true")
    group.add_argument("--retry-failed", action="store_true")
    group.add_argument("--status", action="store_true")
    group.add_argument("--tick", action="store_true")
    args = parser.parse_args(argv)
    config_path = PROJECT_ROOT / "config.json"
    config = load_config(config_path)
    activate_project_environment(config_path)
    service = DailyDeliveryService(config)
    current = beijing_now()
    if args.status:
        item = service.store.get(current.date())
        print(
            f"北京时间：{current:%Y-%m-%d %H:%M}\n今日生成状态：{item.generation_status if item else 'pending'}\n今日推送状态：{item.notification_status if item else 'pending'}\n下次计划时间：{current.date()} 09:00"
        )
        return 0
    if args.tick:
        result = asyncio.run(service.tick(current))
    elif args.retry_failed:
        result = asyncio.run(service.run_for_date(current.date(), retry_only=True))
    else:
        result = asyncio.run(service.run_for_date(current.date()))
    print(f"Daily delivery: {result}")
    return (
        0
        if result in {"sent", "already_sent", "no_subscription", "push_disabled"}
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
