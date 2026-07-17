"""Safe command handling for Weixin-triggered daily report access."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from .weixin_models import WeixinCommandResult
from .weixin_subscription import WeixinSubscriptionStore
from .event_summary import top5_text


HELP_TEXT = """可用命令：

日报：查看最新 AI 日报摘要
完整报告：接收最新 HTML 报告
状态：查看服务和日报状态
订阅每日推送：开启每天北京时间 09:00 推送
取消每日推送：关闭主动推送
订阅状态：查看订阅与会话凭据状态
补发日报：补发最近生成但未成功推送的日报
帮助：查看命令列表"""

COMMAND_ALIASES = {
    "帮助": "help",
    "help": "help",
    "?": "help",
    "日报": "daily",
    "今日日报": "daily",
    "AI 日报": "daily",
    "ai 日报": "daily",
    "最新日报": "daily",
    "完整报告": "report",
    "发送报告": "report",
    "报告文件": "report",
    "状态": "status",
    "服务状态": "status",
    "订阅每日推送": "subscribe",
    "订阅日报": "subscribe",
    "开启每日推送": "subscribe",
    "取消每日推送": "unsubscribe",
    "取消订阅": "unsubscribe",
    "关闭每日推送": "unsubscribe",
    "订阅状态": "subscription_status",
    "推送状态": "subscription_status",
    "补发日报": "retry_delivery",
    "重新发送日报": "retry_delivery",
}


@dataclass(frozen=True)
class LatestReport:
    """A report path proven to remain inside the output directory."""

    path: Path
    metadata: dict[str, Any]
    generated_at: datetime


class ReportLocator:
    """Resolve the newest successful HTML report without trusting messages."""

    def __init__(self, output_dir: Path, allowed_extensions: tuple[str, ...]) -> None:
        self.output_dir = output_dir
        self.allowed_extensions = tuple(value.lower() for value in allowed_extensions)

    def _safe_path(self, filename: str) -> Path | None:
        if not re.fullmatch(r"summary_[A-Za-z0-9_.-]+\.html", filename):
            return None
        candidate = self.output_dir / filename
        try:
            if candidate.is_symlink():
                return None
            root = self.output_dir.resolve()
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(root)
        except (OSError, RuntimeError, ValueError):
            return None
        if (
            resolved.suffix.lower() not in self.allowed_extensions
            or not resolved.is_file()
        ):
            return None
        return resolved

    def latest(self) -> LatestReport | None:
        """Return the newest successful history entry, with safe scan fallback."""
        history_path = self.output_dir / "history.json"
        history: dict[str, Any] = {}
        if history_path.exists():
            try:
                loaded = json.loads(history_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    history = loaded
            except (OSError, UnicodeError, json.JSONDecodeError):
                history = {}

        candidates: list[tuple[datetime, Path, dict[str, Any]]] = []
        for filename, metadata in history.items():
            if not isinstance(filename, str) or not isinstance(metadata, dict):
                continue
            if metadata.get("status") not in {None, "succeeded", "success"}:
                continue
            path = self._safe_path(filename)
            if path is None:
                continue
            generated = self._generated_at(metadata, path)
            candidates.append((generated, path, metadata))

        if not candidates and self.output_dir.exists():
            for candidate in self.output_dir.glob("summary_*.html"):
                path = self._safe_path(candidate.name)
                if path is not None:
                    generated = datetime.fromtimestamp(
                        path.stat().st_mtime
                    ).astimezone()
                    candidates.append((generated, path, {}))
        if not candidates:
            return None
        generated, path, metadata = max(candidates, key=lambda item: item[0])
        return LatestReport(path=path, metadata=metadata, generated_at=generated)

    def find(self, filename: str) -> LatestReport | None:
        """Return one explicitly named report after the same sandbox checks."""
        path = self._safe_path(filename)
        if path is None:
            return None
        generated = datetime.fromtimestamp(path.stat().st_mtime).astimezone()
        return LatestReport(path=path, metadata={}, generated_at=generated)

    @staticmethod
    def _generated_at(metadata: Mapping[str, Any], path: Path) -> datetime:
        value = metadata.get("generated_at")
        if isinstance(value, str):
            try:
                parsed = datetime.fromisoformat(value)
                return parsed.astimezone() if parsed.tzinfo else parsed.astimezone()
            except ValueError:
                pass
        return datetime.fromtimestamp(path.stat().st_mtime).astimezone()


class WeixinCommandHandler:
    """Dispatch the fixed command allowlist without invoking X or an LLM."""

    def __init__(
        self,
        *,
        report_locator: ReportLocator,
        config: Mapping[str, Any],
        cookies_path: Path,
        subscription_store: WeixinSubscriptionStore | None = None,
    ) -> None:
        self.report_locator = report_locator
        self.config = config
        self.cookies_path = cookies_path
        self.subscription_store = subscription_store

    @staticmethod
    def classify(text: str | None) -> str:
        """Classify one exact command, defaulting to short help."""
        normalized = (text or "").strip()
        return COMMAND_ALIASES.get(normalized, "unknown")

    def handle(
        self, text: str | None, *, user_id: str | None = None
    ) -> WeixinCommandResult:
        """Execute a local read-only command."""
        command = self.classify(text)
        if command in {"help", "unknown"}:
            return WeixinCommandResult(command=command, text=HELP_TEXT)
        if command == "daily":
            return self._daily()
        if command == "report":
            return self._report()
        if command == "status":
            return self._status()
        if command == "subscribe" and user_id and self.subscription_store:
            self.subscription_store.set_enabled(user_id, True)
            return WeixinCommandResult(
                command,
                "每日 AI 日报推送已开启。\n\n计划时间：北京时间 09:00\n当前推送依赖最近一次微信会话凭据。\n若凭据失效，我会在你下次发送消息后自动刷新。",
            )
        if command == "unsubscribe" and user_id and self.subscription_store:
            self.subscription_store.set_enabled(user_id, False)
            return WeixinCommandResult(
                command, "每日 AI 日报推送已关闭。历史发送记录会被保留。"
            )
        if command == "subscription_status" and user_id and self.subscription_store:
            return self._subscription_status(user_id)
        if command == "retry_delivery":
            return WeixinCommandResult(
                command, "正在尝试补发最近一份已生成但未成功推送的日报。"
            )
        return self._status()

    def _subscription_status(self, user_id: str) -> WeixinCommandResult:
        subscription = (
            self.subscription_store.get(user_id) if self.subscription_store else None
        )
        enabled = bool(subscription and subscription.enabled)
        context = {"available": "可用", "expired": "已失效"}.get(
            subscription.context_status if subscription else "unknown", "未知"
        )
        latest = "尚未发送"
        if subscription and subscription.last_push_success_at:
            latest = "成功"
        elif subscription and subscription.last_push_error_code:
            latest = "失败"
        return WeixinCommandResult(
            "subscription_status",
            f"每日推送：{'已开启' if enabled else '已关闭'}\n计划时间：北京时间 09:00\n最近会话凭据：{context}\n最近推送：{latest}",
        )

    def _daily(self) -> WeixinCommandResult:
        report = self.report_locator.latest()
        if report is None:
            return WeixinCommandResult("daily", "尚未找到已生成的 AI 日报。")
        structured = top5_text(
            self.report_locator.output_dir.parent / "data", report.generated_at.date()
        )
        if structured:
            return WeixinCommandResult("daily", structured)
        metadata = report.metadata
        text = (
            f"AI 每日情报｜{report.generated_at:%Y-%m-%d}\n\n"
            "最新报告已生成\n"
            f"抓取帖子：{int(metadata.get('tweets', 0) or 0)}\n"
            f"外部链接：{int(metadata.get('links', 0) or 0)}\n"
            f"生成时间：{report.generated_at:%H:%M}\n\n"
            "发送“完整报告”接收 HTML 文件。"
        )
        return WeixinCommandResult("daily", text)

    def _report(self) -> WeixinCommandResult:
        report = self.report_locator.latest()
        if report is None:
            return WeixinCommandResult("report", "尚未找到可发送的 HTML 日报。")
        display_name = f"AI日报-{report.generated_at:%Y-%m-%d}.html"
        return WeixinCommandResult(
            "report",
            "正在发送最新 HTML 日报。",
            file_path=report.path,
            display_name=display_name,
        )

    def _status(self) -> WeixinCommandResult:
        report = self.report_locator.latest()
        twitter = self.config.get("twitter", {})
        summary = self.config.get("summarization", {})
        method = twitter.get("fetch_method", "twikit")
        environment_cookies = bool(os.environ.get("XLS_X_AUTH_TOKEN")) and bool(
            os.environ.get("XLS_X_CT0")
        )
        x_configured = bool(twitter.get("list_urls")) and bool(
            twitter.get("api_bearer_token")
            if method == "api"
            else self.cookies_path.exists() or environment_cookies
        )
        provider = summary.get("provider")
        option = summary.get("options", {}).get(provider, {})
        ai_configured = bool(
            provider in {"ollama", "lmstudio"} or option.get("api_key")
        )
        generated = report.generated_at.strftime("%Y-%m-%d %H:%M") if report else "无"
        text = (
            "微信服务：运行中\n"
            f"最新日报：{'存在' if report else '不存在'}\n"
            f"报告生成时间：{generated}\n"
            f"X 抓取配置：{'已配置' if x_configured else '未配置'}\n"
            f"DeepSeek：{'已配置' if ai_configured else '未配置'}"
        )
        return WeixinCommandResult("status", text)
