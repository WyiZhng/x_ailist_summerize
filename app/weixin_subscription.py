"""Private subscription persistence for proactive Weixin delivery."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from .weixin_models import WeixinSubscription
from .weixin_state import WeixinStateError, WeixinStateStore, _parse_datetime


def _now() -> datetime:
    return datetime.now(timezone.utc)


class WeixinSubscriptionStore:
    """Keep subscriptions in the existing private Weixin data directory."""

    def __init__(self, state_store: WeixinStateStore) -> None:
        self.state_store = state_store
        self.path = state_store.data_dir / "subscriptions.json"

    def _load_raw(self) -> dict[str, Any]:
        return self.state_store._read_object(self.path, missing={})

    @staticmethod
    def _deserialize(user_id: str, value: Any) -> WeixinSubscription:
        if not isinstance(value, dict):
            raise WeixinStateError("Weixin subscriptions are invalid")

        def optional_time(name: str) -> datetime | None:
            item = value.get(name)
            return _parse_datetime(item, name) if item else None

        return WeixinSubscription(
            user_id=user_id,
            enabled=bool(value.get("enabled", False)),
            created_at=optional_time("created_at"),
            updated_at=optional_time("updated_at"),
            timezone=str(value.get("timezone", "Asia/Shanghai")),
            scheduled_hour=int(value.get("scheduled_hour", 9)),
            scheduled_minute=int(value.get("scheduled_minute", 0)),
            last_context_token=value.get("last_context_token")
            if isinstance(value.get("last_context_token"), str)
            else None,
            last_context_received_at=optional_time("last_context_received_at"),
            last_context_message_id=value.get("last_context_message_id")
            if isinstance(value.get("last_context_message_id"), str)
            else None,
            context_status=str(value.get("context_status", "unknown")),
            last_push_attempt_at=optional_time("last_push_attempt_at"),
            last_push_success_at=optional_time("last_push_success_at"),
            last_push_error_code=value.get("last_push_error_code")
            if isinstance(value.get("last_push_error_code"), str)
            else None,
        )

    @staticmethod
    def _serialize(subscription: WeixinSubscription) -> dict[str, Any]:
        value = asdict(subscription)
        value.pop("user_id", None)
        for key, item in list(value.items()):
            if isinstance(item, datetime):
                value[key] = item.isoformat()
        return value

    def get(self, user_id: str) -> WeixinSubscription | None:
        raw = self._load_raw().get(self.state_store.user_key(user_id))
        return self._deserialize(user_id, raw) if raw is not None else None

    def all(self) -> list[WeixinSubscription]:
        values: list[WeixinSubscription] = []
        for key, raw in self._load_raw().items():
            if not isinstance(raw, dict) or not isinstance(raw.get("user_id"), str):
                continue
            values.append(self._deserialize(str(raw["user_id"]), raw))
        return values

    def save(self, subscription: WeixinSubscription) -> WeixinSubscription:
        raw = self._load_raw()
        item = self._serialize(subscription)
        # User ID is private state; its hash is only a storage key, not a substitute.
        item["user_id"] = subscription.user_id
        raw[self.state_store.user_key(subscription.user_id)] = item
        self.state_store._atomic_write(self.path, raw, private=True)
        return subscription

    def refresh_context(
        self, user_id: str, token: str, message_id: str, received_at: datetime
    ) -> WeixinSubscription:
        current = self.get(user_id)
        now = _now()
        return self.save(
            WeixinSubscription(
                user_id=user_id,
                enabled=current.enabled if current else False,
                created_at=current.created_at
                if current and current.created_at
                else now,
                updated_at=now,
                timezone=current.timezone if current else "Asia/Shanghai",
                scheduled_hour=current.scheduled_hour if current else 9,
                scheduled_minute=current.scheduled_minute if current else 0,
                last_context_token=token,
                last_context_received_at=received_at,
                last_context_message_id=message_id,
                context_status="available",
                last_push_attempt_at=current.last_push_attempt_at if current else None,
                last_push_success_at=current.last_push_success_at if current else None,
                last_push_error_code=current.last_push_error_code if current else None,
            )
        )

    def set_enabled(self, user_id: str, enabled: bool) -> WeixinSubscription:
        current = self.get(user_id)
        now = _now()
        if current is None:
            current = WeixinSubscription(
                user_id=user_id, created_at=now, updated_at=now
            )
        return self.save(
            WeixinSubscription(
                **{**asdict(current), "enabled": enabled, "updated_at": now}
            )
        )

    def update_push(
        self,
        subscription: WeixinSubscription,
        *,
        status: str,
        error_code: str | None = None,
    ) -> WeixinSubscription:
        now = _now()
        context_status = (
            "expired" if status == "token_expired" else subscription.context_status
        )
        return self.save(
            WeixinSubscription(
                **{
                    **asdict(subscription),
                    "updated_at": now,
                    "context_status": context_status,
                    "last_push_attempt_at": now,
                    "last_push_success_at": now
                    if status == "sent"
                    else subscription.last_push_success_at,
                    "last_push_error_code": error_code,
                }
            )
        )
