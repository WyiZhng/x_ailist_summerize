"""Durable models and storage for daily report generation and notification."""

from __future__ import annotations

from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from .weixin_models import DailyDelivery
from .weixin_state import WeixinStateError


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class DailyDeliveryStore:
    """Atomically persist delivery state below an ignored private directory."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        self.path = self.data_dir / "daily_deliveries.json"
        self.attempts_path = self.data_dir / "push_attempts.jsonl"
        self.runtime_path = self.data_dir / "runtime_state.json"

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            import json

            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise WeixinStateError("Daily delivery state is unreadable") from exc
        if not isinstance(raw, dict):
            raise WeixinStateError("Daily delivery state must be an object")
        return raw

    def _write(self, raw: dict[str, Any]) -> None:
        from .weixin_state import WeixinStateStore

        WeixinStateStore(self.data_dir)._atomic_write(self.path, raw, private=True)

    @staticmethod
    def _decode(value: Any) -> DailyDelivery:
        if not isinstance(value, dict):
            raise WeixinStateError("Daily delivery entry is invalid")

        def time(name: str) -> datetime | None:
            if not value.get(name):
                return None
            parsed = datetime.fromisoformat(str(value[name]))
            # Earlier local runs wrote runner timestamps without an offset.
            # They are application-owned UTC timestamps, not user input.
            return (
                parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed
            )

        try:
            return DailyDelivery(
                id=str(value["id"]),
                delivery_date=date.fromisoformat(str(value["delivery_date"])),
                timezone=str(value.get("timezone", "Asia/Shanghai")),
                generation_status=str(value.get("generation_status", "pending")),
                report_run_id=value.get("report_run_id"),
                report_path=value.get("report_path"),
                report_generated_at=time("report_generated_at"),
                notification_status=str(value.get("notification_status", "pending")),
                notification_attempts=int(value.get("notification_attempts", 0)),
                notification_last_attempt_at=time("notification_last_attempt_at"),
                notification_sent_at=time("notification_sent_at"),
                notification_error_code=value.get("notification_error_code"),
                created_at=time("created_at"),
                updated_at=time("updated_at"),
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise WeixinStateError("Daily delivery entry is invalid") from exc

    @staticmethod
    def _encode(delivery: DailyDelivery) -> dict[str, Any]:
        raw = asdict(delivery)
        raw["delivery_date"] = delivery.delivery_date.isoformat()
        for key, value in list(raw.items()):
            if isinstance(value, datetime):
                raw[key] = value.isoformat()
        return raw

    def get(self, value: date) -> DailyDelivery | None:
        raw = self._read().get(value.isoformat())
        return self._decode(raw) if raw else None

    def latest(self) -> DailyDelivery | None:
        entries = [self._decode(value) for value in self._read().values()]
        return max(entries, key=lambda item: item.delivery_date) if entries else None

    def get_or_create(
        self, value: date, timezone_name: str = "Asia/Shanghai"
    ) -> DailyDelivery:
        existing = self.get(value)
        if existing:
            return existing
        now = utc_now()
        delivery = DailyDelivery(
            id=f"daily:{value.isoformat()}",
            delivery_date=value,
            timezone=timezone_name,
            created_at=now,
            updated_at=now,
        )
        return self.save(delivery)

    def save(self, delivery: DailyDelivery) -> DailyDelivery:
        raw = self._read()
        raw[delivery.delivery_date.isoformat()] = self._encode(delivery)
        self._write(raw)
        return delivery

    def update(self, delivery: DailyDelivery, **changes: Any) -> DailyDelivery:
        return self.save(
            DailyDelivery(**{**asdict(delivery), **changes, "updated_at": utc_now()})
        )

    def record_attempt(
        self,
        *,
        delivery: DailyDelivery,
        status: str,
        error_code: str | None,
        token_age_seconds: int | None,
    ) -> None:
        import json
        import os

        self.data_dir.mkdir(parents=True, exist_ok=True)
        with self.attempts_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "delivery_date": delivery.delivery_date.isoformat(),
                        "status": status,
                        "error_code": error_code,
                        "token_age_seconds": token_age_seconds,
                        "at": utc_now().isoformat(),
                    }
                )
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(self.attempts_path, 0o600)
