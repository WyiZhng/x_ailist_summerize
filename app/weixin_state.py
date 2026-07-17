"""Private, durable state storage for Weixin credentials and polling."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .weixin_models import WeixinCredentials, WeixinSyncState


class WeixinStateError(RuntimeError):
    """Raised when private state is malformed or cannot be persisted."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_datetime(value: Any, field_name: str) -> datetime:
    if not isinstance(value, str):
        raise WeixinStateError(f"Invalid {field_name} in Weixin state")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise WeixinStateError(f"Invalid {field_name} in Weixin state") from exc
    if parsed.tzinfo is None:
        raise WeixinStateError(f"Timezone required for {field_name}")
    return parsed


class WeixinStateStore:
    """Store all Weixin-private files below one configured directory."""

    def __init__(self, data_dir: Path | str) -> None:
        self.data_dir = Path(data_dir)
        self.credentials_path = self.data_dir / "credentials.json"
        self.sync_state_path = self.data_dir / "sync_state.json"
        self.users_path = self.data_dir / "users.json"
        self.processed_path = self.data_dir / "processed_messages.jsonl"
        self.send_history_path = self.data_dir / "send_history.jsonl"
        self.temporary_dir = self.data_dir / "temporary"

    def ensure_directories(self) -> None:
        """Create private state directories without touching state files."""
        self.temporary_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def user_key(user_id: str) -> str:
        """Return a stable, non-reversible per-user storage key."""
        return hashlib.sha256(user_id.encode("utf-8")).hexdigest()

    @staticmethod
    def masked_message_id(message_id: str) -> str:
        """Return a stable short identifier safe for ordinary logs."""
        return hashlib.sha256(message_id.encode("utf-8")).hexdigest()[:12]

    def _read_object(self, path: Path, *, missing: dict[str, Any]) -> dict[str, Any]:
        if not path.exists():
            return missing.copy()
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise WeixinStateError(f"Weixin state is unreadable: {path.name}") from exc
        if not isinstance(value, dict):
            raise WeixinStateError(f"Weixin state must be an object: {path.name}")
        return value

    def _atomic_write(self, path: Path, value: Any, *, private: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                json.dump(value, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            if private:
                os.chmod(temporary, 0o600)
            os.replace(temporary, path)
            temporary = None
            if private:
                os.chmod(path, 0o600)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def load_credentials(self) -> WeixinCredentials | None:
        """Load credentials, failing explicitly if the file is corrupt."""
        if not self.credentials_path.exists():
            return None
        data = self._read_object(self.credentials_path, missing={})
        try:
            return WeixinCredentials(
                token=str(data["token"]),
                account_id=str(data["account_id"]),
                base_url=str(data["base_url"]),
                created_at=_parse_datetime(data["created_at"], "created_at"),
                updated_at=_parse_datetime(data["updated_at"], "updated_at"),
                expires_at=(
                    _parse_datetime(data["expires_at"], "expires_at")
                    if data.get("expires_at")
                    else None
                ),
                status=str(data.get("status", "active")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise WeixinStateError("Weixin credentials are invalid") from exc

    def save_credentials(self, credentials: WeixinCredentials) -> None:
        """Atomically save credentials with owner-only permissions."""
        data = asdict(credentials)
        for key in ("created_at", "updated_at", "expires_at"):
            value = data[key]
            data[key] = value.isoformat() if value is not None else None
        self._atomic_write(self.credentials_path, data, private=True)

    def mark_credentials_expired(self) -> None:
        """Mark the current credentials unusable without deleting evidence."""
        credentials = self.load_credentials()
        if credentials is None:
            return
        self.save_credentials(
            WeixinCredentials(
                token=credentials.token,
                account_id=credentials.account_id,
                base_url=credentials.base_url,
                created_at=credentials.created_at,
                updated_at=_utc_now(),
                expires_at=credentials.expires_at,
                status="expired",
            )
        )

    def load_sync_state(self) -> WeixinSyncState:
        """Load the polling cursor or return a new timezone-aware state."""
        if not self.sync_state_path.exists():
            return WeixinSyncState(None, _utc_now(), None)
        data = self._read_object(self.sync_state_path, missing={})
        return WeixinSyncState(
            get_updates_buf=(
                str(data["get_updates_buf"])
                if data.get("get_updates_buf") is not None
                else None
            ),
            updated_at=_parse_datetime(data["updated_at"], "updated_at"),
            last_message_id=(
                str(data["last_message_id"])
                if data.get("last_message_id") is not None
                else None
            ),
        )

    def save_sync_state(self, state: WeixinSyncState) -> None:
        """Atomically commit a processed batch cursor."""
        self._atomic_write(
            self.sync_state_path,
            {
                "get_updates_buf": state.get_updates_buf,
                "updated_at": state.updated_at.isoformat(),
                "last_message_id": state.last_message_id,
            },
            private=True,
        )

    def _load_users(self) -> dict[str, Any]:
        return self._read_object(self.users_path, missing={})

    def save_user_context(
        self, user_id: str, context_token: str, message_id: str, received_at: datetime
    ) -> None:
        """Persist the most recent context token, strictly keyed by its user."""
        users = self._load_users()
        users[self.user_key(user_id)] = {
            "user_id": user_id,
            "context_token": context_token,
            "received_at": received_at.isoformat(),
            "last_message_id": message_id,
            "valid": True,
        }
        self._atomic_write(self.users_path, users, private=True)

    def invalidate_user_context(self, user_id: str) -> None:
        """Mark one user's token invalid without touching other users."""
        users = self._load_users()
        key = self.user_key(user_id)
        if key in users and isinstance(users[key], dict):
            users[key]["valid"] = False
            self._atomic_write(self.users_path, users, private=True)

    def is_processed(self, message_id: str) -> bool:
        """Return whether a stable message ID was durably completed."""
        if not self.processed_path.exists():
            return False
        try:
            with self.processed_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    if data.get("message_id") == message_id:
                        return True
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise WeixinStateError("Processed-message state is unreadable") from exc
        return False

    def mark_processed(self, message_id: str, command: str) -> None:
        """Append and fsync one completed message record."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "message_id": message_id,
            "command": command,
            "processed_at": _utc_now().isoformat(),
        }
        with self.processed_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(self.processed_path, 0o600)

    def record_send(self, message_id: str, command: str, success: bool) -> None:
        """Append a private send outcome without user or token data."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "message_id": message_id,
            "command": command,
            "success": success,
            "sent_at": _utc_now().isoformat(),
        }
        with self.send_history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(self.send_history_path, 0o600)
