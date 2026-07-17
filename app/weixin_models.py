"""Typed models for the Weixin ClawBot integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class WeixinCredentials:
    """Private credentials produced by QR-code login."""

    token: str
    account_id: str
    base_url: str
    created_at: datetime
    updated_at: datetime
    expires_at: datetime | None = None
    status: str = "active"


@dataclass(frozen=True)
class WeixinSyncState:
    """Durable cursor for the long-poll stream."""

    get_updates_buf: str | None
    updated_at: datetime
    last_message_id: str | None = None


@dataclass(frozen=True)
class WeixinInboundMessage:
    """Normalized inbound private-chat message."""

    message_id: str
    user_id: str
    message_type: str
    text: str | None
    context_token: str | None
    received_at: datetime
    raw_payload: dict[str, Any] = field(repr=False, compare=False)


@dataclass(frozen=True)
class WeixinUpdateBatch:
    """One long-poll response and the cursor to commit after processing."""

    messages: tuple[WeixinInboundMessage, ...]
    get_updates_buf: str | None
    longpolling_timeout_ms: int | None = None


@dataclass(frozen=True)
class WeixinSendResult:
    """Sanitized result of a send operation."""

    success: bool
    message_id: str | None = None
    error_code: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class WeixinQRSession:
    """QR login session returned by the official API."""

    session_id: str
    content: str


@dataclass(frozen=True)
class WeixinCommandResult:
    """Command output containing text and an optional approved report file."""

    command: str
    text: str
    file_path: Path | None = None
    display_name: str | None = None
