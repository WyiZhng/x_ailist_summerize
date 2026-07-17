"""Async client for the official Weixin ilink Bot API.

Protocol details are based on the MIT-licensed weixin-clawbot-skill project:
https://github.com/liiiiwh/weixin-clawbot-skill
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import re
import secrets
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlparse

import httpx
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad

from .weixin_models import (
    WeixinCredentials,
    WeixinInboundMessage,
    WeixinSendResult,
    WeixinUpdateBatch,
)
from .weixin_state import WeixinStateStore


logger = logging.getLogger(__name__)

OFFICIAL_API_BASE_URL = "https://ilinkai.weixin.qq.com"
OFFICIAL_CDN_BASE_URL = "https://novac2c.cdn.weixin.qq.com/c2c"
ALLOWED_API_HOSTS = frozenset({"ilinkai.weixin.qq.com"})
ALLOWED_CDN_HOSTS = frozenset({"novac2c.cdn.weixin.qq.com"})
MAX_REPORT_BYTES = 20 * 1024 * 1024


class WeixinClientError(RuntimeError):
    """Base error for sanitized Weixin client failures."""


class WeixinAuthenticationError(WeixinClientError):
    """Raised when the Bot Token has expired or is rejected."""


class WeixinRateLimitError(WeixinClientError):
    """Raised when bounded retries cannot satisfy a rate limit."""


def _validate_base_url(value: str, allowed_hosts: frozenset[str]) -> str:
    parsed = urlparse(value)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in allowed_hosts
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise WeixinClientError("Untrusted Weixin service URL")
    return value.rstrip("/")


def _random_uin() -> str:
    value = int.from_bytes(os.urandom(4), "big")
    return base64.b64encode(str(value).encode("ascii")).decode("ascii")


def _message_id(payload: dict[str, Any]) -> str:
    for key in ("message_id", "client_id"):
        value = payload.get(key)
        if value not in (None, ""):
            return str(value)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _received_at(payload: dict[str, Any]) -> datetime:
    value = payload.get("create_time_ms")
    if isinstance(value, (int, float)) and value >= 0:
        return datetime.fromtimestamp(value / 1000, tz=timezone.utc)
    return datetime.now(timezone.utc)


def _extract_text(payload: dict[str, Any]) -> tuple[str, str | None]:
    for item in payload.get("item_list", []):
        if not isinstance(item, dict):
            continue
        if item.get("type") == 1:
            text_item = item.get("text_item")
            if isinstance(text_item, dict) and text_item.get("text") is not None:
                return "text", str(text_item["text"])
        if item.get("type") == 3:
            voice_item = item.get("voice_item")
            if isinstance(voice_item, dict) and voice_item.get("text"):
                return "voice_text", str(voice_item["text"])
    return "unsupported", None


class WeixinClient:
    """Bounded-retry Weixin client with report-only file sending."""

    def __init__(
        self,
        credentials: WeixinCredentials,
        *,
        state_store: WeixinStateStore,
        report_root: Path,
        request_timeout_seconds: int = 45,
        poll_timeout_seconds: int = 35,
        retry_attempts: int = 3,
        maximum_backoff_seconds: int = 120,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        if credentials.status != "active":
            raise WeixinAuthenticationError("Weixin credentials require re-login")
        self.credentials = credentials
        self.state_store = state_store
        self.report_root = report_root.resolve()
        self.base_url = _validate_base_url(credentials.base_url, ALLOWED_API_HOSTS)
        self.cdn_base_url = _validate_base_url(OFFICIAL_CDN_BASE_URL, ALLOWED_CDN_HOSTS)
        self.request_timeout_seconds = request_timeout_seconds
        self.poll_timeout_seconds = poll_timeout_seconds
        self.retry_attempts = max(1, retry_attempts)
        self.maximum_backoff_seconds = max(1, maximum_backoff_seconds)
        self._owns_client = http_client is None
        self.http = http_client or httpx.AsyncClient()

    async def close(self) -> None:
        """Close the internally owned HTTP client."""
        if self._owns_client:
            await self.http.aclose()

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "AuthorizationType": "ilink_bot_token",
            "Authorization": f"Bearer {self.credentials.token}",
            "X-WECHAT-UIN": _random_uin(),
        }

    async def _request(
        self,
        method: str,
        endpoint: str,
        *,
        json_body: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> httpx.Response:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        last_error: Exception | None = None
        for attempt in range(self.retry_attempts):
            try:
                response = await self.http.request(
                    method,
                    url,
                    headers=self._headers(),
                    json=json_body,
                    timeout=timeout or self.request_timeout_seconds,
                )
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                last_error = exc
                if attempt + 1 == self.retry_attempts:
                    break
                await asyncio.sleep(min(2**attempt, self.maximum_backoff_seconds))
                continue

            if response.status_code in {401, 403}:
                raise WeixinAuthenticationError("Weixin credentials require re-login")
            if response.status_code == 429:
                if attempt + 1 == self.retry_attempts:
                    raise WeixinRateLimitError("Weixin rate limit retry exhausted")
                try:
                    retry_after = float(response.headers.get("Retry-After", "1"))
                except ValueError:
                    retry_after = 1
                await asyncio.sleep(min(retry_after, self.maximum_backoff_seconds))
                continue
            if response.status_code >= 500:
                if attempt + 1 == self.retry_attempts:
                    raise WeixinClientError("Weixin service temporarily unavailable")
                await asyncio.sleep(min(2**attempt, self.maximum_backoff_seconds))
                continue
            if response.status_code >= 400:
                raise WeixinClientError(
                    f"Weixin request rejected ({response.status_code})"
                )
            return response
        raise WeixinClientError("Weixin network request failed") from last_error

    @staticmethod
    def _response_data(response: httpx.Response) -> dict[str, Any]:
        try:
            data = response.json()
        except ValueError as exc:
            raise WeixinClientError("Invalid response from Weixin service") from exc
        if not isinstance(data, dict):
            raise WeixinClientError("Invalid response from Weixin service")
        ret = data.get("ret", 0)
        errcode = data.get("errcode", 0)
        if ret == -14 or errcode == -14:
            raise WeixinAuthenticationError("Weixin credentials require re-login")
        if ret not in (None, 0) or errcode not in (None, 0):
            raise WeixinClientError("Weixin API reported an error")
        return data

    async def get_updates(self, get_updates_buf: str | None) -> WeixinUpdateBatch:
        """Long-poll and normalize inbound messages without logging payloads."""
        response = await self._request(
            "POST",
            "/ilink/bot/getupdates",
            json_body={"get_updates_buf": get_updates_buf or ""},
            timeout=self.poll_timeout_seconds + 5,
        )
        data = self._response_data(response)
        messages: list[WeixinInboundMessage] = []
        for payload in data.get("msgs", []):
            if not isinstance(payload, dict) or payload.get("message_type") == 2:
                continue
            user_id = payload.get("from_user_id")
            if not isinstance(user_id, str) or not user_id:
                continue
            message_type, text = _extract_text(payload)
            context_token = payload.get("context_token")
            messages.append(
                WeixinInboundMessage(
                    message_id=_message_id(payload),
                    user_id=user_id,
                    message_type=message_type,
                    text=text,
                    context_token=(
                        context_token if isinstance(context_token, str) else None
                    ),
                    received_at=_received_at(payload),
                    raw_payload=payload,
                )
            )
        next_buf = data.get("get_updates_buf")
        return WeixinUpdateBatch(
            messages=tuple(messages),
            get_updates_buf=next_buf if isinstance(next_buf, str) else get_updates_buf,
            longpolling_timeout_ms=(
                data["longpolling_timeout_ms"]
                if isinstance(data.get("longpolling_timeout_ms"), int)
                else None
            ),
        )

    async def _send_item(
        self, user_id: str, context_token: str, item: dict[str, Any]
    ) -> WeixinSendResult:
        if not context_token:
            return WeixinSendResult(False, error_code="missing_context_token")
        client_id = f"x-daily-{uuid.uuid4()}"
        try:
            response = await self._request(
                "POST",
                "/ilink/bot/sendmessage",
                json_body={
                    "msg": {
                        "from_user_id": "",
                        "to_user_id": user_id,
                        "client_id": client_id,
                        "message_type": 2,
                        "message_state": 2,
                        "context_token": context_token,
                        "item_list": [item],
                    }
                },
            )
            self._response_data(response)
            return WeixinSendResult(True, message_id=client_id)
        except WeixinAuthenticationError:
            raise
        except WeixinClientError as exc:
            return WeixinSendResult(
                False, error_code="send_failed", error_message=str(exc)
            )

    async def send_text(
        self, user_id: str, context_token: str, text: str
    ) -> WeixinSendResult:
        """Reply to the current private-chat message."""
        return await self._send_item(
            user_id, context_token, {"type": 1, "text_item": {"text": text}}
        )

    def _approved_report(self, file_path: Path) -> Path:
        try:
            if file_path.is_symlink():
                raise WeixinClientError("Report symlinks are not allowed")
            resolved = file_path.resolve(strict=True)
            resolved.relative_to(self.report_root)
        except (OSError, RuntimeError, ValueError) as exc:
            raise WeixinClientError(
                "Report file is outside the approved directory"
            ) from exc
        if resolved.suffix.lower() != ".html" or not resolved.is_file():
            raise WeixinClientError("Only HTML reports can be sent")
        if resolved.stat().st_size > MAX_REPORT_BYTES:
            raise WeixinClientError("Report file is too large")
        return resolved

    async def _upload_ciphertext(self, url: str, ciphertext: bytes) -> httpx.Response:
        """Upload encrypted bytes with the same bounded retry policy."""
        last_error: Exception | None = None
        for attempt in range(self.retry_attempts):
            try:
                response = await self.http.post(
                    url,
                    content=ciphertext,
                    headers={"Content-Type": "application/octet-stream"},
                    timeout=self.request_timeout_seconds,
                )
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                last_error = exc
                if attempt + 1 == self.retry_attempts:
                    break
                await asyncio.sleep(min(2**attempt, self.maximum_backoff_seconds))
                continue
            if response.status_code == 429 or response.status_code >= 500:
                if attempt + 1 == self.retry_attempts:
                    break
                try:
                    retry_after = float(response.headers.get("Retry-After", "1"))
                except ValueError:
                    retry_after = 1
                await asyncio.sleep(min(retry_after, self.maximum_backoff_seconds))
                continue
            if response.status_code >= 400:
                raise WeixinClientError("Weixin report upload was rejected")
            return response
        raise WeixinClientError("Weixin report upload failed") from last_error

    def _resolve_upload_url(self, upload_data: dict[str, Any], *, file_key: str) -> str:
        """Accept current and legacy upload responses without allowing SSRF."""
        full_url = upload_data.get("upload_full_url")
        if isinstance(full_url, str) and full_url:
            parsed = urlparse(full_url)
            if (
                parsed.scheme != "https"
                or parsed.hostname not in ALLOWED_CDN_HOSTS
                or parsed.username is not None
                or parsed.password is not None
                or parsed.path != "/c2c/upload"
                or parsed.fragment
            ):
                raise WeixinClientError("Untrusted Weixin upload URL")
            return full_url
        upload_param = upload_data.get("upload_param")
        if not isinstance(upload_param, str) or not upload_param:
            raise WeixinClientError("Weixin upload authorization is missing")
        return f"{self.cdn_base_url}/upload?{urlencode({'encrypted_query_param': upload_param, 'filekey': file_key})}"

    async def send_file(
        self,
        user_id: str,
        context_token: str,
        file_path: Path,
        display_name: str,
    ) -> WeixinSendResult:
        """Encrypt, upload, and send one approved HTML report."""
        report = self._approved_report(file_path)
        safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", Path(display_name).name)
        if not safe_name.lower().endswith(".html"):
            safe_name += ".html"
        plaintext = report.read_bytes()
        aes_key = secrets.token_bytes(16)
        aes_key_hex = aes_key.hex()
        ciphertext = AES.new(aes_key, AES.MODE_ECB).encrypt(pad(plaintext, 16))
        file_key = secrets.token_hex(16)
        temporary_path: Path | None = None
        self.state_store.ensure_directories()
        try:
            temporary_path = self.state_store.temporary_dir / f"{file_key}.encrypted"
            temporary_path.write_bytes(ciphertext)
            os.chmod(temporary_path, 0o600)
            upload_response = await self._request(
                "POST",
                "/ilink/bot/getuploadurl",
                json_body={
                    "filekey": file_key,
                    "media_type": 3,
                    "to_user_id": user_id,
                    "rawsize": len(plaintext),
                    "rawfilemd5": hashlib.md5(
                        plaintext, usedforsecurity=False
                    ).hexdigest(),
                    "filesize": len(ciphertext),
                    "no_need_thumb": True,
                    "aeskey": aes_key_hex,
                },
            )
            upload_data = self._response_data(upload_response)
            upload_url = self._resolve_upload_url(upload_data, file_key=file_key)
            upload = await self._upload_ciphertext(upload_url, ciphertext)
            if upload.status_code != 200:
                raise WeixinClientError("Weixin report upload failed")
            encrypted_param = upload.headers.get("x-encrypted-param")
            if not encrypted_param:
                raise WeixinClientError("Weixin report upload response is incomplete")
            media = {
                "encrypt_query_param": encrypted_param,
                "aes_key": base64.b64encode(aes_key_hex.encode("ascii")).decode(
                    "ascii"
                ),
                "encrypt_type": 1,
            }
            return await self._send_item(
                user_id,
                context_token,
                {
                    "type": 4,
                    "file_item": {
                        "media": media,
                        "file_name": safe_name,
                        "len": str(len(plaintext)),
                    },
                },
            )
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    async def send_typing(self, user_id: str, context_token: str) -> WeixinSendResult:
        """Best-effort typing indicator bound to the current conversation."""
        if not context_token:
            return WeixinSendResult(False, error_code="missing_context_token")
        try:
            response = await self._request(
                "POST",
                "/ilink/bot/getconfig",
                json_body={"ilink_user_id": user_id, "context_token": context_token},
            )
            data = self._response_data(response)
            ticket = data.get("typing_ticket")
            if not isinstance(ticket, str) or not ticket:
                return WeixinSendResult(False, error_code="typing_unavailable")
            response = await self._request(
                "POST",
                "/ilink/bot/sendtyping",
                json_body={
                    "ilink_user_id": user_id,
                    "typing_ticket": ticket,
                    "status": 1,
                },
            )
            self._response_data(response)
            return WeixinSendResult(True)
        except WeixinClientError as exc:
            return WeixinSendResult(
                False, error_code="typing_failed", error_message=str(exc)
            )
