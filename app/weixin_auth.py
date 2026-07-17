"""Interactive QR-code authentication for the Weixin ClawBot service."""

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

import httpx
import qrcode

from .config import load_config
from .weixin_client import (
    ALLOWED_API_HOSTS,
    OFFICIAL_API_BASE_URL,
    WeixinClientError,
    _validate_base_url,
)
from .weixin_models import WeixinCredentials, WeixinQRSession
from .weixin_state import WeixinStateStore


logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class WeixinAuthError(RuntimeError):
    """Raised when QR login cannot complete safely."""


class WeixinAuthClient:
    """Small client dedicated to the unauthenticated QR login endpoints."""

    def __init__(
        self,
        *,
        http_client: httpx.AsyncClient | None = None,
        base_url: str = OFFICIAL_API_BASE_URL,
        request_timeout_seconds: int = 45,
    ) -> None:
        self.base_url = _validate_base_url(base_url, ALLOWED_API_HOSTS)
        self.request_timeout_seconds = request_timeout_seconds
        self._owns_client = http_client is None
        self.http = http_client or httpx.AsyncClient()

    async def close(self) -> None:
        """Close an internally owned client."""
        if self._owns_client:
            await self.http.aclose()

    async def fetch_qr_code(self) -> WeixinQRSession:
        """Request a new official QR login session."""
        try:
            response = await self.http.get(
                f"{self.base_url}/ilink/bot/get_bot_qrcode?bot_type=3",
                timeout=self.request_timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise WeixinAuthError("Unable to obtain Weixin login QR code") from exc
        if not isinstance(data, dict):
            raise WeixinAuthError("Invalid Weixin QR response")
        session_id = data.get("qrcode")
        content = data.get("qrcode_img_content")
        if not isinstance(session_id, str) or not isinstance(content, str):
            raise WeixinAuthError("Incomplete Weixin QR response")
        return WeixinQRSession(session_id=session_id, content=content)

    async def poll_status(self, session_id: str) -> dict[str, Any]:
        """Poll one QR session status with a bounded request timeout."""
        try:
            response = await self.http.get(
                f"{self.base_url}/ilink/bot/get_qrcode_status?qrcode={quote(session_id)}",
                headers={"iLink-App-ClientVersion": "1"},
                timeout=self.request_timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
        except httpx.TimeoutException:
            return {"status": "wait"}
        except (httpx.HTTPError, ValueError) as exc:
            raise WeixinAuthError("Unable to check Weixin QR status") from exc
        if not isinstance(data, dict):
            raise WeixinAuthError("Invalid Weixin QR status response")
        return data


async def run_login(
    store: WeixinStateStore,
    *,
    auth_client: WeixinAuthClient,
    timeout_seconds: int = 480,
    poll_interval_seconds: float = 1,
    on_qr_ready: Callable[[Path], None] | None = None,
) -> bool:
    """Run one bounded login flow and securely persist confirmed credentials."""
    store.ensure_directories()
    qr_path: Path | None = None
    try:
        session = await auth_client.fetch_qr_code()
        qr_path = store.temporary_dir / "weixin-login.png"
        qrcode.make(session.content).save(qr_path)
        qr_path.chmod(0o600)
        if on_qr_ready is not None:
            on_qr_ready(qr_path)

        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_seconds
        while loop.time() < deadline:
            status = await auth_client.poll_status(session.session_id)
            state = status.get("status")
            if state in {"wait", "scaned"}:
                await asyncio.sleep(poll_interval_seconds)
                continue
            if state == "expired":
                raise WeixinAuthError("Weixin login QR code expired")
            if state != "confirmed":
                raise WeixinAuthError("Unexpected Weixin login status")
            token = status.get("bot_token")
            account_id = status.get("ilink_bot_id")
            returned_base = status.get("baseurl") or OFFICIAL_API_BASE_URL
            if not isinstance(token, str) or not token:
                raise WeixinAuthError("Weixin login did not return credentials")
            if not isinstance(account_id, str) or not account_id:
                raise WeixinAuthError("Weixin login did not return a Bot ID")
            base_url = _validate_base_url(str(returned_base), ALLOWED_API_HOSTS)
            now = datetime.now(timezone.utc)
            previous = store.load_credentials()
            store.save_credentials(
                WeixinCredentials(
                    token=token,
                    account_id=account_id,
                    base_url=base_url,
                    created_at=previous.created_at if previous else now,
                    updated_at=now,
                    expires_at=None,
                    status="active",
                )
            )
            return True
        raise WeixinAuthError("Weixin login timed out")
    finally:
        if qr_path is not None:
            qr_path.unlink(missing_ok=True)


def _configured_store() -> tuple[WeixinStateStore, dict[str, Any]]:
    config = load_config(PROJECT_ROOT / "config.json")
    storage_dir = Path(config["storage"]["data_dir"]).expanduser()
    if not storage_dir.is_absolute():
        storage_dir = PROJECT_ROOT / storage_dir
    weixin_dir = Path(config["weixin"]["data_dir"])
    if not weixin_dir.is_absolute():
        weixin_dir = storage_dir.parent / weixin_dir
    return WeixinStateStore(weixin_dir), config["weixin"]


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for user-confirmed QR login."""
    parser = argparse.ArgumentParser(description="Configure Weixin ClawBot login")
    parser.parse_args(argv)
    store, config = _configured_store()

    def show_qr(path: Path) -> None:
        print(f"Scan the local QR image with Weixin: {path}")

    async def authenticate() -> bool:
        client = WeixinAuthClient(
            request_timeout_seconds=config["request_timeout_seconds"]
        )
        try:
            return await run_login(store, auth_client=client, on_qr_ready=show_qr)
        finally:
            await client.close()

    try:
        success = asyncio.run(authenticate())
    except KeyboardInterrupt:
        print("Weixin login cancelled.")
        return 130
    except (WeixinAuthError, WeixinClientError) as exc:
        logger.error("Weixin login failed (%s)", type(exc).__name__)
        print(str(exc))
        return 1
    print(f"Weixin credentials configured: {str(success).lower()}")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
