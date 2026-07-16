"""Network target validation for server-side article requests."""

from __future__ import annotations

import asyncio
import inspect
import ipaddress
import socket
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from .article_models import stable_sha256
from .url_normalizer import UrlNormalizationError, UrlNormalizer


Resolver = Callable[[str, int], Iterable[Any] | Awaitable[Iterable[Any]]]

_BLOCKED_HOSTNAMES = frozenset(
    {
        "localhost",
        "localhost.localdomain",
        "metadata.google.internal",
        "metadata",
    }
)


class NetworkSecurityError(RuntimeError):
    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class ValidatedTarget:
    url: str
    hostname: str
    port: int
    addresses: tuple[str, ...]


async def _system_resolver(hostname: str, port: int) -> Iterable[Any]:
    return await asyncio.to_thread(
        socket.getaddrinfo,
        hostname,
        port,
        type=socket.SOCK_STREAM,
    )


def _address_strings(values: Iterable[Any]) -> tuple[str, ...]:
    addresses: list[str] = []
    for value in values:
        candidate: Any = value
        if isinstance(value, tuple) and len(value) >= 5:
            sockaddr = value[4]
            if isinstance(sockaddr, tuple) and sockaddr:
                candidate = sockaddr[0]
        if isinstance(candidate, (ipaddress.IPv4Address, ipaddress.IPv6Address)):
            text = str(candidate)
        elif isinstance(candidate, str):
            text = candidate
        else:
            continue
        try:
            normalized = str(ipaddress.ip_address(text))
        except ValueError:
            continue
        if normalized not in addresses:
            addresses.append(normalized)
    return tuple(addresses)


def _is_public_address(value: str) -> bool:
    address = ipaddress.ip_address(value)
    return bool(address.is_global)


class NetworkSecurityValidator:
    """Reject non-public targets before requests and after every redirect."""

    def __init__(self, resolver: Resolver | None = None) -> None:
        self._resolver = resolver or _system_resolver
        self._normalizer = UrlNormalizer()

    async def validate_url(self, url: str) -> ValidatedTarget:
        try:
            normalized = self._normalizer.normalize(url).normalized_url
        except UrlNormalizationError as exc:
            raise NetworkSecurityError(str(exc), code=exc.code) from exc
        parsed = urlsplit(normalized)
        hostname = (parsed.hostname or "").rstrip(".").lower()
        if (
            hostname in _BLOCKED_HOSTNAMES
            or hostname.endswith(".local")
            or hostname.endswith(".internal")
            or ("." not in hostname and ":" not in hostname and not hostname.isdigit())
        ):
            raise NetworkSecurityError(
                "Target hostname is not public", code="blocked_private_address"
            )
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        try:
            literal = ipaddress.ip_address(hostname)
        except ValueError:
            literal = None
        if literal is not None:
            addresses = (str(literal),)
        else:
            try:
                result = self._resolver(hostname, port)
                if inspect.isawaitable(result):
                    result = await result
                addresses = _address_strings(result)
            except (OSError, socket.gaierror, asyncio.TimeoutError) as exc:
                raise NetworkSecurityError(
                    "DNS resolution failed", code="dns_failure"
                ) from exc
            if not addresses:
                raise NetworkSecurityError(
                    "DNS resolution returned no addresses", code="dns_failure"
                )
        if any(not _is_public_address(address) for address in addresses):
            raise NetworkSecurityError(
                "Target resolves to a non-public address",
                code="blocked_private_address",
            )
        return ValidatedTarget(
            url=normalized,
            hostname=hostname,
            port=port,
            addresses=addresses,
        )


def safe_url_log_fields(url: str) -> dict[str, str]:
    """Return URL observability fields without query values or credentials."""
    try:
        parsed = urlsplit(url)
        hostname = (parsed.hostname or "invalid").lower()
        path_fingerprint = stable_sha256(parsed.path or "/")[:12]
    except ValueError:
        hostname = "invalid"
        path_fingerprint = "invalid"
    return {"domain": hostname, "path_hash": path_fingerprint}


__all__ = [
    "NetworkSecurityError",
    "NetworkSecurityValidator",
    "Resolver",
    "ValidatedTarget",
    "safe_url_log_fields",
]
