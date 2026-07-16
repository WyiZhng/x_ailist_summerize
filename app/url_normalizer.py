"""Conservative URL normalization for external article identity."""

from __future__ import annotations

import ipaddress
import re
from urllib.parse import unquote_plus, urlsplit, urlunsplit

from .article_models import NormalizedUrl, stable_sha256


TRACKING_PARAMETERS = frozenset(
    {
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_term",
        "utm_content",
        "utm_id",
        "gclid",
        "fbclid",
        "yclid",
        "mc_cid",
        "mc_eid",
        "ref_src",
        "ref_url",
    }
)

SIGNED_QUERY_MARKERS = frozenset(
    {
        "signature",
        "sig",
        "token",
        "access_token",
        "auth",
        "key",
        "x-amz-signature",
        "x-amz-credential",
        "x-goog-signature",
        "expires",
    }
)

_BAD_PERCENT = re.compile(r"%(?![0-9A-Fa-f]{2})")
_CONTROL = re.compile(r"[\x00-\x20\x7f]")


class UrlNormalizationError(ValueError):
    def __init__(self, message: str, *, code: str = "invalid_url") -> None:
        super().__init__(message)
        self.code = code


def _query_key(raw_part: str) -> str:
    raw_key = raw_part.partition("=")[0]
    try:
        return unquote_plus(raw_key).strip().lower()
    except (UnicodeDecodeError, ValueError):
        return raw_key.strip().lower()


def _obviously_blocked_host(hostname: str) -> bool:
    host = hostname.rstrip(".").lower()
    if host == "localhost" or host.endswith(".localhost") or host.endswith(".local"):
        return True
    if host in {"metadata.google.internal", "0.0.0.0"}:
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return not address.is_global


class UrlNormalizer:
    """Normalize identity fields without rewriting content-bearing parameters."""

    def __init__(self, tracking_parameters: frozenset[str] = TRACKING_PARAMETERS) -> None:
        self.tracking_parameters = frozenset(item.lower() for item in tracking_parameters)

    @staticmethod
    def _hostname(parsed) -> str:
        hostname = parsed.hostname
        if not hostname:
            raise UrlNormalizationError("URL must include a hostname")
        try:
            ascii_host = hostname.rstrip(".").encode("idna").decode("ascii").lower()
        except UnicodeError as exc:
            raise UrlNormalizationError("URL hostname is not valid IDNA") from exc
        if not ascii_host or ".." in ascii_host:
            raise UrlNormalizationError("URL hostname is malformed")
        if _obviously_blocked_host(ascii_host):
            raise UrlNormalizationError(
                "URL points to a non-public address", code="blocked_private_address"
            )
        return ascii_host

    @staticmethod
    def _related_hosts(candidate: str, base: str) -> bool:
        candidate = candidate.rstrip(".").lower()
        base = base.rstrip(".").lower()
        return (
            candidate == base
            or candidate.endswith("." + base)
            or base.endswith("." + candidate)
        )

    def _normalize_candidate(self, value: str) -> tuple[str, str, list[str]]:
        if not isinstance(value, str) or not value.strip():
            raise UrlNormalizationError("URL must be a non-empty string")
        raw = value.strip()
        if _CONTROL.search(raw) or "\\" in raw:
            raise UrlNormalizationError("URL contains whitespace, control, or backslash characters")
        if _BAD_PERCENT.search(raw):
            raise UrlNormalizationError("URL contains malformed percent encoding")
        try:
            parsed = urlsplit(raw)
            port = parsed.port
        except ValueError as exc:
            raise UrlNormalizationError("URL could not be parsed safely") from exc
        scheme = parsed.scheme.lower()
        if scheme not in {"http", "https"}:
            raise UrlNormalizationError("Only absolute HTTP and HTTPS URLs are supported")
        if parsed.username is not None or parsed.password is not None:
            raise UrlNormalizationError("Credentials embedded in URLs are not allowed")
        hostname = self._hostname(parsed)
        if ":" in hostname:
            host_part = f"[{hostname}]"
        else:
            host_part = hostname
        if port is not None and not (
            (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
        ):
            host_part = f"{host_part}:{port}"

        raw_parts = parsed.query.split("&") if parsed.query else []
        query_keys = {_query_key(part) for part in raw_parts}
        signed = bool(query_keys.intersection(SIGNED_QUERY_MARKERS))
        removed: list[str] = []
        kept: list[str] = []
        for part in raw_parts:
            key = _query_key(part)
            if not signed and key in self.tracking_parameters:
                removed.append(key)
            else:
                kept.append(part)
        query = "&".join(kept)
        path = parsed.path or "/"
        normalized = urlunsplit((scheme, host_part, path, query, ""))
        return normalized, hostname, list(dict.fromkeys(removed))

    def normalize(
        self,
        original_url: str,
        resolved_url: str | None = None,
        canonical_url: str | None = None,
    ) -> NormalizedUrl:
        original_normalized, original_host, original_removed = self._normalize_candidate(
            original_url
        )
        selected = original_normalized
        selected_host = original_host
        removed = original_removed

        resolved_normalized: str | None = None
        resolved_host: str | None = None
        if resolved_url:
            try:
                resolved_normalized, resolved_host, resolved_removed = self._normalize_candidate(
                    resolved_url
                )
            except UrlNormalizationError:
                resolved_normalized = None
            else:
                selected = resolved_normalized
                selected_host = resolved_host
                removed = resolved_removed

        if canonical_url:
            try:
                canonical_normalized, canonical_host, canonical_removed = (
                    self._normalize_candidate(canonical_url)
                )
            except UrlNormalizationError:
                canonical_normalized = None
            else:
                base_host = resolved_host or original_host
                if self._related_hosts(canonical_host, base_host):
                    selected = canonical_normalized
                    selected_host = canonical_host
                    removed = canonical_removed

        return NormalizedUrl(
            original_url=original_url,
            resolved_url=resolved_url,
            canonical_url=canonical_url,
            normalized_url=selected,
            domain=selected_host,
            url_hash=stable_sha256(selected),
            removed_tracking_params=removed,
        )


__all__ = [
    "SIGNED_QUERY_MARKERS",
    "TRACKING_PARAMETERS",
    "UrlNormalizationError",
    "UrlNormalizer",
]
