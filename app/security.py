"""Central, dependency-free helpers for safely rendering untrusted content.

The helpers in this module deliberately keep URL validation separate from
HTML attribute escaping.  Callers rendering a URL in HTML should normally use
``safe_url_attribute`` or ``safe_image_src_attribute`` rather than composing
those two operations themselves.
"""

from __future__ import annotations

import html
import re
from typing import Any
from urllib.parse import urlsplit, urlunsplit


_ALLOWED_EXTERNAL_SCHEMES = frozenset({"http", "https"})
_ATTRIBUTE_CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f]")
_SAFE_DATA_IMAGE = re.compile(
    r"\Adata:image/(?:png|jpeg|gif|webp);base64,[A-Za-z0-9+/]+={0,2}\Z",
    re.IGNORECASE,
)
_BEARER_CREDENTIAL = re.compile(
    r"(?i)(\bBearer[ \t]+)[A-Za-z0-9._~+/=-]{8,}"
)
_SK_STYLE_CREDENTIAL = re.compile(
    r"(?<![A-Za-z0-9])sk-[A-Za-z0-9][A-Za-z0-9._-]{7,}",
    re.IGNORECASE,
)
_REDACTED = "[REDACTED]"


def _as_text(value: Any) -> str:
    """Convert a possibly absent value to text without producing ``None``."""

    return "" if value is None else str(value)


def escape_html_text(value: Any) -> str:
    """Escape untrusted text for an HTML text node while preserving newlines.

    Line endings are normalized and converted to a known-safe ``<br>`` tag so
    multi-line tweet text remains readable in normal HTML flow.  All markup
    supplied by the input is escaped before those tags are added.
    """

    text = _as_text(value).replace("\r\n", "\n").replace("\r", "\n")
    return html.escape(text, quote=True).replace("\n", "<br>\n")


def safe_html_attribute(value: Any) -> str:
    """Escape a value for use inside a *quoted* HTML attribute.

    ASCII control characters are replaced with spaces before escaping.  The
    caller must still surround the returned value with double or single
    quotes; unquoted HTML attributes are not safe for arbitrary input.
    """

    text = _ATTRIBUTE_CONTROL_CHARACTERS.sub(" ", _as_text(value))
    return html.escape(text, quote=True)


def sanitize_external_url(value: Any, fallback: str = "") -> str:
    """Return an absolute HTTP(S) URL, or ``fallback`` when it is unsafe.

    Protocol-relative URLs, credentials in the authority component, browser
    executable schemes (for example ``javascript:`` and ``data:``), raw HTML
    delimiters, backslashes, and embedded whitespace/control characters are
    rejected.  The URL scheme is normalized to lowercase.

    ``fallback`` is trusted application text and should normally be ``""`` or
    ``"#"``.  Escape it with :func:`safe_html_attribute` before rendering.
    """

    candidate = _as_text(value).strip()
    if not candidate:
        return fallback

    if any(character.isspace() or ord(character) < 0x20 for character in candidate):
        return fallback
    if any(character in candidate for character in ('\\', '"', "<", ">", "`")):
        return fallback

    try:
        parsed = urlsplit(candidate)
        scheme = parsed.scheme.lower()
        if scheme not in _ALLOWED_EXTERNAL_SCHEMES or not parsed.netloc or not parsed.hostname:
            return fallback
        if parsed.username is not None or parsed.password is not None:
            return fallback
        # Accessing ``port`` validates malformed/non-numeric and out-of-range ports.
        parsed.port
    except (TypeError, ValueError):
        return fallback

    return urlunsplit((scheme, parsed.netloc, parsed.path, parsed.query, parsed.fragment))


def sanitize_image_src(
    value: Any,
    fallback: str = "",
    *,
    allow_data_image: bool = False,
) -> str:
    """Validate an image source.

    HTTP(S) is always accepted.  A tightly constrained base64 image data URI
    can be enabled explicitly for embedded report assets; SVG and arbitrary
    data URIs remain rejected.
    """

    candidate = _as_text(value).strip()
    if allow_data_image and _SAFE_DATA_IMAGE.fullmatch(candidate):
        return candidate
    return sanitize_external_url(candidate, fallback=fallback)


def safe_url_attribute(value: Any, fallback: str = "") -> str:
    """Validate an external URL and escape it for a quoted HTML attribute."""

    return safe_html_attribute(sanitize_external_url(value, fallback=fallback))


def safe_image_src_attribute(
    value: Any,
    fallback: str = "",
    *,
    allow_data_image: bool = False,
) -> str:
    """Validate an image source and escape it for a quoted HTML attribute."""

    return safe_html_attribute(
        sanitize_image_src(value, fallback=fallback, allow_data_image=allow_data_image)
    )


def mask_secret(
    value: Any,
    *,
    visible_prefix: int = 3,
    visible_suffix: int = 2,
    mask_character: str = "*",
) -> str:
    """Mask a secret while retaining a small identifier for user interfaces.

    Short values are masked completely.  Longer values retain three leading
    and two trailing characters by default, which is enough to distinguish
    configured keys without exposing a useful credential fragment.
    """

    secret = _as_text(value)
    if not secret:
        return ""
    if visible_prefix < 0 or visible_suffix < 0:
        raise ValueError("visible lengths must be non-negative")
    if len(mask_character) != 1:
        raise ValueError("mask_character must be exactly one character")

    visible_count = visible_prefix + visible_suffix
    if len(secret) <= visible_count + 2:
        return mask_character * len(secret)

    suffix = secret[-visible_suffix:] if visible_suffix else ""
    return (
        secret[:visible_prefix]
        + mask_character * (len(secret) - visible_count)
        + suffix
    )


def redact_sensitive_text(value: Any, secrets: Any = ()) -> str:
    """Remove credentials from text intended for logs or user-facing errors.

    Every non-empty explicitly supplied secret is replaced first.  The result
    is then conservatively scanned for common ``Bearer <token>`` and long
    ``sk-...`` credentials.  Short prose such as ``"sk-model"`` is left alone
    to reduce false positives.
    """

    text = _as_text(value)
    if secrets is None:
        supplied_secrets = ()
    elif isinstance(secrets, (str, bytes)):
        supplied_secrets = (secrets,)
    else:
        supplied_secrets = secrets

    normalized_secrets = {
        secret_text
        for secret in supplied_secrets
        if (secret_text := _as_text(secret))
    }
    for secret in sorted(normalized_secrets, key=len, reverse=True):
        text = text.replace(secret, _REDACTED)

    text = _BEARER_CREDENTIAL.sub(r"\1[REDACTED]", text)
    return _SK_STYLE_CREDENTIAL.sub(_REDACTED, text)


__all__ = [
    "escape_html_text",
    "mask_secret",
    "redact_sensitive_text",
    "safe_html_attribute",
    "safe_image_src_attribute",
    "safe_url_attribute",
    "sanitize_external_url",
    "sanitize_image_src",
]
