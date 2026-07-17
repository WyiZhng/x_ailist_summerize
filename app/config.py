"""Configuration loading, validation, redaction, and atomic persistence.

This module is the single source of truth for application defaults.  It keeps
older, partial ``config.json`` files working by deeply merging them with the
current schema, while ensuring secrets never have to be returned by a public
configuration endpoint.
"""

from __future__ import annotations

import copy
import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import parse_qsl, urlsplit

try:
    from .security import mask_secret
except ImportError:  # Compatibility with `python app/web_ui.py`.
    from security import mask_secret


CONFIG_PATH = Path("config.json")

# Keep all defaults here.  Callers should use ``normalize_config`` rather than
# maintaining a second set of fallbacks in UI, installation, or service code.
DEFAULT_CONFIG: dict[str, Any] = {
    "summarization": {
        "provider": "ollama",
        "options": {
            "ollama": {
                "model": "qwen2.5:7b",
                "endpoint": "http://localhost:11434",
            },
            "lmstudio": {
                "model": "local-model",
                "endpoint": "http://localhost:1234/v1",
            },
            "groq": {
                "model": "llama-3.3-70b-versatile",
                "endpoint": "https://api.groq.com/openai/v1",
                "api_key": "",
            },
            "claude": {
                "model": "claude-3-5-sonnet-20240620",
                "api_key": "",
            },
            "openai": {
                "model": "gpt-4o",
                "api_key": "",
            },
            "gemini": {
                "model": "gemini-1.5-flash",
                "api_key": "",
            },
            "deepseek": {
                "model": "deepseek-chat",
                "api_key": "",
            },
            "openrouter": {
                "model": "google/gemini-2.0-flash-001",
                "api_key": "",
            },
            "grok": {
                "model": "grok-3",
                "api_key": "",
            },
        },
    },
    "twitter": {
        "list_urls": [],
        "max_tweets": 100,
        "list_owner": None,
        "max_scrolls": 5,
        "headless_after_auth": True,
        "fetch_method": "twikit",
        "api_bearer_token": "",
        "incremental_sync": True,
        "initial_fetch_limit": 100,
        "page_size": 100,
        "max_pages": 20,
    },
    "storage": {
        "data_dir": "data",
    },
    "articles": {
        "enabled": True,
        "timeout_seconds": 15,
        "max_redirects": 5,
        "max_response_bytes": 5_242_880,
        "max_article_chars": 50_000,
        "cache_ttl_hours": 168,
        "failure_retry_hours": 6,
        "max_articles_per_run": 20,
        "retry_attempts": 2,
        "user_agent": "x-ai-daily/1.0",
    },
    "weixin": {
        "enabled": False,
        "data_dir": "data/weixin",
        "poll_timeout_seconds": 35,
        "request_timeout_seconds": 45,
        "retry_attempts": 3,
        "maximum_backoff_seconds": 120,
        "allowed_report_extensions": [".html"],
    },
}

SUPPORTED_PROVIDERS = tuple(DEFAULT_CONFIG["summarization"]["options"])
SECRET_KEYS = frozenset({"api_key", "api_bearer_token"})


def _looks_sensitive_key(key: Any) -> bool:
    """Recognize credential-like extension keys without hiding normal fields."""
    normalized = str(key).strip().lower().replace("-", "_")
    return bool(
        normalized in SECRET_KEYS
        or normalized
        in {
            "authorization",
            "auth_token",
            "client_secret",
            "cookie",
            "cookies",
            "ct0",
            "password",
            "refresh_token",
            "secret",
            "token",
        }
        or normalized.endswith("_api_key")
        or normalized.endswith("_password")
        or normalized.endswith("_secret")
        or normalized.endswith("_token")
    )


def _endpoint_contains_credentials(value: Any) -> bool:
    """Return whether a configured endpoint embeds credentials in its URL."""
    if not isinstance(value, str) or not value:
        return False
    try:
        parsed = urlsplit(value)
        if parsed.username is not None or parsed.password is not None:
            return True
        for query_key, _ in parse_qsl(parsed.query, keep_blank_values=True):
            normalized = query_key.strip().lower().replace("-", "_")
            if _looks_sensitive_key(normalized) or normalized in {
                "auth",
                "credential",
                "credentials",
                "key",
            }:
                return True
        fragment = parsed.fragment.lower()
        return any(marker in fragment for marker in ("api_key=", "password=", "token="))
    except ValueError:
        return True


def _valid_scalar(default: Any, value: Any) -> bool:
    """Return whether *value* has the basic type represented by *default*."""
    if default is None:
        # The only nullable field in the current schema is list_owner.
        return value is None or isinstance(value, str)
    if isinstance(default, bool):
        return isinstance(value, bool)
    if isinstance(default, int):
        return isinstance(value, int) and not isinstance(value, bool)
    if isinstance(default, float):
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return isinstance(value, type(default))


def _merge_with_defaults(defaults: Any, supplied: Any) -> Any:
    """Deeply merge and type-check known fields while preserving extensions."""
    if isinstance(defaults, dict):
        if not isinstance(supplied, Mapping):
            return copy.deepcopy(defaults)

        result: dict[str, Any] = {}
        for key, default_value in defaults.items():
            if key in supplied:
                result[key] = _merge_with_defaults(default_value, supplied[key])
            else:
                result[key] = copy.deepcopy(default_value)

        # Preserve unknown fields for forwards compatibility.  Known fields
        # above remain validated against DEFAULT_CONFIG.
        for key, value in supplied.items():
            if key not in defaults:
                result[key] = copy.deepcopy(value)
        return result

    if isinstance(defaults, list):
        return (
            copy.deepcopy(supplied)
            if isinstance(supplied, list)
            else copy.deepcopy(defaults)
        )

    return (
        copy.deepcopy(supplied)
        if _valid_scalar(defaults, supplied)
        else copy.deepcopy(defaults)
    )


def normalize_config(config: Any) -> dict[str, Any]:
    """Return a complete, independent configuration with basic validation.

    Missing fields are filled from :data:`DEFAULT_CONFIG`.  Invalid known
    field types fall back to their defaults; unknown fields are retained so a
    newer config is not destroyed by an older application build.
    """
    if not isinstance(config, Mapping):
        return copy.deepcopy(DEFAULT_CONFIG)

    normalized = _merge_with_defaults(DEFAULT_CONFIG, config)

    summarization = normalized["summarization"]
    if summarization["provider"] not in SUPPORTED_PROVIDERS:
        summarization["provider"] = DEFAULT_CONFIG["summarization"]["provider"]

    twitter = normalized["twitter"]
    urls = twitter["list_urls"]
    if not all(isinstance(item, str) for item in urls):
        twitter["list_urls"] = copy.deepcopy(DEFAULT_CONFIG["twitter"]["list_urls"])
    if twitter["max_tweets"] <= 0:
        twitter["max_tweets"] = DEFAULT_CONFIG["twitter"]["max_tweets"]
    if twitter["max_scrolls"] < 0:
        twitter["max_scrolls"] = DEFAULT_CONFIG["twitter"]["max_scrolls"]
    for positive_field in ("initial_fetch_limit", "page_size", "max_pages"):
        if twitter[positive_field] <= 0:
            twitter[positive_field] = DEFAULT_CONFIG["twitter"][positive_field]
    twitter["page_size"] = min(100, twitter["page_size"])
    if twitter["fetch_method"] not in {"twikit", "api"}:
        twitter["fetch_method"] = DEFAULT_CONFIG["twitter"]["fetch_method"]

    storage = normalized["storage"]
    if not storage["data_dir"].strip():
        storage["data_dir"] = DEFAULT_CONFIG["storage"]["data_dir"]

    articles = normalized["articles"]
    for positive_field in (
        "timeout_seconds",
        "max_response_bytes",
        "max_article_chars",
        "cache_ttl_hours",
        "max_articles_per_run",
    ):
        if articles[positive_field] <= 0:
            articles[positive_field] = DEFAULT_CONFIG["articles"][positive_field]
    for bounded_field, maximum in (("max_redirects", 10), ("retry_attempts", 5)):
        if articles[bounded_field] < 0:
            articles[bounded_field] = DEFAULT_CONFIG["articles"][bounded_field]
        articles[bounded_field] = min(maximum, articles[bounded_field])
    if articles["failure_retry_hours"] < 0:
        articles["failure_retry_hours"] = DEFAULT_CONFIG["articles"][
            "failure_retry_hours"
        ]
    if not articles["user_agent"].strip():
        articles["user_agent"] = DEFAULT_CONFIG["articles"]["user_agent"]

    weixin = normalized["weixin"]
    if not weixin["data_dir"].strip():
        weixin["data_dir"] = DEFAULT_CONFIG["weixin"]["data_dir"]
    for positive_field in (
        "poll_timeout_seconds",
        "request_timeout_seconds",
        "retry_attempts",
        "maximum_backoff_seconds",
    ):
        if weixin[positive_field] <= 0:
            weixin[positive_field] = DEFAULT_CONFIG["weixin"][positive_field]
    allowed_extensions = weixin["allowed_report_extensions"]
    if not allowed_extensions or not all(
        isinstance(item, str) and item.startswith(".") for item in allowed_extensions
    ):
        weixin["allowed_report_extensions"] = copy.deepcopy(
            DEFAULT_CONFIG["weixin"]["allowed_report_extensions"]
        )

    return normalized


def load_config(
    path: Path | str = CONFIG_PATH, *, use_environment: bool = True
) -> dict[str, Any]:
    """Load and normalize a JSON config without ever modifying the source.

    A missing, unreadable, malformed, or non-object file safely falls back to
    a fresh copy of :data:`DEFAULT_CONFIG`.  In particular, a damaged file is
    neither rewritten nor renamed here, preserving it for recovery.
    """
    target = Path(path)
    try:
        with target.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError):
        raw = copy.deepcopy(DEFAULT_CONFIG)

    config = normalize_config(raw)
    if not use_environment:
        return config
    provider = os.environ.get("XLS_LLM_PROVIDER", "").strip().lower()
    if provider in SUPPORTED_PROVIDERS:
        config["summarization"]["provider"] = provider

    active_provider = config["summarization"]["provider"]
    model = os.environ.get("XLS_LLM_MODEL", "").strip()
    api_key = os.environ.get("XLS_LLM_API_KEY", "").strip()
    if model:
        config["summarization"]["options"][active_provider]["model"] = model
    if api_key:
        config["summarization"]["options"][active_provider]["api_key"] = api_key

    list_urls = os.environ.get("XLS_X_LIST_URLS", "")
    if list_urls.strip():
        config["twitter"]["list_urls"] = [
            value.strip()
            for line in list_urls.splitlines()
            for value in line.split(",")
            if value.strip()
        ]
    return config


def _safe_mask(value: Any) -> str:
    """Mask a secret and defensively avoid returning it unchanged."""
    if value in (None, ""):
        return ""
    masked = str(mask_secret(value))
    raw = str(value)
    if masked == raw:
        return "*" * max(4, len(raw))
    return masked


def get_public_config(config: Any) -> dict[str, Any]:
    """Return normalized config with raw secrets replaced by safe metadata.

    For example, ``api_key`` becomes sibling fields
    ``api_key_configured`` and ``api_key_mask``.  The original key is removed
    entirely, so serializing the result cannot expose the credential.
    """

    def redact(node: Any) -> Any:
        if isinstance(node, Mapping):
            public: dict[str, Any] = {}
            for key, value in node.items():
                credentialed_endpoint = str(
                    key
                ).lower() == "endpoint" and _endpoint_contains_credentials(value)
                if _looks_sensitive_key(key) or credentialed_endpoint:
                    configured = (
                        bool(str(value).strip()) if value is not None else False
                    )
                    public[f"{key}_configured"] = configured
                    public[f"{key}_mask"] = (
                        "configured URL (credentials hidden)"
                        if configured and credentialed_endpoint
                        else _safe_mask(value)
                        if configured
                        else ""
                    )
                else:
                    public[key] = redact(value)
            return public
        if isinstance(node, list):
            return [redact(item) for item in node]
        return copy.deepcopy(node)

    return redact(normalize_config(config))


def _clear_requested(update: Mapping[str, Any], secret_key: str) -> bool:
    value = update.get(secret_key)
    value_marker = isinstance(value, Mapping) and (
        value.get("clear") is True or value.get("$clear") is True
    )
    return bool(
        value_marker
        or update.get(f"{secret_key}_clear") is True
        or update.get(f"clear_{secret_key}") is True
    )


def _is_secret_metadata(key: str) -> bool:
    for suffix in ("_configured", "_mask", "_clear"):
        if key.endswith(suffix):
            base_key = key[: -len(suffix)]
            if _looks_sensitive_key(base_key) or base_key == "endpoint":
                return True
    return key.startswith("clear_") and _looks_sensitive_key(key[6:])


def _merge_update(
    current: Mapping[str, Any], update: Mapping[str, Any]
) -> dict[str, Any]:
    result = copy.deepcopy(dict(current))

    # A clear flag is useful when the client received a public config that did
    # not contain the original secret field at all.
    secret_keys = {str(key) for key in result if _looks_sensitive_key(key)}
    secret_keys.update(
        str(key)
        for key in update
        if _looks_sensitive_key(key) and not _is_secret_metadata(str(key))
    )
    for secret_key in secret_keys:
        if _clear_requested(update, secret_key) and (
            secret_key in result or secret_key in update
        ):
            result[secret_key] = ""

    for key, value in update.items():
        if _is_secret_metadata(key):
            continue

        if _looks_sensitive_key(key):
            if _clear_requested(update, key):
                result[key] = ""
                continue
            if value is None or (isinstance(value, str) and not value.strip()):
                # Existing frontends send an empty password input when the
                # user did not intend to rotate the credential.
                continue
            if isinstance(value, str):
                previous = result.get(key, "")
                try:
                    if previous and value == _safe_mask(previous):
                        continue
                except Exception:
                    pass
                result[key] = value
            # Reject non-string secret values unless they were clear markers.
            continue

        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _merge_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)

    return result


def merge_config_update(current: Any, update: Any) -> dict[str, Any]:
    """Merge an incoming partial/public update into an existing config.

    Empty or masked secret values preserve the current credential.  A client
    must explicitly send ``{"api_key": {"clear": true}}`` (or the sibling
    boolean ``api_key_clear``; likewise for Bearer Token) to remove a secret.
    """
    base = normalize_config(current)
    if not isinstance(update, Mapping):
        return base
    return normalize_config(_merge_update(base, update))


def atomic_write_json(path: Path | str, data: Any) -> None:
    """Atomically serialize JSON using a temporary file beside the target."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def save_config(
    config: Any,
    path: Path | str = CONFIG_PATH,
    *,
    current: Any | None = None,
) -> dict[str, Any]:
    """Merge, validate, and atomically save configuration.

    ``current`` is injectable for callers/tests.  When omitted, the existing
    target is loaded.  Empty secret inputs retain values from that base.
    """
    base = (
        load_config(path, use_environment=False)
        if current is None
        else normalize_config(current)
    )
    merged = merge_config_update(base, config)
    atomic_write_json(path, merged)
    return merged


def ensure_config(path: Path | str = CONFIG_PATH) -> bool:
    """Create a default config only when no user configuration exists.

    Returns ``True`` when a new file was created.  Existing files, including
    malformed files that a user may need to recover, are never overwritten.
    """
    target = Path(path)
    if target.exists():
        return False
    atomic_write_json(target, DEFAULT_CONFIG)
    return True


def main(argv: list[str] | None = None) -> int:
    """Small installer-facing CLI for creating a missing config safely."""
    parser = argparse.ArgumentParser(description="X List Summarizer configuration")
    parser.add_argument(
        "--ensure",
        action="store_true",
        help="create config.json from defaults only when it is missing",
    )
    parser.add_argument("--path", default=str(CONFIG_PATH), help="configuration path")
    args = parser.parse_args(argv)
    if not args.ensure:
        parser.error("an action is required (use --ensure)")
    created = ensure_config(args.path)
    print(
        "Created default configuration."
        if created
        else "Existing configuration preserved."
    )
    return 0


__all__ = [
    "CONFIG_PATH",
    "DEFAULT_CONFIG",
    "SUPPORTED_PROVIDERS",
    "atomic_write_json",
    "ensure_config",
    "get_public_config",
    "load_config",
    "merge_config_update",
    "normalize_config",
    "save_config",
]


if __name__ == "__main__":
    raise SystemExit(main())
