import copy
import contextlib
import json
import shutil
import unittest
import uuid
from pathlib import Path
from unittest import mock

from app.config import (
    DEFAULT_CONFIG,
    SUPPORTED_PROVIDERS,
    atomic_write_json,
    ensure_config,
    get_public_config,
    load_config,
    merge_config_update,
    normalize_config,
    save_config,
)


@contextlib.contextmanager
def workspace_temp_directory():
    """Create test storage with inherited workspace ACLs on Windows."""
    path = Path.cwd() / "tests" / f".tmp-config-{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


class ConfigTests(unittest.TestCase):
    def test_missing_config_uses_defaults_and_ensure_creates_it_once(self):
        with workspace_temp_directory() as directory:
            path = directory / "config.json"

            self.assertEqual(load_config(path), DEFAULT_CONFIG)
            self.assertTrue(ensure_config(path))
            first_bytes = path.read_bytes()
            self.assertFalse(ensure_config(path))
            self.assertEqual(path.read_bytes(), first_bytes)

    def test_default_config_has_all_nine_providers_and_current_twitter_schema(self):
        self.assertEqual(
            set(SUPPORTED_PROVIDERS),
            {
                "ollama",
                "lmstudio",
                "groq",
                "claude",
                "openai",
                "gemini",
                "deepseek",
                "openrouter",
                "grok",
            },
        )
        self.assertIn("fetch_method", DEFAULT_CONFIG["twitter"])
        self.assertIn("api_bearer_token", DEFAULT_CONFIG["twitter"])
        self.assertIs(DEFAULT_CONFIG["twitter"]["incremental_sync"], True)
        self.assertEqual(DEFAULT_CONFIG["twitter"]["initial_fetch_limit"], 100)
        self.assertEqual(DEFAULT_CONFIG["twitter"]["page_size"], 100)
        self.assertEqual(DEFAULT_CONFIG["twitter"]["max_pages"], 20)
        self.assertEqual(DEFAULT_CONFIG["storage"]["data_dir"], "data")
        self.assertIs(DEFAULT_CONFIG["articles"]["enabled"], True)
        self.assertEqual(DEFAULT_CONFIG["articles"]["max_articles_per_run"], 20)
        self.assertEqual(DEFAULT_CONFIG["articles"]["cache_ttl_hours"], 168)

    def test_normalize_deep_merges_old_partial_config_without_mutating_default(self):
        defaults_before = copy.deepcopy(DEFAULT_CONFIG)
        old = {
            "summarization": {
                "provider": "openai",
                "options": {"openai": {"api_key": "sk-old"}},
            },
            "twitter": {"list_urls": ["https://x.com/i/lists/1"]},
            "future_section": {"enabled": True},
        }

        result = normalize_config(old)

        self.assertEqual(result["summarization"]["provider"], "openai")
        self.assertEqual(result["summarization"]["options"]["openai"]["api_key"], "sk-old")
        self.assertEqual(result["summarization"]["options"]["openai"]["model"], "gpt-4o")
        self.assertIn("gemini", result["summarization"]["options"])
        self.assertEqual(result["twitter"]["fetch_method"], "twikit")
        self.assertEqual(result["future_section"], {"enabled": True})
        self.assertEqual(DEFAULT_CONFIG, defaults_before)

    def test_normalize_rejects_invalid_known_types_and_values(self):
        result = normalize_config(
            {
                "summarization": {"provider": "not-a-provider", "options": []},
                "twitter": {
                    "list_urls": ["valid", 123],
                    "max_tweets": True,
                    "max_scrolls": -4,
                    "headless_after_auth": "yes",
                    "fetch_method": "scrape-anything",
                    "api_bearer_token": 123,
                    "incremental_sync": "yes",
                    "initial_fetch_limit": 0,
                    "page_size": 500,
                    "max_pages": -1,
                },
                "storage": {"data_dir": ""},
                "articles": {
                    "enabled": "yes",
                    "timeout_seconds": 0,
                    "max_redirects": 99,
                    "max_response_bytes": -1,
                    "max_article_chars": 0,
                    "cache_ttl_hours": 0,
                    "failure_retry_hours": -1,
                    "max_articles_per_run": 0,
                    "retry_attempts": 99,
                    "user_agent": "",
                },
            }
        )

        self.assertEqual(result["summarization"], DEFAULT_CONFIG["summarization"])
        self.assertEqual(result["twitter"]["list_urls"], [])
        self.assertEqual(result["twitter"]["max_tweets"], 100)
        self.assertEqual(result["twitter"]["max_scrolls"], 5)
        self.assertIs(result["twitter"]["headless_after_auth"], True)
        self.assertEqual(result["twitter"]["fetch_method"], "twikit")
        self.assertEqual(result["twitter"]["api_bearer_token"], "")
        self.assertIs(result["twitter"]["incremental_sync"], True)
        self.assertEqual(result["twitter"]["initial_fetch_limit"], 100)
        self.assertEqual(result["twitter"]["page_size"], 100)
        self.assertEqual(result["twitter"]["max_pages"], 20)
        self.assertEqual(result["storage"]["data_dir"], "data")
        self.assertEqual(result["articles"]["timeout_seconds"], 15)
        self.assertEqual(result["articles"]["max_redirects"], 10)
        self.assertEqual(result["articles"]["max_response_bytes"], 5_242_880)
        self.assertEqual(result["articles"]["max_article_chars"], 50_000)
        self.assertEqual(result["articles"]["cache_ttl_hours"], 168)
        self.assertEqual(result["articles"]["failure_retry_hours"], 6)
        self.assertEqual(result["articles"]["max_articles_per_run"], 20)
        self.assertEqual(result["articles"]["retry_attempts"], 5)
        self.assertEqual(result["articles"]["user_agent"], "x-ai-daily/1.0")

    def test_load_corrupt_json_falls_back_without_touching_original(self):
        with workspace_temp_directory() as directory:
            path = directory / "config.json"
            damaged = b'{"summarization": '
            path.write_bytes(damaged)

            result = load_config(path)

            self.assertEqual(result, DEFAULT_CONFIG)
            self.assertEqual(path.read_bytes(), damaged)

    def test_environment_overrides_runtime_configuration(self):
        environment = {
            "XLS_LLM_PROVIDER": "deepseek",
            "XLS_LLM_MODEL": "test-model",
            "XLS_LLM_API_KEY": "test-secret",
            "XLS_X_LIST_URLS": "https://x.com/i/lists/1\nhttps://x.com/i/lists/2",
        }
        with workspace_temp_directory() as directory, mock.patch.dict(
            "os.environ", environment, clear=False
        ):
            result = load_config(directory / "missing.json")

        self.assertEqual(result["summarization"]["provider"], "deepseek")
        self.assertEqual(
            result["summarization"]["options"]["deepseek"]["model"], "test-model"
        )
        self.assertEqual(
            result["summarization"]["options"]["deepseek"]["api_key"], "test-secret"
        )
        self.assertEqual(
            result["twitter"]["list_urls"],
            ["https://x.com/i/lists/1", "https://x.com/i/lists/2"],
        )

    def test_environment_can_be_excluded_from_persistent_configuration(self):
        environment = {
            "XLS_LLM_PROVIDER": "deepseek",
            "XLS_LLM_API_KEY": "must-not-be-persisted",
        }
        with workspace_temp_directory() as directory, mock.patch.dict(
            "os.environ", environment, clear=False
        ):
            path = directory / "config.json"
            saved = save_config({"twitter": {"max_tweets": 25}}, path)
            persisted = path.read_text(encoding="utf-8")

        self.assertEqual(saved["summarization"]["provider"], "ollama")
        self.assertNotIn("api_key", saved["summarization"]["options"]["ollama"])
        self.assertNotIn("must-not-be-persisted", persisted)

    def test_public_config_removes_all_raw_secrets(self):
        config = normalize_config(
            {
                "summarization": {
                    "provider": "openai",
                    "options": {
                        "openai": {"api_key": "sk-super-secret"},
                        "groq": {"api_key": "gsk-another-secret"},
                    },
                },
                "twitter": {"api_bearer_token": "bearer-super-secret"},
                "future_provider": {
                    "client_secret": "future-client-secret",
                    "access_token": "future-access-token",
                    "max_tokens": 2048,
                },
            }
        )

        public = get_public_config(config)
        serialized = json.dumps(public)

        openai = public["summarization"]["options"]["openai"]
        self.assertNotIn("api_key", openai)
        self.assertIs(openai["api_key_configured"], True)
        self.assertTrue(openai["api_key_mask"])
        self.assertNotIn("api_bearer_token", public["twitter"])
        self.assertIs(public["twitter"]["api_bearer_token_configured"], True)
        self.assertNotIn("sk-super-secret", serialized)
        self.assertNotIn("gsk-another-secret", serialized)
        self.assertNotIn("bearer-super-secret", serialized)
        self.assertNotIn("future-client-secret", serialized)
        self.assertNotIn("future-access-token", serialized)
        self.assertTrue(public["future_provider"]["client_secret_configured"])
        self.assertTrue(public["future_provider"]["access_token_configured"])
        self.assertEqual(public["future_provider"]["max_tokens"], 2048)

    def test_public_config_hides_credentials_embedded_in_endpoint(self):
        endpoint = "https://user:password@example.com/v1?api_key=query-secret"
        current = normalize_config(
            {
                "summarization": {
                    "provider": "openai",
                    "options": {"openai": {"endpoint": endpoint}},
                }
            }
        )

        public = get_public_config(current)
        serialized = json.dumps(public)
        openai = public["summarization"]["options"]["openai"]
        self.assertNotIn(endpoint, serialized)
        self.assertNotIn("password", serialized)
        self.assertNotIn("query-secret", serialized)
        self.assertNotIn("endpoint", openai)
        self.assertTrue(openai["endpoint_configured"])

        round_tripped = merge_config_update(current, public)
        self.assertEqual(
            round_tripped["summarization"]["options"]["openai"]["endpoint"],
            endpoint,
        )

    def test_update_preserves_empty_and_masked_secrets_and_accepts_new_values(self):
        current = normalize_config(
            {
                "summarization": {
                    "provider": "openai",
                    "options": {"openai": {"api_key": "sk-existing"}},
                },
                "twitter": {"api_bearer_token": "bearer-existing"},
            }
        )
        public = get_public_config(current)
        public["summarization"]["options"]["openai"]["api_key"] = ""
        public["twitter"]["api_bearer_token"] = "   "

        preserved = merge_config_update(current, public)
        self.assertEqual(
            preserved["summarization"]["options"]["openai"]["api_key"],
            "sk-existing",
        )
        self.assertEqual(preserved["twitter"]["api_bearer_token"], "bearer-existing")
        self.assertNotIn(
            "api_key_mask",
            preserved["summarization"]["options"]["openai"],
        )

        replaced = merge_config_update(
            current,
            {
                "summarization": {"options": {"openai": {"api_key": "sk-new"}}},
                "twitter": {"api_bearer_token": "bearer-new"},
            },
        )
        self.assertEqual(replaced["summarization"]["options"]["openai"]["api_key"], "sk-new")
        self.assertEqual(replaced["twitter"]["api_bearer_token"], "bearer-new")

    def test_explicit_clear_markers_remove_secrets(self):
        current = normalize_config(
            {
                "summarization": {
                    "provider": "openai",
                    "options": {"openai": {"api_key": "sk-existing"}},
                },
                "twitter": {"api_bearer_token": "bearer-existing"},
            }
        )

        cleared = merge_config_update(
            current,
            {
                "summarization": {
                    "options": {"openai": {"api_key": {"clear": True}}}
                },
                "twitter": {"api_bearer_token_clear": True},
            },
        )

        self.assertEqual(cleared["summarization"]["options"]["openai"]["api_key"], "")
        self.assertEqual(cleared["twitter"]["api_bearer_token"], "")
        self.assertFalse(
            get_public_config(cleared)["twitter"]["api_bearer_token_configured"]
        )

    def test_atomic_write_json_replaces_target_and_leaves_no_temp_file(self):
        with workspace_temp_directory() as directory:
            path = directory / "nested" / "data.json"
            atomic_write_json(path, {"text": "中文", "value": 2})

            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {"text": "中文", "value": 2},
            )
            self.assertEqual(list(path.parent.glob(f".{path.name}.*.tmp")), [])

    def test_atomic_write_failure_preserves_old_file_and_cleans_temp(self):
        with workspace_temp_directory() as directory:
            path = directory / "config.json"
            path.write_text('{"old": true}', encoding="utf-8")

            with mock.patch("app.config.os.replace", side_effect=OSError("replace failed")):
                with self.assertRaises(OSError):
                    atomic_write_json(path, {"new": True})

            self.assertEqual(path.read_text(encoding="utf-8"), '{"old": true}')
            self.assertEqual(list(path.parent.glob(f".{path.name}.*.tmp")), [])

    def test_save_config_uses_existing_secret_and_atomic_json(self):
        with workspace_temp_directory() as directory:
            path = directory / "config.json"
            initial = normalize_config(
                {
                    "summarization": {
                        "provider": "openai",
                        "options": {"openai": {"api_key": "sk-keep"}},
                    }
                }
            )
            atomic_write_json(path, initial)

            saved = save_config(
                {
                    "summarization": {
                        "provider": "openai",
                        "options": {"openai": {"api_key": "", "model": "gpt-4.1"}},
                    }
                },
                path,
            )

            self.assertEqual(saved["summarization"]["options"]["openai"]["api_key"], "sk-keep")
            self.assertEqual(saved["summarization"]["options"]["openai"]["model"], "gpt-4.1")
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), saved)


if __name__ == "__main__":
    unittest.main()
