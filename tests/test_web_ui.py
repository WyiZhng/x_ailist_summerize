import json
import os
import shutil
import threading
import unittest
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timedelta, timezone
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from app import web_ui
from app.config import atomic_write_json, load_config, normalize_config
from app.providers import (
    MockReportGenerator,
    MockSummaryProvider,
    MockXIngestionProvider,
    MockXProvider,
    build_mock_ingestion_posts,
)
from app.models import IngestionRun
from app.storage import FilePostStore


class WebConfigurationSecurityTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = Path.cwd() / "tests" / f".tmp-web-{uuid.uuid4().hex}"
        self.directory.mkdir(parents=True)
        self.original_paths = (
            web_ui.CONFIG_PATH,
            web_ui.COOKIES_PATH,
            web_ui.OUTPUT_DIR,
        )
        web_ui.CONFIG_PATH = self.directory / "config.json"
        web_ui.COOKIES_PATH = self.directory / "browser_session" / "cookies.json"
        web_ui.OUTPUT_DIR = self.directory / "output"
        web_ui.COOKIES_PATH.parent.mkdir(parents=True)
        web_ui.OUTPUT_DIR.mkdir(parents=True)

        self.api_key = "sk-test-api-key-123456789"
        self.bearer = "test-bearer-token-123456789"
        self.auth_token = "test-auth-cookie-123456789"
        atomic_write_json(
            web_ui.CONFIG_PATH,
            normalize_config(
                {
                    "summarization": {
                        "provider": "openai",
                        "options": {"openai": {"api_key": self.api_key}},
                    },
                    "twitter": {"api_bearer_token": self.bearer},
                }
            ),
        )
        atomic_write_json(
            web_ui.COOKIES_PATH,
            {"auth_token": self.auth_token, "ct0": "test-ct0-cookie-123456789"},
        )

        self.app_state = {
            "running": False,
            "status_msg": "Ready",
            "progress": 0,
            "error": None,
            "last_report": None,
        }
        handler = lambda *args, **kwargs: web_ui.DashHandler(
            *args, app_state=self.app_state, **kwargs
        )
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        web_ui.CONFIG_PATH, web_ui.COOKIES_PATH, web_ui.OUTPUT_DIR = self.original_paths
        shutil.rmtree(self.directory, ignore_errors=True)

    def request(self, path, *, data=None, headers=None):
        body = None if data is None else json.dumps(data).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + path,
            data=body,
            headers={"Content-Type": "application/json", **(headers or {})},
            method="POST" if data is not None else "GET",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, dict(response.headers), response.read()

    def test_public_config_returns_only_secret_state_and_masks(self):
        status, headers, raw = self.request("/api/config")
        payload = json.loads(raw)
        serialized = raw.decode("utf-8")

        self.assertEqual(status, 200)
        self.assertEqual(headers.get("X-Content-Type-Options"), "nosniff")
        self.assertEqual(headers.get("Cache-Control"), "no-store")
        self.assertNotIn(self.api_key, serialized)
        self.assertNotIn(self.bearer, serialized)
        self.assertNotIn(self.auth_token, serialized)
        self.assertNotIn("api_key", payload["summarization"]["options"]["openai"])
        self.assertTrue(
            payload["summarization"]["options"]["openai"]["api_key_configured"]
        )
        self.assertTrue(payload["twitter"]["api_bearer_token_configured"])
        self.assertTrue(payload["twitter"]["cookies_configured"])

    def test_blank_secret_submission_preserves_and_explicit_clear_removes(self):
        status, _, raw = self.request(
            "/api/save-config",
            data={
                "summarization": {"options": {"openai": {"api_key": ""}}},
                "twitter": {"api_bearer_token": ""},
            },
        )
        self.assertEqual(status, 200)
        self.assertTrue(json.loads(raw)["success"])
        preserved = load_config(web_ui.CONFIG_PATH)
        self.assertEqual(
            preserved["summarization"]["options"]["openai"]["api_key"], self.api_key
        )
        self.assertEqual(preserved["twitter"]["api_bearer_token"], self.bearer)

        status, _, _ = self.request(
            "/api/save-config",
            data={
                "summarization": {
                    "options": {"openai": {"api_key_clear": True, "api_key": ""}}
                },
                "twitter": {"api_bearer_token_clear": True},
            },
        )
        self.assertEqual(status, 200)
        cleared = load_config(web_ui.CONFIG_PATH)
        self.assertEqual(cleared["summarization"]["options"]["openai"]["api_key"], "")
        self.assertEqual(cleared["twitter"]["api_bearer_token"], "")

    def test_root_sandboxes_reports_and_escapes_history_fields_in_renderer(self):
        status, headers, raw = self.request("/")
        page = raw.decode("utf-8")

        self.assertEqual(status, 200)
        self.assertIn("Content-Security-Policy", headers)
        self.assertIn('sandbox="allow-scripts allow-popups allow-popups-to-escape-sandbox"', page)
        self.assertIn("${escapeHtml(h.name)}", page)
        self.assertIn("${escapeHtml(h.username)}", page)
        self.assertNotIn(" onerror=", page.lower())
        self.assertNotIn(self.api_key, page)
        self.assertIn("lastKnownReport = status.last_report || null", page)
        self.assertIn("statusEl.innerText = s.status_msg || 'Complete!'", page)

        report_name = "summary_security.html"
        (web_ui.OUTPUT_DIR / report_name).write_text(
            '<script>window.__legacy_xss = true</script>', encoding="utf-8"
        )
        request = urllib.request.Request(self.base_url + "/output/" + report_name)
        with urllib.request.urlopen(request, timeout=5) as response:
            report_csp = response.headers.get("Content-Security-Policy", "")
        self.assertIn(
            "sandbox allow-scripts allow-popups allow-popups-to-escape-sandbox",
            report_csp,
        )
        self.assertNotIn("allow-same-origin", report_csp)

    def test_root_includes_persistent_light_theme_toggle(self):
        status, _, raw = self.request("/")
        page = raw.decode("utf-8")

        self.assertEqual(status, 200)
        self.assertIn(':root[data-theme="light"]', page)
        self.assertIn('id="theme-toggle"', page)
        self.assertIn("function toggleTheme()", page)
        self.assertIn("localStorage.setItem(THEME_STORAGE_KEY, nextTheme)", page)
        self.assertIn("白天模式", page)
        self.assertIn("夜间模式", page)

    def test_non_local_host_is_rejected_for_control_api(self):
        request = urllib.request.Request(
            self.base_url + "/api/config",
            headers={"Host": "attacker.example"},
        )
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request, timeout=5)
        self.assertEqual(caught.exception.code, 403)

        cross_site = urllib.request.Request(
            self.base_url + "/api/config",
            headers={"Sec-Fetch-Site": "cross-site"},
        )
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(cross_site, timeout=5)
        self.assertEqual(caught.exception.code, 403)

        wrong_local_origin = urllib.request.Request(
            self.base_url + "/api/config",
            headers={"Origin": "http://localhost:9999"},
        )
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(wrong_local_origin, timeout=5)
        self.assertEqual(caught.exception.code, 403)

    def test_state_changing_reset_requires_post(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(self.base_url + "/api/reset-progress", timeout=5)
        self.assertEqual(caught.exception.code, 404)

        self.app_state.update({"progress": 80, "status_msg": "Working", "last_report": "x.html"})
        status, _, raw = self.request("/api/reset-progress", data={})
        self.assertEqual(status, 200)
        self.assertTrue(json.loads(raw)["success"])
        self.assertEqual(self.app_state["progress"], 0)
        self.assertIsNone(self.app_state["last_report"])

    def test_starting_a_run_does_not_clear_the_previous_report(self):
        self.app_state["last_report"] = "summary_previous.html"

        with mock.patch.object(web_ui.DashHandler, "run_task", autospec=True):
            status, _, raw = self.request("/api/run", data={})

        self.assertEqual(status, 200)
        self.assertTrue(json.loads(raw)["success"])
        self.assertEqual(self.app_state["last_report"], "summary_previous.html")
        self.app_state["running"] = False

    def test_active_running_ingestion_is_not_automatically_resumed(self):
        store = FilePostStore(self.directory / "active-run-data")
        post = build_mock_ingestion_posts()["mock://mixed"][0]
        store.save_posts([post])
        run = IngestionRun(
            run_id="active-run",
            started_at=datetime.now(timezone.utc),
            status="running",
            ingestion_owner_id="active-ingestion-owner",
            ingestion_heartbeat_at=datetime.now(timezone.utc),
            requested_lists=["mock://mixed"],
            new_post_count=1,
            new_post_ids=[post.id],
        )
        store.create_run(run)

        self.assertIsNone(web_ui._find_resumable_report_run(store))

        run.status = "failed"
        run.finished_at = datetime.now(timezone.utc)
        store.update_run(run)
        self.assertEqual(web_ui._find_resumable_report_run(store).run_id, run.run_id)

        run.report_status = "generating"
        run.report_claim_id = "active-report-owner"
        run.report_claimed_at = datetime.now(timezone.utc)
        store.update_run(run)
        self.assertIsNone(web_ui._find_resumable_report_run(store))

        run.report_claimed_at = datetime.now(timezone.utc) - timedelta(hours=2)
        store.update_run(run)
        self.assertEqual(web_ui._find_resumable_report_run(store).run_id, run.run_id)

    def test_web_finder_accepts_only_stale_running_ingestion(self):
        store = FilePostStore(self.directory / "stale-running-data")
        post = build_mock_ingestion_posts()["mock://mixed"][0]
        store.save_posts([post])
        now = datetime.now(timezone.utc)
        run = IngestionRun(
            run_id="running-candidate",
            started_at=now - timedelta(hours=2),
            status="running",
            ingestion_owner_id="active-ingestion-owner",
            ingestion_heartbeat_at=now,
            requested_lists=["mock://mixed"],
            new_post_count=1,
            new_post_ids=[post.id],
            report_status="not_started",
        )
        store.create_run(run)

        self.assertIsNone(web_ui._find_resumable_report_run(store))

        run.ingestion_heartbeat_at = now - timedelta(hours=2)
        store.update_run(run)
        selected = web_ui._find_resumable_report_run(store)
        self.assertIsNotNone(selected)
        self.assertEqual(selected.run_id, run.run_id)

        run.report_status = "succeeded"
        store.update_run(run)
        self.assertIsNone(web_ui._find_resumable_report_run(store))

    def test_invalid_cookies_are_not_reported_configured_or_silently_preserved(self):
        damaged = b'{"auth_token": '
        web_ui.COOKIES_PATH.write_bytes(damaged)

        _, _, raw = self.request("/api/config")
        self.assertFalse(json.loads(raw)["twitter"]["cookies_configured"])

        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.request("/api/save-cookies", data={"auth_token": "", "ct0": ""})
        self.assertEqual(caught.exception.code, 400)
        self.assertEqual(web_ui.COOKIES_PATH.read_bytes(), damaged)

        status, _, _ = self.request(
            "/api/save-cookies",
            data={"auth_token": "new-auth-token", "ct0": "new-ct0-token"},
        )
        self.assertEqual(status, 200)
        _, _, raw = self.request("/api/config")
        self.assertTrue(json.loads(raw)["twitter"]["cookies_configured"])

    def test_post_rejects_simple_cross_site_content_type(self):
        request = urllib.request.Request(
            self.base_url + "/api/reset-progress",
            data=b"{}",
            method="POST",
        )
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request, timeout=5)
        self.assertEqual(caught.exception.code, 415)

    def test_history_tolerates_non_object_metadata_and_report_symlinks_are_rejected(self):
        report = web_ui.OUTPUT_DIR / "summary_valid.html"
        report.write_text("safe", encoding="utf-8")
        (web_ui.OUTPUT_DIR / "history.json").write_text("[]", encoding="utf-8")

        status, _, raw = self.request("/api/history")
        self.assertEqual(status, 200)
        self.assertTrue(any(item["filename"] == report.name for item in json.loads(raw)))

        outside = self.directory / "outside.html"
        outside.write_text("outside-secret", encoding="utf-8")
        link = web_ui.OUTPUT_DIR / "summary_symlink.html"
        try:
            os.symlink(outside, link)
        except (OSError, NotImplementedError):
            return

        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(self.base_url + "/output/" + link.name, timeout=5)
        self.assertEqual(caught.exception.code, 404)

    async def test_web_task_orchestration_uses_injected_offline_providers(self):
        config = load_config(web_ui.CONFIG_PATH)
        config["twitter"]["list_urls"] = ["mock://mixed"]
        atomic_write_json(web_ui.CONFIG_PATH, config)

        class WebMockFetcher(MockXProvider):
            def __init__(self):
                super().__init__()
                self.report_generator = MockReportGenerator()

            def generate_html_report(self, *args, **kwargs):
                return self.report_generator.generate_html_report(*args, **kwargs)

        fetcher = WebMockFetcher()
        handler = object.__new__(web_ui.DashHandler)
        handler.app_state = {
            "running": True,
            "status_msg": "Starting",
            "progress": 0,
            "error": None,
            "last_report": None,
        }

        with (
            mock.patch.object(web_ui, "_build_fetcher", return_value=fetcher),
            mock.patch.object(web_ui, "LLMProvider", return_value=MockSummaryProvider()),
        ):
            await handler._run_async_task()

        self.assertFalse(handler.app_state["running"])
        self.assertEqual(handler.app_state["progress"], 100)
        self.assertIsNone(handler.app_state["error"])
        self.assertTrue(handler.app_state["run_id"])
        report_name = handler.app_state["last_report"]
        self.assertTrue(report_name)
        report = web_ui.OUTPUT_DIR / report_name
        self.assertTrue(report.exists())
        self.assertNotIn("<script>window.__mock_xss", report.read_text(encoding="utf-8"))
        history = json.loads((web_ui.OUTPUT_DIR / "history.json").read_text(encoding="utf-8"))
        self.assertIn(report_name, history)

    async def test_incremental_no_new_posts_preserves_previous_report(self):
        config = load_config(web_ui.CONFIG_PATH)
        config["twitter"].update(
            {
                "list_urls": ["mock://mixed"],
                "fetch_method": "api",
                "incremental_sync": True,
            }
        )
        config["storage"]["data_dir"] = str(self.directory / "data")
        atomic_write_json(web_ui.CONFIG_PATH, config)

        class WebIngestionFetcher(MockXIngestionProvider):
            def __init__(self):
                super().__init__()
                self.report_generator = MockReportGenerator()

            def generate_html_report(self, *args, **kwargs):
                return self.report_generator.generate_html_report(*args, **kwargs)

        handler = object.__new__(web_ui.DashHandler)
        handler.app_state = {
            "running": True,
            "status_msg": "Starting",
            "progress": 0,
            "error": None,
            "last_report": None,
        }
        first_summary = MockSummaryProvider()
        with (
            mock.patch.object(web_ui, "_build_fetcher", return_value=WebIngestionFetcher()),
            mock.patch.object(web_ui, "LLMProvider", return_value=first_summary),
        ):
            await handler._run_async_task()
        first_report = handler.app_state["last_report"]
        self.assertTrue(first_report)
        self.assertEqual(len(first_summary.calls), 1)

        handler.app_state["running"] = True
        second_summary = MockSummaryProvider()
        with (
            mock.patch.object(web_ui, "_build_fetcher", return_value=WebIngestionFetcher()),
            mock.patch.object(web_ui, "LLMProvider", return_value=second_summary),
        ):
            await handler._run_async_task()

        self.assertEqual(handler.app_state["status_msg"], "No new posts")
        self.assertEqual(handler.app_state["last_report"], first_report)
        self.assertEqual(second_summary.calls, [])

    async def test_web_automatically_resumes_a_failed_report_without_x_fetch(self):
        config = load_config(web_ui.CONFIG_PATH)
        config["twitter"].update(
            {
                "list_urls": ["mock://mixed"],
                "fetch_method": "api",
                "incremental_sync": True,
            }
        )
        config["storage"]["data_dir"] = str(self.directory / "resume-data")
        atomic_write_json(web_ui.CONFIG_PATH, config)

        class WebFetcher(MockXIngestionProvider):
            def __init__(self, *, fail_report=False, fail_fetch=False):
                super().__init__()
                self.fail_report = fail_report
                self.fail_fetch = fail_fetch
                self.report_generator = MockReportGenerator()

            async def fetch_page(self, *args, **kwargs):
                if self.fail_fetch:
                    raise AssertionError("resume must not call X")
                return await super().fetch_page(*args, **kwargs)

            def generate_html_report(self, *args, **kwargs):
                if self.fail_report:
                    raise RuntimeError("injected web report failure")
                return self.report_generator.generate_html_report(*args, **kwargs)

        handler = object.__new__(web_ui.DashHandler)
        handler.app_state = {
            "running": True,
            "status_msg": "Starting",
            "progress": 0,
            "error": None,
            "last_report": None,
        }
        with (
            mock.patch.object(
                web_ui, "_build_fetcher", return_value=WebFetcher(fail_report=True)
            ),
            mock.patch.object(web_ui, "LLMProvider", return_value=MockSummaryProvider()),
        ):
            await handler._run_async_task()
        self.assertIsNotNone(handler.app_state["error"])

        handler.app_state.update({"running": True, "error": None})
        resumed_summary = MockSummaryProvider()
        with (
            mock.patch.object(
                web_ui, "_build_fetcher", return_value=WebFetcher(fail_fetch=True)
            ),
            mock.patch.object(web_ui, "LLMProvider", return_value=resumed_summary),
        ):
            await handler._run_async_task()

        self.assertIsNone(handler.app_state["error"])
        self.assertEqual(handler.app_state["status_msg"], "Complete!")
        self.assertEqual(len(resumed_summary.calls), 1)
        self.assertTrue(handler.app_state["last_report"])


if __name__ == "__main__":
    unittest.main()
