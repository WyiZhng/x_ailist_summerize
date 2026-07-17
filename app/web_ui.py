import http.server
from http.server import ThreadingHTTPServer
import json
import logging
import mimetypes
import os
import random
import re
import subprocess
import sys
import threading
import time
import webbrowser
import asyncio
from urllib.parse import urlparse
from pathlib import Path
from datetime import datetime

# Keep the legacy `python app/web_ui.py` entry point while using package imports.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
ROOT_DIR = PROJECT_ROOT

from app.x_list_summarizer import XListFetcher, XApiFetcher
from app.article_fetcher import ArticleFetchConfig, ArticleFetcher
from app.article_service import ArticleService, ArticleServiceConfig
from app.article_storage import FileArticleStore
from app.llm_providers import LLMProvider
from app.config import (
    atomic_write_json,
    get_public_config,
    load_config as load_app_config,
    save_config as save_app_config,
)
from app.ingestion import ingestion_run_is_stale
from app.security import redact_sensitive_text, sanitize_image_src
from app.task_runner import (
    DigestTaskRequest,
    DigestTaskRunner,
    TaskProgress,
    report_claim_is_stale,
)
from app.storage import FilePostStore


logger = logging.getLogger(__name__)

def _build_fetcher(config, list_owner=None):
    """Factory: returns the configured fetcher (twikit cookies or X API Bearer)."""
    tw = config.get('twitter', {}) if config else {}
    method = tw.get('fetch_method', 'twikit')
    if method == 'api':
        return XApiFetcher(bearer_token=tw.get('api_bearer_token', ''), list_owner=list_owner)
    return XListFetcher(
        cookies_path=COOKIES_PATH,
        list_owner=list_owner,
        proxy=tw.get('proxy') or None,
    )


def _find_resumable_report_run(store):
    """Return the newest durable ingestion run whose report is still pending."""

    posts = {post.id for post in store.read_posts()}
    candidates = []
    for run in store.read_runs():
        resumable_report = run.report_status in {"not_started", "failed"} or (
            run.report_status == "generating" and report_claim_is_stale(run)
        )
        if not run.new_post_ids or not resumable_report:
            continue
        if not set(run.new_post_ids).issubset(posts):
            continue
        if run.status in {"success", "partial_success", "failed"} or (
            run.status == "running" and ingestion_run_is_stale(run)
        ):
            candidates.append(run)
    return max(candidates, key=lambda item: item.started_at, default=None)

PORT = 8765
HOST = '127.0.0.1'
CONFIG_PATH = ROOT_DIR / 'config.json'
COOKIES_PATH = ROOT_DIR / 'browser_session' / 'cookies.json'
OUTPUT_DIR = ROOT_DIR / 'output'
STATIC_FILES = {
    '/icon.png': ROOT_DIR / 'icon.png',
    '/screenshots/auth_guide.png': ROOT_DIR / 'screenshots' / 'auth_guide.png',
}

class DashHandler(http.server.SimpleHTTPRequestHandler):
    _run_lock = threading.Lock()
    _health_check_lock = threading.Lock()

    def __init__(self, *args, **kwargs):
        self.app_state = kwargs.pop('app_state', {})
        super().__init__(*args, **kwargs)

    def log_message(self, format, *args):
        # Suppress terminal spam
        return

    def _send_security_headers(self, *, report=False):
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('X-Frame-Options', 'SAMEORIGIN')
        if report:
            policy = (
                "default-src 'none'; "
                "sandbox allow-scripts allow-popups allow-popups-to-escape-sandbox; "
                "style-src 'unsafe-inline' https://fonts.googleapis.com; "
                "font-src https://fonts.gstatic.com; "
                "img-src data: https:; media-src https:; "
                "frame-src https://www.youtube.com; "
                "script-src 'unsafe-inline'; base-uri 'none'; form-action 'none'"
            )
        else:
            policy = (
                "default-src 'self'; "
                "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
                "font-src https://fonts.gstatic.com; "
                "img-src 'self' data: https:; media-src https:; "
                "script-src 'self' 'unsafe-inline'; connect-src 'self'; "
                "frame-src 'self'; object-src 'none'; base-uri 'none'; form-action 'self'"
            )
        self.send_header('Content-Security-Policy', policy)

    def _is_trusted_local_request(self):
        """Restrict the local control API to localhost browser requests."""
        local_hosts = {'localhost', '127.0.0.1', '::1'}

        def is_same_local_service(value):
            try:
                parsed_value = urlparse(value)
                if parsed_value.scheme != 'http' or parsed_value.hostname != hostname:
                    return False
                default_port = 443 if parsed_value.scheme == 'https' else 80
                return (parsed_value.port or default_port) == self.server.server_port
            except ValueError:
                return False

        host_header = self.headers.get('Host', '')
        try:
            hostname = urlparse(f'//{host_header}').hostname
        except ValueError:
            return False
        if hostname not in local_hosts:
            return False

        origin = self.headers.get('Origin')
        if origin and not is_same_local_service(origin):
            return False
        referer = self.headers.get('Referer')
        if referer and not is_same_local_service(referer):
            return False
        fetch_site = self.headers.get('Sec-Fetch-Site')
        if fetch_site and fetch_site.lower() not in {'same-origin', 'same-site', 'none'}:
            return False
        return True

    def send_json(self, data, status=200):
        payload = json.dumps(data, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(payload)))
        self.send_header('Cache-Control', 'no-store')
        self._send_security_headers()
        self.end_headers()
        self.wfile.write(payload)

    def _send_static_asset(self, path, *, head_only=False):
        file_path = STATIC_FILES.get(path)
        if not file_path or not file_path.is_file():
            self.send_error(404)
            return
        payload = file_path.read_bytes()
        content_type = mimetypes.guess_type(file_path.name)[0] or 'application/octet-stream'
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(payload)))
        self.send_header('Cache-Control', 'public, max-age=3600')
        self._send_security_headers()
        self.end_headers()
        if not head_only:
            self.wfile.write(payload)

    def _read_json_body(self, max_bytes=1_000_000):
        """Read a bounded JSON object from the current request."""
        try:
            length = int(self.headers.get('Content-Length', '0'))
        except ValueError as exc:
            raise ValueError('Invalid Content-Length') from exc
        if length < 0 or length > max_bytes:
            raise ValueError('Request body is too large')
        if length == 0:
            return {}
        try:
            data = json.loads(self.rfile.read(length).decode('utf-8'))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError('Invalid JSON request body') from exc
        if not isinstance(data, dict):
            raise ValueError('JSON request body must be an object')
        return data

    def _load_cookie_credentials(self):
        """Return valid saved X cookies, without exposing their values."""
        environment_cookies = {
            "auth_token": os.environ.get("XLS_X_AUTH_TOKEN", "").strip(),
            "ct0": os.environ.get("XLS_X_CT0", "").strip(),
        }
        if all(environment_cookies.values()):
            return environment_cookies
        try:
            data = json.loads(COOKIES_PATH.read_text(encoding='utf-8'))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None
        if not isinstance(data, dict):
            return None
        if not all(
            isinstance(data.get(key), str) and data[key].strip()
            for key in ('auth_token', 'ct0')
        ):
            return None
        return data

    @classmethod
    def _refresh_health_status(cls, config, cookies_configured):
        """Refresh external health state without blocking status requests."""
        try:
            method = config.get('twitter', {}).get('fetch_method', 'twikit')
            bearer = config.get('twitter', {}).get('api_bearer_token')
            has_creds = (method == 'api' and bearer) or (
                method == 'twikit' and cookies_configured
            )
            x_status = {'active': False, 'message': 'Not logged in'}
            if has_creds:
                try:
                    fetcher = _build_fetcher(config)
                    success, message = asyncio.run(fetcher.verify_session(retries=0))
                    x_status = {'active': success, 'message': message}
                except Exception as exc:
                    logger.warning("X health check failed (%s)", type(exc).__name__)
                    x_status = {'active': False, 'message': 'Auth Error'}
            elif method == 'api':
                x_status = {'active': False, 'message': 'No Bearer Token'}

            try:
                ai_status = LLMProvider(config).verify()
            except Exception as exc:
                logger.warning("AI health check failed (%s)", type(exc).__name__)
                ai_status = {'active': False, 'message': 'Error'}

            completed_at = time.time()
            cls._x_cache = x_status
            cls._x_cache_time = completed_at
            cls._ai_cache = ai_status
            cls._ai_cache_time = completed_at
        finally:
            cls._health_check_lock.release()

    def _schedule_health_refresh(self, now):
        """Start at most one external health refresh when cached state is stale."""
        if self.app_state.get('running'):
            return
        last_x = getattr(DashHandler, '_x_cache', None)
        last_ai = getattr(DashHandler, '_ai_cache', None)
        last_x_time = getattr(DashHandler, '_x_cache_time', 0)
        last_ai_time = getattr(DashHandler, '_ai_cache_time', 0)
        x_limit = 30 if (last_x and last_x.get('active')) else 5
        ai_limit = 30 if (last_ai and last_ai.get('active')) else 5
        stale = (now - last_x_time > x_limit) or (now - last_ai_time > ai_limit)
        if not stale or not DashHandler._health_check_lock.acquire(blocking=False):
            return
        try:
            config = self.load_config()
            cookies_configured = self._load_cookie_credentials() is not None
            threading.Thread(
                target=DashHandler._refresh_health_status,
                args=(config, cookies_configured),
                daemon=True,
                name="health-refresh",
            ).start()
        except Exception:
            DashHandler._health_check_lock.release()
            raise

    def _resolve_report_path(self, filename):
        """Resolve a generated report without following an escaping symlink."""
        if not re.fullmatch(r'[A-Za-z0-9_.-]+\.html', filename or ''):
            return None
        candidate = OUTPUT_DIR / filename
        try:
            if candidate.is_symlink():
                return None
            output_root = OUTPUT_DIR.resolve()
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(output_root)
            if not resolved.is_file():
                return None
            return resolved
        except (OSError, RuntimeError, ValueError):
            return None

    def _list_report_paths(self):
        """Return safe generated reports, newest first."""
        reports = []
        if not OUTPUT_DIR.exists():
            return reports
        for candidate in OUTPUT_DIR.glob('summary_*.html'):
            resolved = self._resolve_report_path(candidate.name)
            if resolved is not None:
                reports.append(resolved)
        try:
            return sorted(reports, key=lambda item: item.stat().st_mtime, reverse=True)
        except OSError:
            return []

    def _send_root(self):
        html = self.get_reconstructed_html().encode('utf-8')
        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(html)))
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self._send_security_headers()
        self.end_headers()
        return html

    def do_HEAD(self):
        parsed = urlparse(self.path)
        if parsed.path == '/':
            self._send_root()
        elif parsed.path in STATIC_FILES:
            self._send_static_asset(parsed.path, head_only=True)
        else:
            self.send_error(404)

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path.startswith('/api/') and not self._is_trusted_local_request():
            self.send_json({'success': False, 'error': 'Local request required'}, status=403)
            return
        
        if parsed.path == '/':
            html = self._send_root()
            self.wfile.write(html)
            return

        elif parsed.path in STATIC_FILES:
            self._send_static_asset(parsed.path)
            return
            
        elif parsed.path == '/api/status':
            now = time.time()
            self._schedule_health_refresh(now)
            x_status = getattr(
                DashHandler, '_x_cache', {'active': False, 'message': 'Checking...'}
            )
            ai_status = getattr(
                DashHandler, '_ai_cache', {'active': False, 'message': 'Checking...'}
            )

            self.send_json({
                'running': self.app_state.get('running', False),
                'run_id': self.app_state.get('run_id'),
                'status_msg': self.app_state.get('status_msg', 'Ready'),
                'progress': self.app_state.get('progress', 0),
                'error': self.app_state.get('error'),
                'x_auth': x_status,
                'ai_status': ai_status,
                'last_report': self.app_state.get('last_report'),
                'output_path': str(OUTPUT_DIR.resolve())
            })

        elif parsed.path == '/api/config':
            public_config = get_public_config(self.load_config())
            public_config.setdefault('twitter', {})['cookies_configured'] = (
                self._load_cookie_credentials() is not None
            )
            self.send_json(public_config)

        elif parsed.path == '/api/history':
            history = []
            metadata = {}
            meta_path = OUTPUT_DIR / 'history.json'
            if meta_path.exists():
                try:
                    with open(meta_path, 'r', encoding='utf-8') as f:
                        loaded_metadata = json.load(f)
                    if isinstance(loaded_metadata, dict):
                        metadata = loaded_metadata
                    else:
                        logger.warning("History metadata is not an object; using report files only")
                except (OSError, json.JSONDecodeError, TypeError):
                    logger.warning("History metadata is unreadable; using report files only")

            if OUTPUT_DIR.exists():
                for f in self._list_report_paths():
                    file_meta = metadata.get(f.name, {})
                    if not isinstance(file_meta, dict):
                        file_meta = {}
                    report_path = self._resolve_report_path(f.name)
                    if report_path is None:
                        continue
                    try:
                        report_stat = report_path.stat()
                    except OSError:
                        continue
                    fallback_profile = 'https://abs.twimg.com/sticky/default_profile_images/default_profile_normal.png'
                    history.append({
                        'filename': f.name,
                        'date': datetime.fromtimestamp(report_stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                        'size': report_stat.st_size,
                        'name': file_meta.get('name', 'Analysis Report'),
                        'username': file_meta.get('username', 'Unknown'),
                        'tweets': file_meta.get('tweets', 0),
                        'links': file_meta.get('links', 0),
                        'profile_img': sanitize_image_src(
                            file_meta.get('profile_img'), fallback=fallback_profile
                        ),
                        'members': file_meta.get('members', 0)
                    })
            self.send_json(history)
            
        elif parsed.path.startswith('/output/'):
            filename = parsed.path.split('/')[-1]
            if filename == 'latest':
                files = self._list_report_paths()
                filename = next(
                    (item.name for item in files),
                    None,
                )
                if filename is None:
                    self.send_error(404)
                    return

            file_path = self._resolve_report_path(filename)
            if file_path is not None:
                try:
                    payload = file_path.read_bytes()
                except OSError:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header('Content-type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(payload)))
                self.send_header('Cache-Control', 'no-store')
                self._send_security_headers(report=True)
                self.end_headers()
                self.wfile.write(payload)
            else:
                self.send_error(404)
        
        else:
            self.send_error(404)

    def _analyze_word_frequencies(self, memberships):
        import re
        from collections import Counter
        
        stop_words = {
            'the', 'and', 'for', 'with', 'your', 'from', 'this', 'that', 'list', 'lists', 'member',
            'of', 'to', 'in', 'on', 'at', 'by', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
            'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'up', 'down', 'out',
            'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where',
            'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
            'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will',
            'just', 'should', 'now', 'my', 'me', 'our', 'i', 'a', 'it', 'its'
        }
        
        words = []
        for l in memberships:
            name = l.get('name', '')
            cleaned = re.sub(r'[^a-zA-Z0-9\s]', ' ', name.lower())
            tokens = cleaned.split()
            for t in tokens:
                if len(t) > 2 and t not in stop_words:
                    words.append(t)
        return dict(Counter(words).most_common(100))

    def do_POST(self):
        parsed = urlparse(self.path)

        if not self._is_trusted_local_request():
            self.send_json({'success': False, 'error': 'Local request required'}, status=403)
            return

        content_type = self.headers.get('Content-Type', '').partition(';')[0].strip().lower()
        if content_type != 'application/json':
            self.send_json(
                {'success': False, 'error': 'Content-Type must be application/json'},
                status=415,
            )
            return

        try:
            data = self._read_json_body()
        except ValueError as exc:
            self.send_json({'success': False, 'error': str(exc)}, status=400)
            return

        if parsed.path == '/api/open-folder':
            try:
                out_abs = str(OUTPUT_DIR.absolute())
                if sys.platform == 'win32':
                    subprocess.run(['explorer', out_abs], check=False)
                else:
                    cmd = ['open', out_abs] if sys.platform == 'darwin' else ['xdg-open', out_abs]
                    subprocess.run(cmd, check=False)
                self.send_json({'success': True})
            except (OSError, subprocess.SubprocessError) as exc:
                logger.warning("Could not open output directory (%s)", type(exc).__name__)
                self.send_json({'success': False, 'error': 'Output directory could not be opened'}, status=500)
            return

        if parsed.path == '/api/reset-progress':
            self.app_state['progress'] = 0
            self.app_state['status_msg'] = 'Ready'
            self.app_state['last_report'] = None
            self.send_json({'success': True})
            return

        if parsed.path == '/api/profile':
            username = str(data.get('username', '')).strip().replace('@', '')
            
            if not username:
                self.send_json({'success': False, 'error': 'Username required'})
                return
            if not re.fullmatch(r'[A-Za-z0-9_]{1,15}', username):
                self.send_json({'success': False, 'error': 'Invalid X username'}, status=400)
                return
            
            try:
                fetcher = _build_fetcher(self.load_config())
                # Run async membership fetching in a synchronous context
                memberships = asyncio.run(fetcher.get_user_memberships(username))
                word_counts = self._analyze_word_frequencies(memberships)
                
                self.send_json({
                    'success': True,
                    'username': username,
                    'list_count': len(memberships),
                    'word_counts': word_counts,
                    'memberships': memberships
                })
            except Exception as exc:
                logger.warning("Profile analysis failed (%s)", type(exc).__name__)
                self.send_json({'success': False, 'error': 'Profile analysis failed'}, status=502)
            return
        
        if parsed.path == '/api/save-config':
            try:
                self.save_config(data)
                if hasattr(DashHandler, '_ai_cache_time'): DashHandler._ai_cache_time = 0
                if hasattr(DashHandler, '_x_cache_time'): DashHandler._x_cache_time = 0
                self.send_json({'success': True})
            except (OSError, TypeError, ValueError) as exc:
                logger.warning("Configuration update failed (%s)", type(exc).__name__)
                self.send_json({'success': False, 'error': 'Configuration could not be saved'}, status=400)
        elif parsed.path == '/api/save-cookies':
            try:
                if data.get('clear') is True:
                    COOKIES_PATH.unlink(missing_ok=True)
                else:
                    existing = self._load_cookie_credentials() or {}
                    for key in ('auth_token', 'ct0'):
                        value = data.get(key)
                        if isinstance(value, str) and value.strip():
                            existing[key] = value
                    if not all(existing.get(key) for key in ('auth_token', 'ct0')):
                        self.send_json(
                            {
                                'success': False,
                                'error': 'Both auth_token and ct0 are required to replace invalid cookies',
                            },
                            status=400,
                        )
                        return
                    atomic_write_json(COOKIES_PATH, existing)
                if hasattr(DashHandler, '_x_cache_time'): DashHandler._x_cache_time = 0
                self.send_json({'success': True})
            except OSError as exc:
                logger.warning("Cookie update failed (%s)", type(exc).__name__)
                self.send_json({'success': False, 'error': 'Authentication could not be saved'}, status=500)
        elif parsed.path == '/api/run':
            with self._run_lock:
                if self.app_state.get('running'):
                    should_start = False
                else:
                    should_start = True
                    self.app_state.update({
                        'running': True,
                        'run_id': None,
                        'progress': 0,
                        'status_msg': 'Starting...',
                        'error': None,
                    })
            if should_start:
                threading.Thread(target=self.run_task, daemon=True).start()
                self.send_json({'success': True})
            else:
                self.send_json({'success': False, 'error': 'Already running'}, status=409)
        else:
            self.send_error(404)

    def load_config(self):
        return load_app_config(CONFIG_PATH)

    def save_config(self, config):
        current = load_app_config(CONFIG_PATH, use_environment=False)
        save_app_config(config, CONFIG_PATH, current=current)

    def save_history_metadata(self, filename, meta):
        meta_path = OUTPUT_DIR / 'history.json'
        data = {}
        if meta_path.exists():
            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    data = loaded
            except (OSError, json.JSONDecodeError, TypeError):
                logger.warning("History metadata could not be read; rebuilding from reports")
        
        data[filename] = meta
        
        # Clean up stale entries (if file doesn't exist)
        cleaned = {}
        for fname, info in data.items():
            if self._resolve_report_path(fname) is not None and isinstance(info, dict):
                cleaned[fname] = info
        
        atomic_write_json(meta_path, cleaned)

    async def _run_async_task(self):
        """Delegate the digest workflow to the injectable task runner."""
        secrets = []
        fetcher = None
        article_service = None
        try:
            config = self.load_config()
            twitter_config = config.get('twitter', {})
            summary_config = config.get('summarization', {})
            for option in summary_config.get('options', {}).values():
                if isinstance(option, dict) and option.get('api_key'):
                    secrets.append(option['api_key'])
            if twitter_config.get('api_bearer_token'):
                secrets.append(twitter_config['api_bearer_token'])
            if COOKIES_PATH.exists():
                try:
                    cookie_data = json.loads(COOKIES_PATH.read_text(encoding='utf-8'))
                    if isinstance(cookie_data, dict):
                        secrets.extend(
                            value for value in cookie_data.values()
                            if isinstance(value, str) and value
                        )
                except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                    logger.warning("Cookie secrets could not be loaded for error redaction")

            fetcher = _build_fetcher(config, list_owner=twitter_config.get('list_owner'))
            summary_provider = LLMProvider(config)
            provider_name = str(summary_config.get('provider', ''))
            model = summary_config.get('options', {}).get(provider_name, {}).get('model', '')
            ai_label = f"{provider_name.capitalize()} · {model}" if model else provider_name.capitalize()

            def update_progress(event: TaskProgress):
                self.app_state.update({
                    'run_id': event.run_id,
                    'status_msg': event.message,
                    'progress': event.percent,
                    'error': None,
                })

            list_urls = twitter_config.get('list_urls', [])
            incremental = bool(
                twitter_config.get('fetch_method') in {'twikit', 'browser_session', 'api'}
                and twitter_config.get('incremental_sync', True)
                and callable(getattr(fetcher, 'fetch_page', None))
            )
            configured_data_dir = Path(
                config.get('storage', {}).get('data_dir', 'data')
            ).expanduser()
            data_dir = (
                configured_data_dir
                if configured_data_dir.is_absolute()
                else ROOT_DIR / configured_data_dir
            )
            post_store = FilePostStore(data_dir) if incremental else None
            articles_config = config.get('articles', {})
            if incremental and articles_config.get('enabled', True):
                article_service = ArticleService(
                    store=FileArticleStore(
                        data_dir,
                        cache_ttl_hours=articles_config.get('cache_ttl_hours', 168),
                    ),
                    fetcher=ArticleFetcher(
                        config=ArticleFetchConfig.from_mapping(articles_config)
                    ),
                    config=ArticleServiceConfig.from_mapping(articles_config),
                )
            resumable_run = (
                _find_resumable_report_run(post_store)
                if post_store is not None
                else None
            )
            runner = DigestTaskRunner(
                fetcher=fetcher,
                summary_provider=summary_provider,
                progress_callback=update_progress,
                sensitive_values=secrets,
                max_concurrency=max(1, len(list_urls)),
                fetch_delay_factory=lambda index: index * (0.3 + random.random() * 0.5),
                post_store=post_store,
                article_service=article_service,
            )
            request = DigestTaskRequest.from_values(
                resumable_run.requested_lists if resumable_run is not None else list_urls,
                max_tweets=twitter_config.get('max_tweets', 100),
                output_dir=OUTPUT_DIR,
                ai_model=ai_label,
                incremental=incremental,
                data_dir=data_dir,
                page_size=twitter_config.get('page_size', 100),
                max_pages=twitter_config.get('max_pages', 20),
                initial_fetch_limit=twitter_config.get('initial_fetch_limit', 100),
                resume_ingestion_run_id=(
                    resumable_run.run_id if resumable_run is not None else None
                ),
            )
            result = await runner.run(request)
            if not result.success:
                raise RuntimeError(result.error or 'Digest task failed')

            update = {
                'run_id': result.run_id,
                'progress': 100,
                'status_msg': (
                    'No new posts'
                    if result.status == 'no_new_posts'
                    else ('Complete with warnings' if result.partial else 'Complete!')
                ),
                'error': None,
            }
            if result.report_path:
                update['last_report'] = result.report_path.name
            self.app_state.update(update)
        except Exception as e:
            err_msg = redact_sensitive_text(str(e), secrets=secrets)
            # Only rewrite generic low-level rate limit errors that weren't already formatted above
            if ('429' in err_msg or 'rate limit' in err_msg.lower()) and 'Please' not in err_msg:
                err_msg = "X Rate Limit Reached. Please wait 15 minutes before trying again."

            logger.error("Digest task failed (%s)", type(e).__name__)
            self.app_state.update({
                'status_msg': 'Error',
                'progress': 0,
                'error': err_msg,
                'running': False,
            })
        finally:
            if article_service is not None:
                try:
                    await article_service.aclose()
                except Exception:
                    logger.warning("Could not close article fetcher")
            close_provider = getattr(fetcher, 'close_ingestion_provider', None)
            if callable(close_provider):
                try:
                    await close_provider()
                except Exception:
                    logger.warning("Could not close X API ingestion provider")
            self.app_state['running'] = False

    def run_task(self):
        asyncio.run(self._run_async_task())

    def get_reconstructed_html(self):
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>X List Summarizer v1.7</title>
    <script>
        (() => {
            try {
                const savedTheme = localStorage.getItem('xls-theme');
                if (savedTheme === 'light' || savedTheme === 'dark') {
                    document.documentElement.dataset.theme = savedTheme;
                }
            } catch (_) {
                // Storage can be unavailable in privacy-restricted browsers.
            }
        })();
    </script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #0b0e14;
            --card: #151921;
            --header: #0f1219;
            --border: #232a35;
            --text: #eff3f4;
            --text-dim: #949ba4;
            --accent: #1d9bf0;
            --accent-hover: #1a8cd8;
            --green: #00ba7c;
            --red: #f4212e;
            --blue-tip: #1d9bf01a;
            --surface-strong: #000;
            --surface-input: #080a0f;
            --surface-elevated: #151921;
            --surface-action: #1e232b;
            --surface-action-hover: #252b36;
            --surface-muted: rgba(255, 255, 255, 0.03);
            --surface-muted-hover: rgba(255, 255, 255, 0.06);
            --overlay: rgba(0, 0, 0, 0.85);
            --nav-active-text: #fff;
            --on-accent: #fff;
        }
        :root[data-theme="light"] {
            color-scheme: light;
            --bg: #f5f7fa;
            --card: #ffffff;
            --header: #ffffff;
            --border: #d8dee8;
            --text: #111827;
            --text-dim: #5f6b7a;
            --accent: #0878c9;
            --accent-hover: #0668ad;
            --green: #087f5b;
            --red: #d92d3a;
            --blue-tip: rgba(8, 120, 201, 0.08);
            --surface-strong: #eef2f7;
            --surface-input: #ffffff;
            --surface-elevated: #ffffff;
            --surface-action: #eef2f7;
            --surface-action-hover: #e2e8f0;
            --surface-muted: rgba(15, 23, 42, 0.04);
            --surface-muted-hover: rgba(15, 23, 42, 0.08);
            --overlay: rgba(15, 23, 42, 0.58);
            --nav-active-text: #0f172a;
            --on-accent: #ffffff;
        }
        * { box-sizing: border-box; }
        body { 
            font-family: 'Inter', sans-serif; 
            background-color: var(--bg); color: var(--text); 
            margin: 0; min-height: 100vh;
        }

        /* Header Precision Alignment */
        header {
            display: flex; align-items: center; justify-content: center;
            padding: 10px 40px; min-height: 90px; height: auto;
            background: var(--header); border-bottom: 1px solid var(--border);
            position: sticky; top: 0; z-index: 100;
        }
        .main-nav {
            display: flex;
            align-items: center;
            justify-content: space-between;
            width: 100%;
            max-width: 1400px;
            gap: 20px;
            white-space: nowrap;
            flex-wrap: wrap;
            min-width: 0;
        }
        .logo-area { display: flex; align-items: center; gap: 14px; cursor: pointer; flex-shrink: 0; }
        .logo-box { 
            width: 64px; height: 64px; border-radius: 16px; 
            background: url("icon.png") center/cover;
            box-shadow: 0 0 20px rgba(29, 155, 240, 0.2);
        }
        .version { font-size: 11px; font-weight: 800; color: var(--text-dim); }

        .middle-section {
            display: flex; align-items: center; gap: 15px;
            flex: 1 1 420px; min-width: 0;
        }
        .status-container {
            background: var(--surface-strong);
            border: 1px solid var(--border);
            border-radius: 50px;
            padding: 5px 5px 5px 24px;
            display: flex; align-items: center; flex-wrap: wrap; gap: 8px;
            min-width: 0; width: 100%;
        }
        .status-label { font-size: 13px; font-weight: 700; color: var(--text-dim); white-space: normal; overflow-wrap: anywhere; flex: 1 1 120px; min-width: 0; }
        .inline-p-con { width: 100px; height: 4px; background: var(--border); border-radius: 10px; margin: 0 20px; display: none; overflow: hidden; }
        .inline-p-bar { height: 100%; width: 0%; background: var(--accent); border-radius: 10px; transition: 0.4s; }

        .run-btn {
            background: linear-gradient(135deg, #1d9bf0 0%, #1a8cd8 100%);
            color: var(--on-accent); border: none; padding: 12px 28px; border-radius: 40px;
            font-weight: 800; cursor: pointer; display: flex; align-items: center; gap: 10px;
            transition: 0.2s; box-shadow: 0 5px 15px rgba(29, 155, 240, 0.35);
            font-size: 14px; margin-left: auto;
        }
        .run-btn:hover { transform: translateY(-1px); filter: brightness(1.1); }
        #run-btn { flex: 0 0 auto; }
        
        .status-pill { 
            display: flex; align-items: center; gap: 10px; font-size: 12px; font-weight: 700; 
            background: rgba(255, 255, 255, 0.05); padding: 11px 20px; border-radius: 40px;
            border: 1px solid var(--border); color: var(--text-dim);
            height: 48px; white-space: nowrap; flex-shrink: 0;
        }
        .dot { width: 8px; height: 8px; border-radius: 50%; }
        .dot.active { background: var(--green); box-shadow: 0 0 10px var(--green); }
        .dot.error { background: var(--red); box-shadow: 0 0 10px var(--red); }

        .nav-links { display: flex; align-items: center; gap: 10px; flex: 0 1 auto; flex-wrap: wrap; min-width: 0; }
        .nav-link { 
            color: var(--text-dim); text-decoration: none; font-weight: 700; font-size: 14px; 
            cursor: pointer; transition: 0.2s;
            display: flex; align-items: center;
            padding: 10px 18px; border-radius: 12px;
            white-space: nowrap;
        }
        .nav-link:hover { color: var(--text); background: var(--surface-muted); }
        .guide-list { margin-top: 15px; padding-left: 20px; }
        .guide-list li { margin-bottom: 10px; color: var(--text-dim); line-height: 1.6; }
        .nav-link.active { 
            color: var(--nav-active-text);
            background: #1d9bf025;
            border: 1px solid #1d9bf040;
        }
        .feature-grid { min-width: 0; }

        /* Settings Grid Layout */
        .container { max-width: 1200px; margin: 40px auto; padding: 0 40px; }
        .settings-grid { display: grid; grid-template-columns: 1fr 360px; gap: 40px; align-items: start; }
        
        .card { background: var(--card); border: 1px solid var(--border); border-radius: 24px; padding: 32px; margin-bottom: 32px; }
        .sec-title { display: flex; align-items: center; gap: 12px; font-size: 20px; font-weight: 700; margin-bottom: 25px; }
        
        label { display: block; font-size: 13px; font-weight: 600; color: var(--text-dim); margin-bottom: 12px; }
        input, select, textarea { 
            width: 100%; background: var(--surface-input); border: 1px solid var(--border); color: var(--text);
            padding: 15px 18px; border-radius: 12px; margin-bottom: 20px; font-family: inherit; font-size: 14px;
        }
        input:focus, textarea:focus { border-color: var(--accent); outline: none; }
        .hint { font-size: 12px; color: var(--text-dim); margin-top: -15px; margin-bottom: 20px; display: block; }

        /* Sync Tip Box */
        .tip-box { 
            background: var(--blue-tip); border: 1px solid #1d9bf030; border-radius: 12px; 
            padding: 20px; margin-bottom: 25px; 
        }
        .tip-title { color: var(--accent); font-weight: 700; font-size: 13px; margin-bottom: 12px; }
        .tip-list { margin: 0; padding-left: 18px; font-size: 12px; color: var(--text-dim); line-height: 1.8; }

        .btn-full { width: 100%; justify-content: center; }
        .btn-save { background: var(--surface-muted); border: 1px solid var(--border); color: var(--text); }
        .btn-save:hover { background: var(--surface-muted-hover); }

        /* ProgressOverlay */
        #progress-overlay {
            position: fixed; inset: 0; background: var(--overlay); display: none; align-items: center; justify-content: center; z-index: 1000;
        }
        .p-box { width: 440px; background: var(--card); border: 1px solid var(--border); padding: 48px; border-radius: 32px; text-align: center; }
        .p-bar-con { height: 8px; background: var(--surface-strong); border-radius: 10px; margin: 30px 0; overflow: hidden; }
        .p-bar { height: 100%; background: var(--accent); width: 0%; transition: 0.4s; }

        /* Storage & History Styling */
        .storage-card { 
            border: 1px dashed #1d9bf080; background: rgba(29, 155, 240, 0.04); 
            border-radius: 12px; padding: 40px; margin-bottom: 50px;
        }
        .path-display { 
            background: var(--surface-strong); border: 1px solid var(--border); border-radius: 8px;
            padding: 18px 25px; font-family: 'Consolas', monospace; font-size: 13px; color: var(--text-dim);
            margin: 25px 0; width: 100%;
        }
        .history-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 30px; }
        .history-title { font-size: 22px; font-weight: 800; display: flex; align-items: center; gap: 14px; }
        .report-count { font-size: 13px; color: var(--text-dim); }

        .report-card { 
            background: var(--surface-elevated); border: 1px solid var(--border); border-radius: 16px;
            padding: 30px 40px; margin-bottom: 24px; display: flex; justify-content: space-between; align-items: center;
        }
        .report-info .r-title { font-weight: 800; font-size: 19px; color: var(--accent); margin-bottom: 10px; display: block; }
        .report-info .r-date { font-size: 14px; color: var(--text-dim); font-weight: 500; }
        
        .report-actions { display: flex; gap: 15px; }
        .btn-action { 
            background: var(--surface-action); border: 1px solid var(--border); color: var(--text);
            padding: 10px 22px; border-radius: 8px; font-size: 13px; font-weight: 700; cursor: pointer;
            transition: 0.2s; display: flex; align-items: center; gap: 10px;
        }
        .btn-action:hover { background: var(--surface-action-hover); border-color: var(--border); }
        .icon-small { font-size: 14px; opacity: 0.8; }
        .h-img { width: 44px; height: 44px; border-radius: 50%; border: 1px solid var(--border); margin-right: 15px; flex-shrink: 0; }
        .report-info-con { display: flex; align-items: center; flex: 1; }

        .tab-content { display: none; }
        .tab-content.active { display: block; animation: fadeIn 0.3s ease-out; }
        
        /* Profiler Word Cloud Styles */
        .cloud-word {
            transition: 0.3s;
            cursor: pointer;
            padding: 8px 15px;
            border-radius: 12px;
            display: inline-block;
            font-weight: 700;
            background: var(--surface-muted);
            border: 1px solid var(--border);
            user-select: none;
        }
        .cloud-word:hover {
            transform: scale(1.15) rotate(2deg);
            background: rgba(29, 155, 240, 0.15);
            border-color: var(--accent);
            color: #fff !important;
            z-index: 10;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        }
        .cloud-word.active {
            background: var(--accent);
            color: #fff !important;
            border-color: var(--accent);
            box-shadow: 0 0 20px rgba(29, 155, 240, 0.4);
        }

        .prof-detail-card {
            background: var(--surface-muted);
            border: 1px solid var(--border);
            border-radius: 16px;
            overflow: hidden;
            margin-top: 25px;
            animation: fadeIn 0.4s ease-out;
        }
        .prof-table { width: 100%; border-collapse: collapse; }
        .prof-table th { background: var(--surface-muted); padding: 15px; text-align: left; font-size: 11px; text-transform: uppercase; color: var(--text-dim); }
        .prof-table td { padding: 15px; border-top: 1px solid var(--border); font-size: 14px; }
        .prof-table tr:hover { background: var(--surface-muted); }
        .word-tag { 
            background: var(--accent); color: #fff; padding: 4px 12px; border-radius: 20px; 
            font-size: 13px; font-weight: 800; display: inline-block; margin-bottom: 20px;
        }

        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes floatIn { from { opacity: 0; transform: scale(0.5) translateZ(-100px); } to { opacity: 1; transform: scale(1) translateZ(0); } }

        /* Modal Styles */
        .modal {
            display: none;
            position: fixed;
            z-index: 2000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.9);
            backdrop-filter: blur(10px);
            cursor: zoom-out;
            align-items: center; justify-content: center;
        }
        .modal-content {
            margin: auto;
            display: block;
            max-width: 90%;
            max-height: 90%;
            border-radius: 12px;
            box-shadow: 0 0 50px rgba(0,0,0,0.5);
            cursor: default;
        }
        .close-modal {
            position: absolute;
            top: 30px;
            right: 50px;
            color: #fff;
            font-size: 40px;
            font-weight: bold;
            cursor: pointer;
        }
        .enlarge-hint {
            font-size: 11px;
            color: var(--accent);
            text-align: center;
            margin-top: 8px;
            font-weight: 700;
            cursor: pointer;
        }

        /* Ranking Modal Specifics */
        .rank-table { width: 100%; border-collapse: collapse; margin-top: 20px; color: var(--text); }
        .rank-table th { text-align: left; padding: 12px; border-bottom: 2px solid var(--border); color: var(--accent); font-size: 13px; text-transform: uppercase; }
        .rank-table td { padding: 15px 12px; border-bottom: 1px solid var(--border); font-size: 14px; line-height: 1.4; }
        .rank-num { font-weight: 800; color: var(--accent); font-size: 18px; }
        .info-trigger { 
            cursor: pointer; width: 22px; height: 22px; border-radius: 50%; 
            background: var(--accent-dim); color: var(--accent); 
            display: inline-flex; align-items: center; justify-content: center; 
            font-size: 14px; font-weight: 800; border: 1px solid #1d9bf030;
            transition: 0.2s;
        }
        .info-trigger:hover { background: var(--accent); color: #fff; transform: scale(1.1); }

        .theme-toggle {
            position: fixed;
            right: 24px;
            bottom: 24px;
            z-index: 1500;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            min-height: 44px;
            padding: 10px 16px;
            border: 1px solid var(--border);
            border-radius: 999px;
            background: var(--card);
            color: var(--text);
            box-shadow: 0 8px 28px rgba(15, 23, 42, 0.18);
            font: inherit;
            font-size: 13px;
            font-weight: 800;
            cursor: pointer;
        }
        .theme-toggle:hover { background: var(--surface-action-hover); }
        .theme-toggle:focus-visible { outline: 3px solid rgba(29, 155, 240, 0.35); outline-offset: 2px; }
        :root[data-theme="light"] #prof_user,
        :root[data-theme="light"] #meth_toggle_btn {
            background: var(--surface-input) !important;
            color: var(--text) !important;
        }
        :root[data-theme="light"] #profiler > .card {
            background: var(--card) !important;
        }

        @media (max-width: 768px) {
            header { padding: 10px 16px; }
            .main-nav { gap: 12px; }
            .logo-box { width: 52px; height: 52px; border-radius: 13px; }
            .nav-links { flex: 1 1 auto; justify-content: flex-end; gap: 4px; }
            .nav-link { padding: 9px 10px; font-size: 13px; }
            .middle-section { order: 3; flex: 1 1 100%; width: 100%; }
            .status-container { width: 100%; justify-content: flex-start; padding-left: 16px; }
            .container { margin-top: 24px; padding: 0 16px; }
            .settings-grid { grid-template-columns: 1fr; gap: 20px; }
            .feature-grid { grid-template-columns: 1fr !important; }
        }

        @media (max-width: 480px) {
            .nav-links { justify-content: flex-start; flex-basis: 100%; }
            .middle-section { align-items: stretch; }
            .status-container { border-radius: 20px; padding: 8px; }
            .status-label { flex-basis: 100%; padding: 4px 8px; }
            .inline-p-con { flex: 1 1 100%; margin: 4px 8px; }
            #run-btn { width: 100%; margin-left: 0; justify-content: center; }
            .theme-toggle { right: 12px; bottom: 12px; }
        }

    </style>
</head>
<body>
    <button type="button" class="theme-toggle" id="theme-toggle" onclick="toggleTheme()" aria-label="切换到白天模式" aria-pressed="false">
        <span id="theme-toggle-icon" aria-hidden="true">☀️</span>
        <span id="theme-toggle-label">白天模式</span>
    </button>
    <header>
        <div class="main-nav">
            <div class="logo-area" onclick="resetApp()">
                <div class="logo-box"></div>
            </div>

            <div class="middle-section">
                <div class="status-container">
                    <span class="status-label" id="run-status">Preparing...</span>
                    <div class="inline-p-con" id="inline-progress">
                        <div class="inline-p-bar" id="inline-p-bar"></div>
                    </div>
                    <button class="run-btn" id="run-btn" onclick="startAnalysis()">
                        <span style="font-size:11px;">▶</span> Run Analysis
                    </button>
                </div>
            </div>

            <div class="nav-links">
                <a class="nav-link active" id="nav-home" href="javascript:void(0)" onclick="showTab('home')">Dashboard</a>
                <a class="nav-link" id="nav-report" href="javascript:void(0)" onclick="viewLatest()">Report</a>
                <a class="nav-link" id="nav-history" href="javascript:void(0)" onclick="showTab('history')">History</a>
                <a class="nav-link" id="nav-profiler" href="javascript:void(0)" onclick="showTab('profiler')">Profiler</a>
                <a class="nav-link" id="nav-settings" href="javascript:void(0)" onclick="showTab('settings')">Settings</a>
            </div>
        </div>
    </header>

    <div id="home" class="container tab-content active">
        <div style="text-align:center; margin: 60px 0 80px;">
            <h1 style="font-size: 52px; font-weight: 800; margin-bottom: 25px;">X List Summarizer <span style="font-size: 18px; opacity: 0.6; font-weight: 600; margin-left: 10px;">v1.7.0</span></h1>
            <p style="font-size: 18px; color: var(--text-dim); line-height: 1.6; max-width: 650px; margin: 0 auto;">Turn the noise of X into actionable intelligence. This premium tool analyzes curated lists to extract high-signal trends and media.</p>
        </div>
        <div class="feature-grid" style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 24px;">
            <div class="card" style="padding: 35px; border-radius: 28px;">
                <span style="font-size: 36px; display: block; margin-bottom: 20px;">🔍</span>
                <div style="font-weight: 800; font-size: 19px; margin-bottom: 12px;">Deep Extraction</div>
                <div style="font-size: 14px; color: var(--text-dim); line-height: 1.6;">Recursively scans Retweets and Quote Tweets to capture shared links and deduplicated media, ensuring no high-signal content is missed.</div>
            </div>
            <div class="card" style="padding: 35px; border-radius: 28px;">
                <span style="font-size: 36px; display: block; margin-bottom: 20px;">📈</span>
                <div style="font-weight: 800; font-size: 19px; margin-bottom: 12px;">Power Scoring</div>
                <div style="font-size: 14px; color: var(--text-dim); line-height: 1.6;">Identifies trending topics via a weighted algorithm (Likes + RTs + Replies + Quotes + Bookmarks) to filter out low-value noise.</div>
            </div>
            <div class="card" style="padding: 35px; border-radius: 28px;">
                <span style="font-size: 36px; display: block; margin-bottom: 20px;">🤖</span>
                <div style="font-weight: 800; font-size: 19px; margin-bottom: 12px;">AI Intelligence</div>
                <div style="font-size: 14px; color: var(--text-dim); line-height: 1.6;">Harnesses xAI Grok, Claude, and Llama 3 to synthesize hundreds of posts into structured reports with explicit model labeling.</div>
            </div>
        </div>
        <div class="card" style="text-align: center; background: linear-gradient(135deg, rgba(29,155,240,0.06), rgba(29,155,240,0.02)); border: 1px solid rgba(29,155,240,0.15); margin-top: 40px; padding: 45px;">
            <div style="font-weight: 800; font-size: 20px; margin-bottom: 15px;">Ready to begin?</div>
            <div style="font-size: 15px; color: var(--text-dim);">Ensure your <strong>X Authentication</strong> and <strong>AI Model</strong> are configured in Settings, then click <strong>Run Analysis</strong> in the header to start.</div>
        </div>
    </div>

    <div id="profiler" class="container tab-content">
        <div style="text-align:center; margin-bottom: 40px;">
            <h1 style="font-size: 42px; font-weight: 800; margin-bottom: 15px;">Account Profiler</h1>
            <p style="color: var(--text-dim); font-size: 16px;">See how any account is categorized by the X community via list membership analysis.</p>
        </div>

        <div class="card" style="padding: 40px; text-align: center; background: linear-gradient(135deg, #151921 0%, #0b0e14 100%);">
            <div style="max-width: 500px; margin: 0 auto;">
                <div style="font-weight: 800; font-size: 18px; margin-bottom: 20px;">Search X Username</div>
                <div style="position: relative; display: flex; gap: 10px;">
                    <span style="position: absolute; left: 20px; top: 50%; transform: translateY(-50%); color: var(--accent); font-weight: 800; font-size: 18px;">@</span>
                    <input type="text" id="prof_user" placeholder="username" style="width: 100%; background: var(--surface-input); border: 1px solid var(--border); padding: 16px 16px 16px 45px; border-radius: 12px; color: var(--text); font-size: 16px; font-weight: 600; margin-bottom: 0;">
                    <button onclick="generateProfile()" id="prof_btn" class="run-btn" style="margin: 0; padding: 0 30px;">Analyze</button>
                </div>
            </div>
        </div>

        <div id="prof_results" style="display: none; margin-top: 30px;">
            <div class="card" style="padding: 30px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 30px; border-bottom: 1px solid var(--border); padding-bottom: 20px;">
                    <div>
                        <div style="font-size: 13px; color: var(--text-dim); font-weight: 800; text-transform: uppercase; letter-spacing: 1px;">Analysis Results for</div>
                        <div id="prof_res_user" style="font-size: 24px; font-weight: 800; color: var(--accent);">@username</div>
                    </div>
                    <div style="text-align: right;">
                        <div id="prof_res_count" style="font-size: 24px; font-weight: 800; color: var(--text);">0</div>
                        <div style="font-size: 11px; color: var(--text-dim); font-weight: 800; text-transform: uppercase;">List Memberships</div>
                        <a id="prof_x_link" href="#" target="_blank" style="font-size: 10px; color: var(--accent); text-decoration: none; font-weight: 800; display: none; margin-top: 5px;">VIEW ON X ↗</a>
                    </div>
                </div>
                
                <div id="word_cloud" style="min-height: 440px; display: flex; flex-wrap: wrap; align-items: center; justify-content: center; gap: 15px; padding: 30px; background: rgba(0,0,0,0.2); border-radius: 20px; position: relative; overflow: hidden; perspective: 1000px;">
                    <!-- Words will be injected here -->
                </div>

                <div id="prof_details" style="display: none; margin-top: 30px; border-top: 1px dashed var(--border); padding-top: 30px;">
                    <!-- List details will be injected here -->
                </div>
            </div>
        </div>
    </div>


    <div id="settings" class="container tab-content">
        <div style="margin-bottom: 30px; display: flex; gap: 15px; align-items: center; border-bottom: 1px solid var(--border); padding-bottom: 25px;">
            <div style="font-weight: 800; font-size: 20px;">⚙️ App Settings</div>
            <span style="font-size: 10px; font-weight: 800; color: var(--accent); background: var(--blue-tip); border: 1px solid #1d9bf030; padding: 3px 9px; border-radius: 20px; letter-spacing: 0.5px; text-transform: uppercase;">v1.7</span>
            <div style="display: flex; gap: 10px; margin-left: 20px;">
                <div class="status-pill" style="font-size: 11px; padding: 8px 16px; height: auto;">
                    <div id="settings-ai-dot" class="dot active"></div>
                    AI: <span id="settings-ai-txt">Ready</span>
                </div>
                <div class="status-pill" style="font-size: 11px; padding: 8px 16px; height: auto;">
                    <div id="settings-x-dot" class="dot active"></div>
                    X Auth: <span id="settings-x-txt">OK</span>
                </div>
            </div>
            <div style="margin-left: auto; display: flex; gap: 8px;">
                <a href="javascript:void(0)" onclick="toggleMethodology()" id="meth_toggle_btn" class="btn-action" style="font-size: 11px; padding: 6px 14px; background: #1d9bf020; border-color: #1d9bf040; color: #fff;">🧠 View Methodology</a>
            </div>
        </div>

        <!-- Integrated Methodology Section (Collapsible) -->
        <div id="methodology_sec" style="margin-bottom: 40px; border-bottom: 1px solid var(--border); padding-bottom: 40px; display: none;">
            <div style="text-align: center; margin-bottom: 40px; position: relative;">
                <button onclick="toggleMethodology(false)" style="position: absolute; right: 0; top: 0; background: transparent; border: 1px solid var(--border); color: var(--text-dim); padding: 8px 15px; border-radius: 8px; cursor: pointer; font-size: 12px; font-weight: 700;">✖ Close</button>
                <h2 style="font-size: 28px; font-weight: 800; margin-bottom: 10px;">Methodology & Under-the-Hood</h2>
                <p style="color: var(--text-dim); font-size: 14px;">Understanding how the X List Summarizer processes your data for maximum signal.</p>
            </div>

            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 24px;">
                <div class="card" style="margin-bottom: 0; padding: 25px;" id="meth_1">
                    <div style="font-size: 18px; font-weight: 800; margin-bottom: 12px; color: var(--accent); display: flex; align-items: center; gap: 10px;">
                        <span>📊</span> 1. Smart Fetching & Extraction
                    </div>
                    <p style="color: var(--text-dim); line-height: 1.5; font-size: 13px;">The app follows a <strong>"Latest-First"</strong> approach, fetching the newest content backward through history.</p>
                    <ul class="guide-list" style="font-size: 12px;">
                        <li><strong>Deep Link Extraction:</strong> We recursively scan <strong>Retweets and Quote Tweets</strong> to ensure shared links are tracked even when discussed indirectly.</li>
                        <li><strong>Deduplication:</strong> If the same tweet appears in multiple lists, it is only counted once for engagement math.</li>
                    </ul>
                </div>

                <div class="card" style="margin-bottom: 0; padding: 25px;" id="meth_2">
                    <div style="font-size: 18px; font-weight: 800; margin-bottom: 12px; color: var(--accent); display: flex; align-items: center; gap: 10px;">
                        <span>🧠</span> 2. Weighted Ranking
                    </div>
                    <p style="color: var(--text-dim); line-height: 1.5; font-size: 13px;">Low-signal noise is filtered using a weighted scoring algorithm for every grouped link:</p>
                    <div style="background: #000; padding: 10px; border-radius: 6px; font-family: monospace; font-size: 11px; color: var(--accent); margin: 12px 0; text-align: center;">
                        Likes + (RTs*1.5) + (Replies*2.0) + Quotes + Bookmarks
                    </div>
                    <ul class="guide-list" style="font-size: 12px;">
                        <li><strong>Report Visibility:</strong> The top 30 filtered link-groups are displayed in your report.</li>
                        <li><strong>AI Focus:</strong> We feed the top 20 groups to the AI for synthesis to ensure razor-sharp accuracy.</li>
                    </ul>
                </div>

                <div class="card" style="margin-bottom: 0; padding: 25px;" id="meth_3">
                    <div style="font-size: 18px; font-weight: 800; margin-bottom: 12px; color: var(--accent); display: flex; align-items: center; gap: 10px;">
                        <span>🎞️</span> 3. Media Deduplication
                    </div>
                    <p style="color: var(--text-dim); line-height: 1.5; font-size: 13px;">Reports are kept lightweight and professional through advanced media handling:</p>
                    <ul class="guide-list" style="font-size: 12px;">
                        <li><strong>Group Deduplication:</strong> Identical images or videos shared multiple times in a retweet chain are rendered only once per cluster.</li>
                        <li><strong>Click-to-Play:</strong> To bypass X's session-based video authentication, we render videos as clickable thumbnails that open the native tweet on X.</li>
                    </ul>
                </div>

                <div class="card" style="margin-bottom: 0; padding: 25px;" id="meth_4">
                    <div style="font-size: 18px; font-weight: 800; margin-bottom: 12px; color: var(--accent); display: flex; align-items: center; gap: 10px;">
                        <span>🤖</span> 4. Transparent AI Synthesis
                    </div>
                    <p style="color: var(--text-dim); line-height: 1.5; font-size: 13px;">The AI synthesizes the messy stream of raw tweets into cohesive narrative themes:</p>
                    <ul class="guide-list" style="font-size: 12px;">
                        <li><strong>Model Labeling:</strong> Reports now explicitly state the exact provider and model (e.g. Grok-3, Llama-3.3) used for the analysis.</li>
                        <li><strong>Domain Insight:</strong> Section C calculates the mention count and sentiment for trending domains.</li>
                    </ul>
                </div>
            </div>
            <div style="text-align: center; margin-top: 30px;">
                <button class="run-btn" style="background: rgba(255,255,255,0.05); border: 1px solid var(--border); font-size: 12px; padding: 10px 25px;" onclick="toggleMethodology(false)">✖ Close Methodology</button>
            </div>
        </div>

        <div class="settings-grid">
            <div class="left-col">
                <div class="card">
                    <div class="sec-title">📝 Lists & Sources</div>
                    <label>X List URLs (One per line)</label>
                    <textarea id="s_urls" rows="6" style="resize: none;"></textarea>
                    
                    <label>Max Tweets per List</label>
                    <input type="number" id="s_max" value="100">

                    <label>List Owner Username (Optional)</label>
                    <input type="text" id="s_owner" placeholder="Scobleizer">
                    <span class="hint">Use this if the owner is shown as "Unknown" in reports.</span>
                </div>

                <div class="card">
                    <div class="sec-title">📰 External Articles</div>
                    <label style="display:flex; gap:10px; align-items:center;">
                        <input type="checkbox" id="s_articles_enabled" style="width:auto; margin:0;">
                        Extract article text from external post links
                    </label>
                    <span class="hint" style="margin-top:8px;">Uses bounded HTTP requests without X cookies, browser automation, or JavaScript.</span>

                    <label>Max Articles per Run</label>
                    <input type="number" id="s_articles_max" min="1" max="100" value="20">

                    <label>Cache TTL (Hours)</label>
                    <input type="number" id="s_articles_ttl" min="1" value="168">

                    <label>Request Timeout (Seconds)</label>
                    <input type="number" id="s_articles_timeout" min="1" max="120" value="15">
                </div>

                <div class="card">
                    <div class="sec-title" style="justify-content: space-between;">
                        <div style="display: flex; align-items: center; gap: 12px;">
                            <span>🤖</span> AI Intelligence
                        </div>
                        <div class="info-trigger" onclick="openRankingModal()">?</div>
                    </div>
                    <label>Provider</label>
                    <select id="s_prov" onchange="renderProviderOptions()">
                        <option value="groq">Groq (Free Cloud)</option>
                        <option value="ollama">Ollama (Local)</option>
                        <option value="lmstudio">LM Studio (Local)</option>
                        <option value="claude">Anthropic Claude</option>
                        <option value="openai">OpenAI GPT-4o</option>
                        <option value="gemini">Google Gemini</option>
                        <option value="deepseek">DeepSeek (V3)</option>
                        <option value="grok">xAI Grok</option>
                        <option value="openrouter">OpenRouter (All Models)</option>
                    </select>

                    <div id="ai_help" class="tip-box" style="margin-top: -10px; margin-bottom: 25px; display: none;"></div>

                    <label>Model Name</label>
                    <select id="p_mod_select" onchange="toggleCustomModel()"></select>
                    <input type="text" id="p_mod_custom" placeholder="Enter custom model name..." style="display:none; margin-top: -10px;">

                    <div id="p_key_con">
                        <label>API Key</label>
                        <input type="password" id="p_key" placeholder="••••••••••••••••••••••••••••••••••••••••••••••••">
                        <label style="display:flex; gap:8px; align-items:center; font-size:12px; font-weight:600;">
                            <input type="checkbox" id="p_key_clear" style="width:auto; margin:0;"> Clear the saved API key
                        </label>
                    </div>

                    <button class="run-btn btn-full btn-save" onclick="saveConfig()">
                        <span>💾</span> Save App Configuration
                    </button>
                </div>
            </div>

            <div class="right-col" id="auth_sec">
                <div class="card">
                    <div class="sec-title">🔑 X Authentication</div>

                    <label>Fetch Method</label>
                    <select id="s_fetch_method" onchange="renderFetchMethod()">
                        <option value="twikit">Browser Session (twikit — free, ToS risk)</option>
                        <option value="api">Official X API (paid, stable)</option>
                    </select>
                    <span class="hint">Switch between free cookie-based scraping and the official X API v2.</span>

                    <div id="auth_twikit_sec" style="margin-top: 25px;">
                        <p style="font-size: 13px; color: var(--text-dim); line-height: 1.6; margin-bottom: 20px;">
                            Twikit uses cookies from a logged-in X.com session. Free, but may violate X's ToS and can break when X changes endpoints.
                        </p>

                        <label>auth_token</label>
                        <input type="password" id="s_token" placeholder="Paste auth_token">

                        <label>ct0</label>
                        <input type="text" id="s_ct0" placeholder="Paste ct0">

                        <label style="display:flex; gap:8px; align-items:center; font-size:12px; font-weight:600;">
                            <input type="checkbox" id="s_cookies_clear" style="width:auto; margin:0;"> Clear the saved cookies
                        </label>

                        <button class="run-btn btn-full btn-save" style="margin-top: 10px;" onclick="saveCookies()">
                            <span>💾</span> Save Cookies
                        </button>

                        <div class="tip-box" style="margin-top: 20px;">
                            <div class="tip-title">How to find these:</div>
                            <ul class="tip-list">
                                <li>Log in to <strong>x.com</strong> in Chrome/Edge</li>
                                <li>Press <strong>F12</strong> > <strong>Application</strong> tab</li>
                                <li>Under <strong>Cookies</strong>, select <strong>https://x.com</strong></li>
                                <li>Copy values for <strong>auth_token</strong> and <strong>ct0</strong></li>
                            </ul>
                            <img src="screenshots/auth_guide.png" onclick="openModal(this.src)" style="width: 100%; border-radius: 8px; margin-top: 15px; border: 1px solid var(--border); cursor: zoom-in;">
                            <div class="enlarge-hint" onclick="openModal('screenshots/auth_guide.png')">🔍 Click to enlarge image</div>
                        </div>
                    </div>

                    <div id="auth_api_sec" style="display:none; margin-top: 25px;">
                        <p style="font-size: 13px; color: var(--text-dim); line-height: 1.6; margin-bottom: 20px;">
                            Uses the official X API v2 with an app-only Bearer Token. <strong>Public lists only.</strong>
                            Link preview cards are not exposed by v2 (tweets still render, without rich article previews).
                        </p>

                        <label>Bearer Token</label>
                        <input type="password" id="s_bearer" placeholder="Paste your X API Bearer Token">
                        <label style="display:flex; gap:8px; align-items:center; font-size:12px; font-weight:600;">
                            <input type="checkbox" id="s_bearer_clear" style="width:auto; margin:0;"> Clear the saved Bearer Token
                        </label>
                        <span class="hint">Saved together with the other settings via <strong>Save App Configuration</strong>.</span>

                        <div class="tip-box" style="margin-top: 20px;">
                            <div class="tip-title">Setup &amp; Cost:</div>
                            <ul class="tip-list">
                                <li>Create a project at <a href="https://developer.x.com/en/portal/dashboard" target="_blank" style="color:var(--accent);">developer.x.com</a></li>
                                <li>Copy the <strong>Bearer Token</strong> from your app's Keys &amp; Tokens page</li>
                                <li><strong>Pay-per-use pricing:</strong> ~$0.001 per request (Owned Reads, Apr 2026)</li>
                                <li>~100 tweets per request; typical list fetch costs pennies/month</li>
                                <li>CLI helper: <a href="https://github.com/xdevplatform/xurl" target="_blank" style="color:var(--accent);">xurl</a> can generate &amp; test tokens</li>
                            </ul>
                        </div>
                    </div>

                </div>
            </div>
        </div>


    </div>

    <div id="history" class="container tab-content">
        <div class="storage-card">
            <div style="font-weight: 800; font-size: 16px; display: flex; align-items: center; gap: 12px;">
                <span style="color:#ffcc00; font-size: 18px;">📁</span> Storage Location
            </div>
            <p style="font-size: 14px; color: var(--text-dim); margin-top: 15px;">All your generated reports are stored on your local drive at:</p>
                    <div id="storage-path" class="path-display">C:\\...</div>
            <button class="run-btn" style="padding: 12px 24px; font-size: 13px;" onclick="openFolder()">
                🚀 Open Output Folder
            </button>
        </div>

        <div class="history-row">
            <div class="history-title"><span style="font-size: 20px;">📄</span> Report History</div>
            <div id="report-stats" class="report-count">Showing reports 0 - 0 of 0</div>
        </div>
        
        <div id="history-grid"></div>
    </div>



    <div id="report" class="container tab-content" style="max-width: 100%; padding: 0;">
        <iframe id="report-frame" sandbox="allow-scripts allow-popups allow-popups-to-escape-sandbox" referrerpolicy="no-referrer" style="width: 100%; height: calc(100vh - 90px); border: none;"></iframe>
    </div>

    <script>
        let cfg = { summarization: { options: {} }, twitter: { list_urls: [] } };

        function escapeHtml(value) {
            return String(value ?? '').replace(/[&<>"']/g, ch => ({
                '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
            })[ch]);
        }

        function safeHttpUrl(value, fallback = '') {
            try {
                const parsed = new URL(String(value ?? ''));
                return (parsed.protocol === 'http:' || parsed.protocol === 'https:') ? parsed.href : fallback;
            } catch (_) {
                return fallback;
            }
        }

        function safeReportFilename(value) {
            const name = String(value ?? '');
            return name === 'latest' || /^[A-Za-z0-9_.-]+[.]html$/.test(name) ? name : null;
        }

        function safeInteger(value) {
            const parsed = Number.parseInt(value, 10);
            return Number.isFinite(parsed) ? Math.max(0, parsed) : 0;
        }

        const THEME_STORAGE_KEY = 'xls-theme';

        function updateThemeButton() {
            const isLight = document.documentElement.dataset.theme === 'light';
            const button = document.getElementById('theme-toggle');
            const icon = document.getElementById('theme-toggle-icon');
            const label = document.getElementById('theme-toggle-label');
            if (!button || !icon || !label) return;
            icon.textContent = isLight ? '🌙' : '☀️';
            label.textContent = isLight ? '夜间模式' : '白天模式';
            button.setAttribute('aria-label', isLight ? '切换到夜间模式' : '切换到白天模式');
            button.setAttribute('aria-pressed', String(isLight));
        }

        function toggleTheme() {
            const nextTheme = document.documentElement.dataset.theme === 'light' ? 'dark' : 'light';
            document.documentElement.dataset.theme = nextTheme;
            try {
                localStorage.setItem(THEME_STORAGE_KEY, nextTheme);
            } catch (_) {
                // The visual switch still works when storage is unavailable.
            }
            updateThemeButton();
        }

        async function controlPost(path, payload = {}) {
            const response = await fetch(path, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const result = await response.json();
            if (!response.ok || !result.success) {
                throw new Error(result.error || 'Request failed');
            }
            return result;
        }

        async function openFolder() {
            try {
                await controlPost('/api/open-folder');
            } catch (error) {
                alert(error.message);
            }
        }
        
        // Critical: Ensure functions are available globally before everything else
        window.showTab = function(t) {
            console.log("Switching to tab:", t);
            const tabs = document.querySelectorAll('.tab-content');
            const navs = document.querySelectorAll('.nav-link');
            
            tabs.forEach(x => x.classList.remove('active'));
            navs.forEach(x => x.classList.remove('active'));
            
            const targetTab = document.getElementById(t);
            const targetNav = document.getElementById('nav-' + t);
            
            if (targetTab) targetTab.classList.add('active');
            if (targetNav) targetNav.classList.add('active');
            
            if (t === 'history') loadHistory().catch(e => console.error("History error:", e));
        };

        function resetApp() {
            const frame = document.getElementById('report-frame');
            if (frame) frame.src = 'about:blank';
            window.showTab('home');
        }

        let reportOpened = false;
        let lastKnownReport = null;
        
        async function poll() {
            try {
                const r = await fetch('/api/status');
                const s = await r.json();
                
                // Update status indicators (settings page)
                document.getElementById('settings-ai-dot').className = 'dot ' + (s.ai_status.active ? 'active' : 'error');
                document.getElementById('settings-ai-txt').innerText = s.ai_status.message;
                document.getElementById('settings-x-dot').className = 'dot ' + (s.x_auth.active ? 'active' : 'error');
                document.getElementById('settings-x-txt').innerText = s.x_auth.message;

                const statusEl = document.getElementById('run-status');
                const progressCon = document.getElementById('inline-progress');
                const progressBar = document.getElementById('inline-p-bar');
                const runBtn = document.getElementById('run-btn');

                // CRITICAL: Capture and auto-open report as soon as we see it
                if (s.last_report && s.last_report !== lastKnownReport) {
                    lastKnownReport = s.last_report;
                    if (!reportOpened) {
                        console.log("Report detected! Auto-opening:", s.last_report);
                        reportOpened = true;
                        loadInAppReport(s.last_report);
                    }
                }

                // Handle UI states
                if (s.error) {
                    statusEl.innerText = 'Error: ' + s.error;
                    statusEl.style.color = 'var(--red)';
                    statusEl.style.maxWidth = '400px';
                    progressCon.style.display = 'none';
                    runBtn.innerText = '✖ Clear';
                    runBtn.onclick = async () => {
                        await controlPost('/api/reset-progress');
                        location.reload();
                    };
                    runBtn.style.filter = 'none';
                    runBtn.disabled = false;
                } else if (s.running) {
                    statusEl.innerText = s.status_msg;
                    statusEl.style.color = 'var(--text-dim)';
                    progressCon.style.display = 'block';
                    progressBar.style.width = s.progress + '%';
                    runBtn.style.filter = 'grayscale(1) opacity(0.5)';
                    runBtn.disabled = true;
                    runBtn.innerHTML = '<span style="font-size:11px;">▶</span> Run Analysis';
                    runBtn.onclick = null;
                } else if (s.progress === 100) {
                    statusEl.innerText = s.status_msg || 'Complete!';
                    statusEl.style.color = 'var(--green)';
                    progressCon.style.display = 'none';
                    runBtn.style.filter = 'none';
                    runBtn.disabled = false;
                    runBtn.innerHTML = '<span style="font-size:11px;">▶</span> Run Analysis';
                    runBtn.onclick = startAnalysis;
                    
                    // Reset progress after delay
                    setTimeout(() => { controlPost('/api/reset-progress').catch(console.error); }, 3000);
                } else {
                    statusEl.innerText = 'Ready';
                    statusEl.style.color = 'var(--text-dim)';
                    progressCon.style.display = 'none';
                    runBtn.style.filter = 'none';
                    runBtn.disabled = false;
                    runBtn.innerHTML = '<span style="font-size:11px;">▶</span> Run Analysis';
                    runBtn.onclick = startAnalysis;
                }
            } catch(e) { console.error('Poll error:', e); }
        }
 
        function openModal(src) {
            const modal = document.getElementById('imageModal');
            const modalImg = document.getElementById('modalImg');
            modal.style.display = "flex";
            modalImg.src = src;
        }

        function closeModal() {
            document.getElementById('imageModal').style.display = "none";
        }

        async function viewLatest() {
            try {
                const r = await fetch('/api/status');
                const s = await r.json();
                const reportName = s.last_report || 'latest';
                loadInAppReport(reportName);
                const overlay = document.getElementById('progress-overlay');
                if (overlay) overlay.style.display = 'none';
            } catch(e) { console.error('viewLatest error:', e); }
        }

        function loadInAppReport(name) {
            const safeName = safeReportFilename(name);
            if (!safeName) {
                console.error('Rejected invalid report filename');
                return;
            }
            const frame = document.getElementById('report-frame');
            frame.src = '/output/' + encodeURIComponent(safeName);
            showTab('report');
        }


        async function loadConfig() {
            try {
                const r = await fetch('/api/config');
                cfg = await r.json();
                document.getElementById('s_urls').value = (cfg.twitter.list_urls || []).join('\\n');
                document.getElementById('s_max').value = cfg.twitter.max_tweets;
                document.getElementById('s_prov').value = cfg.summarization.provider;
                document.getElementById('s_owner').value = cfg.twitter.list_owner || '';
                document.getElementById('s_fetch_method').value = cfg.twitter.fetch_method || 'twikit';
                const articles = cfg.articles || {};
                document.getElementById('s_articles_enabled').checked = articles.enabled !== false;
                document.getElementById('s_articles_max').value = articles.max_articles_per_run || 20;
                document.getElementById('s_articles_ttl').value = articles.cache_ttl_hours || 168;
                document.getElementById('s_articles_timeout').value = articles.timeout_seconds || 15;
                const bearer = document.getElementById('s_bearer');
                bearer.value = '';
                bearer.placeholder = cfg.twitter.api_bearer_token_configured
                    ? `Configured (${cfg.twitter.api_bearer_token_mask || 'masked'}) — leave blank to keep`
                    : 'Paste your X API Bearer Token';
                document.getElementById('s_bearer_clear').checked = false;
                const cookieHint = cfg.twitter.cookies_configured
                    ? 'Configured — leave blank to keep'
                    : 'Paste value';
                document.getElementById('s_token').placeholder = `auth_token: ${cookieHint}`;
                document.getElementById('s_ct0').placeholder = `ct0: ${cookieHint}`;
                document.getElementById('s_cookies_clear').checked = false;
                renderFetchMethod();
                renderProviderOptions();
            } catch(e) { console.error('loadConfig error:', e); }
        }

        function toggleCustomModel() {
            const sel = document.getElementById('p_mod_select');
            const custom = document.getElementById('p_mod_custom');
            if (sel.value === 'custom') {
                custom.style.display = 'block';
            } else {
                custom.style.display = 'none';
            }
        }

        function renderProviderOptions() {
            const p = document.getElementById('s_prov').value;
            const data = cfg.summarization.options[p] || {};
            const sel = document.getElementById('p_mod_select');
            const custom = document.getElementById('p_mod_custom');
            
            const presets = {
                'groq': ['llama-3.3-70b-versatile', 'llama-3.1-8b-instant', 'openai/gpt-oss-120b'],
                'claude': ['claude-sonnet-4-6', 'claude-opus-4-6', 'claude-haiku-4-5'],
                'openai': ['gpt-4o', 'gpt-4.1', 'gpt-4o-mini', 'gpt-5'],
                'gemini': ['gemini-2.5-flash', 'gemini-2.5-flash-lite', 'gemini-2.5-pro', 'gemini-3-flash-preview'],
                'deepseek': ['deepseek-chat', 'deepseek-reasoner'],
                'grok': ['grok-3-latest', 'grok-2-latest', 'grok-beta'],
                'openrouter': ['google/gemini-2.5-flash', 'anthropic/claude-sonnet-4-6', 'deepseek/deepseek-chat', 'meta-llama/llama-3.3-70b-instruct'],
                'ollama': ['qwen2.5:7b', 'llama3.1', 'mistral', 'phi3'],
                'lmstudio': ['local-model']
            };

            const models = presets[p] || [];
            sel.innerHTML = models.map(m => `<option value="${m}">${m}</option>`).join('') + '<option value="custom">Custom...</option>';
            
            if (models.includes(data.model)) {
                sel.value = data.model;
                custom.style.display = 'none';
            } else if (data.model) {
                sel.value = 'custom';
                custom.value = data.model;
                custom.style.display = 'block';
            } else {
                sel.value = models[0] || 'custom';
                toggleCustomModel();
            }

            const keyInput = document.getElementById('p_key');
            if (keyInput) {
                keyInput.value = '';
                keyInput.placeholder = data.api_key_configured
                    ? `Configured (${data.api_key_mask || 'masked'}) — leave blank to keep`
                    : 'Paste API key';
            }
            document.getElementById('p_key_clear').checked = false;
            const keyCon = document.getElementById('p_key_con');
            if (p === 'ollama' || p === 'lmstudio') {
                keyCon.style.display = 'none';
            } else {
                keyCon.style.display = 'block';
            }

            const helpEl = document.getElementById('ai_help');
            const helpTexts = {
                'groq': '<strong>Setup Groq (Free Cloud):</strong><br>1. Get an API key from the <a href="https://console.groq.com/keys" target="_blank" style="color:var(--accent);">Groq Console</a>.<br>2. Recommended: <code>llama-3.3-70b-versatile</code> (fast, 128K context) or <code>openai/gpt-oss-120b</code> (highest capability)',
                'ollama': '<strong>Setup Ollama (Local):</strong><br>1. Ensure <a href="https://ollama.com" target="_blank" style="color:var(--accent);">Ollama</a> is running.<br>2. Run <code>ollama pull qwen2.5:7b</code> in your terminal.',
                'lmstudio': '<strong>Setup LM Studio (Local):</strong><br>1. Download <a href="https://lmstudio.ai/" target="_blank" style="color:var(--accent);">LM Studio</a>.<br>2. Load a model (e.g., <code>Qwen 2.5 7B</code>) and click <strong>Start Server</strong>.<br>3. Default endpoint: <code>http://localhost:1234/v1</code>',
                'claude': '<strong>Setup Claude:</strong><br>1. Get an API key from the <a href="https://console.anthropic.com/settings/keys" target="_blank" style="color:var(--accent);">Anthropic Console</a>.<br>2. Recommended: <code>claude-sonnet-4-6</code> (default, 1M context) or <code>claude-opus-4-6</code> (most powerful)',
                'openai': '<strong>Setup OpenAI:</strong><br>1. Get an API key from the <a href="https://platform.openai.com/api-keys" target="_blank" style="color:var(--accent);">OpenAI Platform</a>.<br>2. Recommended: <code>gpt-4.1</code> (best value) or <code>gpt-5</code> (most capable)',
                'gemini': '<strong>Setup Google Gemini:</strong><br>1. Get an API key from <a href="https://aistudio.google.com/app/apikey" target="_blank" style="color:var(--accent);">Google AI Studio</a>.<br>2. Recommended: <code>gemini-2.5-flash</code> (Fast, 1M context, reasoning)',
                'deepseek': '<strong>Setup DeepSeek:</strong><br>1. Get an API key from <a href="https://platform.deepseek.com/" target="_blank" style="color:var(--accent);">DeepSeek Platform</a>.<br>2. Recommended: <code>deepseek-chat</code> (V3, general) or <code>deepseek-reasoner</code> (chain-of-thought)',
                'grok': '<strong>Setup xAI Grok:</strong><br>1. Get an API key from the <a href="https://console.x.ai/" target="_blank" style="color:var(--accent);">xAI Console</a>.<br>2. Recommended: <code>grok-3-latest</code> (latest flagship) or <code>grok-2-latest</code> (fast, cost-effective). Earns 20% xAI credit back on X API spend.',
                'openrouter': '<strong>Setup OpenRouter:</strong><br>1. Get an API key from <a href="https://openrouter.ai/keys" target="_blank" style="color:var(--accent);">OpenRouter</a>.<br>2. Access any model through a single API key. Recommended: <code>google/gemini-2.5-flash</code>'
            };
            
            if (helpTexts[p]) {
                helpEl.innerHTML = '<div class="tip-title">Provider Guide:</div><div style="font-size:12px; line-height:1.6; color:var(--text-dim);">' + helpTexts[p] + '</div>';
                helpEl.style.display = 'block';
            } else {
                helpEl.style.display = 'none';
            }
        }

        async function saveConfig() {
            const p = document.getElementById('s_prov').value;
            const newCfg = JSON.parse(JSON.stringify(cfg));
            newCfg.summarization.provider = p;
            newCfg.twitter.list_urls = document.getElementById('s_urls').value.split('\\n').filter(x => x.trim());
            newCfg.twitter.max_tweets = Math.max(1, safeInteger(document.getElementById('s_max').value) || 100);
            newCfg.twitter.list_owner = document.getElementById('s_owner').value || null;
            newCfg.articles = newCfg.articles || {};
            newCfg.articles.enabled = document.getElementById('s_articles_enabled').checked;
            newCfg.articles.max_articles_per_run = Math.max(1, safeInteger(document.getElementById('s_articles_max').value) || 20);
            newCfg.articles.cache_ttl_hours = Math.max(1, safeInteger(document.getElementById('s_articles_ttl').value) || 168);
            newCfg.articles.timeout_seconds = Math.max(1, safeInteger(document.getElementById('s_articles_timeout').value) || 15);
            newCfg.summarization.options[p] = newCfg.summarization.options[p] || {};
            
            const sel = document.getElementById('p_mod_select');
            const custom = document.getElementById('p_mod_custom');
            newCfg.summarization.options[p].model = (sel.value === 'custom') ? custom.value : sel.value;

            if (document.getElementById('p_key_clear').checked) {
                newCfg.summarization.options[p].api_key_clear = true;
                newCfg.summarization.options[p].api_key = '';
            } else {
                newCfg.summarization.options[p].api_key = document.getElementById('p_key').value;
            }
            newCfg.twitter.fetch_method = document.getElementById('s_fetch_method').value;
            if (document.getElementById('s_bearer_clear').checked) {
                newCfg.twitter.api_bearer_token_clear = true;
                newCfg.twitter.api_bearer_token = '';
            } else {
                newCfg.twitter.api_bearer_token = document.getElementById('s_bearer').value;
            }
            const response = await fetch('/api/save-config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(newCfg)
            });
            const result = await response.json();
            if (!response.ok || !result.success) throw new Error(result.error || 'Settings could not be saved');
            await loadConfig();
            alert('Settings Saved');
        }

        function renderFetchMethod() {
            const m = document.getElementById('s_fetch_method').value;
            document.getElementById('auth_twikit_sec').style.display = (m === 'twikit') ? 'block' : 'none';
            document.getElementById('auth_api_sec').style.display = (m === 'api') ? 'block' : 'none';
        }

        async function saveCookies() {
            const cookies = {
                auth_token: document.getElementById('s_token').value,
                ct0: document.getElementById('s_ct0').value,
                clear: document.getElementById('s_cookies_clear').checked
            };
            const response = await fetch('/api/save-cookies', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(cookies)
            });
            const result = await response.json();
            if (!response.ok || !result.success) throw new Error(result.error || 'Authentication could not be saved');
            document.getElementById('s_token').value = '';
            document.getElementById('s_ct0').value = '';
            await loadConfig();
            alert('Authentication Updated');
        }

        async function loadHistory() {
            const r = await fetch('/api/history');
            const data = await r.json();
            
            const sr = await fetch('/api/status');
            const s = await sr.json();
            document.getElementById('storage-path').innerText = s.output_path;
            
            const count = data.length;
            document.getElementById('report-stats').innerText = `Showing reports 1 - ${Math.min(count, 10)} of ${count}`;
            
            document.getElementById('history-grid').innerHTML = data.map(h => {
                const filename = safeReportFilename(h.filename);
                if (!filename || filename === 'latest') return '';
                const dateObj = new Date(h.date.replace(' ', 'T'));
                // Format: February 02, 2026 at 21:26:05
                const formattedDate = dateObj.toLocaleDateString('en-US', { 
                    month: 'long', day: '2-digit', year: 'numeric' 
                }) + ' at ' + dateObj.toLocaleTimeString('en-US', { hour12: false });
                const fallbackProfile = 'https://abs.twimg.com/sticky/default_profile_images/default_profile_normal.png';
                const profileUrl = safeHttpUrl(h.profile_img, fallbackProfile);
                const reportUrl = '/output/' + encodeURIComponent(filename);

                return `
                <div class="report-card">
                    <div class="report-info-con">
                        <img src="${escapeHtml(profileUrl)}" class="h-img" alt="">
                        <div class="report-info">
                            <span class="r-title">${escapeHtml(h.name)}</span>
                            <div style="font-size: 13px; color: var(--text-dim); margin-bottom: 8px; font-weight: 600;">
                                @${escapeHtml(h.username)} • ${safeInteger(h.tweets)} tweets & ${safeInteger(h.links)} links • ${safeInteger(h.members)} members
                            </div>
                            <span class="r-date">${escapeHtml(formattedDate)}</span>
                        </div>
                    </div>
                    <div class="report-actions">
                        <button class="btn-action" onclick="loadInAppReport('${filename}')">
                            Preview <span class="icon-small">👁️</span>
                        </button>
                        <button class="btn-action" onclick="window.open('${reportUrl}', '_blank', 'noopener')">
                            External <span class="icon-small">↗️</span>
                        </button>
                    </div>
                </div>
            `}).join('');
        }

        let currentMemberships = [];

        async function generateProfile() {
            const user = document.getElementById('prof_user').value.trim();
            if (!user) return alert('Please enter a username');
            
            const btn = document.getElementById('prof_btn');
            const results = document.getElementById('prof_results');
            const cloud = document.getElementById('word_cloud');
            const details = document.getElementById('prof_details');
            
            btn.disabled = true;
            btn.innerText = 'Analyzing...';
            results.style.display = 'none';
            details.style.display = 'none';
            cloud.innerHTML = '';
            
            try {
                const r = await fetch('/api/profile', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ username: user })
                });
                const d = await r.json();
                
                if (!d.success) throw new Error(d.error);
                
                currentMemberships = d.memberships || [];
                
                document.getElementById('prof_res_user').innerText = '@' + d.username;
                document.getElementById('prof_res_count').innerText = d.list_count || 0;
                
                const xLink = document.getElementById('prof_x_link');
                xLink.href = `https://x.com/${encodeURIComponent(String(d.username || ''))}/lists/memberships`;
                xLink.style.display = 'block';
                
                // Render Word Cloud
                const counts = d.word_counts;
                const words = Object.keys(counts);
                
                if (words.length === 0) {
                    cloud.innerHTML = '<div style="color: var(--text-dim); font-weight: 600;">No lists found for this account.</div>';
                } else {
                    const maxCount = Math.max(...Object.values(counts));
                    const colors = ['#1d9bf0', '#00ba7c', '#ffd400', '#f91880', '#7856ff', '#ff7a00'];
                    
                    words.forEach((w, i) => {
                        const count = counts[w];
                        const size = 14 + (count / maxCount) * 36; // Scale between 14px and 50px
                        const color = colors[i % colors.length];
                        const opacity = 0.5 + (count / maxCount) * 0.5;
                        
                        const span = document.createElement('span');
                        span.className = 'cloud-word';
                        span.innerText = w;
                        span.style.fontSize = size + 'px';
                        span.style.color = color;
                        span.style.opacity = opacity;
                        span.style.animation = `floatIn 0.5s ease-out ${i * 0.02}s both`;
                        
                        span.onclick = () => showWordDetails(w, span);
                        
                        cloud.appendChild(span);
                    });
                }
                
                results.style.display = 'block';
            } catch (e) {
                alert('Analysis failed: ' + e.message);
            } finally {
                btn.disabled = false;
                btn.innerText = 'Analyze';
            }
        }

        function showWordDetails(word, el) {
            document.querySelectorAll('.cloud-word').forEach(s => s.classList.remove('active'));
            el.classList.add('active');
            
            const results = currentMemberships.filter(m => 
                String(m.name || '').toLowerCase().includes(String(word || '').toLowerCase())
            );
            
            const details = document.getElementById('prof_details');
            details.style.display = 'block';
            
            const rows = results.map(m => {
                const id = /^[0-9]+$/.test(String(m.id || '')) ? String(m.id) : null;
                const action = id
                    ? `<a href="https://x.com/i/lists/${id}" target="_blank" rel="noopener noreferrer" class="btn-action" style="padding: 6px 14px; font-size: 11px; display: inline-flex; text-decoration: none;">VIEW LIST</a>`
                    : '<span style="color:var(--text-dim)">Unavailable</span>';
                return `
                    <tr>
                        <td style="font-weight:700; color:var(--text)">${escapeHtml(m.name)}</td>
                        <td style="color:var(--text-dim)">@${escapeHtml(m.owner)}</td>
                        <td style="text-align:right">${action}</td>
                    </tr>`;
            }).join('');

            details.innerHTML = `
                <div class="word-tag"># ${escapeHtml(word)}</div>
                <div style="font-size: 13px; color: var(--text-dim); margin-bottom: 15px; font-weight: 600;">
                    Found in ${results.length} lists:
                </div>
                <div class="prof-detail-card">
                    <table class="prof-table">
                        <thead>
                            <tr>
                                <th>List Name</th>
                                <th>Owner</th>
                                <th style="text-align:right">Action</th>
                            </tr>
                        </thead>
                        <tbody>${rows}</tbody>
                    </table>
                </div>
            `;
            
            setTimeout(() => {
                details.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }, 100);
        }

        async function startAnalysis() {
            reportOpened = false;
            // Seed the baseline before starting so the preserved previous
            // report is not mistaken for this run's newly generated report.
            try {
                const response = await fetch('/api/status');
                if (response.ok) {
                    const status = await response.json();
                    lastKnownReport = status.last_report || null;
                }
            } catch (error) {
                console.debug('Could not seed report baseline', error);
            }
            await controlPost('/api/run');
        }

        function openRankingModal() {
            document.getElementById('rankingModal').style.display = 'flex';
        }
        function closeRankingModal() {
            document.getElementById('rankingModal').style.display = 'none';
        }

        function toggleMethodology(show, targetId) {
            const sec = document.getElementById('methodology_sec');
            const btn = document.getElementById('meth_toggle_btn');
            
            // If called without arguments, toggle current state
            const shouldShow = (show !== undefined) ? show : (sec.style.display === 'none');
            
            if (shouldShow) {
                sec.style.display = 'block';
                btn.innerHTML = '🧠 Hide Methodology';
                setTimeout(() => {
                    const scrollTarget = targetId ? document.getElementById(targetId) : sec;
                    scrollTarget.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }, 100);
            } else {
                sec.style.display = 'none';
                btn.innerHTML = '🧠 View Methodology';
                window.scrollTo({ top: 0, behavior: 'smooth' });
            }
        }

        updateThemeButton();
        loadConfig();
        setInterval(poll, 1500);
    </script>

    <!-- Image Modal -->
    <div id="imageModal" class="modal" onclick="closeModal()">
        <span class="close-modal" onclick="closeModal()">&times;</span>
        <img class="modal-content" id="modalImg" onclick="event.stopPropagation()">
    </div>

    <!-- Ranking Modal -->
    <div id="rankingModal" class="modal" onclick="closeRankingModal()">
        <span class="close-modal" onclick="closeRankingModal()">&times;</span>
        <div class="modal-content card" style="max-width: 800px; cursor: default; padding: 40px;" onclick="event.stopPropagation()">
            <h2 style="margin-top: 0; font-size: 28px; font-weight: 800; border-bottom: 1px solid var(--border); padding-bottom: 20px;">
                Intelligence Provider Ranking
            </h2>
            <p style="color: var(--text-dim); font-size: 14px; line-height: 1.6; margin-bottom: 25px;">
                Based on latency, context window size, and instruction-following for summarization tasks.
            </p>
            <table class="rank-table">
                <thead>
                    <tr>
                        <th style="width: 60px;">Rank</th>
                        <th>Provider</th>
                        <th>Why it belongs here</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td class="rank-num">1</td>
                        <td><strong>Groq</strong><br><span style="font-size:11px; color:var(--text-dim);">Llama 3.3 70B</span></td>
                        <td><strong>Speed King.</strong> Near-instant reporting. Best for quick summaries.</td>
                    </tr>
                    <tr>
                        <td class="rank-num">2</td>
                        <td><strong>Gemini</strong><br><span style="font-size:11px; color:var(--text-dim);">1.5 Flash</span></td>
                        <td><strong>Context King.</strong> 1.5M token window. Can summarize 1,000+ tweets without truncation.</td>
                    </tr>
                    <tr>
                        <td class="rank-num">3</td>
                        <td><strong>Claude</strong><br><span style="font-size:11px; color:var(--text-dim);">3.5 Sonnet</span></td>
                        <td><strong>Writing Quality.</strong> Best synthesis and capture of conversational nuance.</td>
                    </tr>
                    <tr>
                        <td class="rank-num">4</td>
                        <td><strong>Grok</strong><br><span style="font-size:11px; color:var(--text-dim);">Grok-3</span></td>
                        <td><strong>The Super-Model.</strong> Deeply integrated with X content. Unrivaled reasoning and freshness.</td>
                    </tr>
                    <tr>
                        <td class="rank-num">5</td>
                        <td><strong>DeepSeek</strong><br><span style="font-size:11px; color:var(--text-dim);">V3 (Chat)</span></td>
                        <td><strong>Efficiency Expert.</strong> Matches GPT-4o intelligence at 1/10th the cost.</td>
                    </tr>
                    <tr>
                        <td class="rank-num">5</td>
                        <td><strong>OpenAI</strong><br><span style="font-size:11px; color:var(--text-dim);">GPT-4o</span></td>
                        <td><strong>The Reliability Go-to.</strong> Strong reasoning, widely supported.</td>
                    </tr>
                    <tr>
                        <td class="rank-num">6</td>
                        <td><strong>OpenRouter</strong><br><span style="font-size:11px; color:var(--text-dim);">All Models</span></td>
                        <td><strong>The Safety Net.</strong> Access any model instantly without code changes.</td>
                    </tr>
                    <tr>
                        <td class="rank-num">7</td>
                        <td><strong>Local</strong><br><span style="font-size:11px; color:var(--text-dim);">Ollama/LMStudio</span></td>
                        <td><strong>Privacy First.</strong> Zero data leaves your machine. Slower but secure.</td>
                    </tr>
                </tbody>
            </table>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 30px;">
                <button class="run-btn" style="background: #1d9bf020; border: 1px solid #1d9bf040;" onclick="closeRankingModal(); toggleMethodology(true, 'meth_2')">
                    🧠 View Scoring Logic
                </button>
                <button class="run-btn btn-full" onclick="closeRankingModal()">Got it</button>
            </div>
        </div>
    </div>
</body>
</html>'''

def run_server(app_state):
    handler = lambda *args, **kwargs: DashHandler(*args, app_state=app_state, **kwargs)
    with ThreadingHTTPServer((HOST, PORT), handler) as httpd:
        # Keep the localhost URL shape consumed by the existing Pinokio run.js.
        print(f"Dashboard running at http://localhost:{PORT}", flush=True)
        if os.environ.get('XLS_NO_BROWSER') != '1':
            webbrowser.open(f"http://localhost:{PORT}")
        httpd.serve_forever()

if __name__ == "__main__":
    logging.basicConfig(
        level=os.environ.get('XLS_LOG_LEVEL', 'INFO').upper(),
        format='%(asctime)s %(levelname)s %(name)s %(message)s',
    )
    app_state = {'running': False, 'status_msg': '', 'progress': 0, 'error': None, 'last_report': None}
    (ROOT_DIR / 'logs').mkdir(exist_ok=True)
    run_server(app_state)
