import asyncio
import re
import json
import os
import sys
import logging
import threading
import webbrowser
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from collections import defaultdict
from twikit import Client
import httpx
import base64
from urllib.parse import parse_qs, quote, urlparse

try:
    from .security import (
        escape_html_text,
        redact_sensitive_text,
        safe_html_attribute,
        safe_image_src_attribute,
        safe_url_attribute,
        sanitize_external_url,
        sanitize_image_src,
    )
    from .models import XAuthor, XPost, XPostMetrics, XPostReference
    from .x_api_provider import OfficialXApiProvider, XFetchPage
except ImportError:  # Compatibility with `python app/web_ui.py`.
    from security import (
        escape_html_text,
        redact_sensitive_text,
        safe_html_attribute,
        safe_image_src_attribute,
        safe_url_attribute,
        sanitize_external_url,
        sanitize_image_src,
    )
    from models import XAuthor, XPost, XPostMetrics, XPostReference
    from x_api_provider import OfficialXApiProvider, XFetchPage


logger = logging.getLogger(__name__)


class SessionVerificationState(str, Enum):
    """Result of the optional account-level Twikit session preflight."""

    VALID = "valid"
    INVALID = "invalid"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class SessionVerificationResult:
    """Classified preflight result while preserving the legacy tuple API."""

    state: SessionVerificationState
    can_fetch_timeline: bool
    retryable: bool
    message: str
    reason: str

class XListFetcher:
    """Class to fetch and process tweets from X lists with premium reporting."""

    # Incremental ingestion normally assumes a page provider is already
    # authenticated.  Twikit still needs its cookie preflight before paging.
    requires_login_for_fetch_page = True
    
    def __init__(
        self,
        cookies_path='browser_session/cookies.json',
        list_owner=None,
        proxy=None,
    ):
        proxy = proxy or os.environ.get("XLS_X_PROXY") or os.environ.get(
            "HTTPS_PROXY"
        ) or os.environ.get("https_proxy")
        # Pass the resolved proxy explicitly because Twikit forwards proxy=None
        # to httpx, which is not reliable across httpx versions and environments.
        self.client = Client('en-US', proxy=proxy) if proxy else Client('en-US')
        self.cookies_path = Path(cookies_path)
        self.list_owner_pref = list_owner
        self.proxy = proxy
        self.session_verification_state = SessionVerificationState.INCONCLUSIVE
        self._session_ready = False
        self.last_timeline_request_count = 0
        self.last_pagination_count = 0
        self.list_info = {
            'name': 'X List Summary', 'list_names': [],
            'owner': list_owner or 'Unknown', 
            'owner_name': list_owner or 'Unknown', 'member_count': 0,
            'profile_image_url': None
        }
        self.list_url = ""
        self.cache_dir = Path('cache')
        self.user_cache_path = self.cache_dir / 'user_ids.json'
        self.user_cache = self._load_user_cache()

    def _safe_error(self, error):
        """Return an exception message with X session credentials removed."""
        secrets = []
        try:
            cookies = self.client.get_cookies()
            if isinstance(cookies, dict):
                secrets.extend((cookies.get('auth_token'), cookies.get('ct0')))
        except Exception:
            # Error reporting must never hide the original failure.
            pass
        return redact_sensitive_text(str(error), secrets=secrets)

    def _load_session_cookies(self) -> bool:
        """Load X cookies from environment first, then the saved cookie file."""
        auth_token = os.environ.get("XLS_X_AUTH_TOKEN", "").strip()
        ct0 = os.environ.get("XLS_X_CT0", "").strip()
        if auth_token and ct0:
            self.client.set_cookies({"auth_token": auth_token, "ct0": ct0})
            return True
        if not self.cookies_path.exists():
            return False
        self.client.load_cookies(str(self.cookies_path))
        return True

    @staticmethod
    def _is_cloudflare_forbidden(message: str) -> bool:
        """Return whether a 403 body is recognizably a Cloudflare edge block."""
        lowered = message.lower()
        if "403" not in lowered and "forbidden" not in lowered:
            return False
        return any(
            marker in lowered
            for marker in (
                "cloudflare",
                "attention required",
                "sorry, you have been blocked",
                "cf-ray",
                "cf-wrapper",
            )
        )

    def _classify_session_error(self, error) -> SessionVerificationResult:
        """Classify preflight errors without treating every 403 as harmless."""
        safe_error = self._safe_error(error)
        lowered = safe_error.lower()

        if any(
            marker in lowered
            for marker in (
                "401",
                "unauthorized",
                "invalid cookie",
                "could not authenticate",
                "authentication required",
            )
        ):
            return SessionVerificationResult(
                SessionVerificationState.INVALID,
                False,
                False,
                "Session expired or unauthorized (401). Please re-import your cookies.",
                "unauthorized",
            )
        if "429" in lowered or "rate limit" in lowered or "recursion" in lowered:
            return SessionVerificationResult(
                SessionVerificationState.INCONCLUSIVE,
                False,
                False,
                "Rate limited (429) during session verification. Wait before retrying.",
                "rate_limited",
            )
        if self._is_cloudflare_forbidden(safe_error):
            return SessionVerificationResult(
                SessionVerificationState.INCONCLUSIVE,
                True,
                False,
                "Session verification was blocked by Cloudflare (403); continuing with the List timeline request.",
                "cloudflare_403",
            )
        if "403" in lowered or "forbidden" in lowered:
            return SessionVerificationResult(
                SessionVerificationState.INVALID,
                False,
                False,
                "Session verification was forbidden (403). Timeline access was not attempted.",
                "forbidden",
            )
        if "404" in lowered:
            return SessionVerificationResult(
                SessionVerificationState.INCONCLUSIVE,
                True,
                False,
                "Session verification endpoint returned 404; continuing with the List timeline request.",
                "verification_not_found",
            )
        return SessionVerificationResult(
            SessionVerificationState.INCONCLUSIVE,
            False,
            True,
            f"Session verification failed: {safe_error[:120]}",
            "transient_error",
        )

    def _record_session_verification(self, result: SessionVerificationResult) -> None:
        self.session_verification_state = result.state
        self._session_ready = result.can_fetch_timeline
        logger.info(
            "session_verification=%s reason=%s",
            result.state.value,
            result.reason,
        )

    async def _verify_loaded_session(self) -> SessionVerificationResult:
        try:
            user = await self.client.user()
            screen_name = getattr(user, "screen_name", "unknown")
            return SessionVerificationResult(
                SessionVerificationState.VALID,
                True,
                False,
                f"Logged in as @{screen_name}",
                "verified",
            )
        except Exception as error:
            logger.warning("Twikit login verification failed (%s)", type(error).__name__)
            return self._classify_session_error(error)

    def _load_user_cache(self):
        """Load username -> User ID mapping from local cache."""
        if not self.user_cache_path.exists():
            return {}
        try:
            with open(self.user_cache_path, 'r') as f:
                return json.load(f)
        except (OSError, ValueError, json.JSONDecodeError):
            return {}

    def _save_user_cache(self):
        """Save username -> User ID mapping to local cache."""
        self.cache_dir.mkdir(exist_ok=True)
        try:
            with open(self.user_cache_path, 'w') as f:
                json.dump(self.user_cache, f)
        except OSError:
            pass

    async def get_user_id(self, username: str) -> str:
        """Get User ID for a username, using cache if available to save API requests."""
        username = username.lower().replace('@', '').strip()
        if username in self.user_cache:
            return self.user_cache[username]
            
        try:
            user = await self.client.get_user_by_screen_name(username)
            self.user_cache[username] = user.id
            self._save_user_cache()
            return user.id
        except Exception as e:
            if '429' in str(e) or 'rate limit' in str(e).lower():
                raise Exception("X Rate Limit reached. Please wait 15 minutes before searching new users.")
            raise e
        
    async def login(self):
        """Load cookies and verify login."""
        try:
            loaded = self._load_session_cookies()
        except Exception as error:
            result = SessionVerificationResult(
                SessionVerificationState.INVALID,
                False,
                False,
                f"Cookies could not be loaded: {self._safe_error(error)[:120]}",
                "cookies_unreadable",
            )
            self._record_session_verification(result)
            return False, result.message
        if not loaded:
            result = SessionVerificationResult(
                SessionVerificationState.INVALID, False, False,
                "Cookies not configured. Please use environment variables or settings.",
                "cookies_missing",
            )
            self._record_session_verification(result)
            return False, result.message

        result = await self._verify_loaded_session()
        self._record_session_verification(result)
        return result.can_fetch_timeline, result.message

    async def verify_session(self, retries=1):
        """Lightweight check to see if current session is still valid with retry for transient errors."""
        try:
            loaded = self._load_session_cookies()
        except Exception as error:
            result = SessionVerificationResult(
                SessionVerificationState.INVALID,
                False,
                False,
                f"Cookies could not be loaded: {self._safe_error(error)[:120]}",
                "cookies_unreadable",
            )
            self._record_session_verification(result)
            return False, result.message
        if not loaded:
            result = SessionVerificationResult(
                SessionVerificationState.INVALID, False, False,
                "No cookies", "cookies_missing",
            )
            self._record_session_verification(result)
            return False, result.message

        last_result = None
        for attempt in range(retries + 1):
            last_result = await self._verify_loaded_session()
            self._record_session_verification(last_result)
            if last_result.can_fetch_timeline or not last_result.retryable:
                return last_result.can_fetch_timeline, last_result.message
            if attempt < retries:
                await asyncio.sleep(1)

        return False, last_result.message if last_result else "Session verification failed"

    @staticmethod
    def _twikit_entities(tweet) -> list[object]:
        """Return the source tweet and referenced tweets used for normalization."""
        candidates = [
            tweet,
            getattr(tweet, "retweeted_tweet", None),
            getattr(tweet, "quote", None),
            getattr(tweet, "quoted_tweet", None),
        ]
        return [
            item
            for index, item in enumerate(candidates)
            if item is not None and item not in candidates[:index]
        ]

    def _twikit_post(self, tweet, list_id: str) -> XPost:
        """Normalize one Twikit object into the canonical incremental model."""
        url_map: dict[str, str] = {}
        urls: list[str] = []
        media: list[dict] = []
        seen_media_ids: set[str] = set()

        for source in self._twikit_entities(tweet):
            legacy = getattr(source, "_legacy", {}) or {}
            entities = legacy.get("entities", {}) or getattr(source, "entities", {}) or {}
            if isinstance(entities, dict):
                for item in entities.get("urls", []) or []:
                    if not isinstance(item, dict):
                        continue
                    short = item.get("url")
                    expanded = item.get("expanded_url") or short
                    if short and expanded:
                        url_map[str(short)] = str(expanded)
                    if expanded and not any(
                        domain in str(expanded).lower()
                        for domain in ("x.com", "twitter.com", "twimg.com", "t.co")
                    ):
                        urls.append(str(expanded))

            extended = legacy.get("extended_entities", {}) or entities
            if not isinstance(extended, dict):
                continue
            for item in extended.get("media", []) or []:
                if not isinstance(item, dict):
                    continue
                media_id = str(item.get("id_str") or item.get("media_key") or "")
                if media_id and media_id in seen_media_ids:
                    continue
                media_type = str(item.get("type") or "")
                image_url = item.get("media_url_https") or item.get("media_url")
                normalized: dict[str, object] | None = None
                if media_type == "photo" and image_url:
                    normalized = {"type": "photo", "url": image_url, "id": media_id}
                elif media_type in {"video", "animated_gif"}:
                    variants = item.get("video_info", {}).get("variants", []) or []
                    mp4_variants = [
                        value for value in variants
                        if isinstance(value, dict) and value.get("content_type") == "video/mp4"
                    ]
                    best = max(mp4_variants, key=lambda value: value.get("bitrate", 0), default=None)
                    if best and best.get("url"):
                        normalized = {
                            "type": media_type,
                            "url": best["url"],
                            "thumbnail": image_url,
                            "id": media_id,
                        }
                if normalized is not None:
                    media.append(normalized)
                    if media_id:
                        seen_media_ids.add(media_id)

        text = str(getattr(tweet, "text", "") or "")
        for short, expanded in url_map.items():
            text = text.replace(short, expanded)

        user = getattr(tweet, "user", None)
        username = str(getattr(user, "screen_name", "") or "unknown")
        author_id = str(
            getattr(user, "id", None)
            or getattr(user, "rest_id", None)
            or username
        )
        author = XAuthor(
            id=author_id,
            username=username,
            name=getattr(user, "name", None),
            profile_image_url=getattr(user, "profile_image_url", None),
        )
        legacy = getattr(tweet, "_legacy", {}) or {}
        references: list[XPostReference] = []
        for relation_type, attribute_names in (
            ("retweeted", ("retweeted_tweet",)),
            ("quoted", ("quote", "quoted_tweet")),
        ):
            referenced = next(
                (
                    getattr(tweet, name, None)
                    for name in attribute_names
                    if getattr(tweet, name, None) is not None
                ),
                None,
            )
            referenced_id = getattr(referenced, "id", None)
            if referenced_id:
                references.append(XPostReference(relation_type, str(referenced_id)))
        reply_id = legacy.get("in_reply_to_status_id_str") or legacy.get("in_reply_to_status_id")
        if reply_id:
            references.append(XPostReference("replied_to", str(reply_id)))

        return XPost(
            id=str(getattr(tweet, "id", "") or ""),
            text=text,
            author_id=author_id,
            author=author,
            created_at=getattr(tweet, "created_at", None),
            language=legacy.get("lang"),
            conversation_id=(
                getattr(tweet, "conversation_id", None)
                or legacy.get("conversation_id_str")
            ),
            source_list_id=list_id,
            metrics=XPostMetrics(
                likes=getattr(tweet, "favorite_count", 0),
                retweets=getattr(tweet, "retweet_count", 0),
                replies=getattr(tweet, "reply_count", 0),
                quotes=getattr(tweet, "quote_count", 0),
                bookmarks=getattr(tweet, "bookmark_count", 0),
                impressions=getattr(tweet, "view_count", None),
            ),
            references=references,
            urls=list(dict.fromkeys(urls)),
            media=media,
            raw_payload={"provider": "twikit"},
        )

    async def fetch_page(
        self,
        list_id: str,
        *,
        pagination_token: str | None = None,
        max_results: int = 100,
        page_number: int = 1,
        run_id: str | None = None,
    ) -> XFetchPage:
        """Fetch one cookie-authenticated Twikit page for incremental ingestion."""
        if isinstance(max_results, bool) or not isinstance(max_results, int) or max_results < 1:
            raise ValueError("max_results must be a positive integer")
        if not self._session_ready:
            success, message = await self.login()
            if not success:
                raise RuntimeError(message)

        normalized_list_id = self.extract_list_id(list_id)
        if page_number <= 1:
            self.last_timeline_request_count = 0
            self.last_pagination_count = 0
        self.last_timeline_request_count += 1
        self.last_pagination_count = max(0, self.last_timeline_request_count - 1)
        try:
            batch = await self.client.get_list_tweets(
                normalized_list_id,
                count=min(40, max_results),
                cursor=pagination_token,
            )
            posts = [
                self._twikit_post(tweet, normalized_list_id)
                for tweet in list(batch or [])[:max_results]
            ]
            next_token = getattr(batch, "next_cursor", None) if batch else None
            logger.info(
                "timeline_fetch=success list_id=%s returned_count=%s request_count=%s pagination_count=%s",
                normalized_list_id,
                len(posts),
                self.last_timeline_request_count,
                self.last_pagination_count,
            )
            return XFetchPage(
                list_id=normalized_list_id,
                posts=posts,
                next_token=str(next_token) if next_token else None,
                page_number=page_number,
            )
        except Exception as error:
            safe_error = self._safe_error(error)
            lowered = safe_error.lower()
            if "429" in lowered or "rate limit" in lowered:
                logger.warning("timeline_fetch=rate_limited list_id=%s", normalized_list_id)
                raise RuntimeError("X timeline rate limited (429). Wait before retrying.") from error
            if "401" in lowered or "unauthorized" in lowered:
                logger.warning("timeline_fetch=unauthorized list_id=%s", normalized_list_id)
                raise RuntimeError("X session unauthorized (401). Re-import your cookies.") from error
            if "403" in lowered or "forbidden" in lowered:
                logger.warning("timeline_fetch=forbidden list_id=%s", normalized_list_id)
                raise RuntimeError("X timeline access forbidden (403).") from error
            raise RuntimeError(f"X timeline fetch failed: {safe_error[:120]}") from error

    async def get_user_memberships(self, username: str):
        """Fetch all lists that a specific user is a member of (Profiler feature)."""
        try:
            if not self._load_session_cookies():
                raise RuntimeError("X cookies are not configured")
            user_id = await self.get_user_id(username)
            
            memberships = []
            cursor = '-1' # v1.1 cursor starts at -1
            
            # Fetch memberships (lists the user is in)
            # twikit 2.3.3 lacks get_user_memberships, so we use manual v1.1 call
            while True:
                url = f'https://api.twitter.com/1.1/lists/memberships.json?user_id={user_id}&count=50'
                if cursor and cursor != '-1':
                    url += f'&cursor={cursor}'
                
                # We MUST pass client._base_headers for authentication
                # client.get() handles the transaction IDs automatically
                response, raw_response = await self.client.get(url, headers=self.client._base_headers)
                
                if not response or 'lists' not in response:
                    break
                
                for l in response['lists']:
                    name = l.get('name', '')
                    if name:
                        owner = l.get('user', {}).get('screen_name', 'Unknown')
                        list_id = l.get('id_str', '')
                        memberships.append({
                            'name': name,
                            'owner': owner,
                            'id': list_id
                        })
                
                cursor = str(response.get('next_cursor_str', '0'))
                if cursor == '0' or not cursor or len(memberships) > 500: # Safety cap
                    break
                    
            print(f"✅ Found {len(memberships)} memberships for {username}")
            return memberships
        except Exception as e:
            print(f"❌ Error fetching memberships for {username}: {self._safe_error(e)}")
            return []

    def extract_list_id(self, url_or_id: str) -> str:
        """Extract numeric list ID from URL."""
        if url_or_id.isdigit():
            return url_or_id
        match = re.search(r'/lists/(\d+)', url_or_id)
        return match.group(1) if match else url_or_id

    def extract_owner_from_url(self, url: str) -> str:
        """Extract username from list URL."""
        match = re.search(r'x\.com/([^/]+)/lists/', url)
        if not match:
            match = re.search(r'twitter\.com/([^/]+)/lists/', url)
        return match.group(1) if match else None

    async def _resolve_list_redirect(self, list_id: str) -> str:
        """Find list owner via redirect logic."""
        url = f"https://x.com/i/lists/{list_id}"
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.get(url, follow_redirects=False, timeout=5)
                if resp.status_code in [301, 302]:
                    loc = resp.headers.get('location', '')
                    return self.extract_owner_from_url(loc)
            except httpx.HTTPError:
                pass
        return None

    async def fetch_list_tweets(self, list_url_or_id: str, max_tweets: int = 100, delay: float = 0):
        """Fetch tweets from a list (Aggregates metadata)."""
        if isinstance(max_tweets, bool) or not isinstance(max_tweets, int):
            raise ValueError("max_tweets must be an integer")
        self.last_timeline_request_count = 0
        self.last_pagination_count = 0
        if max_tweets <= 0:
            logger.info("timeline_fetch=success returned_count=0 reason=non_positive_limit")
            return []

        if delay > 0:
            await asyncio.sleep(delay)
            
        list_id = self.extract_list_id(list_url_or_id)
        logger.info("timeline_fetch=starting list_id=%s", list_id)
        
        tweets = []
        cursor = None
        current_url = list_url_or_id if list_url_or_id.startswith('http') else f"https://x.com/i/lists/{list_id}"
        
        try:
            # 1. Get List Info
            list_name = None
            member_count_for_list = 0
            owner_obj = None

            try:
                l_info = await self.client.get_list(list_id)
                list_name = getattr(l_info, 'name', None)
                member_count_for_list = getattr(l_info, 'member_count', 0)
                owner_obj = getattr(l_info, 'user', getattr(l_info, 'creator', None))
                logger.info(
                    "list_metadata=success provider=twikit list_id=%s list_name=%s member_count=%s",
                    list_id,
                    list_name,
                    member_count_for_list,
                )
            except Exception as _e:
                logger.warning(
                    "list_metadata=failed provider=twikit list_id=%s error=%s fallback=v1.1",
                    list_id,
                    self._safe_error(_e),
                )
                try:
                    v1_url = f'https://api.twitter.com/1.1/lists/show.json?list_id={list_id}'
                    v1_resp, _ = await self.client.get(v1_url, headers=self.client._base_headers)
                    if v1_resp:
                        list_name = v1_resp.get('name')
                        member_count_for_list = v1_resp.get('member_count', 0)
                        logger.info(
                            "list_metadata=success provider=v1.1 list_id=%s list_name=%s member_count=%s",
                            list_id,
                            list_name,
                            member_count_for_list,
                        )
                except Exception as _e2:
                    logger.warning(
                        "list_metadata=failed provider=v1.1 list_id=%s error=%s",
                        list_id,
                        self._safe_error(_e2),
                    )

            if list_name:
                self.list_info['list_names'].append(list_name)
                if self.list_info['name'] == 'X List Summary':
                    self.list_info['name'] = list_name
            self.list_info['member_count'] += member_count_for_list

            # Resolve owner profile (independent of list name)
            if self.list_owner_pref and not self.list_info['profile_image_url']:
                try:
                    u_info = await self.client.get_user_by_screen_name(self.list_owner_pref)
                    self.list_info['owner'] = self.list_owner_pref
                    self.list_info['owner_name'] = getattr(u_info, 'name', self.list_owner_pref)
                    self.list_info['profile_image_url'] = getattr(u_info, 'profile_image_url', None)
                except Exception as exc:
                    logger.debug("Could not resolve preferred list owner (%s)", type(exc).__name__)

            if not self.list_info['profile_image_url'] and owner_obj:
                self.list_info['owner'] = getattr(owner_obj, 'screen_name', self.list_info['owner'])
                self.list_info['owner_name'] = getattr(owner_obj, 'name', self.list_info['owner'])
                self.list_info['profile_image_url'] = getattr(owner_obj, 'profile_image_url', None)

            # 2. Fetch Tweets
            while len(tweets) < max_tweets:
                remaining = max_tweets - len(tweets)
                if remaining <= 0:
                    break
                self.last_timeline_request_count += 1
                self.last_pagination_count = max(0, self.last_timeline_request_count - 1)
                batch = await self.client.get_list_tweets(
                    list_id,
                    count=min(40, remaining),
                    cursor=cursor,
                )
                if not batch: break
                
                for tweet in batch:
                    if len(tweets) >= max_tweets:
                        break
                    # Resolve Links & Entities — including retweets and quote tweets
                    resolved_links = set()
                    url_map = {}
                    display_map = {}

                    def extract_urls_from(t_obj):
                        if not t_obj: return
                        leg = getattr(t_obj, '_legacy', {}) or {}
                        ents = leg.get('entities', {}) or getattr(t_obj, 'entities', {}) or {}
                        if isinstance(ents, dict):
                            for u in ents.get('urls', []):
                                short = u.get('url')
                                expanded = u.get('expanded_url')
                                if short:
                                    url_map[short] = expanded or short
                                    display_map[short] = u.get('display_url', expanded or short)

                    extract_urls_from(tweet)
                    extract_urls_from(getattr(tweet, 'retweeted_tweet', None))
                    extract_urls_from(getattr(tweet, 'quote', None))

                    for expanded in url_map.values():
                        if not any(d in expanded.lower() for d in ['x.com', 'twitter.com', 'twimg.com', 't.co']):
                            resolved_links.add(expanded)
                    
                    # Media extraction (with best bitrate video)
                    media = []
                    seen_media_ids = set()
                    
                    def extract_media(t_obj):
                        if not t_obj: return
                        leg = getattr(t_obj, '_legacy', {})
                        ext_ents = leg.get('extended_entities', {}) or leg.get('entities', {}) or {}
                        for m in ext_ents.get('media', []):
                            m_id = m.get('id_str')
                            if m_id in seen_media_ids: continue
                            m_type = m.get('type')
                            m_url = m.get('media_url_https')
                            
                            if m_type == 'photo':
                                media.append({'type': 'photo', 'url': m_url, 'id': m_id})
                            elif m_type in ['video', 'animated_gif']:
                                variants = m.get('video_info', {}).get('variants', [])
                                best = sorted([v for v in variants if v.get('content_type') == 'video/mp4'], 
                                            key=lambda x: x.get('bitrate', 0), reverse=True)
                                if best:
                                    media.append({'type': m_type, 'url': best[0]['url'], 'thumbnail': m_url, 'id': m_id})
                            seen_media_ids.add(m_id)

                    extract_media(tweet)
                    if hasattr(tweet, 'retweeted_tweet'): extract_media(tweet.retweeted_tweet)
                    if hasattr(tweet, 'quote'): extract_media(tweet.quote)
                    
                    # Card Extraction (Link Previews)
                    tweet_card = None
                    def extract_card(t_obj):
                        if not t_obj: return None
                        c = getattr(t_obj, 'card', None)
                        if not c: return None
                        
                        try:
                            # Twikit card object processing
                            bv = getattr(c, 'binding_values', {})
                            if not bv: return None
                            
                            res = {}
                            if 'title' in bv: res['title'] = bv['title'].get('string_value')
                            if 'description' in bv: res['description'] = bv['description'].get('string_value')
                            if 'thumbnail_image' in bv: 
                                res['image'] = bv['thumbnail_image'].get('image_value', {}).get('url')
                            elif 'player_image' in bv:
                                res['image'] = bv['player_image'].get('image_value', {}).get('url')
                                
                            if res.get('title'): return res
                        except (AttributeError, KeyError, TypeError, ValueError):
                            pass
                        return None

                    tweet_card = extract_card(tweet)
                    if not tweet_card and hasattr(tweet, 'retweeted_status'): 
                        tweet_card = extract_card(tweet.retweeted_status)
                    if not tweet_card and hasattr(tweet, 'quoted_status'):
                        tweet_card = extract_card(tweet.quoted_status)
                    
                    # Clean Text
                    clean_text = tweet.text
                    for short, expanded in url_map.items():
                        if short in clean_text:
                            # Use expanded for external, display for internal
                            tgt = expanded if not any(d in expanded.lower() for d in ['x.com', 'twitter.com', 'twimg.com']) else display_map.get(short, short)
                            clean_text = clean_text.replace(short, tgt)

                    tweets.append({
                        'id': tweet.id, 'text': clean_text, 'author': tweet.user.screen_name,
                        'links': list(resolved_links), 'media': media, 'card': tweet_card,
                        'likes': getattr(tweet, 'favorite_count', 0),
                        'retweets': getattr(tweet, 'retweet_count', 0),
                        'replies': getattr(tweet, 'reply_count', 0),
                        'quotes': getattr(tweet, 'quote_count', 0),
                        'bookmarks': getattr(tweet, 'bookmark_count', 0)
                    })

                if len(tweets) >= max_tweets:
                    break
                cursor = batch.next_cursor
                if not cursor: break

            logger.info(
                "timeline_fetch=success list_id=%s returned_count=%s request_count=%s pagination_count=%s",
                list_id,
                len(tweets),
                self.last_timeline_request_count,
                self.last_pagination_count,
            )
            return tweets[:max_tweets]
        except Exception as e:
            err = self._safe_error(e)
            # Re-raise rate limit and auth errors so the caller can show a clear message
            if '429' in err or 'rate limit' in err.lower():
                logger.warning("timeline_fetch=rate_limited list_id=%s", list_id)
                raise Exception(f"X Rate Limit reached fetching list {list_id}. Please wait 15 minutes and try again.") from e
            if '401' in err or 'unauthorized' in err.lower():
                logger.warning("timeline_fetch=unauthorized list_id=%s", list_id)
                raise Exception(f"X session unauthorized (401) fetching list {list_id}. Please re-import your cookies via Settings → X Account.") from e
            if '403' in err or 'forbidden' in err.lower():
                logger.warning("timeline_fetch=forbidden list_id=%s", list_id)
                raise Exception(
                    f"X timeline access forbidden (403) fetching list {list_id}."
                ) from e
            # Preserve already-fetched pages, but surface complete list failures
            # so the task runner can report partial success accurately.
            logger.error("timeline_fetch=failed list_id=%s error=%s", list_id, err)
            if tweets:
                return tweets
            raise RuntimeError(f"X list {list_id} fetch failed: {err}") from e

    def aggregate_by_links(self, tweets: list) -> dict:
        """Group tweets by link and sort by engagement, weighted by author diversity."""
        by_link = defaultdict(list)
        no_links = []
        for t in tweets:
            if t.get('links'):
                for link in t['links']:
                    # Skip unresolved t.co short URLs
                    if 't.co/' in link.lower():
                        continue
                    by_link[link].append(t)
            else: no_links.append(t)

        def _score(link_tweets):
            base = sum(
                t['likes'] + (t['retweets'] * 1.5) + (t['replies'] * 2.0) + t['quotes'] + t['bookmarks']
                for t in link_tweets
            )
            # Multiply by number of unique authors sharing this link.
            # Community consensus (multiple people sharing) ranks above a single curator.
            unique_authors = len({t['author'] for t in link_tweets})
            return base * unique_authors

        sorted_links = sorted(by_link.items(), key=lambda x: _score(x[1]), reverse=True)

        # Per-author cap: limit sole-author links to 2 entries in the final list.
        # Multi-author links (community consensus) are never capped.
        # This prevents one prolific curator from flooding the top-20.
        PER_AUTHOR_CAP = 2
        author_counts: dict = {}
        capped: list = []
        overflow: list = []
        for item in sorted_links:
            link, link_tweets = item
            authors = {t['author'] for t in link_tweets}
            if len(authors) > 1:
                # Community-shared link → always include, no cap applied
                capped.append(item)
            else:
                sole_author = next(iter(authors))
                count = author_counts.get(sole_author, 0)
                if count < PER_AUTHOR_CAP:
                    author_counts[sole_author] = count + 1
                    capped.append(item)
                else:
                    overflow.append(item)

        # Fill remaining slots from overflow (preserving score order)
        final = capped + overflow
        return {'by_link': final, 'no_links': no_links}


    def _build_card_html(self, tweet_data):
        """Build HTML for a link preview card inside a tweet."""
        card = tweet_data.get('card')
        if not card: return ""

        # Determine target URL from links if not in card
        url = sanitize_external_url(
            tweet_data['links'][0] if tweet_data.get('links') else ""
        )
        if not url:
            return ""

        title = escape_html_text(card.get('title', ''))
        if not title:
            return ""

        image_url = sanitize_image_src(card.get('image', ''))
        img_html = (
            f'<div class="tc-img"><img src="{safe_image_src_attribute(image_url)}" '
            'loading="lazy" alt=""></div>'
            if image_url else ""
        )
        description = escape_html_text(card.get('description', ''))
        desc_html = f'<div class="tc-desc">{description}</div>' if description else ""
        url_attr = safe_url_attribute(url)
        domain = escape_html_text(self._extract_domain(url))

        return f'''
        <a href="{url_attr}" target="_blank" rel="noopener noreferrer" class="tweet-card-link">
            <div class="tc-container">
                {img_html}
                <div class="tc-content">
                    <div class="tc-title">{title}</div>
                    {desc_html}
                    <div class="tc-site">{domain}</div>
                </div>
            </div>
        </a>'''

    def _build_media_html(self, tweet, seen_urls=None):
        """Build HTML for images and videos in a tweet. seen_urls deduplicates within a group."""
        media = tweet.get('media', [])
        if not media: return ""

        html_parts = []
        for m in media:
            # Use thumbnail URL as the dedup key (stable CDN image, same across retweets)
            url_key = m.get('thumbnail') or m.get('url')
            if seen_urls is not None:
                if url_key in seen_urls:
                    continue  # duplicate — already shown in this group
                seen_urls.add(url_key)

            if m.get('type') == 'photo':
                image_url = sanitize_image_src(m.get('url', ''))
                if image_url:
                    html_parts.append(
                        f'<div class="media-item"><img src="{safe_image_src_attribute(image_url)}" '
                        'loading="lazy" alt=""></div>'
                    )
            elif m.get('type') == 'animated_gif':
                video_url = sanitize_external_url(m.get('url', ''))
                poster_url = sanitize_image_src(m.get('thumbnail', ''))
                if not video_url:
                    continue
                poster_attr = (
                    f' poster="{safe_image_src_attribute(poster_url)}"'
                    if poster_url else ""
                )
                html_parts.append(f'''
                <div class="media-item">
                    <video playsinline autoplay loop muted{poster_attr}>
                        <source src="{safe_url_attribute(video_url)}" type="video/mp4">
                    </video>
                </div>''')
            elif m.get('type') == 'video':
                # X video CDN requires session auth — show poster thumbnail with play overlay
                author_path = quote(str(tweet.get('author', 'i')), safe='')
                tweet_id_path = quote(str(tweet.get('id', '')), safe='')
                tweet_url = sanitize_external_url(
                    f"https://x.com/{author_path}/status/{tweet_id_path}"
                )
                thumb = sanitize_image_src(m.get('thumbnail', ''))
                if thumb:
                    html_parts.append(f'''
                    <div class="media-item">
                        <a href="{safe_url_attribute(tweet_url)}" target="_blank" rel="noopener noreferrer" class="video-thumb-link" title="Watch on X">
                            <img src="{safe_image_src_attribute(thumb)}" loading="lazy" alt="">
                            <div class="play-overlay">&#9654;</div>
                        </a>
                    </div>''')

        if not html_parts: return ""
        return '<div class="tweet-media">' + "".join(html_parts) + '</div>'

    def _extract_domain(self, url):
        """Extract domain from URL."""
        try:
            safe_url = sanitize_external_url(url)
            if not safe_url:
                return ""
            domain = urlparse(safe_url).hostname or ""
            return domain.replace('www.', '')
        except (TypeError, ValueError):
            return ""

    def _build_link_card(self, url):
        """Build a link card component with a high-quality favicon."""
        url = sanitize_external_url(url)
        if not url:
            return ""
        domain = self._extract_domain(url)
        if not domain:
            return ""

        url_attr = safe_url_attribute(url)
        domain_text = escape_html_text(domain)
        
        # YouTube Special Case
        if 'youtube.com' in domain or 'youtu.be' in domain:
            parsed = urlparse(url)
            if domain.endswith('youtu.be'):
                y_id = parsed.path.strip('/').split('/')[0]
            else:
                y_id = parse_qs(parsed.query).get('v', [''])[0]
            if re.fullmatch(r'[A-Za-z0-9_-]{6,20}', y_id or ''):
                embed_url = safe_url_attribute(f"https://www.youtube.com/embed/{y_id}")
                return f'''
                <div class="l-card">
                    <div class="l-dom">YOUTUBE</div>
                    <div class="v-con">
                        <iframe src="{embed_url}" allowfullscreen></iframe>
                    </div>
                </div>'''
            
        # Standard Link Card with Favicon
        favicon_url = f"https://www.google.com/s2/favicons?domain={domain}&sz=64"
        
        return f'''
        <div class="shared-content">
            <div class="link-card">
                <a href="{url_attr}" target="_blank" rel="noopener noreferrer" class="link-icon-wrap">
                    <img src="{safe_image_src_attribute(favicon_url)}" class="link-icon-img" alt="">
                </a>
                <div class="link-details">
                    <div class="link-domain">{domain_text}</div>
                    <a href="{url_attr}" target="_blank" rel="noopener noreferrer" class="link-url-text">{escape_html_text(url)}</a>
                </div>
            </div>
        </div>'''

    def _parse_ai_insights(self, ai_summary):
        """Parse AI output into {key: why} dict. Keys may be full URLs or domains."""
        insights = {}
        if not ai_summary:
            return insights
        for line in ai_summary.strip().split('\n'):
            line = line.strip()
            if ' :: ' in line:
                parts = line.split(' :: ', 1)
                raw_key = parts[0].strip().lstrip('0123456789. -*#[]')
                why = parts[1].strip()
                if not raw_key or not why:
                    continue
                # Store by the exact key the AI used (could be a full URL or domain)
                insights[raw_key] = why
                # Also store by lowercased version for case-insensitive matching
                insights[raw_key.lower()] = why
                # Extract and store by domain as a fallback (only if not already set)
                try:
                    from urllib.parse import urlparse as _up
                    if raw_key.startswith('http'):
                        domain = _up(raw_key).netloc.replace('www.', '').lower()
                    else:
                        domain = raw_key.lower()
                    if domain and domain not in insights:
                        insights[domain] = why
                    # Also index by base domain (e.g. blog.example.com → example.com)
                    base = '.'.join(domain.split('.')[-2:]) if domain.count('.') >= 2 else domain
                    if base and base not in insights:
                        insights[base] = why
                except Exception:
                    pass
        return insights

    def generate_html_report(self, aggregated, ai_summary, output_path, tweet_count=0, ai_model=''):
        """Standardized HTML Report generator."""
        timestamp = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        link_count = len(aggregated['by_link'])

        def metric(value):
            """Coerce untrusted metric values to safe non-negative integers."""
            try:
                return max(0, int(value))
            except (TypeError, ValueError):
                return 0
        
        # Logo as Base64
        logo_uri = "icon.png"
        try:
            icon_path = Path("icon.png")
            if icon_path.exists():
                with open(icon_path, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode()
                    logo_uri = f"data:image/png;base64,{encoded_string}"
        except (OSError, ValueError) as exc:
            logger.warning("Could not embed report icon (%s)", type(exc).__name__)

        # Build display title from all fetched list names
        list_names = self.list_info.get('list_names', [])
        if not list_names:
            list_names = [self.list_info.get('name', 'X List Summary')]
        display_title = safe_html_attribute(
            ' & '.join(str(name) for name in list_names if name)
            or 'X List Summary'
        )

        owner_name = escape_html_text(self.list_info.get('owner_name') or 'Unknown')
        owner_handle = escape_html_text(self.list_info.get('owner') or 'Unknown')
        member_count = metric(self.list_info.get('member_count', 0))
        owner_info = (
            f"by {owner_name} (@{owner_handle}) &bull; "
            f"{member_count:,} members total"
        )
        
        # Parse AI insights into lookup dict (keyed by URL and/or domain)
        insights = self._parse_ai_insights(ai_summary)
        
        # Build "Most Shared Content & Why" table with inline expandable tweet rows
        table_rows = ""
        for i, (link, tweets) in enumerate(aggregated['by_link'][:20]):
            safe_link = sanitize_external_url(link)
            domain = self._extract_domain(safe_link)
            # Lookup priority: 1) exact URL, 2) truncated URL (as sent to AI), 3) domain, 4) base domain
            key_80 = link[:80] if len(link) > 80 else link
            why = (insights.get(link)
                   or insights.get(link.lower())
                   or insights.get(key_80)
                   or insights.get(key_80.lower())
                   or insights.get(domain))
            if not why:
                base = '.'.join(domain.split('.')[-2:]) if domain.count('.') >= 2 else domain
                why = insights.get(base)
            why_html = escape_html_text(why) if why else '&mdash;'
            count = len(tweets)
            label = str(count)
            favicon = (
                f"https://www.google.com/s2/favicons?domain={domain}&sz=32"
                if domain else ""
            )
            if safe_link:
                content_label = escape_html_text(domain or safe_link)
                content_html = (
                    f'<a href="{safe_url_attribute(safe_link)}" target="_blank" '
                    f'rel="noopener noreferrer" class="t-link">{content_label}</a>'
                )
            else:
                content_html = (
                    f'<span class="t-link">{escape_html_text(link)}</span>'
                )
            favicon_html = (
                f'<img src="{safe_image_src_attribute(favicon)}" class="t-fav" alt="">'
                if favicon else ""
            )

            # Build the tweet list for this link's expand row (dedup media across retweets)
            tweet_list = ""
            group_seen_media = set()
            for t in tweets[:5]:
                author_raw = str(t.get('author', 'unknown'))
                author_path = quote(author_raw, safe='')
                tweet_id_path = quote(str(t.get('id', '')), safe='')
                tweet_url = sanitize_external_url(
                    f"https://x.com/{author_path}/status/{tweet_id_path}"
                )
                tweet_url_attr = safe_url_attribute(tweet_url)
                author_text = escape_html_text(author_raw)
                tweet_text = escape_html_text(t.get('text', ''))
                media_html = self._build_media_html(t, seen_urls=group_seen_media)
                card_html = self._build_card_html(t)
                tweet_list += f'''
                    <div class="tweet">
                        <div class="tweet-header">
                            <a href="{tweet_url_attr}" target="_blank" rel="noopener noreferrer" class="author">@{author_text} &#8599;</a>
                            <div class="tweet-meta">
                                <span class="metrics">&#10084;&#65039; {metric(t.get('likes'))} | &#128260; {metric(t.get('retweets'))} | &#128172; {metric(t.get('replies'))} | &#128279; {metric(t.get('bookmarks'))}</span>
                                <a href="{tweet_url_attr}" target="_blank" rel="noopener noreferrer" class="view-tweet">View Tweet</a>
                            </div>
                        </div>
                        <p class="tweet-text">{tweet_text}</p>
                        {media_html}
                        {card_html}
                    </div>'''

            table_rows += f'''
                <tr class="insight-row" id="insight-row-{i}">
                    <td class="t-name">
                        <div class="t-domain-wrap">
                            {favicon_html}
                            {content_html}
                        </div>
                    </td>
                    <td class="t-count-cell">
                        <a class="tweet-expand-link" data-idx="{i}" onclick="toggleRow({i}); return false;">{label} <span class="expand-arrow" id="arrow-{i}">&#9660;</span></a>
                    </td>
                    <td class="t-why">{why_html}</td>
                </tr>
                <tr class="tweet-expand-row" id="tweets-{i}">
                    <td colspan="3" class="tweet-expand-cell">
                        {tweet_list}
                        {self._build_link_card(link)}
                    </td>
                </tr>'''

        insights_html = f'''
        <div class="insights-card">
            <div class="insights-title">&#128293; Most Shared Content &amp; Why</div>
            <div class="table-con">
                <table>
                    <tr class="t-head">
                        <th>Content</th>
                        <th>Mentions</th>
                        <th>Why It&rsquo;s Trending</th>
                    </tr>
                    {table_rows}
                </table>
            </div>
        </div>'''

        # Build individual tweets section (compact grid, 3 per row)
        individual_html = ""
        for t in aggregated['no_links'][:30]:
            author_raw = str(t.get('author', 'unknown'))
            author_path = quote(author_raw, safe='')
            tweet_id_path = quote(str(t.get('id', '')), safe='')
            tweet_url = sanitize_external_url(
                f"https://x.com/{author_path}/status/{tweet_id_path}"
            )
            tweet_url_attr = safe_url_attribute(tweet_url)
            text_raw = t.get('text', '') or ''
            text_snip = (text_raw[:220] + '…') if len(text_raw) > 220 else text_raw
            media_html = self._build_media_html(t)
            card_html = self._build_card_html(t)
            individual_html += f'''
            <div class="tweet-mini">
                <div class="tm-head">
                    <a href="{tweet_url_attr}" target="_blank" rel="noopener noreferrer" class="tm-author">@{escape_html_text(author_raw)} <span class="tm-arrow">&#8599;</span></a>
                </div>
                <p class="tm-text">{escape_html_text(text_snip)}</p>
                {media_html}
                {card_html}
                <div class="tm-metrics">&#10084;&#65039; {metric(t.get('likes'))} &nbsp;&#128260; {metric(t.get('retweets'))} &nbsp;&#128172; {metric(t.get('replies'))}</div>
            </div>'''

        ai_model_html = (
            f'<div class="gen-model">&#129302; AI Analysis by {escape_html_text(ai_model)}</div>'
            if ai_model else ''
        )

        fallback_profile = 'https://abs.twimg.com/sticky/default_profile_images/default_profile_normal.png'
        profile_img = sanitize_image_src(
            self.list_info.get('profile_image_url'), fallback=fallback_profile
        )
        safe_logo_uri = safe_image_src_attribute(
            logo_uri,
            fallback=fallback_profile,
            allow_data_image=True,
        )

        report_html = self._get_report_template().format(
            title=display_title,
            owner_line=owner_info,
            tweet_count=metric(tweet_count),
            link_count=link_count,
            timestamp=timestamp,
            ai_model_html=ai_model_html,
            logo_uri=safe_logo_uri,
            insights=insights_html,
            individual=individual_html,
            profile_img=safe_image_src_attribute(profile_img)
        )
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_html)

    def _md_to_html(self, text):
        """Kept for backward compatibility — no longer called in report generation."""
        return ""

    def _get_report_template(self):
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Report - {title}</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg: #0b0e14; --card: #151921; --border: #232a35; --text: #eff3f4;
            --dim: #949ba4; --accent: #1d9bf0; --green: #00ba7c;
            --purple-grad: linear-gradient(135deg, #a855f7 0%, #1d9bf0 100%);
            --fire-grad: linear-gradient(135deg, #f97316 0%, #ef4444 50%, #a855f7 100%);
        }}
        * {{ box-sizing: border-box; }}
        body {{ font-family: 'Inter', sans-serif; background: var(--bg); color: var(--text); margin: 0; padding: 40px 20px; line-height: 1.6; }}
        .con {{ max-width: 1000px; margin: 0 auto; }}

        /* Header */
        .page-header {{ text-align: center; margin-bottom: 60px; }}
        .main-logo {{ width: 100px; height: 100px; border-radius: 20px; margin-bottom: 30px; box-shadow: 0 0 40px rgba(29,155,240,0.2); }}
        .main-title {{ font-size: 48px; font-weight: 800; margin: 0; background: var(--purple-grad); -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: -1px; }}
        .gen-date {{ color: var(--dim); font-size: 14px; margin-top: 10px; font-weight: 500; }}
        .gen-model {{ color: var(--dim); font-size: 12px; margin-top: 6px; font-weight: 500; opacity: 0.7; }}

        /* List Card */
        .list-card {{ background: var(--card); border: 1px solid var(--border); border-radius: 20px; padding: 24px; display: flex; align-items: center; gap: 20px; margin-bottom: 40px; }}
        .l-img {{ width: 56px; height: 56px; border-radius: 50%; border: 2px solid var(--border); }}
        .l-title {{ font-size: 18px; font-weight: 700; margin-bottom: 4px; display: block; }}
        .l-meta {{ color: var(--dim); font-size: 13px; }}

        /* Stats */
        .stats-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 40px; }}
        .stat-box {{ background: #11151c; border: 1px solid var(--border); border-radius: 24px; padding: 40px; text-align: center; }}
        .stat-val {{ font-size: 48px; font-weight: 800; color: var(--accent); display: block; margin-bottom: 8px; }}
        .stat-lbl {{ font-size: 12px; font-weight: 800; color: var(--dim); letter-spacing: 2px; text-transform: uppercase; }}

        /* Insights Card (Most Shared Content & Why) */
        .insights-card {{
            background: linear-gradient(135deg, rgba(249,115,22,0.06) 0%, rgba(239,68,68,0.06) 50%, rgba(168,85,247,0.06) 100%);
            border: 1px solid rgba(249,115,22,0.35);
            border-radius: 32px; padding: 40px; margin-bottom: 60px;
            box-shadow: 0 0 40px rgba(249,115,22,0.06);
        }}
        .insights-title {{ font-size: 24px; font-weight: 800; margin-bottom: 25px; background: var(--fire-grad); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}

        /* Table */
        .table-con {{ background: rgba(0,0,0,0.3); border: 1px solid var(--border); border-radius: 16px; overflow: hidden; }}
        table {{ width: 100%; border-collapse: collapse; text-align: left; font-size: 14px; }}
        th {{ background: rgba(26,32,42,0.8); padding: 14px 20px; color: var(--accent); font-weight: 700; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; }}
        td {{ padding: 16px 20px; border-bottom: 1px solid var(--border); vertical-align: middle; }}
        tr:last-child td {{ border-bottom: none; }}
        tr:hover td {{ background: rgba(29,155,240,0.03); }}
        .t-name {{ width: 220px; }}
        .t-domain-wrap {{ display: flex; align-items: center; gap: 10px; }}
        .t-fav {{ width: 20px; height: 20px; border-radius: 4px; flex-shrink: 0; }}
        .t-link {{ color: var(--text); font-weight: 700; text-decoration: none; font-size: 13px; }}
        .t-link:hover {{ color: var(--accent); }}
        .t-count-cell {{ width: 130px; white-space: nowrap; }}
        .tweet-expand-link {{
            display: inline-flex; align-items: center; gap: 6px;
            color: var(--accent); font-weight: 700; font-size: 13px; text-decoration: none;
            background: rgba(29,155,240,0.1); border: 1px solid rgba(29,155,240,0.25);
            border-radius: 20px; padding: 5px 12px; transition: 0.2s; cursor: pointer;
        }}
        .tweet-expand-link:hover {{ background: rgba(29,155,240,0.2); border-color: var(--accent); }}
        .t-why {{ color: var(--dim); font-size: 13px; line-height: 1.5; }}

        /* Section Labels */
        .sec-label {{ font-size: 20px; font-weight: 800; margin: 0 0 20px; color: var(--text); display: flex; align-items: center; gap: 10px; }}

        /* Inline tweet expand rows */
        .tweet-expand-row {{ display: none; }}
        .tweet-expand-row.open {{ display: table-row; }}
        .tweet-expand-cell {{
            padding: 0 !important; border-top: 2px solid rgba(29,155,240,0.2);
            background: rgba(0,0,0,0.25);
        }}
        .expand-arrow {{ display: inline-block; transition: transform 0.2s; font-style: normal; }}
        .tweet-expand-link.open .expand-arrow {{ transform: rotate(180deg); }}
        .insight-row.open td {{ background: rgba(29,155,240,0.04); }}

        /* Tweets */
        .tweet {{ padding: 22px 24px; border-bottom: 1px solid var(--border); }}
        .tweet:last-of-type {{ border-bottom: none; }}
        .tweet-header {{ display: flex; justify-content: space-between; margin-bottom: 10px; align-items: flex-start; gap: 12px; }}
        .author {{ font-weight: 800; color: var(--text); text-decoration: none; font-size: 15px; white-space: nowrap; }}
        .author:hover {{ color: var(--accent); }}
        .tweet-meta {{ display: flex; flex-direction: column; align-items: flex-end; gap: 4px; flex-shrink: 0; }}
        .metrics {{ color: var(--dim); font-size: 12px; font-weight: 500; white-space: nowrap; }}
        .view-tweet {{ color: var(--accent); font-size: 12px; text-decoration: none; font-weight: 600; }}
        .view-tweet:hover {{ text-decoration: underline; }}
        .tweet-text {{ margin: 0; white-space: pre-wrap; font-size: 14px; color: var(--text); line-height: 1.55; }}

        /* Media */
        .tweet-media {{ margin-top: 12px; border-radius: 12px; overflow: hidden; border: 1px solid var(--border); }}
        .media-item img, .media-item video {{ width: 100%; display: block; object-fit: cover; max-height: 450px; }}

        /* Link card */
        .shared-content {{ padding: 0 24px 20px 24px; }}
        .link-card {{ background: #0b0e14; border: 1px solid var(--border); border-radius: 14px; padding: 14px; display: flex; gap: 14px; align-items: center; margin-top: 8px; }}
        .link-icon-wrap {{ width: 48px; height: 48px; display: flex; align-items: center; justify-content: center; border-radius: 12px; border: 1px solid var(--border); background: #151921; transition: 0.2s; overflow: hidden; flex-shrink: 0; }}
        .link-icon-wrap:hover {{ border-color: var(--accent); }}
        .link-icon-img {{ width: 28px; height: 28px; object-fit: contain; }}
        .link-details {{ display: flex; flex-direction: column; overflow: hidden; }}
        .link-domain {{ color: var(--dim); font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 2px; }}
        .link-url-text {{ color: var(--accent); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; text-decoration: none; font-size: 13px; font-weight: 500; }}
        .link-url-text:hover {{ text-decoration: underline; }}
        .l-card {{ background: #000; padding: 20px; margin: 20px; border-radius: 14px; border: 1px solid var(--border); }}
        .l-dom {{ color: var(--dim); font-size: 11px; font-weight: 800; text-transform: uppercase; margin-bottom: 6px; letter-spacing: 1px; }}

        /* Video thumbnail with play overlay */
        .video-thumb-link {{ position: relative; display: block; }}
        .video-thumb-link img {{ width: 100%; border-radius: 8px; display: block; }}
        .play-overlay {{ position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 52px; height: 52px; background: rgba(0,0,0,0.65); border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 22px; color: #fff; transition: background 0.2s; pointer-events: none; }}
        .video-thumb-link:hover .play-overlay {{ background: rgba(29,155,240,0.85); }}
        .v-con {{ position: relative; padding-bottom: 56.25%; height: 0; background: #000; }}
        .v-con iframe {{ position: absolute; width: 100%; height: 100%; border: 0; }}

        /* Tweet card preview */
        .tweet-card-link {{ text-decoration: none; color: inherit; display: block; margin-top: 10px; }}
        .tc-container {{ border: 1px solid var(--border); border-radius: 14px; overflow: hidden; background: #0b0e14; transition: 0.2s; }}
        .tc-container:hover {{ border-color: var(--accent); }}
        .tc-img img {{ width: 100%; aspect-ratio: 1.91/1; object-fit: cover; border-bottom: 1px solid var(--border); }}
        .tc-content {{ padding: 10px 14px; }}
        .tc-title {{ font-weight: 700; font-size: 14px; margin-bottom: 4px; }}
        .tc-desc {{ color: var(--dim); font-size: 12px; line-height: 1.4; margin-bottom: 6px; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }}
        .tc-site {{ font-size: 11px; text-transform: uppercase; color: var(--dim); letter-spacing: 0.5px; }}

        /* Other tweets wrapper */
        .no-links-group {{ background: var(--card); border: 1px solid var(--border); border-radius: 20px; overflow: hidden; }}

        /* Compact tweet grid (Other Relevant Tweets) */
        .tweet-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; }}
        @media (max-width: 860px) {{ .tweet-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
        @media (max-width: 560px) {{ .tweet-grid {{ grid-template-columns: 1fr; }} }}
        .tweet-mini {{
            display: flex; flex-direction: column; gap: 8px;
            background: var(--card); border: 1px solid var(--border); border-radius: 14px;
            padding: 14px 16px; text-decoration: none; color: var(--text);
            transition: border-color 0.15s, transform 0.15s;
            min-height: 120px;
        }}
        .tweet-mini:hover {{ border-color: var(--accent); transform: translateY(-1px); }}
        .tm-head {{ display: flex; justify-content: space-between; align-items: center; }}
        .tm-author {{ font-weight: 800; font-size: 13px; color: var(--text); }}
        .tm-arrow {{ font-size: 12px; color: var(--dim); }}
        .tweet-mini:hover .tm-author {{ color: var(--accent); }}
        .tm-text {{ margin: 0; font-size: 13px; line-height: 1.45; color: var(--text);
                    display: -webkit-box; -webkit-line-clamp: 5; -webkit-box-orient: vertical;
                    overflow: hidden; white-space: pre-wrap; word-break: break-word; }}
        .tm-metrics {{ margin-top: auto; font-size: 11px; color: var(--dim); font-weight: 500; }}

        /* Scoped overrides so previews fit inside compact cards */
        .tweet-mini .tweet-media {{ margin-top: 4px; border-radius: 10px; }}
        .tweet-mini .tweet-media .media-item img,
        .tweet-mini .tweet-media .media-item video {{ max-height: 180px; }}
        .tweet-mini .tweet-card-link {{ margin-top: 4px; }}
        .tweet-mini .tc-container {{ border-radius: 10px; }}
        .tweet-mini .tc-img img {{ aspect-ratio: 1.91/1; max-height: 140px; }}
        .tweet-mini .tc-title {{ font-size: 12px; }}
        .tweet-mini .tc-desc {{ font-size: 11px; -webkit-line-clamp: 2; }}
        .tweet-mini .tc-site {{ font-size: 10px; }}
        .tweet-mini .tc-content {{ padding: 8px 10px; }}

        footer {{ text-align: center; color: var(--dim); font-size: 13px; margin-top: 100px; padding: 40px; border-top: 1px solid var(--border); }}
    </style>
</head>
<body>
    <div class="con">
        <div class="page-header">
            <img src="{logo_uri}" class="main-logo" alt="">
            <h1 class="main-title">X List Summary</h1>
            <div class="gen-date">Generated on {timestamp}</div>
            {ai_model_html}
        </div>

        <div class="list-card">
            <img src="{profile_img}" class="l-img">
            <div>
                <span class="l-title">{title}</span>
                <span class="l-meta">{owner_line}</span>
            </div>
        </div>

        <div class="stats-row">
            <div class="stat-box">
                <span class="stat-val">{tweet_count}</span>
                <span class="stat-lbl">Tweets Analyzed</span>
            </div>
            <div class="stat-box">
                <span class="stat-val">{link_count}</span>
                <span class="stat-lbl">Shared Links</span>
            </div>
        </div>

        {insights}

        <h2 class="sec-label">&#128172; Other Relevant Tweets</h2>
        <div class="tweet-grid">{individual}</div>

        <footer>
            Generated by X List Summarizer &bull; {timestamp}<br>
            All data fetched directly from official X API via browser session.
        </footer>
    </div>

    <script>
        function toggleRow(idx) {{
            var row = document.getElementById('tweets-' + idx);
            var link = document.querySelector('.tweet-expand-link[data-idx="' + idx + '"]');
            var insightRow = document.getElementById('insight-row-' + idx);
            if (!row) return;
            var isOpen = row.classList.contains('open');
            row.classList.toggle('open', !isOpen);
            if (link) link.classList.toggle('open', !isOpen);
            if (insightRow) insightRow.classList.toggle('open', !isOpen);
        }}
    </script>
</body>
</html>'''


class XApiFetcher(XListFetcher):
    """Fetcher that uses the official X API v2 (Bearer Token) instead of twikit scraping.

    Only public lists are supported in this mode (app-only auth). Link preview cards
    are not exposed by v2, so the 'card' field is always None.
    """

    API_BASE = "https://api.x.com/2"

    def __init__(self, bearer_token: str = '', list_owner=None):
        super().__init__(cookies_path='browser_session/cookies.json', list_owner=list_owner)
        self.bearer_token = (bearer_token or '').strip()
        self._ingestion_provider = (
            OfficialXApiProvider(self.bearer_token) if self.bearer_token else None
        )

    async def fetch_page(
        self,
        list_id: str,
        *,
        pagination_token: str | None = None,
        max_results: int = 100,
        page_number: int = 1,
        run_id: str | None = None,
    ):
        """Delegate stage-1 canonical paging while retaining legacy rendering."""

        if self._ingestion_provider is None:
            raise RuntimeError("No Bearer Token configured for official X API ingestion")
        return await self._ingestion_provider.fetch_page(
            list_id,
            pagination_token=pagination_token,
            max_results=max_results,
            page_number=page_number,
            run_id=run_id,
        )

    async def close_ingestion_provider(self):
        """Close the shared stage-1 HTTP client when an owner lifecycle provides it."""

        if self._ingestion_provider is not None:
            await self._ingestion_provider.aclose()

    def _safe_error(self, error):
        """Return an API exception message with the Bearer Token removed."""
        return redact_sensitive_text(
            super()._safe_error(error),
            secrets=(getattr(self, 'bearer_token', ''),),
        )

    def _headers(self):
        return {'Authorization': f'Bearer {self.bearer_token}', 'User-Agent': 'x-list-summarizer/1.0'}

    async def _get(self, path: str, params: dict = None):
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(f"{self.API_BASE}{path}", headers=self._headers(), params=params or {})
            if resp.status_code == 401:
                raise Exception("401 Unauthorized — invalid or missing Bearer Token.")
            if resp.status_code == 429:
                raise Exception("429 Rate limit — X API quota exceeded.")
            if resp.status_code >= 400:
                raise Exception(f"X API error {resp.status_code}.")
            return resp.json()

    async def login(self):
        if not self.bearer_token:
            return False, "No Bearer Token configured. Please add one in Settings → X Authentication."
        try:
            # Minimal probe: fetch a public handle to validate the token
            await self._get('/users/by/username/x')
            return True, "Bearer Token OK"
        except Exception as e:
            safe_error = self._safe_error(e)
            return False, f"Login failed: {safe_error[:120]}"

    async def verify_session(self, retries=1):
        if not self.bearer_token:
            return False, "No Bearer Token"
        for attempt in range(retries + 1):
            try:
                await self._get('/users/by/username/x')
                return True, "OK (API)"
            except Exception as e:
                err = self._safe_error(e)
                if '401' in err:
                    return False, "Invalid Bearer Token (401)"
                if attempt < retries:
                    await asyncio.sleep(1)
                    continue
                return False, f"API: {err[:30]}"
        return False, "Unknown"

    async def get_user_memberships(self, username: str):
        try:
            username = username.lower().replace('@', '').strip()
            user_resp = await self._get(f'/users/by/username/{username}')
            user_id = user_resp.get('data', {}).get('id')
            if not user_id:
                return []

            memberships = []
            pagination_token = None
            while True:
                params = {
                    'max_results': 100,
                    'list.fields': 'name,owner_id',
                    'expansions': 'owner_id',
                    'user.fields': 'username',
                }
                if pagination_token:
                    params['pagination_token'] = pagination_token

                resp = await self._get(f'/users/{user_id}/list_memberships', params=params)
                users_by_id = {u['id']: u for u in resp.get('includes', {}).get('users', [])}

                for l in resp.get('data', []) or []:
                    owner_user = users_by_id.get(l.get('owner_id'), {})
                    memberships.append({
                        'name': l.get('name', ''),
                        'owner': owner_user.get('username', 'Unknown'),
                        'id': l.get('id', ''),
                    })

                pagination_token = resp.get('meta', {}).get('next_token')
                if not pagination_token or len(memberships) > 500:
                    break

            print(f"✅ Found {len(memberships)} memberships for {username} (API)")
            return memberships
        except Exception as e:
            print(f"❌ Error fetching memberships for {username} via API: {self._safe_error(e)}")
            return []

    async def _fetch_list_metadata(self, list_id: str):
        try:
            resp = await self._get(f'/lists/{list_id}', params={
                'list.fields': 'name,member_count,owner_id',
                'expansions': 'owner_id',
                'user.fields': 'name,username,profile_image_url',
            })
            data = resp.get('data', {}) or {}
            owner = (resp.get('includes', {}).get('users') or [{}])[0]
            return {
                'name': data.get('name'),
                'member_count': data.get('member_count', 0),
                'owner_screen_name': owner.get('username'),
                'owner_name': owner.get('name'),
                'profile_image_url': owner.get('profile_image_url'),
            }
        except Exception as e:
            print(f"⚠️ get_list via API failed for {list_id}: {self._safe_error(e)}")
            return {}

    async def fetch_list_tweets(self, list_url_or_id: str, max_tweets: int = 100, delay: float = 0):
        if delay > 0:
            await asyncio.sleep(delay)

        list_id = self.extract_list_id(list_url_or_id)
        print(f"📋 Fetching list {list_id} via X API...")

        # 1. List metadata
        meta = await self._fetch_list_metadata(list_id)
        if meta.get('name'):
            self.list_info['list_names'].append(meta['name'])
            if self.list_info['name'] == 'X List Summary':
                self.list_info['name'] = meta['name']
        self.list_info['member_count'] += meta.get('member_count', 0) or 0

        if self.list_owner_pref and not self.list_info['profile_image_url']:
            try:
                u_resp = await self._get(f"/users/by/username/{self.list_owner_pref}",
                                         params={'user.fields': 'name,profile_image_url'})
                u = u_resp.get('data', {}) or {}
                self.list_info['owner'] = self.list_owner_pref
                self.list_info['owner_name'] = u.get('name', self.list_owner_pref)
                self.list_info['profile_image_url'] = u.get('profile_image_url')
            except Exception as exc:
                logger.debug("Could not resolve preferred API list owner (%s)", type(exc).__name__)

        if not self.list_info['profile_image_url'] and meta.get('owner_screen_name'):
            self.list_info['owner'] = meta['owner_screen_name']
            self.list_info['owner_name'] = meta.get('owner_name') or meta['owner_screen_name']
            self.list_info['profile_image_url'] = meta.get('profile_image_url')

        # 2. Tweets
        tweets = []
        pagination_token = None
        try:
            while len(tweets) < max_tweets:
                remaining = max_tweets - len(tweets)
                params = {
                    'max_results': min(100, max(5, remaining)),
                    'tweet.fields': 'public_metrics,entities,attachments,author_id,created_at',
                    'expansions': 'author_id,attachments.media_keys',
                    'user.fields': 'username',
                    'media.fields': 'type,url,preview_image_url,variants',
                }
                if pagination_token:
                    params['pagination_token'] = pagination_token

                resp = await self._get(f'/lists/{list_id}/tweets', params=params)
                data = resp.get('data', []) or []
                if not data: break

                includes = resp.get('includes', {}) or {}
                users_by_id = {u['id']: u for u in includes.get('users', [])}
                media_by_key = {m['media_key']: m for m in includes.get('media', [])}

                for t in data:
                    author = users_by_id.get(t.get('author_id'), {})
                    author_handle = author.get('username', 'unknown')

                    # URLs / resolved links
                    resolved_links = set()
                    url_map = {}
                    display_map = {}
                    for u in (t.get('entities', {}).get('urls', []) or []):
                        short = u.get('url')
                        expanded = u.get('expanded_url') or short
                        if short:
                            url_map[short] = expanded
                            display_map[short] = u.get('display_url', expanded)
                    for expanded in url_map.values():
                        if not any(d in expanded.lower() for d in ['x.com', 'twitter.com', 'twimg.com', 't.co']):
                            resolved_links.add(expanded)

                    # Media
                    media = []
                    for mk in (t.get('attachments', {}).get('media_keys', []) or []):
                        m = media_by_key.get(mk)
                        if not m: continue
                        m_type = m.get('type')
                        if m_type == 'photo':
                            media.append({'type': 'photo', 'url': m.get('url'), 'id': mk})
                        elif m_type in ('video', 'animated_gif'):
                            variants = m.get('variants', []) or []
                            best = sorted([v for v in variants if v.get('content_type') == 'video/mp4'],
                                          key=lambda x: x.get('bit_rate', 0), reverse=True)
                            if best:
                                media.append({'type': m_type, 'url': best[0]['url'],
                                              'thumbnail': m.get('preview_image_url'), 'id': mk})

                    # Clean text (replace t.co with expanded/display)
                    clean_text = t.get('text', '')
                    for short, expanded in url_map.items():
                        if short in clean_text:
                            tgt = expanded if not any(d in expanded.lower() for d in ['x.com', 'twitter.com', 'twimg.com']) else display_map.get(short, short)
                            clean_text = clean_text.replace(short, tgt)

                    pm = t.get('public_metrics', {}) or {}
                    tweets.append({
                        'id': t.get('id'), 'text': clean_text, 'author': author_handle,
                        'links': list(resolved_links), 'media': media, 'card': None,
                        'likes': pm.get('like_count', 0),
                        'retweets': pm.get('retweet_count', 0),
                        'replies': pm.get('reply_count', 0),
                        'quotes': pm.get('quote_count', 0),
                        'bookmarks': pm.get('bookmark_count', 0),
                    })

                pagination_token = resp.get('meta', {}).get('next_token')
                if not pagination_token: break

            return tweets
        except Exception as e:
            err = self._safe_error(e)
            if '429' in err:
                raise Exception(f"X API rate limit fetching list {list_id}. Please wait and try again.") from e
            if '401' in err:
                raise Exception(f"X API unauthorized (401) fetching list {list_id}. Please check your Bearer Token in Settings.") from e
            print(f"❌ Error fetching list {list_id} via API: {err}")
            if tweets:
                return tweets
            raise RuntimeError(f"X API list {list_id} fetch failed: {err}") from e
