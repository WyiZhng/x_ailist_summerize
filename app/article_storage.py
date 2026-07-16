"""Atomic file and in-memory stores for extracted article data."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterator, Protocol, Sequence, runtime_checkable

from .article_models import (
    Article,
    ArticleFetchAttempt,
    PostArticleLink,
    article_id_for_url,
    stable_sha256,
)
from .models import ensure_aware_datetime, utc_now


class ArticleStorageError(RuntimeError):
    pass


class ArticleStorageCorruptionError(ArticleStorageError):
    pass


@runtime_checkable
class ArticleStore(Protocol):
    def get_by_normalized_url(self, normalized_url: str) -> Article | None: ...

    def get_by_original_url(self, original_url: str) -> Article | None: ...

    def save_article(self, article: Article) -> None: ...

    def link_post_article(self, link: PostArticleLink) -> None: ...

    def save_fetch_attempt(self, attempt: ArticleFetchAttempt) -> None: ...

    def should_fetch(self, normalized_url: str, now: datetime) -> bool: ...

    def read_articles(self) -> list[Article]: ...

    def read_links(self) -> list[PostArticleLink]: ...

    def read_attempts(self) -> list[ArticleFetchAttempt]: ...


def _json_line(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _render_jsonl(values: Sequence[dict[str, Any]]) -> str:
    return "".join(f"{_json_line(value)}\n" for value in values)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    values: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("JSONL row is not an object")
                values.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ArticleStorageCorruptionError(
            f"Article storage is malformed: {path.name}"
        ) from exc
    return values


class FileArticleStore:
    """Normalized article metadata plus content-addressed body files."""

    _locks_guard = threading.Lock()
    _locks: dict[str, threading.RLock] = {}

    def __init__(self, data_dir: str | Path = "data", *, cache_ttl_hours: int = 168) -> None:
        self.data_dir = Path(data_dir).expanduser()
        self.root = self.data_dir / "articles"
        self.articles_path = self.root / "articles.jsonl"
        self.links_path = self.root / "post_articles.jsonl"
        self.attempts_path = self.root / "fetch_attempts.jsonl"
        self.content_dir = self.root / "content"
        self.lock_path = self.root / ".articles.lock"
        self.cache_ttl = timedelta(hours=max(0, int(cache_ttl_hours)))
        self.content_dir.mkdir(parents=True, exist_ok=True)
        lock_key = os.path.normcase(str(self.root.resolve(strict=False)))
        with self._locks_guard:
            self._lock = self._locks.setdefault(lock_key, threading.RLock())

    @contextmanager
    def _guard(self) -> Iterator[None]:
        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            with self.lock_path.open("a+b", buffering=0) as handle:
                if handle.seek(0, os.SEEK_END) == 0:
                    handle.write(b"\0")
                    handle.flush()
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    handle.seek(0)
                    if os.name == "nt":
                        import msvcrt

                        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        import fcntl

                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _content_path(self, content_hash: str) -> Path:
        if len(content_hash) != 64 or any(ch not in "0123456789abcdef" for ch in content_hash):
            raise ArticleStorageError("content_hash must be a lowercase SHA-256 digest")
        return self.content_dir / f"{content_hash}.txt"

    def _hydrate(self, article: Article) -> Article:
        if article.content_hash:
            path = self._content_path(article.content_hash)
            try:
                content = path.read_text(encoding="utf-8")
            except FileNotFoundError as exc:
                raise ArticleStorageCorruptionError(
                    "Article metadata references a missing content file"
                ) from exc
            except (OSError, UnicodeError) as exc:
                raise ArticleStorageCorruptionError(
                    "Article content file could not be read"
                ) from exc
            if stable_sha256(content) != article.content_hash:
                raise ArticleStorageCorruptionError("Article content hash does not match")
            article.content_text = content
        return article

    def _article_rows(self) -> list[Article]:
        values = [Article.from_dict(item) for item in _read_jsonl(self.articles_path)]
        ids: set[str] = set()
        urls: set[str] = set()
        for value in values:
            if value.id in ids or value.normalized_url in urls:
                raise ArticleStorageCorruptionError("Duplicate article identity in storage")
            ids.add(value.id)
            urls.add(value.normalized_url)
        return values

    def read_articles(self) -> list[Article]:
        with self._guard():
            return [self._hydrate(item) for item in self._article_rows()]

    def get_by_normalized_url(self, normalized_url: str) -> Article | None:
        return next(
            (item for item in self.read_articles() if item.normalized_url == normalized_url),
            None,
        )

    def get_by_original_url(self, original_url: str) -> Article | None:
        return next(
            (item for item in self.read_articles() if original_url in item.original_urls),
            None,
        )

    def save_article(self, article: Article) -> None:
        if not isinstance(article, Article) or not article.id or not article.normalized_url:
            raise ArticleStorageError("A valid Article is required")
        if article.id != article_id_for_url(article.normalized_url):
            raise ArticleStorageError("Article id must be derived from normalized_url")
        with self._guard():
            articles = self._article_rows()
            existing = next(
                (item for item in articles if item.normalized_url == article.normalized_url),
                None,
            )
            clone = article.clone()
            if existing:
                clone.original_urls = list(
                    dict.fromkeys([*existing.original_urls, *clone.original_urls])
                )
                articles = [
                    value
                    for value in articles
                    if value.normalized_url != clone.normalized_url
                ]
            if clone.content_text is not None:
                calculated = stable_sha256(clone.content_text)
                if clone.content_hash and clone.content_hash != calculated:
                    raise ArticleStorageError("Article content_hash does not match content_text")
                clone.content_hash = calculated
                content_path = self._content_path(calculated)
                if content_path.exists():
                    try:
                        current = content_path.read_text(encoding="utf-8")
                    except (OSError, UnicodeError) as exc:
                        raise ArticleStorageCorruptionError(
                            "Existing article content could not be read"
                        ) from exc
                    if current != clone.content_text:
                        raise ArticleStorageCorruptionError(
                            "A content hash maps to different text"
                        )
                else:
                    _atomic_write(content_path, clone.content_text)
            articles.append(clone)
            articles.sort(key=lambda value: value.id)
            _atomic_write(
                self.articles_path,
                _render_jsonl([value.to_dict(include_content=False) for value in articles]),
            )

    def read_links(self) -> list[PostArticleLink]:
        with self._guard():
            links = [
                PostArticleLink.from_dict(item) for item in _read_jsonl(self.links_path)
            ]
            article_ids = {item.id for item in self._article_rows()}
            if any(link.article_id not in article_ids for link in links):
                raise ArticleStorageCorruptionError(
                    "Post/article relation references a missing article"
                )
            return links

    def link_post_article(self, link: PostArticleLink) -> None:
        if not isinstance(link, PostArticleLink) or not link.post_id or not link.article_id:
            raise ArticleStorageError("A valid PostArticleLink is required")
        with self._guard():
            article_ids = {item.id for item in self._article_rows()}
            if link.article_id not in article_ids:
                raise ArticleStorageError("Cannot create a link to a missing article")
            links = [PostArticleLink.from_dict(item) for item in _read_jsonl(self.links_path)]
            key = (link.post_id, link.article_id, link.original_url)
            if key not in {
                (item.post_id, item.article_id, item.original_url) for item in links
            }:
                links.append(link)
                links.sort(key=lambda item: (item.post_id, item.article_id, item.original_url))
                _atomic_write(
                    self.links_path,
                    _render_jsonl([item.to_dict() for item in links]),
                )

    def read_attempts(self) -> list[ArticleFetchAttempt]:
        with self._guard():
            return [
                ArticleFetchAttempt.from_dict(item)
                for item in _read_jsonl(self.attempts_path)
            ]

    def save_fetch_attempt(self, attempt: ArticleFetchAttempt) -> None:
        if not isinstance(attempt, ArticleFetchAttempt) or not attempt.id:
            raise ArticleStorageError("A valid ArticleFetchAttempt is required")
        with self._guard():
            attempts = [
                ArticleFetchAttempt.from_dict(item)
                for item in _read_jsonl(self.attempts_path)
            ]
            if attempt.id not in {item.id for item in attempts}:
                attempts.append(attempt)
                _atomic_write(
                    self.attempts_path,
                    _render_jsonl([item.to_dict() for item in attempts]),
                )

    def should_fetch(self, normalized_url: str, now: datetime) -> bool:
        now = ensure_aware_datetime(now)
        article = self.get_by_normalized_url(normalized_url)
        if article is None or article.status not in {"fetched", "unchanged"}:
            return True
        return article.checked_at + self.cache_ttl <= now


class InMemoryArticleStore:
    def __init__(self, *, cache_ttl_hours: int = 168) -> None:
        self.cache_ttl = timedelta(hours=max(0, int(cache_ttl_hours)))
        self._articles: dict[str, Article] = {}
        self._links: dict[tuple[str, str, str], PostArticleLink] = {}
        self._attempts: dict[str, ArticleFetchAttempt] = {}

    def read_articles(self) -> list[Article]:
        return [item.clone() for item in self._articles.values()]

    def get_by_normalized_url(self, normalized_url: str) -> Article | None:
        value = self._articles.get(normalized_url)
        return value.clone() if value else None

    def get_by_original_url(self, original_url: str) -> Article | None:
        return next(
            (item.clone() for item in self._articles.values() if original_url in item.original_urls),
            None,
        )

    def save_article(self, article: Article) -> None:
        if not isinstance(article, Article):
            raise ArticleStorageError("A valid Article is required")
        if article.id != article_id_for_url(article.normalized_url):
            raise ArticleStorageError("Article id must be derived from normalized_url")
        clone = article.clone()
        existing = self._articles.get(clone.normalized_url)
        if existing:
            clone.original_urls = list(
                dict.fromkeys([*existing.original_urls, *clone.original_urls])
            )
        if clone.content_text is not None:
            calculated = stable_sha256(clone.content_text)
            if clone.content_hash and clone.content_hash != calculated:
                raise ArticleStorageError("Article content_hash does not match content_text")
            clone.content_hash = calculated
        self._articles[clone.normalized_url] = clone

    def read_links(self) -> list[PostArticleLink]:
        return [PostArticleLink.from_dict(item.to_dict()) for item in self._links.values()]

    def link_post_article(self, link: PostArticleLink) -> None:
        if link.article_id not in {item.id for item in self._articles.values()}:
            raise ArticleStorageError("Cannot create a link to a missing article")
        key = (link.post_id, link.article_id, link.original_url)
        self._links.setdefault(key, PostArticleLink.from_dict(link.to_dict()))

    def read_attempts(self) -> list[ArticleFetchAttempt]:
        return [ArticleFetchAttempt.from_dict(item.to_dict()) for item in self._attempts.values()]

    def save_fetch_attempt(self, attempt: ArticleFetchAttempt) -> None:
        self._attempts.setdefault(
            attempt.id, ArticleFetchAttempt.from_dict(attempt.to_dict())
        )

    def should_fetch(self, normalized_url: str, now: datetime) -> bool:
        now = ensure_aware_datetime(now)
        article = self._articles.get(normalized_url)
        if article is None or article.status not in {"fetched", "unchanged"}:
            return True
        return article.checked_at + self.cache_ttl <= now


__all__ = [
    "ArticleStorageCorruptionError",
    "ArticleStorageError",
    "ArticleStore",
    "FileArticleStore",
    "InMemoryArticleStore",
]
