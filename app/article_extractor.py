"""Pure-text metadata and body extraction from untrusted HTML."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from bs4 import BeautifulSoup

from .article_models import stable_sha256
from .models import parse_datetime


_SPACE = re.compile(r"[ \t\f\v]+")
_BLANK_LINES = re.compile(r"\n{3,}")
_COOKIE_NOISE = re.compile(r"(?:cookie|consent|gdpr|privacy-banner)", re.IGNORECASE)


@dataclass(slots=True)
class ExtractedArticle:
    title: str | None
    author: str | None
    published_at: datetime | None
    site_name: str | None
    language: str | None
    excerpt: str | None
    content_text: str | None
    content_hash: str | None
    word_count: int | None
    canonical_urls: list[str] = field(default_factory=list)
    extractor: str = "beautifulsoup"
    content_truncated: bool = False


def _clean_scalar(value: Any, limit: int) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = _SPACE.sub(" ", value).strip()
    return cleaned[:limit] if cleaned else None


def _meta(soup: BeautifulSoup, *keys: str) -> str | None:
    wanted = {item.lower() for item in keys}
    for tag in soup.find_all("meta"):
        key = str(
            tag.get("property") or tag.get("name") or tag.get("itemprop") or ""
        ).lower()
        if key in wanted:
            value = _clean_scalar(tag.get("content"), 2000)
            if value:
                return value
    return None


def _parse_published(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return parse_datetime(value, field_name="published_at")
    except (TypeError, ValueError):
        return None


def _visible_text_blocks(container) -> list[str]:
    blocks: list[str] = []
    for tag in container.find_all(["h1", "h2", "h3", "p", "blockquote", "li", "pre"]):
        text = _clean_scalar(tag.get_text(" ", strip=True), 20_000)
        if not text or len(text) < 2:
            continue
        if not blocks or blocks[-1] != text:
            blocks.append(text)
    if not blocks:
        fallback = container.get_text("\n", strip=True)
        blocks = [line.strip() for line in fallback.splitlines() if line.strip()]
    return blocks


def extract_article(html: str, *, max_article_chars: int = 50_000) -> ExtractedArticle:
    if not isinstance(html, str) or not html.strip():
        return ExtractedArticle(None, None, None, None, None, None, None, None, None)
    # BeautifulSoup 4.15 keeps the remainder of a malformed document inside an
    # unclosed title element. Repair only this narrow, deterministic pattern so
    # body/article fallbacks remain available across parser versions.
    title_open = re.search(r"<title(?:\s[^>]*)?>", html, flags=re.IGNORECASE)
    head_close = re.search(r"</head\s*>", html, flags=re.IGNORECASE)
    if title_open and head_close and title_open.end() < head_close.start():
        between = html[title_open.end() : head_close.start()]
        if not re.search(r"</title\s*>", between, flags=re.IGNORECASE):
            html = html[: head_close.start()] + "</title>" + html[head_close.start() :]
    soup = BeautifulSoup(html, "html.parser")

    canonical_urls = list(
        dict.fromkeys(
            str(tag.get("href")).strip()
            for tag in soup.find_all("link")
            if tag.get("href")
            and "canonical" in [str(item).lower() for item in (tag.get("rel") or [])]
        )
    )
    title = _meta(soup, "og:title") or _meta(soup, "twitter:title")
    if not title and soup.title:
        title = _clean_scalar(soup.title.get_text(" ", strip=True), 500)
    if not title:
        heading = soup.find("h1")
        title = (
            _clean_scalar(heading.get_text(" ", strip=True), 500) if heading else None
        )
    author = _meta(soup, "author", "article:author", "byl")
    published_at = _parse_published(
        _meta(
            soup,
            "article:published_time",
            "datepublished",
            "pubdate",
            "date",
            "dc.date",
        )
    )
    site_name = _meta(soup, "og:site_name", "application-name")
    excerpt = _meta(soup, "og:description", "description", "twitter:description")
    html_tag = soup.find("html")
    language = _clean_scalar(html_tag.get("lang"), 40) if html_tag else None

    for tag in soup.find_all(
        [
            "script",
            "style",
            "nav",
            "header",
            "footer",
            "form",
            "noscript",
            "svg",
            "iframe",
        ]
    ):
        tag.decompose()
    for tag in soup.find_all(True):
        # Decomposing a noise container also invalidates descendants that were
        # already captured by find_all(). BeautifulSoup leaves those detached
        # tags in the snapshot with attrs=None, so skip them safely.
        if tag.attrs is None:
            continue
        style = str(tag.get("style") or "").replace(" ", "").lower()
        classes = " ".join(str(item) for item in (tag.get("class") or []))
        identity = f"{tag.get('id') or ''} {classes}"
        if (
            tag.has_attr("hidden")
            or str(tag.get("aria-hidden") or "").lower() == "true"
            or "display:none" in style
            or "visibility:hidden" in style
            or _COOKIE_NOISE.search(identity)
        ):
            tag.decompose()

    container = soup.find("article") or soup.find("main") or soup.body
    if container is None:
        return ExtractedArticle(
            title,
            author,
            published_at,
            site_name,
            language,
            excerpt,
            None,
            None,
            None,
            canonical_urls,
        )
    content = "\n\n".join(_visible_text_blocks(container))
    content = _BLANK_LINES.sub("\n\n", content).strip()
    if len(content) < 40:
        content = ""
    truncated = len(content) > max_article_chars
    if truncated:
        content = content[:max_article_chars].rstrip()
    content_text = content or None
    content_hash = stable_sha256(content_text) if content_text else None
    word_count = (
        len(re.findall(r"\b\w+\b", content_text, flags=re.UNICODE))
        if content_text
        else None
    )
    return ExtractedArticle(
        title=title,
        author=author,
        published_at=published_at,
        site_name=site_name,
        language=language,
        excerpt=excerpt,
        content_text=content_text,
        content_hash=content_hash,
        word_count=word_count,
        canonical_urls=canonical_urls,
        content_truncated=truncated,
    )


__all__ = ["ExtractedArticle", "extract_article"]
