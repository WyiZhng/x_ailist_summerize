"""Build bounded, injection-delimited analysis inputs from stored evidence."""

from __future__ import annotations
import hashlib
from .content_models import ContentItem
from .models import XPost
from .article_models import Article, PostArticleLink


def build_content_items(
    posts: list[XPost],
    articles: list[Article],
    links: list[PostArticleLink],
    *,
    max_chars: int,
    model_name: str,
    prompt_version: str = "event_synthesis_zh_v1",
) -> list[ContentItem]:
    article_by_id = {a.id: a for a in articles}
    by_post: dict[str, list[Article]] = {}
    for link in links:
        if link.article_id in article_by_id:
            by_post.setdefault(link.post_id, []).append(article_by_id[link.article_id])
    items = []
    for post in sorted(posts, key=lambda p: p.id):
        attached = by_post.get(post.id, [])
        article = attached[0] if attached else None
        excerpt = (
            (article.content_text or article.excerpt or "")[: max_chars // 2]
            if article
            else ""
        )
        source = f"<UNTRUSTED_SOURCE>帖子：{post.text}\n文章标题：{article.title if article else ''}\n文章摘录：{excerpt}</UNTRUSTED_SOURCE>"
        digest = hashlib.sha256(
            (source + prompt_version + model_name).encode()
        ).hexdigest()
        items.append(
            ContentItem(
                id=f"content:{post.id}",
                item_type="post_with_article" if article else "post_only",
                post_ids=[post.id],
                article_ids=[a.id for a in attached],
                author_ids=[post.author_id],
                source_list_ids=[post.source_list_id],
                created_at=post.created_at,
                original_language=post.language,
                post_text_original=post.text,
                article_title_original=article.title if article else None,
                article_excerpt_original=excerpt or None,
                analysis_text=source[:max_chars],
                analysis_text_hash=digest,
                source_urls=list(
                    dict.fromkeys(
                        [*post.urls, *([article.normalized_url] if article else [])]
                    )
                ),
                engagement={
                    "likes": post.metrics.likes,
                    "retweets": post.metrics.retweets,
                    "replies": post.metrics.replies,
                },
            )
        )
    return items
