from datetime import datetime, timezone

from app.content_models import ContentItem
from app.event_recall import recall_pairs


def item(
    identifier: str, text: str, *, article: str = "", url: str = ""
) -> ContentItem:
    return ContentItem(
        id=identifier,
        item_type="post_only",
        post_ids=[identifier],
        article_ids=[article] if article else [],
        author_ids=["a"],
        source_list_ids=["l"],
        created_at=datetime(2026, 7, 17, tzinfo=timezone.utc),
        original_language="en",
        post_text_original=text,
        article_title_original=None,
        article_excerpt_original=None,
        analysis_text=text,
        analysis_text_hash=identifier,
        source_urls=[url] if url else [],
        engagement={},
    )


def test_recall_marks_shared_article_as_strong_and_stable():
    values = [
        item("b", "Comment on Claude Code", article="article-1"),
        item("a", "Official Claude Code release", article="article-1"),
    ]
    first = recall_pairs(values)
    second = recall_pairs(reversed(values))
    assert len(first) == 1 and first[0].strong and "相同已保存文章" in first[0].reasons
    assert first == second


def test_recall_does_not_use_company_name_alone_as_high_signal():
    pairs = recall_pairs(
        [
            item("a", "OpenAI changes API pricing"),
            item("b", "OpenAI hires a finance executive"),
        ]
    )
    assert pairs == []


def test_recall_keeps_same_product_multilingual_pair_for_review():
    pairs = recall_pairs(
        [
            item("a", "Anthropic released Claude Code version 2"),
            item("b", "Anthropic 发布 Claude Code 版本 2"),
        ]
    )
    assert len(pairs) == 1 and not pairs[0].strong and "核心词重叠" in pairs[0].reasons
