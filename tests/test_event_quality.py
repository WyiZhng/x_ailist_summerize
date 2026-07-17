from datetime import datetime, timezone

from app.content_models import ContentItem
from app.event_quality import evaluate_event_quality


def item(text: str, *, likes: int = 0) -> ContentItem:
    return ContentItem(
        id="content:1",
        item_type="post_only",
        post_ids=["1"],
        article_ids=[],
        author_ids=["author"],
        source_list_ids=["list"],
        created_at=datetime(2026, 7, 17, tzinfo=timezone.utc),
        original_language="zh",
        post_text_original=text,
        article_title_original=None,
        article_excerpt_original=None,
        analysis_text=text,
        analysis_text_hash="hash",
        source_urls=[],
        engagement={"likes": likes},
    )


def test_marketing_growth_case_stays_in_digest_but_not_must_read():
    result = evaluate_event_quality(
        event_type="technical_analysis",
        factuality="fact",
        items=[item("我用 AI 视频让浏览量提高 10 倍，这是个人增长案例", likes=10000)],
    )
    assert result.eligible_for_digest
    assert not result.eligible_for_must_read
    assert result.final_score < 70
    assert any(
        "增长" in reason or "个人" in reason
        for reason in result.quality_gate_reasons_zh
    )


def test_low_engagement_security_event_is_must_read():
    result = evaluate_event_quality(
        event_type="security",
        factuality="fact",
        items=[item("OpenAI 修复 API 身份验证安全漏洞")],
    )
    assert result.eligible_for_digest and result.eligible_for_must_read
    assert result.final_score >= 70


def test_obvious_marketing_is_not_digest_eligible():
    result = evaluate_event_quality(
        event_type="marketing",
        factuality="opinion",
        items=[item("订阅课程并加入社群可获得优惠")],
    )
    assert not result.eligible_for_digest
