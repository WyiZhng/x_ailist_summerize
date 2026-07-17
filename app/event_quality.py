"""Deterministic quality gates and explainable event scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .content_models import ContentItem

_HIGH_VALUE = {"security", "model_release", "policy", "acquisition", "funding"}
_PRODUCT_VALUE = {
    "product_release",
    "developer_tool",
    "open_source",
    "research",
    "benchmark",
}
_ANALYSIS_VALUE = {"technical_analysis", "industry_analysis", "opinion", "tutorial"}
_DROP_TYPES = {"marketing", "job_post", "personal_chat", "meme"}
_PROMOTION_MARKERS = {
    "课程",
    "社群",
    "订阅",
    "优惠",
    "折扣",
    "推广",
    "流量",
    "浏览量",
    "涨粉",
    "growth hack",
    "join my",
    "subscribe",
    "discount",
    "course",
    "community",
}
_PERSONAL_CASE_MARKERS = {
    "我的",
    "我用",
    "一人",
    "个人",
    "经验分享",
    "案例",
    "倍产出",
    "i used",
    "my workflow",
    "my revenue",
    "my views",
}
_ACTION_MARKERS = {
    "发布",
    "推出",
    "更新",
    "修复",
    "开源",
    "收购",
    "融资",
    "宣布",
    "披露",
    "下调",
    "上涨",
    "released",
    "launched",
    "updated",
    "fixed",
    "open-sourced",
    "acquired",
    "announced",
}


@dataclass(frozen=True)
class QualityGateResult:
    eligible_for_digest: bool
    eligible_for_must_read: bool
    final_score: float
    score_reasons_zh: list[str]
    quality_gate_reasons_zh: list[str]


def evaluate_event_quality(
    *,
    event_type: str,
    factuality: str,
    items: Iterable[ContentItem],
    synthesized_text: str = "",
) -> QualityGateResult:
    """Calculate a bounded score and must-read eligibility without an LLM."""
    values = list(items)
    text = (
        " ".join(
            [item.post_text_original for item in values]
            + [item.article_title_original or "" for item in values]
        ).lower()
        + " "
        + synthesized_text.lower()
    )
    promotion = sum(marker in text for marker in _PROMOTION_MARKERS)
    personal_case = sum(marker in text for marker in _PERSONAL_CASE_MARKERS)
    has_action = any(marker in text for marker in _ACTION_MARKERS)
    source_chains = {
        item.article_ids[0] if item.article_ids else item.author_ids[0]
        for item in values
        if item.article_ids or item.author_ids
    }
    independent_sources = max(1, len(source_chains)) if values else 0
    engagement = sum(
        sum(
            int(item.engagement.get(key) or 0)
            for key in ("likes", "retweets", "replies")
        )
        for item in values
    )

    importance = (
        24
        if event_type in _HIGH_VALUE
        else 19
        if event_type in _PRODUCT_VALUE
        else 17
        if event_type in _ANALYSIS_VALUE
        else 12
    )
    credibility = 15 if factuality == "fact" else 12 if factuality == "mixed" else 8
    credibility += min(5, max(0, independent_sources - 1) * 2)
    novelty = 17 if has_action else 10
    relevance = (
        18
        if event_type in (_HIGH_VALUE | _PRODUCT_VALUE)
        else 17
        if event_type in _ANALYSIS_VALUE
        else 12
    )
    diversity = min(10, independent_sources * 3)
    engagement_score = min(5, engagement // 200)
    score = (
        importance + credibility + novelty + relevance + diversity + engagement_score
    )
    reasons = [
        f"事件类型贡献 {importance} 分。",
        f"事实支持与来源可信度贡献 {credibility} 分。",
        f"明确动作与新颖度贡献 {novelty} 分。",
        f"与 AI 技术情报相关性贡献 {relevance} 分。",
        f"独立来源多样性贡献 {diversity} 分。",
        f"互动量贡献 {engagement_score} 分，且上限为 5 分。",
    ]
    gate_reasons: list[str] = []
    digest = event_type not in _DROP_TYPES
    must_read = digest
    if event_type in _DROP_TYPES:
        gate_reasons.append("内容类型属于营销、招聘、个人闲聊或 Meme，不进入日报。")
    promotion_dominates = promotion >= 2 or event_type not in (
        _HIGH_VALUE | _PRODUCT_VALUE
    )
    if promotion and promotion_dominates:
        must_read = False
        score -= min(20, 8 + promotion * 4)
        gate_reasons.append("内容包含推广、流量增长或转化导向信号，不具备必读资格。")
    if personal_case:
        must_read = False
        score -= min(12, personal_case * 4)
        gate_reasons.append("内容主要是单一来源的个人经验或增长案例，不具备必读资格。")
    if not has_action and event_type not in _HIGH_VALUE:
        must_read = False
        score -= 6
        gate_reasons.append("缺少清晰、可核验的事件动作，不具备必读资格。")
    if (
        factuality in {"opinion", "speculation", "prediction", "unclear"}
        and independent_sources == 1
    ):
        must_read = False
        score -= 8
        gate_reasons.append("单一来源的观点或推测可信度有限，不具备必读资格。")
    if event_type in _HIGH_VALUE and factuality in {"fact", "mixed"}:
        must_read = digest
        gate_reasons.append("模型发布、安全、政策或公司重大事件具备优先阅读价值。")
    if must_read:
        gate_reasons.append("主体、动作和技术情报价值满足今日必读要求。")
    return QualityGateResult(
        digest,
        must_read,
        float(max(0, min(100, score))),
        reasons,
        gate_reasons or ["内容可进入普通日报事件列表。"],
    )
