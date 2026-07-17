"""Conservative deterministic filter before any LLM request."""

from __future__ import annotations
import re
from .content_models import ContentItem, RuleFilterResult

_DROP = {
    "招聘": "job_post",
    "hiring": "job_post",
    "discount": "promotion",
    "优惠码": "promotion",
    "抽奖": "giveaway",
    "join my community": "promotion",
}
_KEEP = (
    "release",
    "发布",
    "security",
    "漏洞",
    "benchmark",
    "paper",
    "论文",
    "api",
    "open source",
    "开源",
    "price",
    "价格",
)


def filter_item(item: ContentItem) -> RuleFilterResult:
    text = item.post_text_original.strip()
    lower = text.lower()
    if not text or re.fullmatch(r"[\W_]+", text):
        return RuleFilterResult(item.id, "drop", ["empty_or_emoji"], 0.99)
    if any(word in lower for word in _KEEP):
        return RuleFilterResult(item.id, "keep", ["high_value_signal"], 0.9)
    for word, reason in _DROP.items():
        if word.lower() in lower:
            return RuleFilterResult(item.id, "drop", [reason], 0.9)
    if len(re.sub(r"https?://\S+", "", text).strip()) < 8:
        return RuleFilterResult(item.id, "drop", ["contextless_link"], 0.8)
    return RuleFilterResult(
        item.id, "needs_semantic_review", ["requires_semantic_review"], 0.5
    )
