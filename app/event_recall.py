"""Deterministic, bounded candidate-pair recall for same-day events."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from typing import Iterable

from .content_models import ContentItem

_STOP = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "this",
    "that",
    "ai",
    "agent",
    "model",
    "api",
    "open",
    "发布",
    "更新",
    "一个",
    "有关",
}


@dataclass(frozen=True)
class CandidatePair:
    """A locally recalled pair, with deterministic evidence for review."""

    candidate_a_id: str
    candidate_b_id: str
    score: float
    reasons: list[str]
    strong: bool

    @property
    def id(self) -> str:
        return hashlib.sha256(
            (self.candidate_a_id + "|" + self.candidate_b_id).encode()
        ).hexdigest()[:20]

    def to_dict(self) -> dict:
        value = asdict(self)
        value["id"] = self.id
        return value


def _tokens(item: ContentItem) -> set[str]:
    source = " ".join(
        x for x in (item.post_text_original, item.article_title_original or "") if x
    ).lower()
    words = re.findall(r"[a-z][a-z0-9._-]{1,}|[\u4e00-\u9fff]{2,}", source)
    return {word for word in words if word not in _STOP}


def _similarity(left: ContentItem, right: ContentItem) -> tuple[float, list[str], bool]:
    reasons: list[str] = []
    if set(left.article_ids) & set(right.article_ids):
        reasons.append("相同已保存文章")
    if set(left.source_urls) & set(right.source_urls):
        reasons.append("相同规范化来源 URL")
    if set(left.post_ids) & set(right.post_ids):
        reasons.append("相同原始帖子")
    if re.sub(r"\W+", "", left.post_text_original.lower()) == re.sub(
        r"\W+", "", right.post_text_original.lower()
    ):
        reasons.append("规范化帖子正文一致")
    strong = bool(reasons)
    left_tokens, right_tokens = _tokens(left), _tokens(right)
    overlap = left_tokens & right_tokens
    union = left_tokens | right_tokens
    lexical = len(overlap) / len(union) if union else 0.0
    if len(overlap) >= 2:
        reasons.append("核心词重叠")
    if left.created_at and right.created_at:
        hours = abs((left.created_at - right.created_at).total_seconds()) / 3600
        if hours <= 24 and len(overlap) >= 2:
            reasons.append("发布时间接近")
            lexical += 0.05
    # A company-name match alone is intentionally weak: it must be accompanied
    # by another signal before a pair is recalled.
    capitals = {x for x in left_tokens & right_tokens if len(x) > 3}
    if len(capitals) >= 2 and lexical >= 0.12:
        reasons.append("实体与文本相似")
        lexical += 0.08
    return min(1.0, lexical), reasons, strong


def recall_pairs(
    items: Iterable[ContentItem], *, top_k: int = 6, minimum_score: float = 0.18
) -> list[CandidatePair]:
    """Return stable Top-K pairs without an all-pairs LLM comparison."""
    ordered = sorted(items, key=lambda item: item.id)
    per_item: dict[str, list[CandidatePair]] = {item.id: [] for item in ordered}
    for index, left in enumerate(ordered):
        for right in ordered[index + 1 :]:
            score, reasons, strong = _similarity(left, right)
            if strong or (
                len(_tokens(left) & _tokens(right)) >= 2 and score >= minimum_score
            ):
                pair = CandidatePair(
                    left.id, right.id, round(score, 4), reasons, strong
                )
                per_item[left.id].append(pair)
                per_item[right.id].append(pair)
    selected: dict[str, CandidatePair] = {}
    for values in per_item.values():
        for pair in sorted(
            values, key=lambda pair: (-pair.strong, -pair.score, pair.id)
        )[:top_k]:
            selected[pair.id] = pair
    return sorted(
        selected.values(), key=lambda pair: (pair.candidate_a_id, pair.candidate_b_id)
    )
