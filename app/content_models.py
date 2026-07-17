"""Models for evidence-preserving Chinese event analysis."""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from typing import Any


@dataclass(frozen=True)
class ContentItem:
    id: str
    item_type: str
    post_ids: list[str]
    article_ids: list[str]
    author_ids: list[str]
    source_list_ids: list[str]
    created_at: datetime | None
    original_language: str | None
    post_text_original: str
    article_title_original: str | None
    article_excerpt_original: str | None
    analysis_text: str
    analysis_text_hash: str
    source_urls: list[str]
    engagement: dict[str, int | None]


@dataclass(frozen=True)
class RuleFilterResult:
    item_id: str
    decision: str
    reason_codes: list[str]
    confidence: float


@dataclass(frozen=True)
class Claim:
    text_zh: str
    claim_type: str
    confidence: float
    supporting_item_ids: list[str]
    contradicting_item_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class EventCluster:
    id: str
    event_date: date
    title_zh: str
    summary_zh: str
    why_it_matters_zh: str
    event_type: str
    factuality: str
    key_facts: list[Claim]
    item_ids: list[str]
    post_ids: list[str]
    article_ids: list[str]
    source_urls: list[str]
    final_score: float
    score_reasons_zh: list[str]
    eligible_for_digest: bool = True
    eligible_for_must_read: bool = True
    quality_gate_reasons_zh: list[str] = field(default_factory=list)
    must_read: bool = False
    rank: int | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["event_date"] = self.event_date.isoformat()
        return value
