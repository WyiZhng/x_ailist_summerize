"""Safe mobile-friendly Chinese HTML rendering for structured events."""

from __future__ import annotations
from html import escape
from pathlib import Path
from typing import Mapping
from .content_models import EventCluster
from .security import sanitize_external_url


def render_chinese_report(
    events: list[EventCluster],
    path: Path,
    *,
    source_text_by_item: Mapping[str, str] | None = None,
) -> None:
    source_text_by_item = source_text_by_item or {}
    cards = []
    for event in events:
        links = " ".join(
            f'<a rel="noopener noreferrer" target="_blank" href="{escape(sanitize_external_url(url), quote=True)}">查看原文</a>'
            for url in event.source_urls
            if sanitize_external_url(url)
        )
        facts = "".join(f"<li>{escape(x.text_zh)}</li>" for x in event.key_facts)
        originals = "".join(
            f"<li>{escape(source_text_by_item[item_id][:800])}</li>"
            for item_id in event.item_ids
            if source_text_by_item.get(item_id)
        )
        original_section = (
            f"<details><summary>查看原始来源摘录</summary><ul>{originals}</ul></details>"
            if originals
            else ""
        )
        gate = "".join(
            f"<li>{escape(reason)}</li>" for reason in event.quality_gate_reasons_zh
        )
        cards.append(
            f"<article><h2>{escape(event.title_zh)}</h2><p>{escape(event.summary_zh)}</p><h3>为什么值得关注</h3><p>{escape(event.why_it_matters_zh)}</p><ul>{facts}</ul><p>价值评分：{event.final_score:.0f}</p><ul>{gate}</ul><p>{links}</p>{original_section}</article>"
        )
    top = "".join(f"<li>{escape(e.title_zh)}</li>" for e in events if e.must_read)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AI 每日情报</title><style>body{{font-family:-apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif;max-width:760px;margin:auto;padding:16px;line-height:1.65}}article{{border:1px solid #ddd;border-radius:12px;padding:16px;margin:12px 0}}</style><h1>AI 每日情报</h1><h2>今日必读</h2><ol>{top}</ol>{"".join(cards)}',
        encoding="utf-8",
    )
