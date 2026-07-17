"""Read saved event rankings for bounded Weixin and command summaries."""

from __future__ import annotations
import json
from datetime import date
from pathlib import Path


def top5_text(
    data_dir: Path, day: date, *, limit: int = 5, maximum: int = 1200
) -> str | None:
    try:
        events = json.loads(
            (data_dir / "events" / f"{day.isoformat()}.json").read_text(
                encoding="utf-8"
            )
        )
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(events, list):
        return None
    rows = []
    for index, event in enumerate(events[:limit], 1):
        if not isinstance(event, dict):
            continue
        title = str(event.get("title_zh", "")).strip()
        why = str(event.get("why_it_matters_zh", "")).strip()
        if title:
            rows.append(f"{index}. {title}\n{why[:100]}")
    if not rows:
        return None
    text = (
        f"AI 每日情报｜{day.isoformat()}\n\n今日收录：{len(events)} 个事件\n今日必读：{len(rows)} 个\n\n"
        + "\n\n".join(rows)
        + "\n\n发送“完整报告”接收 HTML 文件。"
    )
    return text[:maximum]
