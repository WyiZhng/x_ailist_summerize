"""Offline-first daily Chinese event pipeline; never fetches X or articles."""

from __future__ import annotations
import argparse
import hashlib
import json
from datetime import date
from pathlib import Path
from .article_storage import FileArticleStore
from .config import load_config
from .content_builder import build_content_items
from .content_filter import filter_item
from .content_models import Claim, EventCluster
from .content_analysis import (
    DeepSeekContentAnalysisProvider,
    MockContentAnalysisProvider,
)
from .storage import FilePostStore

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _zh_title(text: str) -> str:
    names = [
        x
        for x in ("OpenAI", "Anthropic", "Claude", "Gemini", "DeepSeek", "LangGraph")
        if x.lower() in text.lower()
    ]
    subject = names[0] if names else "AI 生态"
    return (
        f"{subject} 发布或更新相关技术动态"
        if "发布" not in text
        else f"{subject} 发布新动态"
    )


def run_pipeline(
    day: date,
    *,
    data_dir: Path,
    config: dict,
    max_items: int = 20,
    use_llm: bool = False,
    mock: bool = False,
) -> tuple[list[EventCluster], dict[str, int]]:
    posts = FilePostStore(data_dir).read_posts()[:max_items]
    article_store = FileArticleStore(data_dir)
    items = build_content_items(
        posts,
        article_store.read_articles(),
        article_store.read_links(),
        max_chars=config["events"]["max_analysis_chars_per_item"],
        model_name=config["summarization"]["options"][
            config["summarization"]["provider"]
        ].get("model", ""),
    )
    provider = (
        (
            MockContentAnalysisProvider()
            if mock
            else DeepSeekContentAnalysisProvider(
                config,
                data_dir / "events" / "llm_cache",
                max_requests=config["events"]["max_llm_requests_per_run"],
                max_tokens=config["events"]["max_prompt_tokens_per_run"],
            )
        )
        if use_llm
        else None
    )
    analyzed: dict[str, object] = {}
    kept = []
    filtered = 0
    for item in items:
        decision = filter_item(item)
        if decision.decision == "drop":
            filtered += 1
        else:
            if provider and decision.decision == "needs_semantic_review":
                try:
                    result = provider.analyze(item)
                    analyzed[item.id] = result
                    if not result.keep:
                        filtered += 1
                        continue
                except Exception:
                    pass
            kept.append(item)
    groups: dict[str, list] = {}
    for item in kept:
        key = (
            item.article_ids[0]
            if item.article_ids
            else hashlib.sha256(
                (
                    " ".join(
                        sorted(
                            set(item.post_text_original.lower().split())
                            - {"the", "a", "an"}
                        )
                    )[:120]
                ).encode()
            ).hexdigest()[:16]
        )
        groups.setdefault(key, []).append(item)
    events = []
    for key, group in sorted(groups.items()):
        primary = group[0]
        score = min(
            100,
            45
            + min(20, len(group) * 8)
            + min(5, sum((x.engagement.get("likes") or 0) for x in group) // 100),
        )
        result = analyzed.get(primary.id)
        title = result.title_zh if result else _zh_title(primary.post_text_original)
        eid = (
            "event:" + hashlib.sha256((day.isoformat() + key).encode()).hexdigest()[:16]
        )
        events.append(
            EventCluster(
                eid,
                day,
                title,
                result.summary_zh
                if result
                else f"来源内容显示：{primary.post_text_original[:120]}",
                result.why_it_matters_zh
                if result
                else "该信息与 AI、开发工具或技术生态相关，建议结合原始来源进一步判断。",
                result.event_type if result else "developer_tool",
                result.factuality if result else "fact",
                [Claim("来源内容已被收录。", "fact", 0.6, [x.id for x in group])],
                [x.id for x in group],
                [p for x in group for p in x.post_ids],
                [a for x in group for a in x.article_ids],
                list(dict.fromkeys(u for x in group for u in x.source_urls)),
                score,
                ["包含可追溯技术来源。"],
            )
        )
    events = sorted(events, key=lambda e: (-e.final_score, e.id))[
        : config["events"]["max_events_per_day"]
    ]
    events = [
        EventCluster(
            **{
                **e.__dict__,
                "must_read": i < config["events"]["must_read_limit"],
                "rank": i + 1,
            }
        )
        for i, e in enumerate(events)
        if e.final_score >= config["events"]["minimum_final_score"]
    ]
    path = data_dir / "events"
    path.mkdir(parents=True, exist_ok=True)
    target = path / f"{day.isoformat()}.json"
    payload = [e.to_dict() for e in events]
    if not target.exists() or json.loads(target.read_text(encoding="utf-8")) != payload:
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return events, {
        "input_posts": len(posts),
        "filtered": filtered,
        "items": len(items),
        "clusters": len(groups),
        "events": len(events),
        "must_read": sum(e.must_read for e in events),
        "llm_requests": provider.requests if provider else 0,
        "cache_hits": provider.hits if provider else 0,
        "prompt_tokens": provider.prompt_tokens if provider else 0,
        "completion_tokens": provider.completion_tokens if provider else 0,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--date", required=True)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--mock", action="store_true")
    p.add_argument("--no-llm", action="store_true")
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--max-items", type=int, default=20)
    p.add_argument("--language", default="zh-CN")
    a = p.parse_args(argv)
    c = load_config(PROJECT_ROOT / "config.json")
    events, stats = run_pipeline(
        date.fromisoformat(a.date),
        data_dir=PROJECT_ROOT / a.data_dir,
        config=c,
        max_items=min(20, a.max_items),
        use_llm=not a.no_llm,
        mock=a.mock,
    )
    print("中文事件分析完成：" + json.dumps(stats, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
