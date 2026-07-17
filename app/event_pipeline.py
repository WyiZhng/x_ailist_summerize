"""Offline-first daily Chinese event pipeline; never fetches X or articles."""

from __future__ import annotations
import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
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
from .event_recall import recall_pairs
from .event_quality import evaluate_event_quality
from .event_report import render_chinese_report
from .storage import FilePostStore

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = handle.name
    os.replace(temporary, path)


def _atomic_json_list(path: Path, value: list[dict]) -> None:
    _atomic_json(path, value)


def _load_manifest(path: Path, day: date, items: list) -> dict:
    """Keep completed work when a same-day run is resumed."""
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("date") == day.isoformat():
            saved.setdefault("completed_item_ids", [])
            saved.setdefault("filtered_item_ids", [])
            saved.setdefault("failed_item_ids", [])
            saved.setdefault("pending_item_ids", [])
            saved.setdefault("failures", {})
            saved["failures"] = {
                item_id: (
                    value
                    if isinstance(value, dict)
                    else {
                        "category": "permanent_invalid"
                        if value == "ValueError"
                        else "retryable_failed",
                        "retry_count": 2 if value == "ValueError" else 1,
                        "last_error": str(value),
                    }
                )
                for item_id, value in saved["failures"].items()
            }
            saved["failed_item_ids"] = sorted(saved["failures"])
            return saved
    except (OSError, ValueError, json.JSONDecodeError):
        pass
    return {
        "run_id": f"event:{day.isoformat()}",
        "date": day.isoformat(),
        "status": "pending",
        "stage": "content_load",
        "input_content_ids": [x.id for x in items],
        "completed_item_ids": [],
        "filtered_item_ids": [],
        "failed_item_ids": [],
        "pending_item_ids": [x.id for x in items],
        "failures": {},
        "llm_requests": 0,
        "cache_hits": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def _failure_category(exc: Exception) -> str:
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    if "budget" in message:
        return "budget_pending"
    if (
        "timeout" in name
        or "timeout" in message
        or "429" in message
        or "5" in message[:1]
    ):
        return "retryable_failed"
    if isinstance(exc, ValueError):
        return "permanent_invalid"
    return "retryable_failed"


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
    analysis_provider=None,
) -> tuple[list[EventCluster], dict[str, int]]:
    posts = [
        post
        for post in FilePostStore(data_dir).read_posts()
        if post.created_at is None or post.created_at.date() == day
    ][:max_items]
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
    provider = analysis_provider or (
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
    manifest_path = data_dir / "events" / f"{day.isoformat()}.manifest.json"
    manifest = _load_manifest(manifest_path, day, items)
    manifest.update(
        {
            "status": "running",
            "stage": "semantic_filter",
            "input_content_ids": [x.id for x in items],
        }
    )
    _atomic_json(manifest_path, manifest)
    analyzed: dict[str, object] = {}
    kept = []
    filtered = 0
    for item in items:
        decision = filter_item(item)
        if decision.decision == "drop":
            filtered += 1
            if item.id not in manifest["filtered_item_ids"]:
                manifest["filtered_item_ids"].append(item.id)
        else:
            failure = manifest.get("failures", {}).get(item.id)
            if failure and (
                failure.get("category") == "permanent_invalid"
                or int(failure.get("retry_count", 0)) >= 3
            ):
                continue
            if provider and decision.decision == "needs_semantic_review":
                try:
                    result = provider.analyze(item)
                    analyzed[item.id] = result
                    if not result.keep:
                        filtered += 1
                        if item.id not in manifest["filtered_item_ids"]:
                            manifest["filtered_item_ids"].append(item.id)
                        if item.id not in manifest["completed_item_ids"]:
                            manifest["completed_item_ids"].append(item.id)
                        manifest["failures"].pop(item.id, None)
                        if item.id in manifest["failed_item_ids"]:
                            manifest["failed_item_ids"].remove(item.id)
                        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
                        manifest["llm_requests"] = provider.requests
                        manifest["cache_hits"] = provider.hits
                        manifest["prompt_tokens"] = provider.prompt_tokens
                        manifest["completion_tokens"] = provider.completion_tokens
                        _atomic_json(manifest_path, manifest)
                        continue
                except Exception as exc:
                    previous = manifest.setdefault("failures", {}).get(item.id, {})
                    retry_count = int(previous.get("retry_count", 0)) + 1
                    category = _failure_category(exc)
                    if category == "retryable_failed" and retry_count >= 3:
                        category = "permanent_invalid"
                    if item.id not in manifest["failed_item_ids"]:
                        manifest["failed_item_ids"].append(item.id)
                    manifest["failures"][item.id] = {
                        "category": category,
                        "retry_count": retry_count,
                        "last_error": type(exc).__name__,
                    }
                    _atomic_json(manifest_path, manifest)
                    # A failed semantic decision is not evidence for a public
                    # event; preserve it for resume without emitting a fallback.
                    continue
                else:
                    if item.id not in manifest["completed_item_ids"]:
                        manifest["completed_item_ids"].append(item.id)
                    manifest["failures"].pop(item.id, None)
                    if item.id in manifest["failed_item_ids"]:
                        manifest["failed_item_ids"].remove(item.id)
                    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
                    manifest["llm_requests"] = provider.requests
                    manifest["cache_hits"] = provider.hits
                    manifest["prompt_tokens"] = provider.prompt_tokens
                    manifest["completion_tokens"] = provider.completion_tokens
                    _atomic_json(manifest_path, manifest)
            kept.append(item)
    manifest["stage"] = "candidate_recall"
    pairs = recall_pairs(kept, top_k=int(config["events"].get("pair_review_top_k", 6)))
    event_dir = data_dir / "events"
    _atomic_json_list(
        event_dir / f"{day.isoformat()}.candidate_pairs.json",
        [x.to_dict() for x in pairs],
    )
    manifest["candidate_pair_count"] = len(pairs)
    manifest["strong_pair_count"] = sum(pair.strong for pair in pairs)
    # Strong links are safe deterministic joins. Grey pairs receive a bounded
    # semantic decision only when an LLM provider explicitly supports it.
    parent = {item.id: item.id for item in kept}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def join(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for pair in pairs:
        if pair.strong:
            join(pair.candidate_a_id, pair.candidate_b_id)
    by_id = {item.id: item for item in kept}
    decision_path = event_dir / f"{day.isoformat()}.pair_decisions.json"
    try:
        saved_decisions = json.loads(decision_path.read_text(encoding="utf-8"))
        decisions: list[dict] = (
            saved_decisions if isinstance(saved_decisions, list) else []
        )
    except (OSError, ValueError, json.JSONDecodeError):
        decisions = []
    decisions_by_id = {
        str(item.get("id")): item for item in decisions if isinstance(item, dict)
    }
    if provider and hasattr(provider, "analyze_pair"):
        manifest["stage"] = "pair_review"
        for pair in pairs:
            if pair.strong:
                continue
            existing = decisions_by_id.get(pair.id)
            if existing and existing.get("relationship") in {
                "same_event",
                "follow_up",
                "related_topic",
                "contradictory_report",
                "unrelated",
                "unclear",
            }:
                if (
                    existing.get("relationship") == "same_event"
                    and float(existing.get("confidence", 0)) >= 0.7
                ):
                    join(pair.candidate_a_id, pair.candidate_b_id)
                continue
            try:
                decision = provider.analyze_pair(
                    by_id[pair.candidate_a_id], by_id[pair.candidate_b_id]
                )
                record = {
                    **pair.to_dict(),
                    "relationship": decision.relationship,
                    "confidence": decision.confidence,
                    "reason_zh": decision.reason_zh,
                }
                decisions.append(record)
                decisions_by_id[pair.id] = record
                if decision.same_event:
                    join(pair.candidate_a_id, pair.candidate_b_id)
            except Exception as exc:
                decisions.append(
                    {
                        **pair.to_dict(),
                        "relationship": "unclear",
                        "confidence": 0.0,
                        "reason_zh": "候选关系暂未完成判断。",
                        "error": type(exc).__name__,
                    }
                )
                decisions_by_id[pair.id] = decisions[-1]
            _atomic_json_list(
                event_dir / f"{day.isoformat()}.pair_decisions.json", decisions
            )
            manifest["llm_requests"] = provider.requests
            manifest["cache_hits"] = provider.hits
            manifest["prompt_tokens"] = provider.prompt_tokens
            manifest["completion_tokens"] = provider.completion_tokens
            _atomic_json(manifest_path, manifest)
    groups: dict[str, list] = {}
    for item in kept:
        groups.setdefault(find(item.id), []).append(item)
    _atomic_json_list(
        event_dir / f"{day.isoformat()}.clusters.draft.json",
        [
            {
                "cluster_id": key,
                "item_ids": sorted(item.id for item in value),
                "merge_reasons": [
                    pair.to_dict()
                    for pair in pairs
                    if pair.strong
                    and pair.candidate_a_id in {item.id for item in value}
                    and pair.candidate_b_id in {item.id for item in value}
                ],
            }
            for key, value in sorted(groups.items())
        ],
    )
    events = []
    for key, group in sorted(groups.items()):
        primary = group[0]
        result = analyzed.get(primary.id)
        quality = evaluate_event_quality(
            event_type=result.event_type if result else "developer_tool",
            factuality=result.factuality if result else "fact",
            items=group,
            synthesized_text=(
                f"{result.title_zh} {result.summary_zh} {result.why_it_matters_zh}"
                if result
                else ""
            ),
        )
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
                quality.final_score,
                quality.score_reasons_zh,
                quality.eligible_for_digest,
                quality.eligible_for_must_read,
                quality.quality_gate_reasons_zh,
            )
        )
    events = sorted(
        (event for event in events if event.eligible_for_digest),
        key=lambda e: (-e.final_score, e.id),
    )[: config["events"]["max_events_per_day"]]
    ranked: list[EventCluster] = []
    must_read_count = 0
    for event in events:
        if event.final_score < config["events"]["minimum_final_score"]:
            continue
        is_must_read = (
            event.eligible_for_must_read
            and must_read_count < config["events"]["must_read_limit"]
        )
        if is_must_read:
            must_read_count += 1
        ranked.append(
            EventCluster(
                **{**event.__dict__, "must_read": is_must_read, "rank": len(ranked) + 1}
            )
        )
    events = ranked
    path = event_dir
    path.mkdir(parents=True, exist_ok=True)
    target = path / f"{day.isoformat()}.json"
    payload = [e.to_dict() for e in events]
    if not target.exists() or json.loads(target.read_text(encoding="utf-8")) != payload:
        _atomic_json_list(target, payload)
    output_dir = (
        data_dir.parent / "output" if data_dir.name == "data" else data_dir / "output"
    )
    report_path = output_dir / f"summary_events_{day.isoformat()}.html"
    render_chinese_report(
        events,
        report_path,
        source_text_by_item={
            item.id: (
                f"原帖：{item.post_text_original}"
                + (
                    f"\n外部文章标题：{item.article_title_original}"
                    if item.article_title_original
                    else ""
                )
                + (
                    f"\n外部文章摘录：{item.article_excerpt_original[:500]}"
                    if item.article_excerpt_original
                    else ""
                )
            )
            for item in items
        },
    )
    manifest.update(
        {
            "status": "partial_success" if manifest["failed_item_ids"] else "success",
            "stage": "complete",
            "events": len(events),
            "eligible_for_digest": sum(event.eligible_for_digest for event in events),
            "eligible_for_must_read": sum(
                event.eligible_for_must_read for event in events
            ),
            "pending_item_ids": [
                x.id
                for x in items
                if x.id not in manifest["completed_item_ids"]
                and x.id not in manifest["filtered_item_ids"]
                and x.id not in manifest["failed_item_ids"]
            ],
            "report_filename": report_path.name,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    _atomic_json(manifest_path, manifest)
    return events, {
        "input_posts": len(posts),
        "filtered": filtered,
        "items": len(items),
        "clusters": len(groups),
        "candidate_pairs": len(pairs),
        "strong_pairs": sum(pair.strong for pair in pairs),
        "pair_reviews": sum(
            pair.id in decisions_by_id for pair in pairs if not pair.strong
        ),
        "same_event_pairs": sum(
            decisions_by_id.get(pair.id, {}).get("relationship") == "same_event"
            for pair in pairs
            if not pair.strong
        ),
        "events": len(events),
        "eligible_for_digest": sum(event.eligible_for_digest for event in events),
        "eligible_for_must_read": sum(event.eligible_for_must_read for event in events),
        "must_read": sum(e.must_read for e in events),
        "kept": len(kept),
        "failed": len(manifest["failed_item_ids"]),
        "pending": len(manifest["pending_item_ids"]),
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
    p.add_argument("--resume", action="store_true")
    p.add_argument("--status", action="store_true")
    a = p.parse_args(argv)
    c = load_config(PROJECT_ROOT / "config.json")
    manifest = PROJECT_ROOT / a.data_dir / "events" / f"{a.date}.manifest.json"
    if a.status:
        try:
            value = json.loads(manifest.read_text(encoding="utf-8"))
            print(
                "事件分析状态："
                + json.dumps(
                    {
                        k: value.get(k)
                        for k in (
                            "status",
                            "stage",
                            "llm_requests",
                            "cache_hits",
                            "prompt_tokens",
                            "completion_tokens",
                        )
                    },
                    ensure_ascii=False,
                )
            )
            return 0
        except (OSError, json.JSONDecodeError):
            print("事件分析状态：尚未运行")
            return 0
    events, stats = run_pipeline(
        date.fromisoformat(a.date),
        data_dir=PROJECT_ROOT / a.data_dir,
        config=c,
        max_items=min(50, a.max_items),
        use_llm=not a.no_llm,
        mock=a.mock,
    )
    print("中文事件分析完成：" + json.dumps(stats, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
