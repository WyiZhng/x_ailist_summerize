from datetime import datetime, timezone, date
from pathlib import Path
from app.config import normalize_config
from app.content_builder import build_content_items
from app.content_filter import filter_item
from app.chinese_content_validator import validate_chinese
from app.event_pipeline import run_pipeline
from app.event_report import render_chinese_report
from app.event_summary import top5_text
from app.models import XPost, XPostMetrics
from app.storage import FilePostStore


def post(identifier, text):
    return XPost(
        id=identifier,
        text=text,
        author_id="a",
        author=None,
        created_at=datetime.now(timezone.utc),
        language="en",
        conversation_id=None,
        source_list_id="l",
        metrics=XPostMetrics(likes=1),
    )


def test_filter_keeps_security_and_drops_marketing():
    keep = build_content_items(
        [post("1", "Security vulnerability fixed in OpenAI API")],
        [],
        [],
        max_chars=1000,
        model_name="m",
    )[0]
    drop = build_content_items(
        [post("2", "Join my community with discount")],
        [],
        [],
        max_chars=1000,
        model_name="m",
    )[0]
    assert filter_item(keep).decision == "keep"
    assert filter_item(drop).decision == "drop"


def test_chinese_validator_rejects_english_and_secrets():
    assert validate_chinese("OpenAI 发布新的 API 价格")
    assert not validate_chinese("OpenAI released a new API price")
    assert not validate_chinese("密钥 sk-secret 不应显示")


def test_pipeline_is_stable_and_preserves_sources(tmp_path: Path):
    store = FilePostStore(tmp_path)
    store.save_posts(
        [
            post("1", "Anthropic released Claude Code update"),
            post("2", "Anthropic released Claude Code update"),
        ]
    )
    config = normalize_config({})
    events, stats = run_pipeline(date(2026, 7, 18), data_dir=tmp_path, config=config)
    again, _ = run_pipeline(date(2026, 7, 18), data_dir=tmp_path, config=config)
    assert (
        len(events) == 1
        and events[0].title_zh.startswith("Anthropic")
        and events[0].post_ids == ["1", "2"]
    )
    assert events[0].id == again[0].id and stats["llm_requests"] == 0
    report = tmp_path / "report.html"
    render_chinese_report(events, report)
    assert "AI 每日情报" in report.read_text(encoding="utf-8")
    assert top5_text(tmp_path, date(2026, 7, 18)) and "今日必读" in top5_text(
        tmp_path, date(2026, 7, 18)
    )
