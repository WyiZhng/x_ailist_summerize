from datetime import datetime, timezone, date
from pathlib import Path
from app.config import normalize_config
from app.content_builder import build_content_items
from app.content_filter import filter_item
from app.chinese_content_validator import validate_chinese
from app.event_pipeline import run_pipeline
from app.event_report import render_chinese_report
from app.event_summary import top5_text
from app.content_analysis import AnalysisResult
from app.models import XPost, XPostMetrics
from app.storage import FilePostStore


def post(identifier, text, created_at=None):
    return XPost(
        id=identifier,
        text=text,
        author_id="a",
        author=None,
        created_at=created_at or datetime.now(timezone.utc),
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
            post(
                "1",
                "Anthropic released Claude Code update",
                datetime(2026, 7, 18, tzinfo=timezone.utc),
            ),
            post(
                "2",
                "Anthropic released Claude Code update",
                datetime(2026, 7, 18, tzinfo=timezone.utc),
            ),
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


def test_semantic_provider_can_filter_without_network(tmp_path: Path):
    class Provider:
        requests = hits = prompt_tokens = completion_tokens = 0

        def analyze(self, _item):
            self.requests += 1
            return AnalysisResult(
                "中文标题",
                "中文摘要内容",
                "中文理由内容",
                "marketing",
                "opinion",
                False,
                {},
            )

    store = FilePostStore(tmp_path)
    store.save_posts(
        [
            post(
                "x",
                "ordinary update",
                datetime.combine(
                    date.today(), datetime.min.time(), tzinfo=timezone.utc
                ),
            )
        ]
    )
    events, stats = run_pipeline(
        date.today(),
        data_dir=tmp_path,
        config=normalize_config({}),
        analysis_provider=Provider(),
    )
    assert events == [] and stats["filtered"] == 1 and stats["llm_requests"] == 1


def test_date_filter_does_not_mix_days(tmp_path: Path):
    store = FilePostStore(tmp_path)
    store.save_posts([post("today", "security update")])
    events, stats = run_pipeline(
        date(2000, 1, 1), data_dir=tmp_path, config=normalize_config({})
    )
    assert events == [] and stats["input_posts"] == 0


def test_permanent_invalid_is_not_retried(tmp_path: Path):
    class Provider:
        requests = hits = prompt_tokens = completion_tokens = 0

        def analyze(self, _item):
            self.requests += 1
            raise ValueError("analysis_schema_invalid")

    store = FilePostStore(tmp_path)
    store.save_posts(
        [post("bad", "ordinary update", datetime(2026, 7, 18, tzinfo=timezone.utc))]
    )
    provider = Provider()
    run_pipeline(
        date(2026, 7, 18),
        data_dir=tmp_path,
        config=normalize_config({}),
        analysis_provider=provider,
    )
    run_pipeline(
        date(2026, 7, 18),
        data_dir=tmp_path,
        config=normalize_config({}),
        analysis_provider=provider,
    )
    assert provider.requests == 1


def test_retryable_timeout_can_resume(tmp_path: Path):
    class Provider:
        requests = hits = prompt_tokens = completion_tokens = 0

        def analyze(self, _item):
            self.requests += 1
            if self.requests == 1:
                raise TimeoutError("temporary timeout")
            return AnalysisResult(
                "OpenAI 发布模型更新",
                "OpenAI 发布新的模型能力更新。",
                "该更新影响模型开发与应用。",
                "model_release",
                "fact",
                True,
                {},
            )

    store = FilePostStore(tmp_path)
    store.save_posts(
        [post("retry", "ordinary update", datetime(2026, 7, 18, tzinfo=timezone.utc))]
    )
    provider = Provider()
    first, stats = run_pipeline(
        date(2026, 7, 18),
        data_dir=tmp_path,
        config=normalize_config({}),
        analysis_provider=provider,
    )
    second, resumed = run_pipeline(
        date(2026, 7, 18),
        data_dir=tmp_path,
        config=normalize_config({}),
        analysis_provider=provider,
    )
    assert first == [] and stats["failed"] == 1
    assert len(second) == 1 and resumed["failed"] == 0 and provider.requests == 2


def test_top5_does_not_fill_with_ineligible_growth_case(tmp_path: Path):
    class Provider:
        requests = hits = prompt_tokens = completion_tokens = 0

        def analyze(self, item):
            self.requests += 1
            if "views" in item.post_text_original:
                return AnalysisResult(
                    "作者发布 AI 视频增长方法",
                    "作者分享个人浏览量增长案例。",
                    "该案例可作为内容运营参考。",
                    "technical_analysis",
                    "fact",
                    True,
                    {},
                )
            return AnalysisResult(
                "OpenAI 修复 API 安全漏洞",
                "OpenAI 修复 API 安全问题。",
                "该漏洞可能影响开发者服务安全。",
                "security",
                "fact",
                True,
                {},
            )

    store = FilePostStore(tmp_path)
    store.save_posts(
        [
            post(
                "growth",
                "my views increased 10x with AI videos",
                datetime(2026, 7, 18, tzinfo=timezone.utc),
            ),
            post(
                "security",
                "OpenAI completed important maintenance",
                datetime(2026, 7, 18, tzinfo=timezone.utc),
            ),
        ]
    )
    events, stats = run_pipeline(
        date(2026, 7, 18),
        data_dir=tmp_path,
        config=normalize_config({}),
        analysis_provider=Provider(),
    )
    assert len(events) == 2 and stats["must_read"] == 1
    assert (
        next(event for event in events if "增长" in event.title_zh).must_read is False
    )
    assert next(event for event in events if "漏洞" in event.title_zh).must_read is True
    preview = top5_text(tmp_path, date(2026, 7, 18))
    assert preview and "安全漏洞" in preview and "增长方法" not in preview
