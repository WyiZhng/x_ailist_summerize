from __future__ import annotations

import json
from pathlib import Path

from app.config import normalize_config
from app.weixin_commands import HELP_TEXT, ReportLocator, WeixinCommandHandler


def handler(tmp_path: Path) -> tuple[WeixinCommandHandler, Path]:
    output = tmp_path / "output"
    output.mkdir()
    config = normalize_config(
        {
            "summarization": {
                "provider": "deepseek",
                "options": {"deepseek": {"api_key": "TEST_ONLY_KEY_DO_NOT_USE"}},
            },
            "twitter": {"list_urls": ["https://x.com/i/lists/TEST_ONLY"]},
        }
    )
    command_handler = WeixinCommandHandler(
        report_locator=ReportLocator(output, (".html",)),
        config=config,
        cookies_path=tmp_path / "cookies.json",
    )
    return command_handler, output


def test_help_and_unknown_commands_never_need_an_llm(tmp_path: Path) -> None:
    command_handler, _ = handler(tmp_path)
    assert command_handler.handle("帮助").text == HELP_TEXT
    assert command_handler.handle("arbitrary prompt").text == HELP_TEXT


def test_daily_reads_latest_successful_history(tmp_path: Path) -> None:
    command_handler, output = handler(tmp_path)
    older = output / "summary_older.html"
    newer = output / "summary_newer.html"
    older.write_text("older", encoding="utf-8")
    newer.write_text("newer", encoding="utf-8")
    (output / "history.json").write_text(
        json.dumps(
            {
                older.name: {
                    "status": "succeeded",
                    "generated_at": "2026-07-17T08:00:00+08:00",
                    "tweets": 10,
                },
                newer.name: {
                    "status": "succeeded",
                    "generated_at": "2026-07-18T09:30:00+08:00",
                    "tweets": 100,
                    "links": 15,
                },
            }
        ),
        encoding="utf-8",
    )

    result = command_handler.handle("日报")
    assert "抓取帖子：100" in result.text
    assert "外部链接：15" in result.text
    assert result.file_path is None


def test_full_report_returns_only_safe_html_and_corrupt_history_falls_back(
    tmp_path: Path,
) -> None:
    command_handler, output = handler(tmp_path)
    report = output / "summary_valid.html"
    report.write_text("report", encoding="utf-8")
    (output / "history.json").write_text("{damaged", encoding="utf-8")

    result = command_handler.handle("完整报告")
    assert result.file_path == report.resolve()
    assert result.display_name and result.display_name.endswith(".html")


def test_report_locator_blocks_symlink_escape(tmp_path: Path) -> None:
    _, output = handler(tmp_path)
    outside = tmp_path / "summary_private.html"
    outside.write_text("private", encoding="utf-8")
    link = output / "summary_link.html"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        return
    locator = ReportLocator(output, (".html",))
    assert locator.latest() is None


def test_status_contains_no_sensitive_values(tmp_path: Path) -> None:
    command_handler, _ = handler(tmp_path)
    result = command_handler.handle("状态")
    assert "TEST_ONLY_KEY_DO_NOT_USE" not in result.text
    assert "DeepSeek：已配置" in result.text
    assert str(tmp_path) not in result.text
