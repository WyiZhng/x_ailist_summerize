"""Cached, budgeted structured Chinese analysis through the configured LLM."""

from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from openai import OpenAI
from .chinese_content_validator import validate_chinese
from .event_prompts import BASE_PROMPT
from .structured_llm import parse_structured_json

SCHEMA_VERSION = "content_schema_v1"


@dataclass(frozen=True)
class AnalysisResult:
    title_zh: str
    summary_zh: str
    why_it_matters_zh: str
    event_type: str
    factuality: str
    keep: bool
    usage: dict[str, int | None]


class AnalysisProvider(Protocol):
    def analyze(self, item: Any) -> AnalysisResult: ...


@dataclass(frozen=True)
class PairDecision:
    relationship: str
    confidence: float
    reason_zh: str

    @property
    def same_event(self) -> bool:
        return self.relationship == "same_event" and self.confidence >= 0.7


class DeepSeekContentAnalysisProvider:
    def __init__(
        self, config: dict, cache_dir: Path, *, max_requests: int, max_tokens: int
    ):
        from .llm_providers import LLMProvider

        self.llm = LLMProvider(config)
        self.cache_dir = cache_dir
        self.max_requests = max_requests
        self.max_tokens = max_tokens
        self.requests = self.hits = 0
        self.prompt_tokens = self.completion_tokens = 0

    def _key(self, item: Any) -> str:
        model = self.llm._get_effective_config()[2]
        return hashlib.sha256(
            (
                item.analysis_text_hash
                + "event_synthesis_zh_v1"
                + SCHEMA_VERSION
                + model
            ).encode()
        ).hexdigest()

    def analyze(self, item: Any) -> AnalysisResult:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_dir / (self._key(item) + ".json")
        if path.exists():
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                result = AnalysisResult(**value["result"])
                if all(
                    validate_chinese(x)
                    for x in (
                        result.title_zh,
                        result.summary_zh,
                        result.why_it_matters_zh,
                    )
                ):
                    self.hits += 1
                    return result
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass
        if (
            self.requests >= self.max_requests
            or self.prompt_tokens + self.completion_tokens >= self.max_tokens
        ):
            raise RuntimeError("analysis_budget_exhausted")
        endpoint, key, model = self.llm._get_effective_config()
        prompt = (
            BASE_PROMPT
            + '\nSchema: {"keep":bool,"title_zh":str,"summary_zh":str,"why_it_matters_zh":str,"event_type":str,"factuality":str}。event_type 只能使用 developer_tool、model_release、product_release、open_source、research、company_news、security、technical_analysis、industry_analysis、benchmark、funding、acquisition、policy、opinion、tutorial、job_post、marketing、personal_chat、meme、other 之一；factuality 只能使用 fact、opinion、speculation、prediction、mixed、unclear 之一。枚举必须是英文，所有 *_zh 必须是简体中文。\n'
            + item.analysis_text
        )
        response = OpenAI(
            base_url=endpoint, api_key=key, timeout=60
        ).chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=700,
        )
        self.requests += 1
        usage = response.usage
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion = getattr(usage, "completion_tokens", None)
        if isinstance(prompt_tokens, int):
            self.prompt_tokens += prompt_tokens
        if isinstance(completion, int):
            self.completion_tokens += completion
        value = parse_structured_json(response.choices[0].message.content or "")
        allowed = {
            "model_release",
            "product_release",
            "open_source",
            "research",
            "company_news",
            "security",
            "developer_tool",
            "technical_analysis",
            "industry_analysis",
            "benchmark",
            "funding",
            "acquisition",
            "policy",
            "opinion",
            "tutorial",
            "job_post",
            "marketing",
            "personal_chat",
            "meme",
            "other",
        }
        factual = {"fact", "opinion", "speculation", "prediction", "mixed", "unclear"}
        if (
            not isinstance(value.get("keep"), bool)
            or value.get("event_type") not in allowed
            or value.get("factuality") not in factual
        ):
            raise ValueError("analysis_schema_invalid")
        fields = [value.get(x) for x in ("title_zh", "summary_zh", "why_it_matters_zh")]
        if not all(isinstance(x, str) and validate_chinese(x) for x in fields):
            raise ValueError("language_validation_failed")
        result = AnalysisResult(
            fields[0],
            fields[1],
            fields[2],
            value["event_type"],
            value["factuality"],
            value["keep"],
            {"prompt_tokens": prompt_tokens, "completion_tokens": completion},
        )
        path.write_text(
            json.dumps({"result": result.__dict__}, ensure_ascii=False),
            encoding="utf-8",
        )
        return result

    def analyze_pair(self, left: Any, right: Any) -> PairDecision:
        """Classify only a recalled grey pair; cache it independently."""
        endpoint, key, model = self.llm._get_effective_config()
        cache_key = hashlib.sha256(
            (
                "|".join(
                    sorted((left.analysis_text_hash, right.analysis_text_hash))
                    + ["pair_cluster_zh_v1", model]
                )
            ).encode()
        ).hexdigest()
        path = self.cache_dir / (cache_key + ".pair.json")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            result = PairDecision(**value["result"])
            if result.relationship in {
                "same_event",
                "follow_up",
                "related_topic",
                "contradictory_report",
                "unrelated",
                "unclear",
            } and validate_chinese(result.reason_zh):
                self.hits += 1
                return result
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
        if (
            self.requests >= self.max_requests
            or self.prompt_tokens + self.completion_tokens >= self.max_tokens
        ):
            raise RuntimeError("analysis_budget_exhausted")
        prompt = BASE_PROMPT + (
            '\n比较两个候选是否描述同一事件。只输出 JSON：{"relationship":"same_event|follow_up|related_topic|contradictory_report|unrelated|unclear","confidence":0到1数字,"reason_zh":"中文理由"}。只有主体、动作、产品/版本和时间兼容时才用 same_event；低置信度用 unclear。\n'
            f"<UNTRUSTED_SOURCE>A: {left.post_text_original[:900]}\nB: {right.post_text_original[:900]}</UNTRUSTED_SOURCE>"
        )
        response = OpenAI(
            base_url=endpoint, api_key=key, timeout=60
        ).chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=350,
        )
        self.requests += 1
        usage = response.usage
        prompt_tokens, completion = (
            getattr(usage, "prompt_tokens", None),
            getattr(usage, "completion_tokens", None),
        )
        if isinstance(prompt_tokens, int):
            self.prompt_tokens += prompt_tokens
        if isinstance(completion, int):
            self.completion_tokens += completion
        value = parse_structured_json(response.choices[0].message.content or "")
        relationship = value.get("relationship")
        confidence = value.get("confidence")
        reason = value.get("reason_zh")
        if (
            relationship
            not in {
                "same_event",
                "follow_up",
                "related_topic",
                "contradictory_report",
                "unrelated",
                "unclear",
            }
            or not isinstance(confidence, (int, float))
            or not 0 <= confidence <= 1
            or not isinstance(reason, str)
            or not validate_chinese(reason)
        ):
            raise ValueError("pair_schema_invalid")
        result = PairDecision(relationship, float(confidence), reason)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"result": result.__dict__}, ensure_ascii=False),
            encoding="utf-8",
        )
        return result


class MockContentAnalysisProvider:
    def __init__(self):
        self.calls = 0

    def analyze(self, item: Any) -> AnalysisResult:
        self.calls += 1
        return AnalysisResult(
            "OpenAI 发布开发者工具更新",
            "来源内容涉及 AI 开发工具更新。",
            "该更新可能影响开发者工具链。",
            "developer_tool",
            "fact",
            True,
            {"prompt_tokens": None, "completion_tokens": None},
        )

    def analyze_pair(self, left: Any, right: Any) -> PairDecision:
        self.calls += 1
        return PairDecision("unrelated", 0.9, "离线模拟默认不合并灰区候选。")
