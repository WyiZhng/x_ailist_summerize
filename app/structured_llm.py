"""Strict JSON extraction seam for future DeepSeek event calls."""

from __future__ import annotations
import json
import re


def parse_structured_json(value: str) -> dict:
    value = value.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*|\s*```$", "", value, flags=re.I)
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("Structured response must be an object")
    return parsed
