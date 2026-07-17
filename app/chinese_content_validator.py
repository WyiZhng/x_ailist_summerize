"""Reject unsafe or non-Chinese user-visible LLM fields."""

from __future__ import annotations
import re

_FORBIDDEN = re.compile(
    r"(?:sk-[\w-]+|Bearer\s+\S+|context_token|/Users/|api[_ -]?key)", re.I
)
_HYPE = ("震撼", "重磅", "炸裂", "颠覆", "史诗级", "遥遥领先")


def validate_chinese(value: str, *, minimum: int = 4) -> bool:
    if (
        not isinstance(value, str)
        or len(value.strip()) < minimum
        or _FORBIDDEN.search(value)
    ):
        return False
    chinese = len(re.findall(r"[\u4e00-\u9fff]", value))
    letters = len(re.findall(r"[A-Za-z]", value))
    return (
        chinese >= 2
        and chinese >= letters * 0.08
        and sum(value.count(x) for x in _HYPE) <= 1
    )
