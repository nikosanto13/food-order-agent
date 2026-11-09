"""Turn a chat reply into text that sounds right when spoken.

Flash TTS models read text literally (number normalization is not available on
every plan), so "$11.50" and "ORD-48271" would come out wrong. The conversion is
deterministic and keeps the agent's text replies unchanged: the reply guards
still check for "$11.50" verbatim, and only the spoken copy is rewritten.
"""

from __future__ import annotations

import re

_MONEY = re.compile(r"\$(\d+)(?:\.(\d{1,2}))?")
_ORDER_ID = re.compile(r"#?\b([A-Z]{2,5})-(\d+)\b")
_RANGE = re.compile(r"\b(\d+)\s*[-\u2013]\s*(\d+)\b")
_QUANTITY = re.compile(r"\b(\d+)x\b")
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_HEADING = re.compile(r"^\s*#+\s*")
_SNAKE = re.compile(r"\b([a-z]+)_([a-z_]+)\b")
_SPACES = re.compile(r"\s+")
_DASH = re.compile(r"\s*\u2014\s*|\s+-\s+")  # em dash or spaced hyphen: a pause
_ONE_DIGIT_CENTS = 10  # "11 dollars 5" would be ambiguous


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def _say_money(m: re.Match[str]) -> str:
    dollars = int(m[1])
    cents = int((m[2] or "0").ljust(2, "0"))
    if not cents:
        return _plural(dollars, "dollar")
    if not dollars:
        return _plural(cents, "cent")
    if cents < _ONE_DIGIT_CENTS:
        return f"{_plural(dollars, 'dollar')} and {_plural(cents, 'cent')}"
    return f"{_plural(dollars, 'dollar')} {cents}"  # "11 dollars 50", as people say it


def _say_order_id(m: re.Match[str]) -> str:
    # Spelled out character by character: an order id is read back, not pronounced.
    return f"{' '.join(m[1])}, {' '.join(m[2])}"


def to_speech(text: str) -> str:
    """Return a TTS-friendly rendering of ``text`` (one paragraph, no markup)."""
    sentences: list[str] = []
    for raw in text.splitlines():
        line = _HEADING.sub("", _BULLET.sub("", raw))
        line = line.replace("**", "").replace("__", "").replace("`", "")
        line = line.replace("(", "").replace(")", "").strip()
        if not line:
            continue
        if line[-1] not in ".!?:":
            line += "."
        sentences.append(line)
    spoken = " ".join(sentences)
    spoken = _MONEY.sub(_say_money, spoken)
    spoken = _ORDER_ID.sub(_say_order_id, spoken)
    spoken = _RANGE.sub(r"\1 to \2", spoken)
    spoken = _QUANTITY.sub(r"\1", spoken)
    spoken = _SNAKE.sub(lambda m: f"{m[1]} {m[2].replace('_', ' ')}", spoken)
    spoken = _DASH.sub(", ", spoken.replace("&", " and "))
    return _SPACES.sub(" ", spoken).strip()
