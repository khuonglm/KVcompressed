"""Answer normalisation for behavioural scoring. Likelihood scoring uses surface strings instead."""

from __future__ import annotations

import re

_EQUIVALENT = {"grey": "gray"}
_ARTICLES = {"a", "an", "the"}


def normalize(text: str) -> str:
    words = re.sub(r"[^\w\s-]", " ", text.lower()).split()
    words = [word for word in words if word not in _ARTICLES]
    return " ".join(_EQUIVALENT.get(word, word) for word in words)


def outcome(answer: str, a_true: str, a_alt: str) -> str:
    normalized = normalize(answer)
    if normalized == normalize(a_true):
        return "correct"
    if normalized == normalize(a_alt):
        return "alt"
    return "other"
