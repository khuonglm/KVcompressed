"""What every family's prompts share: the scored surface forms and the PW-attn window's composition.

Prompt text, tokens per context and candidate tokens are each family's (`kvsel.models`).
"""

from __future__ import annotations

from .models.family import Tokens
from .selectors import WINDOW_SIZE



def surface_forms(answer: str) -> dict[str, str]:
    """The exact strings scored for a semantic answer. Each form applies to true and alt alike."""
    return {
        "lower": answer.lower(),
        "capitalized": answer[:1].upper() + answer[1:].lower(),
        "upper": answer.upper(),
    }


def window_composition(tokens: Tokens, window_size: int = WINDOW_SIZE) -> dict[str, float]:
    """Share of window tokens from the question, the description and fixed instruction text."""
    total = len(tokens.suffix_ids)
    window = range(total - window_size, total)
    q_start, q_stop = tokens.question_span
    d_start, d_stop = tokens.description_span or (0, 0)
    question = sum(q_start <= i < q_stop for i in window)
    description = sum(d_start <= i < d_stop for i in window)
    return {
        "question": question / window_size,
        "description": description / window_size,
        "instruction": (window_size - question - description) / window_size,
    }
