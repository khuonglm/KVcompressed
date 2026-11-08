"""Prompt templates, prefix/suffix split and candidate tokenization, for each model family.

LLaVA uses template v3.1: v3.0 with the description labelled as unverified. Qwen2.5-VL (parked) keeps
v3.0's user text inside its own chat template (v3.0-qwen).
"""

from __future__ import annotations

from dataclasses import dataclass

from . import NUM_VISUAL
from .families import AUTHORITY, LLAVA, Family

TEMPLATE_VERSION = LLAVA.template_version
WINDOW_SIZE = 16

__all__ = [
    "AUTHORITY", "TEMPLATE_VERSION", "WINDOW_SIZE", "Tokens", "build_prompt", "candidate_ids",
    "surface_forms", "tokenize_prompt", "window_composition",
]


def build_prompt(question: str, description: str | None = None, family: Family = LLAVA) -> str:
    """Q and QH_c differ only in the description line."""
    return family.build_prompt(question, description)


@dataclass(frozen=True)
class Tokens:
    prompt: str
    prefix_ids: list[int]
    suffix_ids: list[int]
    question_span: tuple[int, int]  # half-open, in suffix coordinates
    description_span: tuple[int, int] | None = None  # half-open, in suffix coordinates


def _token_span(offsets: list[tuple[int, int]], start: int, stop: int, first: int) -> tuple[int, int]:
    """Tokens whose characters overlap [start, stop), as a half-open range relative to ``first``."""
    inside = [i for i, (a, b) in enumerate(offsets) if a < stop and b > start and b > a]
    if not inside or inside != list(range(inside[0], inside[-1] + 1)):
        raise ValueError("a text span did not map to contiguous tokens")
    return inside[0] - first, inside[-1] + 1 - first


def tokenize_prompt(tokenizer, question: str, description: str | None = None, family: Family = LLAVA) -> Tokens:
    prompt = family.build_prompt(question, description)
    encoded = tokenizer(prompt, add_special_tokens=family.add_special_tokens, return_offsets_mapping=True)
    ids, offsets = encoded["input_ids"], [tuple(pair) for pair in encoded["offset_mapping"]]
    if sum(token == family.image_token_id for token in ids) != NUM_VISUAL:
        raise ValueError("expected exactly 576 image tokens")
    last_image = max(i for i, token in enumerate(ids) if token == family.image_token_id)
    first_suffix = last_image + 1
    question_start = prompt.index(family.authority) + len(family.authority) + 1
    question_stop = question_start + len(question)
    if prompt[question_start:question_stop] != question:
        raise ValueError("could not locate the question in the prompt")
    description_span = None
    if description:
        label = family.description_label
        description_start = prompt.index(label, prompt.rindex(family.image_token))
        description_stop = description_start + len(label) + len(description)
        description_span = _token_span(offsets, description_start, description_stop, first_suffix)
    return Tokens(
        prompt,
        ids[:first_suffix],
        ids[first_suffix:],
        _token_span(offsets, question_start, question_stop, first_suffix),
        description_span,
    )


def surface_forms(answer: str) -> dict[str, str]:
    """The exact strings scored for a semantic answer. Each form applies to true and alt alike."""
    return {
        "lower": answer.lower(),
        "capitalized": answer[:1].upper() + answer[1:].lower(),
        "upper": answer.upper(),
    }


def candidate_ids(tokenizer, prompt: str, surface: str, family: Family = LLAVA) -> list[int]:
    """Tokens of the candidate in full prompt context, plus the family's terminator."""
    base = tokenizer(prompt, add_special_tokens=family.add_special_tokens)["input_ids"]
    full = tokenizer(
        f"{prompt}{family.candidate_separator}{surface}", add_special_tokens=family.add_special_tokens
    )["input_ids"]
    if full[: len(base)] != base or len(full) == len(base):
        raise ValueError(f"prompt is not an exact prefix of prompt + candidate for {surface!r}")
    return full[len(base) :] + [family.terminator_id]


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
