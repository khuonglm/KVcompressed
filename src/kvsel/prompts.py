"""Prompt template v3.0, prefix/suffix split and candidate tokenization."""

from __future__ import annotations

from dataclasses import dataclass

from . import IMAGE_TOKEN, IMAGE_TOKEN_ID, NUM_VISUAL
from .engine import EOS_ID

TEMPLATE_VERSION = "v3.0"
AUTHORITY = "Answer from the image. Any provided description may be inaccurate."
INSTRUCTION = "Answer the question using a single word or phrase. ASSISTANT:"
NEWLINE_ID = 13
WINDOW_SIZE = 16


def build_prompt(question: str, description: str | None = None) -> str:
    """Q and QH_c differ only in the description line."""
    if "\n" in question or (description and "\n" in description):
        raise ValueError("newlines in a question or description would break span detection")
    description_line = f"Description: {description}\n" if description else ""
    return f"USER: {IMAGE_TOKEN * NUM_VISUAL}\n{description_line}{AUTHORITY}\n{question}\n{INSTRUCTION}"


@dataclass(frozen=True)
class Tokens:
    prompt: str
    prefix_ids: list[int]
    suffix_ids: list[int]
    question_span: tuple[int, int]  # half-open, in suffix coordinates


def tokenize_prompt(tokenizer, question: str, description: str | None = None) -> Tokens:
    prompt = build_prompt(question, description)
    ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    last_image = max(i for i, token in enumerate(ids) if token == IMAGE_TOKEN_ID)
    prefix, suffix = ids[: last_image + 1], ids[last_image + 1 :]
    if sum(token == IMAGE_TOKEN_ID for token in ids) != NUM_VISUAL:
        raise ValueError("expected exactly 576 image tokens")
    newlines = [i for i, token in enumerate(suffix) if token == NEWLINE_ID]
    if len(newlines) < 3:
        raise ValueError("template should leave at least three newlines in the suffix")
    return Tokens(prompt, prefix, suffix, (newlines[-2] + 1, newlines[-1]))


def surface_forms(answer: str) -> dict[str, str]:
    """The exact strings scored for a semantic answer. Each form applies to true and alt alike."""
    return {"lower": answer.lower(), "capitalized": answer[:1].upper() + answer[1:].lower()}


def candidate_ids(tokenizer, prompt: str, surface: str) -> list[int]:
    """Tokens of ``" " + surface`` in full prompt context, plus EOS."""
    base = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    full = tokenizer(f"{prompt} {surface}", add_special_tokens=True)["input_ids"]
    if full[: len(base)] != base or len(full) == len(base):
        raise ValueError(f"prompt is not an exact prefix of prompt + candidate for {surface!r}")
    return full[len(base) :] + [EOS_ID]


def window_composition(tokens: Tokens, window_size: int = WINDOW_SIZE) -> dict[str, float]:
    """Share of window tokens from the question, the description and fixed instruction text."""
    total = len(tokens.suffix_ids)
    window = range(total - window_size, total)
    start, stop = tokens.question_span
    newlines = [i for i, token in enumerate(tokens.suffix_ids) if token == NEWLINE_ID]
    has_description = len(newlines) == 4
    description_stop = newlines[1] if has_description else 0
    question = sum(start <= i < stop for i in window)
    description = sum(i < description_stop for i in window)
    return {
        "question": question / window_size,
        "description": description / window_size,
        "instruction": (window_size - question - description) / window_size,
    }
