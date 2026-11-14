"""Model families: everything that differs between vision-language models.

The replay engine, selectors, cells and analysis are shared. A family supplies the prompt text,
special token ids, how a scored answer is joined to the prompt, numerics, and where files live.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from . import IMAGE_TOKEN, NUM_VISUAL

AUTHORITY = "Answer from the image. Any provided description may be inaccurate."
INSTRUCTION = "Answer the question using a single word or phrase."


@dataclass(frozen=True)
class Family:
    name: str
    model_id: str
    default_dir: Path
    template_version: str
    image_token: str
    image_token_id: int
    image_size: int  # side of the square the image is padded and resized to
    terminator_id: int  # scored after every candidate's content tokens
    stop_ids: tuple[int, ...]  # any of these ends greedy decoding
    candidate_separator: str  # joins the prompt and a scored candidate
    dtype: torch.dtype
    default_head_precision: str
    add_special_tokens: bool
    description_label: str
    authority: str = AUTHORITY

    def user_text(self, question: str, description: str | None) -> str:
        if "\n" in question or (description and "\n" in description):
            raise ValueError("newlines in a question or description would break span detection")
        description_line = f"{self.description_label}{description}\n" if description else ""
        return f"{description_line}{self.authority}\n{question}\n{INSTRUCTION}"

    def build_prompt(self, question: str, description: str | None = None) -> str:
        return f"USER: {IMAGE_TOKEN * NUM_VISUAL}\n{self.user_text(question, description)} ASSISTANT:"


LLAVA = Family(
    name="llava",
    model_id="llava-hf/llava-1.5-7b-hf",
    default_dir=Path("models/llava-1.5-7b-hf"),
    template_version="v3.1",  # v3.0 with the description labelled unverified (prompt test, 9 Nov 2026)
    image_token=IMAGE_TOKEN,
    image_token_id=32000,
    image_size=336,
    terminator_id=2,
    stop_ids=(2,),
    candidate_separator=" ",
    dtype=torch.float16,
    default_head_precision="fp32",
    add_special_tokens=True,
    description_label="Unverified description, possibly wrong: ",
)

FAMILIES = {family.name: family for family in (LLAVA,)}


def get_family(name: str) -> Family:
    try:
        return FAMILIES[name]
    except KeyError:
        raise ValueError(f"unknown model family {name!r}; choose from {sorted(FAMILIES)}") from None
