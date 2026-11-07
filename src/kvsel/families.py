"""Model families: everything that differs between LLaVA-1.5-7B and Qwen2.5-VL-7B.

The replay engine, selectors, cells and analysis are shared. A family supplies the prompt text,
special token ids, how a scored answer is joined to the prompt, numerics, and where files live.
Both families give a 24 x 24 grid of 576 visual tokens: LLaVA at 336 px with 14 px patches, Qwen at
672 px with 14 px patches merged 2 x 2.
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

    def user_text(self, question: str, description: str | None) -> str:
        if "\n" in question or (description and "\n" in description):
            raise ValueError("newlines in a question or description would break span detection")
        description_line = f"Description: {description}\n" if description else ""
        return f"{description_line}{AUTHORITY}\n{question}\n{INSTRUCTION}"

    def build_prompt(self, question: str, description: str | None = None) -> str:
        text = self.user_text(question, description)
        if self.name == "llava":
            return f"USER: {IMAGE_TOKEN * NUM_VISUAL}\n{text} ASSISTANT:"
        return (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
            f"<|vision_start|>{self.image_token * NUM_VISUAL}<|vision_end|>{text}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )


LLAVA = Family(
    name="llava",
    model_id="llava-hf/llava-1.5-7b-hf",
    default_dir=Path("models/llava-1.5-7b-hf"),
    template_version="v3.0",
    image_token=IMAGE_TOKEN,
    image_token_id=32000,
    image_size=336,
    terminator_id=2,
    stop_ids=(2,),
    candidate_separator=" ",
    dtype=torch.float16,
    default_head_precision="fp32",
    add_special_tokens=True,
)

# Qwen2.5-VL is trained in bf16 and is known to be unstable in fp16. Its 152k vocabulary makes an
# fp32 copy of the head 2.2 GB, which does not fit beside 16.6 GB of weights on a 20 GB GPU.
QWEN = Family(
    name="qwen",
    model_id="Qwen/Qwen2.5-VL-7B-Instruct",
    default_dir=Path("models/qwen2.5-vl-7b-instruct"),
    template_version="v3.0-qwen",
    image_token="<|image_pad|>",
    image_token_id=151655,
    image_size=672,
    terminator_id=151645,  # <|im_end|>
    stop_ids=(151645, 151643),  # <|im_end|>, <|endoftext|>
    candidate_separator="",
    dtype=torch.bfloat16,
    default_head_precision="native",
    add_special_tokens=False,
)

FAMILIES = {family.name: family for family in (LLAVA, QWEN)}


def get_family(name: str) -> Family:
    try:
        return FAMILIES[name]
    except KeyError:
        raise ValueError(f"unknown model family {name!r}; choose from {sorted(FAMILIES)}") from None
