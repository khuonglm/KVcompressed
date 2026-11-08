"""Model families: everything that differs between LLaVA-1.5-7B and Qwen2.5-VL-7B.

The replay engine, selectors, cells and analysis are shared. A family supplies the prompt text,
special token ids, how a scored answer is joined to the prompt, numerics, and where files live.
Both families give a 24 x 24 grid of 576 visual tokens: LLaVA at 336 px with 14 px patches, Qwen at
672 px with 14 px patches merged 2 x 2.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import torch

from . import IMAGE_TOKEN, NUM_VISUAL

AUTHORITY = "Answer from the image. Any provided description may be inaccurate."
INSTRUCTION = "Answer the question using a single word or phrase."

# Prompt variants for the adoption test (plan, decision record of 9 November 2026). Each changes one
# thing about how the description is presented. "base" is template v3.0. Only "tiebreak" changes the
# Q prompt, since it is the only one that edits the authority line.
PROMPT_VARIANTS = {
    "base": {"label": "Description: ", "authority": AUTHORITY, "after_question": False},
    "unverified": {"label": "Unverified description, possibly wrong: ", "authority": AUTHORITY, "after_question": False},
    "tiebreak": {
        "label": "Description: ",
        "authority": "Answer from the image. Any provided description may be inaccurate; "
        "if it conflicts with the image, trust the image.",
        "after_question": False,
    },
    "after": {"label": "Description: ", "authority": AUTHORITY, "after_question": True},
}


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
    fp32_head_copy: bool  # False: upcast the head slice by slice instead of keeping an fp32 copy
    attention_precision: str  # "native": the library's eager attention; "fp32": kvsel.attention
    add_special_tokens: bool
    prompt_variant: str = "base"  # a key of PROMPT_VARIANTS; "base" is template v3.0 itself

    @property
    def authority(self) -> str:
        return PROMPT_VARIANTS[self.prompt_variant]["authority"]

    @property
    def description_label(self) -> str:
        return PROMPT_VARIANTS[self.prompt_variant]["label"]

    def user_text(self, question: str, description: str | None) -> str:
        if "\n" in question or (description and "\n" in description):
            raise ValueError("newlines in a question or description would break span detection")
        description_line = f"{self.description_label}{description}\n" if description else ""
        if PROMPT_VARIANTS[self.prompt_variant]["after_question"]:
            return f"{self.authority}\n{question}\n{description_line}{INSTRUCTION}"
        return f"{description_line}{self.authority}\n{question}\n{INSTRUCTION}"

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
    fp32_head_copy=True,
    attention_precision="native",
    add_special_tokens=True,
)

# Qwen2.5-VL is trained in bf16 and is known to be unstable in fp16. Its 152k vocabulary makes an
# fp32 copy of the head 2.2 GB, which does not fit beside 16.6 GB of weights on a 20 GB GPU, so the
# head is upcast slice by slice. E3 failed on 7 Nov 2026 with the bf16 head and again with the fp32
# head, so the language model's attention runs in fp32 as well (tests/README.md).
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
    default_head_precision="fp32",
    fp32_head_copy=False,
    attention_precision="fp32",
    add_special_tokens=False,
)

FAMILIES = {family.name: family for family in (LLAVA, QWEN)}


def with_prompt_variant(family: Family, variant: str) -> Family:
    """The family with another way of presenting the description. Template versions stay distinct."""
    if variant not in PROMPT_VARIANTS:
        raise ValueError(f"unknown prompt variant {variant!r}; choose from {sorted(PROMPT_VARIANTS)}")
    if variant == "base":
        return family
    return replace(family, prompt_variant=variant, template_version=f"{family.template_version}+{variant}")


def get_family(name: str) -> Family:
    try:
        return FAMILIES[name]
    except KeyError:
        raise ValueError(f"unknown model family {name!r}; choose from {sorted(FAMILIES)}") from None
