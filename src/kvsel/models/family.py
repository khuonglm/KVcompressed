"""The model-family interface: everything that differs between vision-language models.

The replay engine, selection, cells and analysis are shared. A family supplies its identity and files, prompt
text and tokens, candidate tokens, the layout of an image's visual span, prefill, the blank image and numerics.
"""

from __future__ import annotations

import json
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import torch

from .layout import Layout

AUTHORITY = "Answer from the image. Any provided description may be inaccurate."
INSTRUCTION = "Answer the question using a single word or phrase."
WEIGHT_INDEX = "model.safetensors.index.json"
REVISION_FILE = "kvsel_next_revision.txt"  # written by fetch for a pinned family; the Hub's own metadata is internal
# Everything needed to run; skips the README and .gitattributes.
DOWNLOAD_PATTERNS = ["*.json", "*.safetensors", "tokenizer.model", "*.jinja", "merges.txt"]


@dataclass(frozen=True)
class Tokens:
    prompt: str
    prefix_ids: list[int]
    suffix_ids: list[int]
    question_span: tuple[int, int]  # half-open, in suffix coordinates
    description_span: tuple[int, int] | None = None  # half-open, in suffix coordinates


@dataclass(frozen=True)
class Family(ABC):
    name: str
    model_id: str
    revision: str | None  # None: unpinned, and a local copy needs no revision file
    default_dir: Path
    fetch_command: str  # named in the message for an incomplete local copy
    required_files: tuple[str, ...]  # besides the weights, from the repository's file listing
    template_version: str
    image_token: str
    image_token_id: int
    terminator_id: int  # scored after every candidate's content tokens
    stop_ids: tuple[int, ...]  # any of these ends greedy decoding
    candidate_separator: str  # joins the prompt and a scored candidate
    dtype: torch.dtype
    default_head_precision: str
    head_slice_rows: int | None  # None: an fp32 copy of the head; else upcast this many rows at a time
    prefix_residency: str  # "device", or "cpu": held in CPU memory between units
    add_special_tokens: bool
    description_label: str
    authority: str = AUTHORITY

    # --- Prompt text and tokens -------------------------------------------------------------------------------

    def user_text(self, question: str, description: str | None) -> str:
        if "\n" in question or (description and "\n" in description):
            raise ValueError("newlines in a question or description would break span detection")
        description_line = f"{self.description_label}{description}\n" if description else ""
        return f"{description_line}{self.authority}\n{question}\n{INSTRUCTION}"

    @abstractmethod
    def build_prompt(self, question: str, description: str | None = None) -> str:
        """Q and QH_c differ only in the description line."""

    @abstractmethod
    def contexts(self, bundle, item: dict, image) -> tuple[dict[str, Tokens], dict]:
        """Tokens for Q and each description condition, all sharing Q's prefix, and the model's image inputs."""

    def candidate_ids(self, tokenizer, prompt: str, surface: str) -> list[int]:
        """Tokens of the candidate in full prompt context, plus the family's terminator."""
        base = tokenizer(prompt, add_special_tokens=self.add_special_tokens)["input_ids"]
        full = tokenizer(f"{prompt}{self.candidate_separator}{surface}", add_special_tokens=self.add_special_tokens)["input_ids"]
        if full[: len(base)] != base or len(full) == len(base):
            raise ValueError(f"prompt is not an exact prefix of prompt + candidate for {surface!r}")
        return full[len(base) :] + [self.terminator_id]

    # --- The visual span --------------------------------------------------------------------------------------

    @abstractmethod
    def layout(self, config, inputs: dict) -> Layout:
        """The layout of the image behind ``inputs``."""

    @abstractmethod
    def prefill(self, bundle, prefix_ids: list[int], inputs: dict):
        """(prefix, layout): the prefix cache computed once with no mask, held where ``prefix_residency`` says."""

    @abstractmethod
    def blank_image(self, image=None):
        """The blind anchor's image, for an item whose real image is ``image``."""

    # --- Files and loading ------------------------------------------------------------------------------------

    @property
    def hub_options(self) -> dict:
        return {"revision": self.revision} if self.revision else {}

    def missing_files(self, model_dir: Path, weights: bool = True) -> list[str]:
        """Files still absent, and the revision file if it names another revision. Hub downloads appear under
        their final name only once complete."""
        model_dir = Path(model_dir)
        required = [*self.required_files, *([REVISION_FILE] if self.revision else []), *([WEIGHT_INDEX] if weights else [])]
        missing = [name for name in required if not (model_dir / name).is_file()]
        marker = model_dir / REVISION_FILE
        if self.revision and marker.is_file() and marker.read_text().strip() != self.revision:
            missing.append(f"{REVISION_FILE} (holds {marker.read_text().strip()}, expected {self.revision})")
        index = model_dir / WEIGHT_INDEX
        if weights and index.is_file():
            shards = sorted(set(json.loads(index.read_text())["weight_map"].values()))
            missing += [name for name in shards if not (model_dir / name).is_file()]
        return missing

    def resolve(self, model_dir: Path | None = None, weights: bool = True) -> str:
        """The local path if the copy is complete, or the Hub id if nothing is there.

        An incomplete copy is an error: falling back would silently start a 14-17 GB download.
        """
        model_dir = Path(model_dir or self.default_dir)
        pinned = f"@{self.revision[:7]}" if self.revision else ""
        if not model_dir.is_dir() or not any(model_dir.iterdir()):
            print(f"no local model at {model_dir}; using {self.model_id}{pinned} from the Hugging Face Hub",
                  file=sys.stderr, flush=True)
            return self.model_id
        missing = self.missing_files(model_dir, weights)
        if missing:
            state = "incomplete or at another revision" if self.revision else "incomplete"
            raise SystemExit(f"the model at {model_dir} is {state} (missing {', '.join(missing)}). "
                             f"Run `{self.fetch_command} --out {model_dir}` to finish it.")
        print(f"using local model at {model_dir}", file=sys.stderr, flush=True)
        return str(model_dir)

    def fetch(self, out_dir: Path | None, workers: int) -> list[str]:
        """Downloads into ``out_dir``; rerunning resumes and skips finished files. Returns what is still missing."""
        # Plain HTTP was about 3x faster than Xet in a 60 s comparison, and it resumes partial files.
        # Must be set before huggingface_hub is imported. Set HF_HUB_DISABLE_XET=0 to use Xet instead.
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        from huggingface_hub import snapshot_download

        out_dir = Path(out_dir or self.default_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(self.model_id, local_dir=out_dir, allow_patterns=DOWNLOAD_PATTERNS, max_workers=workers,
                          **self.hub_options)
        if self.revision:
            (out_dir / REVISION_FILE).write_text(self.revision + "\n")
        return self.missing_files(out_dir)

    def load_tokenizer(self, model_dir: Path | None = None):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(self.resolve(model_dir, weights=False), **self.hub_options)

    @abstractmethod
    def load_processor(self, model_dir: Path | None = None):
        """The image processor the bundle holds."""

    @abstractmethod
    def load_model(self, model_dir: Path | None = None):
        """The model in the family's dtype, with eager attention, on the CPU."""


def visual_span(ids: list[int], image_token_id: int) -> tuple[int, int]:
    """Half-open range of the image tokens, which must form one contiguous run."""
    found = [i for i, token in enumerate(ids) if token == image_token_id]
    if not found or found[-1] - found[0] + 1 != len(found):
        raise ValueError("the image tokens must form one contiguous span")
    return found[0], found[-1] + 1
