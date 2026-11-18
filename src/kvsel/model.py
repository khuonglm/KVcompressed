"""A loaded model family: the model, its decoder with the family's numerics, tokenizer and image processor."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import torch

from .engine import Decoder
from .models.family import Family
from .models.llava15 import LLAVA


@dataclass
class Bundle:
    model: torch.nn.Module
    decoder: Decoder
    tokenizer: object
    image_processor: object
    family: Family = field(default=LLAVA)

    @property
    def device(self) -> torch.device:
        return self.decoder.device


def bundle_from_model(model, tokenizer, image_processor, head_precision: str | None = None, family: Family = LLAVA) -> Bundle:
    precision = head_precision or family.default_head_precision
    if model.config.text_config._attn_implementation != "eager":
        raise RuntimeError("the language model must use eager attention")  # probes need attention weights
    decoder = Decoder(model.model.language_model, model.lm_head, precision, family.stop_ids, family.head_slice_rows)
    return Bundle(model, decoder, tokenizer, image_processor, family)


def load_bundle(
    device: str = "cuda", head_precision: str | None = None, model_dir: Path | None = None, family: Family = LLAVA,
) -> Bundle:
    # Rerunning an item must reproduce its rows (E7). The variable must be set before CUDA starts.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    model = family.load_model(model_dir).to(device).eval()
    return bundle_from_model(model, family.load_tokenizer(model_dir), family.load_processor(model_dir), head_precision,
                             family)


@torch.inference_mode()
def full_forward_logits(bundle: Bundle, ids: list[int], inputs: dict[str, torch.Tensor], first_position: int) -> torch.Tensor:
    """One uncached pass over the whole sequence. The reference for the noise floor and E2."""
    output = bundle.model.model(input_ids=torch.tensor([ids], device=bundle.device), use_cache=False, **inputs)
    return bundle.decoder.logits(output.last_hidden_state[:, first_position:])
