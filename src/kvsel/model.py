"""Loads a model family and runs the prefix through the library's own image merge."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
from PIL import Image
from transformers import (
    AutoTokenizer,
    CLIPImageProcessor,
    LlavaForConditionalGeneration,
)

from . import NUM_VISUAL
from .cache import entries_from_cache
from .engine import Decoder, Prefix
from .families import LLAVA, Family
from .geometry import pad_to_square
from .model_files import resolve_model


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


def load_tokenizer(model_dir: Path | None = None, family: Family = LLAVA):
    return AutoTokenizer.from_pretrained(resolve_model(model_dir or family.default_dir, family.model_id, weights=False))


def load_image_processor(model_dir: Path | None = None, family: Family = LLAVA):
    source = resolve_model(model_dir or family.default_dir, family.model_id, weights=False)
    return CLIPImageProcessor.from_pretrained(source, do_center_crop=False)


def bundle_from_model(model, tokenizer, image_processor, head_precision: str | None = None, family: Family = LLAVA) -> Bundle:
    precision = head_precision or family.default_head_precision
    if model.config.text_config._attn_implementation != "eager":
        raise RuntimeError("the language model must use eager attention")  # probes need attention weights
    decoder = Decoder(model.model.language_model, model.lm_head, precision, family.stop_ids)
    return Bundle(model, decoder, tokenizer, image_processor, family)


def load_bundle(
    device: str = "cuda", head_precision: str | None = None, model_dir: Path | None = None, family: Family = LLAVA,
) -> Bundle:
    # Rerunning an item must reproduce its rows (E7). The variable must be set before CUDA starts.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    source = resolve_model(model_dir or family.default_dir, family.model_id)
    model = LlavaForConditionalGeneration.from_pretrained(source, dtype=family.dtype, attn_implementation="eager", low_cpu_mem_usage=True)
    model = model.to(device).eval()
    tokenizer = load_tokenizer(model_dir, family)
    return bundle_from_model(model, tokenizer, load_image_processor(model_dir, family), head_precision, family)


def pixel_values(bundle: Bundle, image: Image.Image) -> dict[str, torch.Tensor]:
    """The model's vision inputs for one image, always a 24 x 24 token grid."""
    dtype = next(bundle.model.parameters()).dtype
    square = pad_to_square(image)
    processed = bundle.image_processor(square, return_tensors="pt")["pixel_values"]
    if processed.shape[-2:] != (336, 336):
        raise ValueError(f"expected a 336 x 336 image, got {tuple(processed.shape[-2:])}")
    return {"pixel_values": processed.to(bundle.device, dtype)}


@torch.inference_mode()
def prefill(bundle: Bundle, prefix_ids: list[int], inputs: dict[str, torch.Tensor]) -> Prefix:
    """Compute the prefix cache once, with no mask. Its states are never recomputed."""
    image_token = bundle.family.image_token_id
    if sum(token == image_token for token in prefix_ids) != NUM_VISUAL or prefix_ids[-NUM_VISUAL:] != [image_token] * NUM_VISUAL:
        raise ValueError("the prefix must end with exactly 576 image tokens")
    ids = torch.tensor([prefix_ids], device=bundle.device)
    output = bundle.model.model(input_ids=ids, use_cache=True, **inputs)
    return Prefix(
        entries_from_cache(output.past_key_values),
        len(prefix_ids) - NUM_VISUAL,
        NUM_VISUAL,
    )


@torch.inference_mode()
def full_forward_logits(bundle: Bundle, ids: list[int], inputs: dict[str, torch.Tensor], first_position: int) -> torch.Tensor:
    """One uncached pass over the whole sequence. The reference for the noise floor and E2."""
    output = bundle.model.model(input_ids=torch.tensor([ids], device=bundle.device), use_cache=False, **inputs)
    return bundle.decoder.logits(output.last_hidden_state[:, first_position:])
