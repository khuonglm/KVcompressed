"""Loads LLaVA and runs the prefix through the library's own image merge."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoTokenizer, CLIPImageProcessor, LlavaForConditionalGeneration

from . import IMAGE_TOKEN_ID, NUM_VISUAL
from .cache import entries_from_cache
from .engine import Decoder, Prefix
from .geometry import pad_to_square
from .model_files import DEFAULT_MODEL_DIR, resolve_model


@dataclass
class Bundle:
    model: LlavaForConditionalGeneration
    decoder: Decoder
    tokenizer: object
    image_processor: CLIPImageProcessor

    @property
    def device(self) -> torch.device:
        return self.decoder.device


def load_tokenizer(model_dir: Path = DEFAULT_MODEL_DIR):
    return AutoTokenizer.from_pretrained(resolve_model(model_dir, weights=False))


def load_image_processor(model_dir: Path = DEFAULT_MODEL_DIR) -> CLIPImageProcessor:
    return CLIPImageProcessor.from_pretrained(resolve_model(model_dir, weights=False), do_center_crop=False)


def bundle_from_model(model: LlavaForConditionalGeneration, tokenizer, image_processor, head_precision="fp32") -> Bundle:
    return Bundle(model, Decoder(model.model.language_model, model.lm_head, head_precision), tokenizer, image_processor)


def load_bundle(device: str = "cuda", head_precision: str = "fp32", model_dir: Path = DEFAULT_MODEL_DIR) -> Bundle:
    # Rerunning an item must reproduce its rows (E7). The variable must be set before CUDA starts.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    model = LlavaForConditionalGeneration.from_pretrained(
        resolve_model(model_dir), dtype=torch.float16, attn_implementation="eager", low_cpu_mem_usage=True
    )
    model = model.to(device).eval()
    if model.config.text_config._attn_implementation != "eager":
        raise RuntimeError("the language model must use eager attention")
    return bundle_from_model(model, load_tokenizer(model_dir), load_image_processor(model_dir), head_precision)


def pixel_values(bundle: Bundle, image: Image.Image) -> torch.Tensor:
    processed = bundle.image_processor(pad_to_square(image), return_tensors="pt")["pixel_values"]
    if processed.shape[-2:] != (336, 336):
        raise ValueError(f"expected a 336 x 336 image, got {tuple(processed.shape[-2:])}")
    return processed.to(bundle.device, next(bundle.model.parameters()).dtype)


@torch.inference_mode()
def prefill(bundle: Bundle, prefix_ids: list[int], pixels: torch.Tensor) -> Prefix:
    """Compute the prefix cache once, with no mask. Its states are never recomputed."""
    ids = torch.tensor([prefix_ids], device=bundle.device)
    output = bundle.model.model(input_ids=ids, pixel_values=pixels, use_cache=True)
    num_visual = sum(token == IMAGE_TOKEN_ID for token in prefix_ids)
    if num_visual != NUM_VISUAL or prefix_ids[-NUM_VISUAL:] != [IMAGE_TOKEN_ID] * NUM_VISUAL:
        raise ValueError("the prefix must end with exactly 576 image tokens")
    return Prefix(entries_from_cache(output.past_key_values), len(prefix_ids) - NUM_VISUAL, NUM_VISUAL)


@torch.inference_mode()
def full_forward_logits(bundle: Bundle, ids: list[int], pixels: torch.Tensor, first_position: int) -> torch.Tensor:
    """One uncached pass over the whole sequence. The reference for the noise floor and E2."""
    output = bundle.model.model(input_ids=torch.tensor([ids], device=bundle.device), pixel_values=pixels, use_cache=False)
    return bundle.decoder.logits(output.last_hidden_state[:, first_position:])
