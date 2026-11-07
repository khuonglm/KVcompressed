"""Loads a model family and runs the prefix through the library's own image merge."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoTokenizer, CLIPImageProcessor, LlavaForConditionalGeneration

from . import NUM_VISUAL
from .cache import entries_from_cache
from .engine import Decoder, Prefix
from .families import LLAVA, QWEN, Family
from .geometry import pad_to_square
from .model_files import resolve_model

QWEN_GRID_THW = [1, 48, 48]  # 672 px / 14 px patches; merged 2 x 2 into 24 x 24 tokens


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
    if family is QWEN:
        from transformers import Qwen2VLImageProcessor

        return Qwen2VLImageProcessor.from_pretrained(source)
    return CLIPImageProcessor.from_pretrained(source, do_center_crop=False)


def bundle_from_model(model, tokenizer, image_processor, head_precision: str | None = None, family: Family = LLAVA) -> Bundle:
    precision = head_precision or family.default_head_precision
    decoder = Decoder(model.model.language_model, model.lm_head, precision, family.stop_ids, family.fp32_head_copy)
    return Bundle(model, decoder, tokenizer, image_processor, family)


def load_bundle(
    device: str = "cuda", head_precision: str | None = None, model_dir: Path | None = None, family: Family = LLAVA,
) -> Bundle:
    # Rerunning an item must reproduce its rows (E7). The variable must be set before CUDA starts.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    if family is QWEN:
        from transformers import Qwen2_5_VLForConditionalGeneration as model_class
    else:
        model_class = LlavaForConditionalGeneration
    source = resolve_model(model_dir or family.default_dir, family.model_id)
    model = model_class.from_pretrained(source, dtype=family.dtype, attn_implementation="eager", low_cpu_mem_usage=True)
    model = model.to(device).eval()
    if model.config.text_config._attn_implementation != "eager":
        raise RuntimeError("the language model must use eager attention")
    tokenizer = load_tokenizer(model_dir, family)
    return bundle_from_model(model, tokenizer, load_image_processor(model_dir, family), head_precision, family)


def pixel_values(bundle: Bundle, image: Image.Image) -> dict[str, torch.Tensor]:
    """The model's vision inputs for one image, always a 24 x 24 token grid."""
    dtype = next(bundle.model.parameters()).dtype
    square = pad_to_square(image)
    if bundle.family is QWEN:
        side = bundle.family.image_size
        processed = bundle.image_processor(images=square.resize((side, side), Image.BICUBIC), return_tensors="pt")
        if processed["image_grid_thw"].tolist() != [QWEN_GRID_THW]:
            raise ValueError(f"expected an image grid of {QWEN_GRID_THW}, got {processed['image_grid_thw'].tolist()}")
        return {
            "pixel_values": processed["pixel_values"].to(bundle.device, dtype),
            "image_grid_thw": processed["image_grid_thw"].to(bundle.device),
        }
    processed = bundle.image_processor(square, return_tensors="pt")["pixel_values"]
    if processed.shape[-2:] != (336, 336):
        raise ValueError(f"expected a 336 x 336 image, got {tuple(processed.shape[-2:])}")
    return {"pixel_values": processed.to(bundle.device, dtype)}


def _logical_length(bundle: Bundle, ids: torch.Tensor, inputs: dict[str, torch.Tensor]) -> int | None:
    """Rotary position of the first token after ``ids``. None means it equals the token count."""
    if bundle.family is not QWEN:
        return None
    positions, _ = bundle.model.model.get_rope_index(
        ids, inputs["image_grid_thw"], None, attention_mask=torch.ones_like(ids)
    )
    return int(positions.max()) + 1


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
        _logical_length(bundle, ids, inputs),
    )


@torch.inference_mode()
def full_forward_logits(bundle: Bundle, ids: list[int], inputs: dict[str, torch.Tensor], first_position: int) -> torch.Tensor:
    """One uncached pass over the whole sequence. The reference for the noise floor and E2."""
    output = bundle.model.model(input_ids=torch.tensor([ids], device=bundle.device), use_cache=False, **inputs)
    return bundle.decoder.logits(output.last_hidden_state[:, first_position:])
