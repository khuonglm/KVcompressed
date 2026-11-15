"""Loads LLaVA-NeXT and prefills its prefix: the text before the image and the whole expanded visual span."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

from kvsel.cache import entries_from_cache
from kvsel.engine import Prefix
from kvsel.model import Bundle, bundle_from_model

from .family import NEXT
from .layout import Layout, derive_layout, visual_span
from .model_files import REVISION, resolve_model


def load_processor(model_dir: Path | None = None) -> LlavaNextProcessor:
    source = resolve_model(model_dir or NEXT.default_dir, weights=False)
    return LlavaNextProcessor.from_pretrained(source, revision=REVISION)


def load_bundle(device: str = "cuda", model_dir: Path | None = None) -> Bundle:
    """fp16 backbone and cache, fp32 head; eager attention is checked by `bundle_from_model`.

    The bundle's `image_processor` holds the whole processor, which expands the image token.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # must be set before CUDA starts
    torch.use_deterministic_algorithms(True)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        resolve_model(model_dir or NEXT.default_dir), revision=REVISION, dtype=NEXT.dtype,
        attn_implementation="eager", low_cpu_mem_usage=True,
    )
    processor = load_processor(model_dir)
    return bundle_from_model(model.to(device).eval(), processor.tokenizer, processor, family=NEXT)


@torch.inference_mode()
def prefill(bundle: Bundle, prefix_ids: list[int], inputs) -> tuple[Prefix, Layout]:
    """Compute the prefix cache once, with no mask, after checking the derived layout against the span."""
    if len(inputs["image_sizes"]) != 1:
        raise ValueError("expected exactly one image")
    start, stop = visual_span(prefix_ids, bundle.family.image_token_id)
    if stop != len(prefix_ids):
        raise ValueError("the prefix must end with the visual span")
    layout = derive_layout(bundle.model.config, inputs["image_sizes"][0].tolist())
    if stop - start != layout.total:
        raise ValueError(f"the visual span has {stop - start} tokens but the derived layout has {layout.total}")
    output = bundle.model.model(
        input_ids=torch.tensor([prefix_ids], device=bundle.device),
        pixel_values=inputs["pixel_values"].to(bundle.device, next(bundle.model.parameters()).dtype),
        image_sizes=inputs["image_sizes"].to(bundle.device),
        use_cache=True,
    )
    return Prefix(entries_from_cache(output.past_key_values), start, stop - start), layout
