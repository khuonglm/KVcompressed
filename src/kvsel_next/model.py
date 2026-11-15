"""Loads LLaVA-NeXT and prefills its prefix: the text before the image and the whole expanded visual span."""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

from kvsel.cache import entries_from_cache
from kvsel.engine import Prefix, greedy, replay
from kvsel.model import Bundle, bundle_from_model
from kvsel.normalize import outcome
from kvsel.runner import Unit, cell_name, score_candidates

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
    # No PYTORCH_CUDA_ALLOC_CONF=expandable_segments: its virtual-memory API is unsupported on the 20 GiB GPU slice.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # must be set before CUDA starts
    torch.use_deterministic_algorithms(True)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        resolve_model(model_dir or NEXT.default_dir), revision=REVISION, dtype=NEXT.dtype,
        attn_implementation="eager", low_cpu_mem_usage=True,
    )
    processor = load_processor(model_dir)
    bundle = bundle_from_model(model.to(device).eval(), processor.tokenizer, processor, family=NEXT)
    drop_fp32_head_copy(bundle)
    return bundle


def drop_fp32_head_copy(bundle: Bundle) -> None:
    """Upcast the head a slice of rows at a time instead of keeping a 0.49 GiB fp32 copy on a 20 GiB GPU.
    kvsel's Decoder supports both; each logit is the same fp32 dot product of the fp32 hidden state."""
    bundle.decoder.fp32_head_copy = False
    bundle.decoder.fp32_head = None


def run_unit(bundle: Bundle, item: dict, unit: Unit, prefix: Prefix, kept: np.ndarray, tokens: dict) -> dict:
    """`kvsel.runner.run_unit`'s row, with one cache copy fewer at the peak: candidates are scored before greedy
    decoding (neither changes the state), and the replay state is let go as greedy takes it. kvsel's keeps the
    state alive during greedy, so a Full unit holds four copies of the cache; this holds three."""
    answering = tokens[unit.answering]
    suffix = torch.tensor([answering.suffix_ids])
    held = [replay(bundle.decoder, prefix, torch.from_numpy(kept), suffix, "compact")[0]]
    scores = score_candidates(bundle, held[0], answering.prompt, item["a_true"], item["a_alt"])
    decoded = greedy(bundle.decoder, held.pop())  # greedy's frame now holds the only reference
    text = bundle.tokenizer.decode(decoded.token_ids, skip_special_tokens=True).strip()
    return {
        **asdict(unit),
        "cell": cell_name(unit),
        "kept": kept.tolist() if unit.selector != "full" else None,
        "scores": scores,
        "greedy_ids": decoded.token_ids,
        "greedy_text": text,
        "outcome": "other" if decoded.truncated else outcome(text, item["a_true"], item["a_alt"]),
        "truncated": decoded.truncated,
        "top_two_gaps": decoded.top_two_gaps,
    }


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
