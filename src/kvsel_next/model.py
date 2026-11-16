"""Loads LLaVA-NeXT and prefills its prefix: the text before the image and the whole expanded visual span."""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

from kvsel.cache import compact, entries_from_cache
from kvsel.engine import Decoder, Prefix, State, greedy
from kvsel.model import Bundle
from kvsel.normalize import outcome
from kvsel.runner import Unit, cell_name, score_candidates

from .family import NEXT
from .layout import Layout, derive_layout, visual_span
from .model_files import REVISION, resolve_model


def load_processor(model_dir: Path | None = None) -> LlavaNextProcessor:
    source = resolve_model(model_dir or NEXT.default_dir, weights=False)
    return LlavaNextProcessor.from_pretrained(source, revision=REVISION)


def load_bundle(device: str = "cuda", model_dir: Path | None = None) -> Bundle:
    """fp16 backbone and cache, fp32 head; see `make_bundle`.

    The bundle's `image_processor` holds the whole processor, which expands the image token.
    """
    # No PYTORCH_CUDA_ALLOC_CONF=expandable_segments: its virtual-memory API is unsupported on the 20 GiB GPU slice.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # must be set before CUDA starts
    torch.use_deterministic_algorithms(True)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        resolve_model(model_dir or NEXT.default_dir), revision=REVISION, dtype=NEXT.dtype,
        attn_implementation="eager", low_cpu_mem_usage=True,
    )
    return make_bundle(model.to(device).eval(), load_processor(model_dir))


def make_bundle(model, processor) -> Bundle:
    """`kvsel.model.bundle_from_model`, but the fp32 head is upcast a slice of rows at a time on each call instead of
    kept as a 0.49 GiB fp32 copy (kvsel's Decoder supports both; each logit is the same fp32 dot product of the fp32
    hidden state), and in 64 MiB slices rather than 256 MiB: square images leave no more room on the GPU slice."""
    if model.config.text_config._attn_implementation != "eager":
        raise RuntimeError("the language model must use eager attention")  # probes need attention weights
    decoder = Decoder(model.model.language_model, model.lm_head, NEXT.default_head_precision, NEXT.stop_ids,
                      fp32_head_copy=False)
    decoder.HEAD_ROWS_PER_SLICE = 4096
    return Bundle(model, decoder, processor.tokenizer, processor, NEXT)


def release_cached_memory(device: torch.device) -> None:
    """Return PyTorch's cached, fragmented blocks to the driver before a pass whose eager attention needs one large
    block (about 1 GiB of fp32 softmax at 2,928 visual tokens). Numerics are unchanged."""
    if device.type == "cuda":
        torch.cuda.empty_cache()


# The prefix cache lives in CPU memory between units (a square image's is 1.45 GiB; the 20 GiB GPU slice has room
# for two cache copies beside the weights, and kvsel's engine needs three with the prefix on the GPU). Engine calls
# that take a prefix get a copy through `on_device` or `gather`; gathering and moving copy values exactly.


def on_device(prefix: Prefix, device) -> Prefix:
    """The same prefix with its entries on ``device``: a copy, or the prefix itself if already there."""
    target = torch.empty(0, device=device).device  # "cuda" -> cuda:0, so the comparison below is exact
    if prefix.entries[0][0].device == target:
        return prefix
    entries = tuple((k.to(target), v.to(target)) for k, v in prefix.entries)
    return Prefix(entries, prefix.num_nonvisual, prefix.num_visual, prefix.logical_length)


def gather(decoder: Decoder, prefix: Prefix, kept_visual: torch.Tensor) -> Prefix:
    """The non-visual and kept visual entries of a CPU-held prefix, gathered on the CPU as `kvsel.cache.compact`
    does and moved to the decoder's device: a prefix of its own that keeps the original positions."""
    entries = compact(prefix.entries, kept_visual, prefix.num_nonvisual)
    gathered = Prefix(entries, prefix.num_nonvisual, len(kept_visual), prefix.logical_start)
    return on_device(gathered, decoder.device)


@torch.inference_mode()
def replay_gathered(decoder: Decoder, gathered: Prefix, suffix_ids: torch.Tensor) -> tuple[State, torch.Tensor]:
    """`kvsel.engine.replay`'s compact step on entries already gathered: same positions, no mask."""
    physical = gathered.length
    hidden, new_entries, _ = decoder.step(suffix_ids, gathered.entries, gathered.logical_start, physical, None)
    logits = decoder.logits(hidden)
    tokens = suffix_ids.shape[1]
    return State(new_entries, gathered.logical_start + tokens, physical + tokens, None, logits[0, -1]), logits


def replay_offloaded(decoder: Decoder, prefix: Prefix, kept_visual: torch.Tensor,
                     suffix_ids: torch.Tensor) -> tuple[State, torch.Tensor]:
    """`kvsel.engine.replay` in compact mode for a CPU-held prefix. The GPU holds the gathered entries and the
    step's concatenated copy (two cache copies); the gathered ones are freed when the step returns."""
    return replay_gathered(decoder, gather(decoder, prefix, kept_visual), suffix_ids)


def greedy_releasing(decoder: Decoder, box: list[State], **options):
    """`kvsel.engine.greedy` on the one state in ``box``, emptied first: greedy's frame then holds the only
    reference, so each decoding step frees the state before it. The caller must keep no other reference."""
    return greedy(decoder, box.pop(), **options)


def run_unit(bundle: Bundle, item: dict, unit: Unit, prefix: Prefix, kept: np.ndarray, tokens: dict) -> dict:
    """`kvsel.runner.run_unit`'s row for a CPU-held prefix. Candidates are scored before greedy decoding (neither
    changes the state) and greedy frees the replay state as it steps, where kvsel's keeps it alive: the GPU holds
    at most two cache copies."""
    answering = tokens[unit.answering]
    suffix = torch.tensor([answering.suffix_ids])
    box = [replay_offloaded(bundle.decoder, prefix, torch.from_numpy(kept), suffix)[0]]
    scores = score_candidates(bundle, box[0], answering.prompt, item["a_true"], item["a_alt"])
    decoded = greedy_releasing(bundle.decoder, box)
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
    """Compute the prefix cache once, with no mask, after checking the derived layout against the span.
    The returned prefix is in CPU memory."""
    if len(inputs["image_sizes"]) != 1:
        raise ValueError("expected exactly one image")
    start, stop = visual_span(prefix_ids, bundle.family.image_token_id)
    if stop != len(prefix_ids):
        raise ValueError("the prefix must end with the visual span")
    layout = derive_layout(bundle.model.config, inputs["image_sizes"][0].tolist())
    if stop - start != layout.total:
        raise ValueError(f"the visual span has {stop - start} tokens but the derived layout has {layout.total}")
    release_cached_memory(bundle.device)
    output = bundle.model.model(
        input_ids=torch.tensor([prefix_ids], device=bundle.device),
        pixel_values=inputs["pixel_values"].to(bundle.device, next(bundle.model.parameters()).dtype),
        image_sizes=inputs["image_sizes"].to(bundle.device),
        use_cache=True,
    )
    prefix = Prefix(entries_from_cache(output.past_key_values), start, stop - start)
    return on_device(prefix, "cpu"), layout  # see `on_device`: the prefix waits in CPU memory
