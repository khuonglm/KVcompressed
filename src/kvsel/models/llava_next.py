"""LLaVA-NeXT (v1.6, Vicuna-7B): the square base map, then the unpadded high-resolution map with a newline
ending each row. LLaVA-1.5's template v3.1 with a single image token, which the processor expands per image."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import torch
from PIL import Image

from ..cache import entries_from_cache
from ..engine import Prefix, on_device, release_cached_memory
from .family import Family, Tokens, visual_span
from .layout import Layout, Map
from .llava15 import LLAVA, PAD_COLOR

# The Hub's main for llava-hf/llava-v1.6-vicuna-7b-hf (last changed 27 January 2025), checked 16 November 2026.
REVISION = "c916e6cdcd760b4cecd1dd4907f84ac649f93b23"


def anyres_layout(side: int, rows: int, cols: int) -> Layout:
    """A ``side`` x ``side`` base map, then a ``rows`` x ``cols`` map with a newline after each row."""
    base = side * side
    newlines = tuple(base + row * (cols + 1) + cols for row in range(rows))
    return Layout((Map(0, side, side), Map(base, rows, cols)), newlines)


def derive_layout(config, image_size: list[int]) -> Layout:
    """From the model's own AnyRes grid and unpadding, for an image of (height, width) as the processor reports it."""
    from transformers.models.llava_next.modeling_llava_next import get_anyres_image_grid_shape, unpad_image

    if config.vision_feature_select_strategy != "default":
        raise ValueError("expected the 'default' feature strategy, which drops CLS and leaves a square base tile")
    tile = config.vision_config.image_size
    side = tile // config.vision_config.patch_size
    rows, cols = get_anyres_image_grid_shape(list(image_size), config.image_grid_pinpoints, tile)
    unpadded = unpad_image(torch.empty(0, rows * side, cols * side), list(image_size))
    return anyres_layout(side, *unpadded.shape[1:])


@dataclass(frozen=True)
class LlavaNext(Family):
    def build_prompt(self, question: str, description: str | None = None) -> str:
        return f"USER: {self.image_token}\n{self.user_text(question, description)} ASSISTANT:"

    def tokenize(self, processor, image, question: str, description: str | None = None) -> tuple[Tokens, dict]:
        """(tokens, processor inputs). Spans come from LLaVA-1.5's tokenization of the same text, same tokenizer."""
        reference = LLAVA.tokenize(processor.tokenizer, question, description)
        prompt = self.build_prompt(question, description)
        inputs = processor(images=image, text=prompt, return_tensors="pt")
        ids = inputs["input_ids"][0].tolist()
        start, stop = visual_span(ids, self.image_token_id)
        first_reference_image = reference.prefix_ids.index(LLAVA.image_token_id)
        if ids[:start] != reference.prefix_ids[:first_reference_image] or ids[stop:] != reference.suffix_ids:
            raise ValueError("the processor's expansion changed the tokens outside the image span")
        return replace(reference, prompt=prompt, prefix_ids=ids[:stop]), inputs

    def contexts(self, bundle, item: dict, image) -> tuple[dict[str, Tokens], dict]:
        """Inputs on the CPU; `prefill` moves them. The bundle's image processor is the whole NeXT processor."""
        tokens, inputs = {}, None
        for name, description in (("Q", None), *item["descriptions"].items()):
            tokens[name], context_inputs = self.tokenize(bundle.image_processor, image, item["question"], description)
            inputs = inputs or context_inputs
            if tokens[name].prefix_ids != tokens["Q"].prefix_ids:
                raise ValueError(f"the {name} prefix differs from Q's")
        return tokens, inputs

    def device_inputs(self, bundle, inputs: dict) -> dict:
        return {"pixel_values": inputs["pixel_values"].to(bundle.device, next(bundle.model.parameters()).dtype),
                "image_sizes": inputs["image_sizes"].to(bundle.device)}

    def layout(self, config, inputs: dict) -> Layout:
        if len(inputs["image_sizes"]) != 1:
            raise ValueError("expected exactly one image")
        return derive_layout(config, inputs["image_sizes"][0].tolist())

    @torch.inference_mode()
    def prefill(self, bundle, prefix_ids: list[int], inputs: dict) -> tuple[Prefix, Layout]:
        """After checking the derived layout against the span. The returned prefix is in CPU memory."""
        layout = self.layout(bundle.model.config, inputs)
        start, stop = visual_span(prefix_ids, self.image_token_id)
        if stop != len(prefix_ids):
            raise ValueError("the prefix must end with the visual span")
        if stop - start != layout.total:
            raise ValueError(f"the visual span has {stop - start} tokens but the derived layout has {layout.total}")
        release_cached_memory(bundle.device)
        output = bundle.model.model(input_ids=torch.tensor([prefix_ids], device=bundle.device), use_cache=True,
                                    **self.device_inputs(bundle, inputs))
        prefix = Prefix(entries_from_cache(output.past_key_values), start, stop - start)
        return (on_device(prefix, "cpu") if self.prefix_residency == "cpu" else prefix), layout

    def blank_image(self, image=None) -> Image.Image:
        """The padding colour at the real image's size, so it gets the same AnyRes layout."""
        return Image.new("RGB", image.size, PAD_COLOR)

    def load_tokenizer(self, model_dir: Path | None = None):
        return self.load_processor(model_dir).tokenizer

    def load_processor(self, model_dir: Path | None = None):
        from transformers import LlavaNextProcessor

        return LlavaNextProcessor.from_pretrained(self.resolve(model_dir, weights=False), revision=self.revision)

    def load_model(self, model_dir: Path | None = None):
        # No PYTORCH_CUDA_ALLOC_CONF=expandable_segments: its virtual-memory API is unsupported on the 20 GiB GPU slice.
        from transformers import LlavaNextForConditionalGeneration

        return LlavaNextForConditionalGeneration.from_pretrained(
            self.resolve(model_dir), revision=self.revision, dtype=self.dtype, attn_implementation="eager",
            low_cpu_mem_usage=True)


# Same tokenizer ids, fp16 backbone, fp32 head, labels and candidate joining as LLaVA-1.5. The prefix waits in CPU
# memory between units (a square image's is 1.45 GiB; the 20 GiB GPU slice has room for two cache copies beside the
# weights), and the fp32 head is upcast in 4,096-row (64 MiB) slices instead of kept as a 0.49 GiB fp32 copy.
NEXT = LlavaNext(**{
    **vars(LLAVA),
    "name": "llava_next",
    "model_id": "llava-hf/llava-v1.6-vicuna-7b-hf",
    "revision": REVISION,
    "default_dir": Path("models/llava-v1.6-vicuna-7b-hf"),
    "fetch_command": "kvsel model fetch --model llava_next",
    "required_files": (
        "tokenizer.json", "tokenizer_config.json", "tokenizer.model", "special_tokens_map.json", "added_tokens.json",
        "chat_template.json", "config.json", "generation_config.json", "preprocessor_config.json",
        "processor_config.json",
    ),
    "head_slice_rows": 4096,
    "prefix_residency": "cpu",
})
