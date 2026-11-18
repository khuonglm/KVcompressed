"""LLaVA-1.5-7B: one 24 x 24 map, no newlines. Its image transform and evidence-box mapping live here too.

As in the original LLaVA-1.5 evaluation: pad to square, then resize to 336 x 336, with no centre crop.
The transform and the box mapping are kept together so a box can't drift from its pixels.
Prompts use template v3.1: v3.0 with the description labelled as unverified.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from ..cache import entries_from_cache
from ..engine import Prefix
from .family import Family, Tokens, visual_span
from .layout import Layout, Map

MODEL_ID = "llava-hf/llava-1.5-7b-hf"
IMAGE_TOKEN = "<image>"
IMAGE_TOKEN_ID = 32000
GRID = 24  # side of the one map
NUM_VISUAL = GRID * GRID
LAYOUT = Layout((Map(0, GRID, GRID),))

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
PAD_COLOR = tuple(int(channel * 255) for channel in CLIP_MEAN)
IMAGE_SIZE = 336
PATCH_SIZE = IMAGE_SIZE // GRID
MIN_OVERLAP_FRACTION = 0.25
MAPPING_RULES = ("primary", "fallback", "center", "expanded")

Box = tuple[float, float, float, float]  # x0, y0, x1, y1


# --- Image transform and evidence boxes ---------------------------------------------------------------------


def pad_offsets(width: int, height: int) -> tuple[int, int]:
    side = max(width, height)
    return (side - width) // 2, (side - height) // 2


def pad_to_square(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size
    side = max(width, height)
    canvas = Image.new("RGB", (side, side), PAD_COLOR)
    canvas.paste(image, pad_offsets(width, height))
    return canvas


def box_to_square(box: dict, width: int, height: int) -> Box:
    """A box given as x, y, w, h in original pixels, in 336 x 336 model coordinates."""
    pad_x, pad_y = pad_offsets(width, height)
    scale = IMAGE_SIZE / max(width, height)
    x0, y0 = (box["x"] + pad_x) * scale, (box["y"] + pad_y) * scale
    return x0, y0, x0 + box["w"] * scale, y0 + box["h"] * scale


def _cells():
    """(span index, row, col) of every cell of the layout's one map."""
    positions = LAYOUT.positions(0)
    return [(int(positions[row, col]), row, col) for row in range(GRID) for col in range(GRID)]


def _overlap_patches(box: Box) -> list[int]:
    x0, y0, x1, y1 = box
    patches = []
    for index, row, col in _cells():
        overlap_w = min(x1, (col + 1) * PATCH_SIZE) - max(x0, col * PATCH_SIZE)
        overlap_h = min(y1, (row + 1) * PATCH_SIZE) - max(y0, row * PATCH_SIZE)
        if overlap_w > 0 and overlap_h > 0 and overlap_w * overlap_h >= MIN_OVERLAP_FRACTION * PATCH_SIZE**2:
            patches.append(index)
    return patches


def _point_patch(x: float, y: float) -> int:
    col = min(max(int(x // PATCH_SIZE), 0), GRID - 1)
    row = min(max(int(y // PATCH_SIZE), 0), GRID - 1)
    return int(LAYOUT.positions(0)[row, col])


def _fallback_patches(box: Box) -> list[int]:
    """Patches containing the box's centre and corners."""
    x0, y0, x1, y1 = box
    points = [((x0 + x1) / 2, (y0 + y1) / 2), (x0, y0), (x1, y0), (x0, y1), (x1, y1)]
    return sorted({_point_patch(x, y) for x, y in points})


def _center_patches(box: Box) -> list[int]:
    x0, y0, x1, y1 = box
    return [
        index for index, row, col in _cells()
        if x0 <= (col + 0.5) * PATCH_SIZE <= x1 and y0 <= (row + 0.5) * PATCH_SIZE <= y1
    ]


def evidence_patches(box: Box, rule: str = "primary") -> list[int]:
    """Visual-span indices for a box in model coordinates, under one of the proposal's four mapping rules.

    primary: overlap of at least 25% of the patch, or the fallback if that is empty.
    fallback: patches containing the box centre and corners.
    center: patch centre inside the box.
    expanded: the primary rule, on the box grown by one patch on each side.
    """
    if rule == "primary":
        return _overlap_patches(box) or _fallback_patches(box)
    if rule == "fallback":
        return _fallback_patches(box)
    if rule == "center":
        return _center_patches(box)
    if rule == "expanded":
        x0, y0, x1, y1 = box
        grown = (x0 - PATCH_SIZE, y0 - PATCH_SIZE, x1 + PATCH_SIZE, y1 + PATCH_SIZE)
        return _overlap_patches(grown) or _fallback_patches(grown)
    raise ValueError(f"unknown mapping rule {rule!r}")


# --- The family ----------------------------------------------------------------------------------------------


def _token_span(offsets: list[tuple[int, int]], start: int, stop: int, first: int) -> tuple[int, int]:
    """Tokens whose characters overlap [start, stop), as a half-open range relative to ``first``."""
    inside = [i for i, (a, b) in enumerate(offsets) if a < stop and b > start and b > a]
    if not inside or inside != list(range(inside[0], inside[-1] + 1)):
        raise ValueError("a text span did not map to contiguous tokens")
    return inside[0] - first, inside[-1] + 1 - first


@dataclass(frozen=True)
class Llava15(Family):
    def build_prompt(self, question: str, description: str | None = None) -> str:
        return f"USER: {self.image_token * NUM_VISUAL}\n{self.user_text(question, description)} ASSISTANT:"

    def tokenize(self, tokenizer, question: str, description: str | None = None) -> Tokens:
        prompt = self.build_prompt(question, description)
        encoded = tokenizer(prompt, add_special_tokens=self.add_special_tokens, return_offsets_mapping=True)
        ids, offsets = encoded["input_ids"], [tuple(pair) for pair in encoded["offset_mapping"]]
        if sum(token == self.image_token_id for token in ids) != NUM_VISUAL:
            raise ValueError(f"expected exactly {NUM_VISUAL} image tokens")
        last_image = max(i for i, token in enumerate(ids) if token == self.image_token_id)
        first_suffix = last_image + 1
        question_start = prompt.index(self.authority) + len(self.authority) + 1
        question_stop = question_start + len(question)
        if prompt[question_start:question_stop] != question:
            raise ValueError("could not locate the question in the prompt")
        description_span = None
        if description:
            label = self.description_label
            description_start = prompt.index(label, prompt.rindex(self.image_token))
            description_stop = description_start + len(label) + len(description)
            description_span = _token_span(offsets, description_start, description_stop, first_suffix)
        return Tokens(
            prompt,
            ids[:first_suffix],
            ids[first_suffix:],
            _token_span(offsets, question_start, question_stop, first_suffix),
            description_span,
        )

    def tokens(self, tokenizer, item: dict) -> dict[str, Tokens]:
        """Tokens for Q and each description condition; the image needs no tokenizing."""
        tokens = {"Q": self.tokenize(tokenizer, item["question"])}
        for condition, description in item["descriptions"].items():
            tokens[condition] = self.tokenize(tokenizer, item["question"], description)
        return tokens

    def pixel_values(self, bundle, image: Image.Image) -> dict[str, torch.Tensor]:
        """The model's vision inputs for one image, on the bundle's device."""
        dtype = next(bundle.model.parameters()).dtype
        processed = bundle.image_processor(pad_to_square(image), return_tensors="pt")["pixel_values"]
        if processed.shape[-2:] != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"expected a {IMAGE_SIZE} x {IMAGE_SIZE} image, got {tuple(processed.shape[-2:])}")
        return {"pixel_values": processed.to(bundle.device, dtype)}

    def contexts(self, bundle, item: dict, image) -> tuple[dict[str, Tokens], dict]:
        return self.tokens(bundle.tokenizer, item), self.pixel_values(bundle, image)

    def layout(self, config=None, inputs: dict | None = None) -> Layout:
        return LAYOUT

    @torch.inference_mode()
    def prefill(self, bundle, prefix_ids: list[int], inputs: dict) -> tuple[Prefix, Layout]:
        start, stop = visual_span(prefix_ids, self.image_token_id)
        if stop != len(prefix_ids) or stop - start != LAYOUT.total:
            raise ValueError(f"the prefix must end with exactly {LAYOUT.total} image tokens")
        output = bundle.model.model(input_ids=torch.tensor([prefix_ids], device=bundle.device), use_cache=True, **inputs)
        return Prefix(entries_from_cache(output.past_key_values), start, LAYOUT.total), LAYOUT

    def blank_image(self, image=None) -> Image.Image:
        """A blank image of the padding colour: the blind baseline (PREREG, amendment 1)."""
        return Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), PAD_COLOR)

    def load_processor(self, model_dir: Path | None = None):
        from transformers import CLIPImageProcessor

        return CLIPImageProcessor.from_pretrained(self.resolve(model_dir, weights=False), do_center_crop=False)

    def load_model(self, model_dir: Path | None = None):
        from transformers import LlavaForConditionalGeneration

        return LlavaForConditionalGeneration.from_pretrained(
            self.resolve(model_dir), dtype=self.dtype, attn_implementation="eager", low_cpu_mem_usage=True)


LLAVA = Llava15(
    name="llava",
    model_id=MODEL_ID,
    revision=None,  # unpinned: the records name no revision, and the local copy has no revision file
    default_dir=Path("models/llava-1.5-7b-hf"),
    fetch_command="kvsel fetch-model",
    required_files=(
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
        "config.json", "generation_config.json", "preprocessor_config.json", "processor_config.json",
    ),
    template_version="v3.1",  # v3.0 with the description labelled unverified (prompt test, 9 Nov 2026)
    image_token=IMAGE_TOKEN,
    image_token_id=IMAGE_TOKEN_ID,
    terminator_id=2,
    stop_ids=(2,),
    candidate_separator=" ",
    dtype=torch.float16,
    default_head_precision="fp32",
    head_slice_rows=None,
    prefix_residency="device",
    add_special_tokens=True,
    description_label="Unverified description, possibly wrong: ",
)
