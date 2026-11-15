"""Where each visual token of an AnyRes image sits: the base tile, then high-resolution rows each ended by a newline."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers.models.llava_next.modeling_llava_next import get_anyres_image_grid_shape, unpad_image


@dataclass(frozen=True)
class Layout:
    base: int  # side of the base tile's token grid
    grid: tuple[int, int]  # high-resolution (rows, columns) after unpadding, newlines excluded

    @property
    def total(self) -> int:
        rows, cols = self.grid
        return self.base * self.base + rows * (cols + 1)

    @property
    def newlines(self) -> tuple[int, ...]:
        """Visual-span indices of the newline tokens, one after each high-resolution row."""
        rows, cols = self.grid
        return tuple(self.base * self.base + r * (cols + 1) + cols for r in range(rows))

    def positions(self) -> list[tuple[str, int, int]]:
        """(map, row, column) for each visual-span index; a newline is ("newline", its row, columns)."""
        rows, cols = self.grid
        base = [("base", r, c) for r in range(self.base) for c in range(self.base)]
        high = [("high" if c < cols else "newline", r, c) for r in range(rows) for c in range(cols + 1)]
        return base + high


def derive_layout(config, image_size: list[int]) -> Layout:
    """From the model's own AnyRes grid and unpadding, for an image of (height, width) as the processor reports it."""
    if config.vision_feature_select_strategy != "default":
        raise ValueError("expected the 'default' feature strategy, which drops CLS and leaves a square base tile")
    tile = config.vision_config.image_size
    side = tile // config.vision_config.patch_size
    rows, cols = get_anyres_image_grid_shape(list(image_size), config.image_grid_pinpoints, tile)
    unpadded = unpad_image(torch.empty(0, rows * side, cols * side), list(image_size))
    return Layout(side, tuple(unpadded.shape[1:]))


def visual_span(ids: list[int], image_token_id: int) -> tuple[int, int]:
    """Half-open range of the image tokens, which must form one contiguous run."""
    found = [i for i, token in enumerate(ids) if token == image_token_id]
    if not found or found[-1] - found[0] + 1 != len(found):
        raise ValueError("the image tokens must form one contiguous span")
    return found[0], found[-1] + 1
