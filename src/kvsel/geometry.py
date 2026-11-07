"""The image transform and box mapping, kept together so a box can't drift from its pixels.

As in the original LLaVA-1.5 evaluation: pad to square, then resize to 336 x 336, with no centre crop.
"""

from __future__ import annotations

from PIL import Image

from . import GRID

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
PAD_COLOR = tuple(int(channel * 255) for channel in CLIP_MEAN)
IMAGE_SIZE = 336
PATCH_SIZE = 14
MIN_OVERLAP_FRACTION = 0.25
MAPPING_RULES = ("primary", "fallback", "center", "expanded")

Box = tuple[float, float, float, float]  # x0, y0, x1, y1


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


def _overlap_patches(box: Box) -> list[int]:
    x0, y0, x1, y1 = box
    patches = []
    for row in range(GRID):
        for col in range(GRID):
            overlap_w = min(x1, (col + 1) * PATCH_SIZE) - max(x0, col * PATCH_SIZE)
            overlap_h = min(y1, (row + 1) * PATCH_SIZE) - max(y0, row * PATCH_SIZE)
            if overlap_w > 0 and overlap_h > 0 and overlap_w * overlap_h >= MIN_OVERLAP_FRACTION * PATCH_SIZE**2:
                patches.append(row * GRID + col)
    return patches


def _point_patch(x: float, y: float) -> int:
    col = min(max(int(x // PATCH_SIZE), 0), GRID - 1)
    row = min(max(int(y // PATCH_SIZE), 0), GRID - 1)
    return row * GRID + col


def _fallback_patches(box: Box) -> list[int]:
    """Patches containing the box's centre and corners."""
    x0, y0, x1, y1 = box
    points = [((x0 + x1) / 2, (y0 + y1) / 2), (x0, y0), (x1, y0), (x0, y1), (x1, y1)]
    return sorted({_point_patch(x, y) for x, y in points})


def _center_patches(box: Box) -> list[int]:
    x0, y0, x1, y1 = box
    return [
        row * GRID + col
        for row in range(GRID)
        for col in range(GRID)
        if x0 <= (col + 0.5) * PATCH_SIZE <= x1 and y0 <= (row + 0.5) * PATCH_SIZE <= y1
    ]


def evidence_patches(box: Box, rule: str = "primary") -> list[int]:
    """Grid cells for a box in model coordinates, under one of the proposal's four mapping rules.

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
