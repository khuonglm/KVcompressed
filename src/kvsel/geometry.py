"""Image preprocessing as in the original LLaVA-1.5 evaluation: pad to square, no centre crop."""

from __future__ import annotations

from PIL import Image

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
PAD_COLOR = tuple(int(channel * 255) for channel in CLIP_MEAN)


def pad_to_square(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size
    side = max(width, height)
    canvas = Image.new("RGB", (side, side), PAD_COLOR)
    canvas.paste(image, ((side - width) // 2, (side - height) // 2))
    return canvas
