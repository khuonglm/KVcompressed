"""Rules every builder shares: the development split, JSONL writing and the description-length rule."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from ..models.llava15 import LLAVA

SPLIT_SEED = 0
MAX_DESCRIPTION_TOKEN_SPREAD = 3


def descriptions_within_spread(tokenizer, question: str, descriptions: dict[str, str]) -> bool:
    """The three descriptions' suffix lengths differ by at most MAX_DESCRIPTION_TOKEN_SPREAD tokens."""
    lengths = [len(LLAVA.tokenize(tokenizer, question, d).suffix_ids) for d in descriptions.values()]
    return max(lengths) - min(lengths) <= MAX_DESCRIPTION_TOKEN_SPREAD


def split_by_image(items: list[dict], dev_items: int) -> tuple[list[dict], list[dict]]:
    """Whole images go to one split. Items within a split follow the shuffled image order."""
    by_image: dict[str, list[dict]] = {}
    for item in items:
        by_image.setdefault(item["image_id"], []).append(item)
    image_ids = sorted(by_image)
    random.Random(SPLIT_SEED).shuffle(image_ids)
    development, confirmatory = [], []
    for image_id in image_ids:
        target = development if len(development) < dev_items else confirmatory
        target.extend({**item, "split": "dev" if target is development else "confirm"} for item in by_image[image_id])
    return development, confirmatory


def split_development(items: list[dict], dev_items: int, dev_manifest: Path | None, dev_path: Path,
                      funnel: dict) -> tuple[list[dict], list[dict], str]:
    """With dev_manifest, development items are kept byte for byte (their verification stays valid)
    and the confirmatory pool is every built item on an image not used in development. Without it,
    a fresh split. Writes the development manifest and returns its sha256."""
    if dev_manifest is None:
        development, confirmatory = split_by_image(items, dev_items)
        return development, confirmatory, write_jsonl(dev_path, development)
    dev_bytes = Path(dev_manifest).read_bytes()
    development = [json.loads(line) for line in dev_bytes.decode().splitlines() if line.strip()]
    dev_images = {item["image_id"] for item in development}
    _, confirmatory = split_by_image([item for item in items if item["image_id"] not in dev_images], 0)
    if Path(dev_manifest).resolve() != dev_path.resolve():
        dev_path.write_bytes(dev_bytes)
    funnel["dev_frozen_from"] = {"builder": development[0]["builder"], "file": Path(dev_manifest).name}
    return development, confirmatory, hashlib.sha256(dev_bytes).hexdigest()


def write_jsonl(path: Path, items: list[dict]) -> str:
    text = "".join(json.dumps(item, sort_keys=True) + "\n" for item in items)
    path.write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()
