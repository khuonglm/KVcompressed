"""Reading an item manifest: one JSON item per line, in manifest order."""

from __future__ import annotations

import json
from pathlib import Path

from ..protocol import Refusal


def load_items(path: Path) -> list[dict]:
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def require_confirmatory(path: Path) -> None:
    """Refuses a manifest holding any non-confirmatory item: freezes take confirmatory manifests only."""
    splits = {item.get("split") for item in load_items(path)}
    if splits != {"confirm"}:
        raise Refusal(f"{path} is not a confirmatory manifest (splits: {sorted(map(str, splits))})")
