"""Reading an item manifest: one JSON item per line, in manifest order."""

from __future__ import annotations

import json
from pathlib import Path


def load_items(path: Path) -> list[dict]:
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]
