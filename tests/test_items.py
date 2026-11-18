"""Items keep their identity across the reorganization: verification decisions stay bound to the same items."""

import hashlib
from pathlib import Path

import pytest

from kvsel.items.manifest import load_items
from kvsel.verification import item_digest

MANIFESTS = Path(__file__).parents[1] / "manifests"

# sha256 of "item_id<TAB>digest\n" per item in manifest order, captured from kvsel.verify before the move.
DIGESTS = {
    "gqa_dev": "e65cded71333b74e9d2dfb73fec0aa876fbdc3135b7204f11838a28c4baa6250",
    "gqa_confirm": "ce582b0b454447e284957eeccfe879bef15d192659f4f972a76912b74a809433",
    "textvqa_dev": "65642c9111d7bc56c4533e5020c048b1e3595b15b8fda5d4733f478d4f5a3e8c",
    "textvqa_confirm": "88494ed32d08ebc96382fdcf0dc5b447669192ba04997462d40469dcd2817503",
}


@pytest.mark.parametrize("name", DIGESTS)
def test_item_digests_are_unchanged(name):
    items = load_items(MANIFESTS / f"{name}.jsonl")
    lines = "".join(f"{item['item_id']}\t{item_digest(item)}\n" for item in items)
    assert hashlib.sha256(lines.encode()).hexdigest() == DIGESTS[name]
