"""The replication run (PREREG section 3.2): each item once, resumable, printing progress only."""

from __future__ import annotations

from pathlib import Path

from ..confirm_analysis import load_run
from ..model import Bundle
from ..runner import open_image, run_item, write_json_atomic
from ..screening import blank_image
from .protocol import BLIND, checks, units


def run_replication(bundle: Bundle, items: list[dict], images_dir: Path, out_dir: Path, selection: dict) -> dict:
    """`run_item` probes each selection context once, so every budget's kept set comes from the same scores."""
    by_id = {item["item_id"]: item for item in items}
    ids = selection["replication_ids"]
    missing = [by_id[i]["image_file"] for i in ids if not (Path(images_dir) / by_id[i]["image_file"]).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} images are missing from {images_dir} (first: {missing[0]})")
    control = set(selection["control_ids"])
    (out_dir / "items").mkdir(parents=True, exist_ok=True)
    blank = blank_image(bundle)
    for position, item_id in enumerate(ids, 1):
        target = out_dir / "items" / f"{item_id}.json"
        if not target.exists():
            item = by_id[item_id]
            result, _ = run_item(bundle, item, open_image(images_dir, item), units(control=item_id in control))
            blind, _ = run_item(bundle, item, blank, [BLIND])
            result["rows"].append({**blind["rows"][0], "selector": "blind"})
            result["subset"] = "control" if item_id in control else "main"
            write_json_atomic(target, result)
        if position % 25 == 0 or position == len(ids):
            print(f"{position}/{len(ids)} items", flush=True)
    # Consistency with the screen (flags and counts only, never margins per item).
    found = checks(load_run(out_dir)[0])
    return {"items": len(ids), "control": len(control), "full_q_no_longer_eligible": found["full_q_no_longer_eligible"],
            "blind_now_eligible": found["blind_now_eligible"]}
