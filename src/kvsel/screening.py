"""Stage 2 of the freeze: the eligibility screen on confirmatory items, exposing only eligibility.

Each item runs Full_Q only (full cache, no description, no selection). Nothing but the item id and
its eligible flag is written: no answers, scores or margins, so no outcome can steer verification or
the comparative experiment. The comparative run later recomputes Full_Q itself.
"""

from __future__ import annotations

import json
from pathlib import Path

from .model import Bundle
from .runner import (
    NO_CONTEXT,
    Q,
    Unit,
    eligible,
    open_image,
    run_item,
    write_json_atomic,
)

FULL_Q = Unit("full", 576, False, NO_CONTEXT, Q)


def run_confirm_screen(bundle: Bundle, items: list[dict], images_dir: Path, out_dir: Path, frozen: dict) -> dict:
    settings = frozen["settings"]
    if bundle.family.template_version != settings["template"]:
        raise SystemExit(f"template {bundle.family.template_version} differs from the frozen {settings['template']}")
    if bundle.decoder.head_precision != settings["head_precision"]:
        raise SystemExit(f"head precision {bundle.decoder.head_precision} differs from the frozen {settings['head_precision']}")
    not_confirm = [item["item_id"] for item in items if item.get("split") != "confirm"]
    if not_confirm:
        raise SystemExit(f"{len(not_confirm)} items are not confirmatory (first: {not_confirm[0]})")
    missing = [item["image_file"] for item in items if not (images_dir / item["image_file"]).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} images are missing from {images_dir} (first: {missing[0]}); "
                         "fetch them or screen fewer items with --limit")

    (out_dir / "items").mkdir(parents=True, exist_ok=True)
    flags: dict[str, bool] = {}
    for position, item in enumerate(items, 1):
        target = out_dir / "items" / f"{item['item_id']}.json"
        if target.exists():
            flags[item["item_id"]] = json.loads(target.read_text())["eligible"]
            continue
        result, _ = run_item(bundle, item, open_image(images_dir, item), [FULL_Q])
        flag = eligible(result["rows"][0], settings["surface"])
        write_json_atomic(target, {"item_id": item["item_id"], "eligible": flag})
        flags[item["item_id"]] = flag
        if position % 100 == 0 or position == len(items):
            print(f"{position}/{len(items)} items screened", flush=True)

    with open(out_dir / "eligible.jsonl", "w") as handle:  # manifest order
        for item in items:
            handle.write(json.dumps({"item_id": item["item_id"], "eligible": flags[item["item_id"]]}) + "\n")
    funnel = {
        "screened": len(items),
        "eligible": sum(flags.values()),
        "not_eligible": len(items) - sum(flags.values()),
        "frozen_at": frozen["frozen_at"],
        "code_sha256": frozen["code_sha256"],
        "settings": settings,
    }
    write_json_atomic(out_dir / "funnel.json", funnel)
    return funnel


def eligible_ids(screen_dir: Path) -> set[str]:
    """Ids flagged eligible in a confirmatory screen directory (works while the screen is running)."""
    found = set()
    for path in (screen_dir / "items").glob("*.json"):
        record = json.loads(path.read_text())
        if record["eligible"]:
            found.add(record["item_id"])
    return found
