"""Stage 3 of the freeze and the confirmatory run (PREREG.md, sections 1.6 and 2).

`main_set` applies the frozen rule: the first 1,000 items in manifest order that are eligible in the
screen and whose latest verification decision is "pass"; the first 300 form the control subset.
`freeze_comparative` records hashes of PREREG.md, the stage-1 record, the manifest, the screen flags,
the main set with its decisions, and all code. `run_confirm` checks that record and runs each item
once. This module is deliberately outside the stage-1 screening modules.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .freeze import SCREENING, check_screening, sha256_file
from .model import Bundle
from .runner import (
    Q,
    Unit,
    eligible,
    open_image,
    pilot_units,
    run_item,
    write_json_atomic,
)
from .screening import blank_image
from .verify import PASS, load_decisions

MAIN_SIZE = 1000
CONTROL_SIZE = 300
K = 144


def comparative_settings() -> dict:
    return {
        "k": K,
        "selectors": ["pw", "qspan"],
        "primary_selector": "pw",
        "surface": SCREENING["surface"],
        "main_units": [unit.__dict__ for unit in pilot_units(("mis",), K)],
        "control_units": [unit.__dict__ for unit in pilot_units(k=K)],
        "blind": "Full_Q on a blank image of the padding colour; consistency check, should never be eligible",
        "main_size": MAIN_SIZE,
        "control_size": CONTROL_SIZE,
    }


def code_hash_all() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def load_flags(screen_dir: Path) -> dict[str, bool]:
    path = Path(screen_dir) / "eligible.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} is missing: the screen has not finished")
    return {r["item_id"]: r["eligible"] for r in map(json.loads, path.read_text().splitlines())}


def main_set(items: list[dict], screen_dir: Path, decisions_path: Path) -> dict:
    """The frozen selection rule. Refuses while eligible items are left to verify and the set is short."""
    flags = load_flags(screen_dir)
    decisions, stale = load_decisions(decisions_path, items)
    chosen, unverified, failed = [], [], 0
    for item in items:
        if not flags.get(item["item_id"], False):
            continue
        decision = decisions.get(item["item_id"])
        if decision is None:
            unverified.append(item["item_id"])
        elif decision["decision"] == PASS:
            if not unverified:  # manifest order: an unverified eligible item may not be skipped
                chosen.append(item["item_id"])
            else:
                break
        else:
            failed += 1
        if len(chosen) == MAIN_SIZE:
            break
    screened_all = len(flags) == len(items)
    if len(chosen) < MAIN_SIZE and (unverified or not screened_all):
        where = f"next unverified eligible item: {unverified[0]}" if unverified else "screen the rest of the manifest"
        raise SystemExit(f"main set has {len(chosen)} of {MAIN_SIZE} items; verify more ({where})")
    return {"main": chosen, "control": chosen[:CONTROL_SIZE], "failed_before_cutoff": failed, "stale_decisions": stale}


def current_state(prereg: Path, manifest: Path, screen_dir: Path, decisions_path: Path, stage1: Path) -> dict:
    items = [json.loads(line) for line in Path(manifest).read_text().splitlines() if line.strip()]
    selection = main_set(items, screen_dir, decisions_path)
    decisions, _ = load_decisions(decisions_path, items)
    chosen = [[i, decisions[i]["digest"]] for i in selection["main"]]
    return {
        "prereg_sha256": sha256_file(prereg),
        "stage1_record_sha256": sha256_file(stage1),
        "manifest": {Path(manifest).name: sha256_file(manifest)},
        "screen_flags_sha256": sha256_file(Path(screen_dir) / "eligible.jsonl"),
        "main_set_sha256": hashlib.sha256(json.dumps(chosen).encode()).hexdigest(),
        "main_items": len(selection["main"]),
        "control_items": len(selection["control"]),
        "failed_before_cutoff": selection["failed_before_cutoff"],
        "code_sha256": code_hash_all(),
        "settings": comparative_settings(),
    }


def freeze_comparative(prereg: Path, manifest: Path, screen_dir: Path, decisions: Path, record: Path,
                       stage1: Path) -> dict:
    check_screening(prereg, manifest, stage1)
    if "DRAFT" in Path(prereg).read_text().split("\n## ", 1)[0]:
        raise SystemExit(f"{prereg} is still marked DRAFT; finish section 2 first")
    state = current_state(prereg, manifest, screen_dir, decisions, stage1)
    if record.exists():
        frozen = json.loads(record.read_text())
        if {key: frozen.get(key) for key in state} == state:
            return frozen
        raise SystemExit(f"{record} already exists with different contents; the comparative protocol is frozen")
    payload = {"stage": "comparative", "frozen_at": datetime.now(UTC).isoformat(timespec="seconds"), **state}
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def check_comparative(prereg: Path, manifest: Path, screen_dir: Path, decisions: Path, record: Path,
                      stage1: Path) -> dict:
    if not record.exists():
        raise SystemExit(f"no comparative freeze at {record}: run `kvsel freeze-comparative` first")
    frozen = json.loads(record.read_text())
    now = current_state(prereg, manifest, screen_dir, decisions, stage1)
    changed = [key for key in now if frozen.get(key) != now[key]]
    if changed:
        raise SystemExit("Refusing to run: changed since the comparative freeze: " + ", ".join(changed))
    return frozen


def run_confirm(bundle: Bundle, items: list[dict], images_dir: Path, out_dir: Path, selection: dict) -> dict:
    """Runs the main set once; an item's file appears only when complete, so a crash resumes safely.

    Prints progress only, never margins or answers.
    """
    by_id = {item["item_id"]: item for item in items}
    missing = [by_id[i]["image_file"] for i in selection["main"] if not (images_dir / by_id[i]["image_file"]).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} images are missing from {images_dir} (first: {missing[0]})")
    control = set(selection["control"])
    (out_dir / "items").mkdir(parents=True, exist_ok=True)
    (out_dir / "attention").mkdir(exist_ok=True)
    main_units, control_units = pilot_units(("mis",), K), pilot_units(k=K)
    blank = blank_image(bundle)
    for position, item_id in enumerate(selection["main"], 1):
        target = out_dir / "items" / f"{item_id}.json"
        if target.exists():
            continue
        item = by_id[item_id]
        units = control_units if item_id in control else main_units
        result, attention = run_item(bundle, item, open_image(images_dir, item), units)
        blind, _ = run_item(bundle, item, blank, [Unit("full", 576, False, "none", Q)])
        result["rows"].append({**blind["rows"][0], "selector": "blind"})
        result["subset"] = "control" if item_id in control else "main"
        np.savez_compressed(out_dir / "attention" / f"{item_id}.npz", **attention)
        write_json_atomic(target, result)
        if position % 25 == 0 or position == len(selection["main"]):
            print(f"{position}/{len(selection['main'])} items", flush=True)
    # Consistency with the screen: Full_Q must still be eligible (a determinism check, flags only).
    not_eligible = blind_eligible = 0
    for item_id in selection["main"]:
        rows = json.loads((out_dir / "items" / f"{item_id}.json").read_text())["rows"]
        full_q = next(r for r in rows if r["selector"] == "full" and r["cell"] == "Q")
        not_eligible += not eligible(full_q, SCREENING["surface"])
        blind = next(r for r in rows if r["selector"] == "blind")
        blind_eligible += eligible(blind, SCREENING["surface"])
    return {"items": len(selection["main"]), "control": len(control), "full_q_no_longer_eligible": not_eligible,
            "blind_now_eligible": blind_eligible}
