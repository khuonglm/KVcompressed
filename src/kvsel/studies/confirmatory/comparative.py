"""Confirmatory study, stage 3 and the run (PREREG 1.6, 2.1, 2.2).

`main_set` applies the frozen rule: the first 1,000 items in manifest order that are eligible in the screen and whose
latest verification decision is "pass"; the first 300 form the control subset. Stage 3 freezes PREREG, the stage-1
record, the manifest, the screen flags, the main set with its decisions, the code and the settings. `run_confirm`
runs each main-set item once.
"""

from __future__ import annotations

from pathlib import Path

from ...analysis.tables import checks, load_run
from ...items.manifest import load_items
from ...model import Bundle
from ...models.llava15 import LAYOUT
from ...protocol import Prereg, Refusal, Stage, check, freeze, manifests, selected_set, sha256_file
from ...runner import (
    NO_CONTEXT,
    ItemFile,
    Q,
    Unit,
    item_loop,
    open_image,
    require_images,
    row_units,
    run_item,
    screen_flags,
)
from ...verification import PASS, load_decisions
from ..pilot import pilot_units
from .screening import SCREENING, check_screening

SECTION = "## 2. Comparative experiment"
MAIN_SIZE = 1000
CONTROL_SIZE = 300
K = 144
BLIND = Unit("full", LAYOUT.total, False, NO_CONTEXT, Q)  # PREREG 2.2: on a blank image of the padding colour


def main_units() -> list[Unit]:
    """PREREG 2.1, units on every main-set item (Blind_Q aside)."""
    return pilot_units(("mis",), K)


def control_units() -> list[Unit]:
    """PREREG 2.1, the control subset's units: the main units plus match and irr."""
    return pilot_units(k=K)


def comparative_settings() -> dict:
    return {
        "k": K,
        "selectors": ["pw", "qspan"],
        "primary_selector": "pw",
        "surface": SCREENING["surface"],
        "main_units": [unit.record() for unit in main_units()],
        "control_units": [unit.record() for unit in control_units()],
        "blind": "Full_Q on a blank image of the padding colour; consistency check, should never be eligible",
        "main_size": MAIN_SIZE,
        "control_size": CONTROL_SIZE,
    }


def main_set(items: list[dict], screen_dir: Path, decisions_path: Path) -> dict:
    """PREREG 1.6: the first ``MAIN_SIZE`` eligible items in manifest order whose latest decision is "pass"; the first
    ``CONTROL_SIZE`` are the control subset. An unverified eligible item may not be skipped, so the rule refuses while
    one is left before the set is full."""
    flags = screen_flags(screen_dir)
    decisions, stale = load_decisions(decisions_path, items)
    chosen, unverified, failed = [], [], 0
    for item in items:
        if not flags.get(item["item_id"], False):
            continue
        decision = decisions.get(item["item_id"])
        if decision is None:
            unverified.append(item["item_id"])
        elif decision["decision"] == PASS:
            if not unverified:
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
        raise Refusal(f"main set has {len(chosen)} of {MAIN_SIZE} items; verify more ({where})")
    return {"main": chosen, "control": chosen[:CONTROL_SIZE], "failed_before_cutoff": failed, "stale_decisions": stale}


def stage(prereg: Path, manifest: Path, screen_dir: Path, decisions_path: Path, record: Path, stage1: Path) -> Stage:
    """Stage 3's declaration (PREREG, stage 3 in the preamble): it freezes the whole file, and refuses while section 2
    has a DRAFT line or a "[to fill" blank."""
    items = load_items(manifest)
    selection = main_set(items, screen_dir, decisions_path)
    decisions, _ = load_decisions(decisions_path, items)
    inputs = {
        "prereg_sha256": Prereg(prereg),
        "stage1_record_sha256": sha256_file(stage1),
        "manifest": manifests([manifest]),
        "screen_flags_sha256": sha256_file(Path(screen_dir) / "eligible.jsonl"),
        "main_set_sha256": selected_set(selection["main"], decisions),
        "main_items": len(selection["main"]),
        "control_items": len(selection["control"]),
        "failed_before_cutoff": selection["failed_before_cutoff"],
    }
    return Stage("comparative", record, inputs, comparative_settings(), draft=Prereg(prereg, SECTION))


def freeze_comparative(prereg: Path, manifest: Path, screen_dir: Path, decisions: Path, record: Path,
                       stage1: Path) -> dict:
    check_screening(prereg, manifest, stage1)
    return freeze(stage(prereg, manifest, screen_dir, decisions, record, stage1))


def check_comparative(prereg: Path, manifest: Path, screen_dir: Path, decisions: Path, record: Path,
                      stage1: Path) -> dict:
    if not Path(record).exists():  # before reading the inputs, which may not exist yet either
        raise Refusal(f"no comparative freeze at {record}: run `kvsel freeze-comparative` first")
    return check(stage(prereg, manifest, screen_dir, decisions, record, stage1))


def run_confirm(bundle: Bundle, items: list[dict], images_dir: Path, out_dir: Path, selection: dict,
                record_sha256: str) -> dict:
    """PREREG 2.1: runs the main set once, saving probe attention; a crash resumes safely. ``record_sha256``: the
    stage-3 record's. Prints progress only, never margins or answers; then the consistency counts of 2.1 and 2.2."""
    by_id = {item["item_id"]: item for item in items}
    require_images([by_id[i] for i in selection["main"]], images_dir)
    control = set(selection["control"])
    main, controls = main_units(), control_units()

    def units(item_id):
        return controls if item_id in control else main

    def subset(item_id):
        return "control" if item_id in control else "main"

    def files(item):
        return [ItemFile(out_dir / "items" / f"{item['item_id']}.json", subset(item["item_id"]),
                         row_units(units(item["item_id"]), BLIND))]

    def produce(item, missing):
        result, attention = run_item(bundle, item, open_image(images_dir, item), units(item["item_id"]), blind=BLIND)
        return [({**result, "subset": subset(item["item_id"])}, attention)]

    chosen = [by_id[i] for i in selection["main"]]
    for position, _, _ in item_loop(chosen, record_sha256, files, produce):
        if position % 25 == 0 or position == len(chosen):
            print(f"{position}/{len(chosen)} items", flush=True)
    found = checks(load_run(out_dir)[0])  # flags and counts only, never margins per item
    return {"items": len(selection["main"]), "control": len(control),
            "full_q_no_longer_eligible": found["full_q_no_longer_eligible"], "blind_now_eligible": found["blind_now_eligible"]}
