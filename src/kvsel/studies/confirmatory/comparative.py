"""Confirmatory study, stage 3 and the run (PREREG 1.6, 2.1, 2.2).

`main_set` applies the frozen rule: the first 1,000 items in manifest order that are eligible in the screen and whose
latest verification decision is "pass"; the first 300 form the control subset. Stage 3 freezes PREREG, the stage-1
record, the manifest, the screen flags, the main set with its decisions, the code and the settings. `run_confirm`
runs each main-set item once.
"""

from __future__ import annotations

from pathlib import Path

from ...items.manifest import load_items
from ...model import Bundle
from ...protocol import FrozenInputs, Prereg, Refusal, Stage, check, freeze, manifests, selected_set, sha256_file
from ...runner import Unit, screen_flags
from ...runs import MainSet, run_main_set
from ...verification import PASS, load_decisions
from ..pilot import FULL_Q, pilot_units
from .screening import SCREENING, check_screening

SECTION = "## 2. Comparative experiment"
MAIN_SIZE = 1000
CONTROL_SIZE = 300
K = 144
BLIND = FULL_Q  # PREREG 2.2: on a blank image of the padding colour


def units(*, control: bool) -> list[Unit]:
    """PREREG 2.1: the units on every main-set item, Blind_Q aside; the control subset adds match and irr."""
    return pilot_units(k=K) if control else pilot_units(("mis",), K)


def comparative_settings() -> dict:
    return {
        "k": K,
        "selectors": ["pw", "qspan"],
        "primary_selector": "pw",
        "surface": SCREENING["surface"],
        "main_units": [unit.record() for unit in units(control=False)],
        "control_units": [unit.record() for unit in units(control=True)],
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


def stage(given: FrozenInputs) -> Stage:
    """Stage 3's declaration (PREREG, stage 3 in the preamble): it freezes the whole file, and refuses while section 2
    has a DRAFT line or a "[to fill" blank."""
    items = load_items(given.manifest)
    selection = main_set(items, given.screen_dir, given.decisions)
    decisions, _ = load_decisions(given.decisions, items)
    inputs = {
        "prereg_sha256": Prereg(given.prereg),
        "stage1_record_sha256": sha256_file(given.stage1),
        "manifest": manifests([given.manifest]),
        "screen_flags_sha256": sha256_file(Path(given.screen_dir) / "eligible.jsonl"),
        "main_set_sha256": selected_set(selection["main"], decisions),
        "main_items": len(selection["main"]),
        "control_items": len(selection["control"]),
        "failed_before_cutoff": selection["failed_before_cutoff"],
    }
    return Stage("comparative", given.record, inputs, comparative_settings(), draft=Prereg(given.prereg, SECTION))


def freeze_comparative(given: FrozenInputs) -> dict:
    check_screening(given.prereg, given.manifest, given.stage1)
    return freeze(stage(given))


def check_comparative(given: FrozenInputs) -> dict:
    if not Path(given.record).exists():  # before reading the inputs, which may not exist yet either
        raise Refusal(f"no comparative freeze at {given.record}: run `kvsel confirmatory freeze` first")
    return check(stage(given))


def frozen_main_set(given: FrozenInputs) -> MainSet:
    """The main set and control subset the rule gives now, which `check_comparative` has shown to be the frozen one."""
    selection = main_set(load_items(given.manifest), given.screen_dir, given.decisions)
    return MainSet.of(selection["main"], selection["control"])


def run_confirm(bundle: Bundle, items: list[dict], images_dir: Path, out_dir: Path, main_set: MainSet,
                record_sha256: str) -> dict:
    """PREREG 2.1: the main set once, saving probe attention; a crash resumes safely. ``record_sha256``: the stage-3
    record's. Prints progress only, never margins or answers; then the consistency counts of 2.1 and 2.2."""
    return run_main_set(bundle, items, images_dir, out_dir, main_set, units, BLIND, record_sha256, keep_attention=True)
