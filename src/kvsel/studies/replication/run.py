"""The GQA replication's units, its freeze and its run (PREREG 3.2, 3.5).

The freeze records PREREG, the stage-1 record, the GQA stage-3 record, the manifest, the screen flags, the
replication set with its verification digests, the earlier run folders and the exclusions, the settings and the code.
The run and the report recheck it; the report doesn't compare PREREG, which may grow by dated amendments (3.5).
"""

from __future__ import annotations

import json
from pathlib import Path

from ...analysis.statistics import DELTA, SURFACE
from ...items.manifest import load_items
from ...model import Bundle
from ...protocol import FrozenInputs, Prereg, Refusal, Stage, check, freeze, manifests, selected_set, sha256_file
from ...runner import Q, Unit, screen_flags
from ...runs import MainSet, run_main_set
from ...verification import load_decisions
from ..confirmatory.comparative import main_set
from ..confirmatory.screening import check_screening
from ..pilot import FULL_Q, full
from . import select

RECORD = Path("freeze/replication_gqa.json")
SECTION = "## 3. GQA replication"
BUDGETS = (72, 144, 288)
CONDITIONS = ("mis", "irr")
CONTROL_CONDITIONS = ("match",)
BLIND = FULL_Q  # on the blank image (PREREG 3.2, as 2.2)
ALPHA = 0.05
HOLM_FAMILY = (("R1", "S_H^(mis)", "72"), ("R1", "S_H^(mis)", "144"), ("R2", "S_H^(irr)", "72"),
          ("R2", "S_H^(irr)", "144"), ("R3", "E_harm^(mis)", "72"), ("R5", "T_irr", "72"))
EQUIVALENCE = ("R4", "S_H^(mis)", "288")


def units(*, control: bool) -> list[Unit]:
    """PREREG 3.2: PW-attn cells A, B_c, C_c, D_c at every budget, then the full-cache anchors. The run adds Blind_Q."""
    conditions = CONDITIONS + (CONTROL_CONDITIONS if control else ())
    out = []
    for k in BUDGETS:
        out.append(Unit("pw", k, False, Q, Q))
        for c in conditions:
            out += [Unit("pw", k, False, c, c), Unit("pw", k, False, Q, c), Unit("pw", k, False, c, Q)]
    out += [full(answering) for answering in (Q, *conditions)]
    return out


def settings() -> dict:
    """PREREG 3.5: units, budgets, Blind_Q, surface rule, the hypotheses, α and δ."""
    return {"budgets": list(BUDGETS), "size": select.REPLICATION_SIZE, "control_size": select.CONTROL_SIZE,
            "main_units": [u.record() for u in units(control=False)],
            "control_units": [u.record() for u in units(control=True)], "blind_unit": BLIND.record(),
            "surface": SURFACE, "family": [list(t) for t in HOLM_FAMILY], "alpha": ALPHA,
            "equivalence": list(EQUIVALENCE), "delta": DELTA}


def stage(given: FrozenInputs, gqa_record: Path, exclude_runs: list[Path], report: bool = False) -> Stage:
    """The replication's declaration (PREREG 3.5). It needs the stage-1 check to pass and the confirmatory main set
    to be the one the GQA record froze; it refuses while section 3 has a DRAFT line or a "[to fill" blank."""
    check_screening(given.prereg, given.manifest, given.stage1, report)
    items = load_items(given.manifest)
    current, _ = load_decisions(given.decisions, items)
    main = main_set(items, given.screen_dir, given.decisions)["main"]
    if selected_set(main, current) != json.loads(Path(gqa_record).read_text())["main_set_sha256"]:
        raise Refusal(f"the main set no longer matches {gqa_record}; a decision before its cutoff changed")
    chosen = select.replication_set(items, screen_flags(given.screen_dir), current, main,
                                    select.earlier_run_ids(exclude_runs))
    if len(chosen["replication"]) < select.REPLICATION_SIZE:
        where = f"verify {chosen['next_unverified']} next" if chosen["next_unverified"] else "the pool is exhausted"
        raise Refusal(f"replication set has {len(chosen['replication'])} of {select.REPLICATION_SIZE}; {where}")
    inputs = {
        "prereg_sha256": Prereg(given.prereg),
        "stage1_record_sha256": sha256_file(given.stage1),
        "gqa_record_sha256": sha256_file(gqa_record),
        "manifest": manifests([given.manifest]),
        "screen_flags_sha256": sha256_file(Path(given.screen_dir) / "eligible.jsonl"),
        "replication_set_sha256": selected_set(chosen["replication"], current),
        "replication_ids": chosen["replication"],
        "control_ids": chosen["control"],
        "failed_before_cutoff": chosen["failed"],
        "exclude_runs": [str(run) for run in exclude_runs],
        "excluded": chosen["excluded"],
    }
    return Stage("replication", given.record, inputs, settings(), draft=Prereg(given.prereg, SECTION))


def freeze_replication(given: FrozenInputs, gqa_record: Path, exclude_runs: list[Path]) -> dict:
    return freeze(stage(given, gqa_record, exclude_runs))


def check_replication(given: FrozenInputs, gqa_record: Path, exclude_runs: list[Path], report: bool = False) -> dict:
    """The frozen record, if nothing changed; ``report``: the report-time check, which doesn't compare PREREG."""
    if not Path(given.record).exists():  # before reading the inputs
        raise Refusal(f"no replication freeze at {given.record}: run `kvsel replication freeze` first")
    return check(stage(given, gqa_record, exclude_runs, report), report)


def frozen_main_set(record: dict) -> MainSet:
    """The record's replication set and its control subset."""
    return MainSet.of(record["replication_ids"], record["control_ids"])


def run_replication(bundle: Bundle, items: list[dict], images_dir: Path, out_dir: Path, main_set: MainSet,
                    record_sha256: str) -> dict:
    """PREREG 3.2: each item once, resumable, printing progress only, then the consistency counts. `run_item` probes
    each selection context once, so every budget's kept set comes from the same scores. ``record_sha256``: the
    replication record's."""
    return run_main_set(bundle, items, images_dir, out_dir, main_set, units, BLIND, record_sha256)
