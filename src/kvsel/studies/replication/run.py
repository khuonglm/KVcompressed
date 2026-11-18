"""The GQA replication's units, its freeze and its run (PREREG 3.2, 3.5).

The freeze records PREREG, the stage-1 record, the GQA stage-3 record, the manifest, the screen flags, the
replication set with its verification digests, the earlier run folders and the exclusions, the settings and the code.
The run and the report recheck it; the report doesn't compare PREREG, which may grow by dated amendments (3.5).
"""

from __future__ import annotations

import json
from pathlib import Path

from ...analysis.statistics import DELTA, SURFACE
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
from ...verification import load_decisions
from ..confirmatory.comparative import main_set
from ..confirmatory.screening import check_screening
from . import select

RECORD = Path("freeze/replication_gqa.json")
SECTION = "## 3. GQA replication"
BUDGETS = (72, 144, 288)
CONDITIONS = ("mis", "irr")
CONTROL_CONDITIONS = ("match",)
BLIND = Unit("full", LAYOUT.total, False, NO_CONTEXT, Q)
ALPHA = 0.05
FAMILY = (("R1", "S_H^(mis)", "72"), ("R1", "S_H^(mis)", "144"), ("R2", "S_H^(irr)", "72"),
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
    out += [Unit("full", LAYOUT.total, False, NO_CONTEXT, answering) for answering in (Q, *conditions)]
    return out


def settings() -> dict:
    """PREREG 3.5: units, budgets, Blind_Q, surface rule, the hypotheses, α and δ."""
    return {"budgets": list(BUDGETS), "size": select.REPLICATION_SIZE, "control_size": select.CONTROL_SIZE,
            "main_units": [u.record() for u in units(control=False)],
            "control_units": [u.record() for u in units(control=True)], "blind_unit": BLIND.record(),
            "surface": SURFACE, "family": [list(t) for t in FAMILY], "alpha": ALPHA,
            "equivalence": list(EQUIVALENCE), "delta": DELTA}


def stage(prereg: Path, manifest: Path, screen: Path, decisions: Path, stage1: Path, gqa_record: Path,
          exclude_runs: list[Path], record: Path, report: bool = False) -> Stage:
    """The replication's declaration (PREREG 3.5). It needs the stage-1 check to pass and the confirmatory main set
    to be the one the GQA record froze; it refuses while section 3 has a DRAFT line or a "[to fill" blank."""
    check_screening(prereg, manifest, stage1, report)
    items = load_items(manifest)
    current, _ = load_decisions(decisions, items)
    main = main_set(items, screen, decisions)["main"]
    if selected_set(main, current) != json.loads(Path(gqa_record).read_text())["main_set_sha256"]:
        raise Refusal(f"the main set no longer matches {gqa_record}; a decision before its cutoff changed")
    chosen = select.replication_set(items, screen_flags(screen), current, main, select.earlier_run_ids(exclude_runs))
    if len(chosen["replication"]) < select.REPLICATION_SIZE:
        where = f"verify {chosen['next_unverified']} next" if chosen["next_unverified"] else "the pool is exhausted"
        raise Refusal(f"replication set has {len(chosen['replication'])} of {select.REPLICATION_SIZE}; {where}")
    inputs = {
        "prereg_sha256": Prereg(prereg),
        "stage1_record_sha256": sha256_file(stage1),
        "gqa_record_sha256": sha256_file(gqa_record),
        "manifest": manifests([manifest]),
        "screen_flags_sha256": sha256_file(Path(screen) / "eligible.jsonl"),
        "replication_set_sha256": selected_set(chosen["replication"], current),
        "replication_ids": chosen["replication"],
        "control_ids": chosen["control"],
        "failed_before_cutoff": chosen["failed"],
        "exclude_runs": [str(run) for run in exclude_runs],
        "excluded": chosen["excluded"],
    }
    return Stage("replication", record, inputs, settings(), draft=Prereg(prereg, SECTION))


def freeze_replication(prereg: Path, manifest: Path, screen: Path, decisions: Path, stage1: Path, gqa_record: Path,
                       exclude_runs: list[Path], record: Path) -> dict:
    return freeze(stage(prereg, manifest, screen, decisions, stage1, gqa_record, exclude_runs, record))


def check_replication(prereg: Path, manifest: Path, screen: Path, decisions: Path, stage1: Path, gqa_record: Path,
                      exclude_runs: list[Path], record: Path, report: bool = False) -> dict:
    """The frozen record, if nothing changed; ``report``: the report-time check, which doesn't compare PREREG."""
    if not Path(record).exists():  # before reading the inputs
        raise Refusal(f"no replication freeze at {record}: run `python -m kvsel.replication freeze` first")
    return check(stage(prereg, manifest, screen, decisions, stage1, gqa_record, exclude_runs, record, report), report)


def run_replication(bundle: Bundle, items: list[dict], images_dir: Path, out_dir: Path, selection: dict,
                    record_sha256: str) -> dict:
    """PREREG 3.2: each item once, resumable, printing progress only, then the consistency counts. `run_item` probes
    each selection context once, so every budget's kept set comes from the same scores. ``record_sha256``: the
    replication record's."""
    by_id = {item["item_id"]: item for item in items}
    ids = selection["replication_ids"]
    require_images([by_id[i] for i in ids], images_dir)
    control = set(selection["control_ids"])

    def subset(item_id):
        return "control" if item_id in control else "main"

    def files(item):
        return [ItemFile(out_dir / "items" / f"{item['item_id']}.json", subset(item["item_id"]),
                         row_units(units(control=item["item_id"] in control), BLIND))]

    def produce(item, missing):
        result, _ = run_item(bundle, item, open_image(images_dir, item), units(control=item["item_id"] in control),
                             blind=BLIND)
        return [({**result, "subset": subset(item["item_id"])}, None)]

    for position, _, _ in item_loop([by_id[i] for i in ids], record_sha256, files, produce):
        if position % 25 == 0 or position == len(ids):
            print(f"{position}/{len(ids)} items", flush=True)
    # Consistency with the screen (flags and counts only, never margins per item).
    found = checks(load_run(out_dir)[0])
    return {"items": len(ids), "control": len(control), "full_q_no_longer_eligible": found["full_q_no_longer_eligible"],
            "blind_now_eligible": found["blind_now_eligible"]}
