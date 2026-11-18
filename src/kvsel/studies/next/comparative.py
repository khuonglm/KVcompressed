"""NeXT stage 3 (PREREG 4.2, 4.4): the comparative freeze, the run and the validation plan.

At each share, cell A and cells B, C, D for mis and irr (and match on control items), then the Full
anchors and Blind_Q: the replication's unit lists with shares for budgets.
"""

from __future__ import annotations

from pathlib import Path

from ...analysis.statistics import SURFACE
from ...items.manifest import load_items
from ...model import Bundle
from ...protocol import FrozenInputs, Prereg, Refusal, Stage, check, freeze, manifests, selected_set, sha256_file
from ...runner import NO_CONTEXT, Q, Unit, screen_flags
from ...runs import MainSet, run_main_set
from ...selectors import Budget
from ...validation import Plan, Projection
from ...verification import load_decisions
from . import screening as screen

assert screen.SCREENING["surface"] == SURFACE, "the consistency counts score under the analysis's surface rule"

RECORD = Path("freeze/next_gqa.json")
SHARES = (0.125, 0.25, 0.5)  # PREREG 4.2: budgets as shares of each image's feature tokens
FULL = 1.0  # a Full unit's share: every visual token
CONDITIONS = ("mis", "irr")
CONTROL_CONDITIONS = ("match",)
ALPHA = 0.05
HOLM_FAMILY = (("N1", "S_H^(mis)", 0.25), ("N2", "S_H^(irr)", 0.25), ("N3", "E_harm^(mis)", 0.25), ("N4", "T_irr", 0.25))


def share_unit(selector: str, share: float, selection: str, answering: str) -> Unit:
    """A NeXT unit: its budget is a share of each image's feature tokens (PREREG 4.2)."""
    return Unit(selector, None, False, selection, answering, share=share)


def units(*, control: bool) -> list[Unit]:
    """PREREG 4.2: PW-attn cells A, B_c, C_c, D_c at every share, then the Full anchors (the replication's lists)."""
    conditions = CONDITIONS + (CONTROL_CONDITIONS if control else ())
    out = []
    for share in SHARES:
        out.append(share_unit("pw", share, Q, Q))
        for c in conditions:
            out += [share_unit("pw", share, c, c), share_unit("pw", share, Q, c), share_unit("pw", share, c, Q)]
    return out + [share_unit("full", FULL, NO_CONTEXT, answering) for answering in (Q, *conditions)]


BLIND = screen.FULL_Q  # on the blank image the size of the real one


def settings() -> dict:
    """PREREG 4.4: shares, unit lists, Blind_Q, surface rule, the Holm family and α."""
    return {"shares": list(SHARES), "main_units": [u.record() for u in units(control=False)],
            "control_units": [u.record() for u in units(control=True)], "blind_unit": BLIND.record(),
            "surface": screen.SCREENING["surface"], "family": [list(t) for t in HOLM_FAMILY], "alpha": ALPHA,
            "equivalence": None}  # PREREG 4.3: not tested


def run_comparative(bundle: Bundle, items: list[dict], images_dir: Path, out_dir: Path, main_set: MainSet,
                    record_sha256: str) -> dict:
    """PREREG 4.2: each item once, resumable; prints progress and the two consistency counts only. One prefill and
    one probe per selection context per item, so each share's kept set comes from that probe."""
    found = run_main_set(bundle, items, images_dir, out_dir, main_set, units, BLIND, record_sha256, contexts=False)
    return {key: found[key] for key in ("full_q_no_longer_eligible", "blind_now_eligible")}


# --- Freeze ----------------------------------------------------------------------------------------


def main_set(items: list[dict], flags: dict[str, bool], decisions: Path, size: int) -> tuple[list[str], bool]:
    """PREREG 4.1: the first `size` eligible items of the pool, in manifest order, and whether that set is final: it
    is full, or every pool item is screened (4.5(d): then a short set is every eligible item)."""
    chosen = []
    for item in screen.pool(items, decisions):
        if len(chosen) == size:
            break
        if item["item_id"] not in flags:
            return chosen, False  # an unscreened pool item may not be skipped
        if flags[item["item_id"]]:
            chosen.append(item["item_id"])
    return chosen, True


def stage(given: FrozenInputs, report: bool = False) -> Stage:
    """NeXT stage 3 (PREREG 4.4): the whole PREREG file, the stage-1 record, the manifest, the screen flags, the main
    set with its verification digests and its size frozen at stage 1, the code and the settings."""
    stage1_record = screen.check_screening(given.prereg, given.manifest, given.stage1, report)
    size, control_size = stage1_record["settings"]["main_size"], stage1_record["settings"]["control_size"]
    items = load_items(given.manifest)
    chosen, final = main_set(items, screen_flags(given.screen_dir), given.decisions, size)
    if not final:
        raise Refusal(f"the main set has {len(chosen)} of {size} items and pool items are unscreened; screen the "
                      "rest of the pool (`kvsel next count` shows what is left)")
    current, _ = load_decisions(given.decisions, items)
    inputs = {
        "prereg_sha256": Prereg(given.prereg),
        "stage1_record_sha256": sha256_file(given.stage1),
        "manifest": manifests([given.manifest]),
        "screen_flags_sha256": sha256_file(Path(given.screen_dir) / "eligible.jsonl"),
        "main_set_sha256": selected_set(chosen, current),
        "main_ids": chosen,
        "main_size_frozen_at_stage1": size,  # more than len(main_ids) means a shortfall (PREREG 4.5(d))
        "control_ids": chosen[:control_size],
    }
    return Stage("next_comparative", given.record, inputs, settings(), draft=screen.section(given.prereg))


def freeze_comparative(given: FrozenInputs) -> dict:
    return freeze(stage(given))


def check_comparative(given: FrozenInputs, report: bool = False) -> dict:
    """The frozen record, if nothing changed; ``report``: the report-time check, which doesn't compare PREREG."""
    if not Path(given.record).exists():  # before reading the inputs
        raise Refusal(f"no NeXT comparative freeze at {given.record}: run `kvsel next freeze` first")
    return check(stage(given, report), report)


def frozen_main_set(record: dict) -> MainSet:
    """The record's main set and its control subset."""
    return MainSet.of(record["main_ids"], record["control_ids"])


# --- Validation ------------------------------------------------------------------------------------

POOL = 2335  # passed GQA confirmatory items, 16 November 2026 (PREREG 4.5(d)); for the projection only

VALIDATION = Plan(  # tests/README.md, "LLaVA-NeXT-7B, Level 2"
    budgets=tuple(Budget(share=share) for share in SHARES),
    kinds=("random", "concentrated", "tail"),  # E3 compares compact and dense on any sparse sets; tail: PREREG 4.6
    units=tuple(units(control=True)),
    projection=Projection(POOL, screen.SCREENING["main_size"], screen.SCREENING["control_size"],
                          tuple(units(control=False)), tuple(units(control=True))),
)
