"""NeXT stage 3 (PREREG 4.2, 4.4): the comparative freeze and the run.

At each share, cell A and cells B, C, D for mis and irr (and match on control items), then the Full
anchors and Blind_Q: the replication's unit lists with shares for budgets. It mirrors `kvsel.replication.protocol` and `.run` instead of calling them: their units assume LLaVA-1.5's fixed grid.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from kvsel.confirm import load_flags
from kvsel.confirm_analysis import SURFACE, checks, load_run
from kvsel.engine import probe
from kvsel.freeze import sha256_file
from kvsel.model import Bundle
from kvsel.models.llava_next import NEXT, high_grid
from kvsel.runner import NO_CONTEXT, Q, Unit, load_items, open_image, write_json_atomic
from kvsel.selectors import Budget
from kvsel.verification import load_decisions

from . import screen
from .model import run_unit
from .selection import SHARES, kept_sets_per_share

assert screen.SCREENING["surface"] == SURFACE, "kvsel's `checks` scores consistency under its own surface rule"

FULL = 1.0  # a Full unit's share: every visual token


@dataclass(frozen=True)
class NextUnit:
    """kvsel's Unit with a share in place of an absolute k, which differs per item."""

    selector: str  # pw or full
    share: float
    selection: str  # Q, a condition, or none
    answering: str  # Q or a condition


CONDITIONS = ("mis", "irr")
CONTROL_CONDITIONS = ("match",)
ALPHA = 0.05
FAMILY = (("N1", "S_H^(mis)", 0.25), ("N2", "S_H^(irr)", 0.25), ("N3", "E_harm^(mis)", 0.25), ("N4", "T_irr", 0.25))


def units(*, control: bool) -> list[NextUnit]:
    """PW-attn cells A, B_c, C_c, D_c at every share, then the Full anchors, as `kvsel.replication.protocol.units`."""
    conditions = CONDITIONS + (CONTROL_CONDITIONS if control else ())
    out = []
    for share in SHARES:
        out.append(NextUnit("pw", share, Q, Q))
        for c in conditions:
            out += [NextUnit("pw", share, c, c), NextUnit("pw", share, Q, c), NextUnit("pw", share, c, Q)]
    return out + [NextUnit("full", FULL, NO_CONTEXT, answering) for answering in (Q, *conditions)]


BLIND = NextUnit("full", FULL, NO_CONTEXT, Q)  # on NEXT.blank_image(image)


def settings() -> dict:
    return {"shares": list(SHARES), "main_units": [asdict(u) for u in units(control=False)],
            "control_units": [asdict(u) for u in units(control=True)], "blind_unit": asdict(BLIND),
            "surface": screen.SCREENING["surface"], "family": [list(t) for t in FAMILY], "alpha": ALPHA,
            "equivalence": None}  # PREREG 4.3: not tested


def run_item(bundle: Bundle, item: dict, image: Image.Image, plan: list[NextUnit]) -> dict:
    """One prefill, one probe per selection context; each share's kept set comes from that probe."""
    tokens, inputs = NEXT.contexts(bundle, item, image)
    prefix, layout = NEXT.prefill(bundle, tokens[Q].prefix_ids, inputs)
    contexts = sorted({u.selection for u in plan if u.selector == "pw"})
    kept = {c: kept_sets_per_share(layout, probe(bundle.decoder, prefix, torch.tensor([tokens[c].suffix_ids])))
            for c in contexts}  # each probe moves its own GPU copy of the prefix, freed when it returns
    record = {"base": layout.maps[0].rows, "grid": list(high_grid(layout)), "total": layout.total}
    rows = []
    for u in plan:
        if u.selector == "full":
            keep, k = np.arange(layout.total), layout.total
        else:
            keep, k = kept[u.selection][u.share], Budget(share=u.share).resolve(layout)
        row = run_unit(bundle, item, Unit(u.selector, k, False, u.selection, u.answering), prefix, keep, tokens)
        rows.append({**row, "share": u.share, "layout": record})
    return {"item_id": item["item_id"], "image_id": item["image_id"], "dataset": item["dataset"],
            "a_true": item["a_true"], "a_alt": item["a_alt"], "model": bundle.family.name,
            "template": bundle.family.template_version, "head_precision": bundle.decoder.head_precision, "rows": rows}


def run_comparative(bundle: Bundle, items: list[dict], images_dir: Path, out_dir: Path, selection: dict) -> dict:
    """Each item once, resumable; prints progress and the two consistency counts only."""
    by_id = {item["item_id"]: item for item in items}
    ids, control = selection["main_ids"], set(selection["control_ids"])
    (out_dir / "items").mkdir(parents=True, exist_ok=True)
    for position, item_id in enumerate(ids, 1):
        target = out_dir / "items" / f"{item_id}.json"
        if not target.exists():
            item = by_id[item_id]
            image = open_image(images_dir, item)
            result = run_item(bundle, item, image, units(control=item_id in control))
            blind = run_item(bundle, item, NEXT.blank_image(image), [BLIND])["rows"][0]
            result["rows"].append({**blind, "selector": "blind"})
            result["subset"] = "control" if item_id in control else "main"
            write_json_atomic(target, result)
        if position % 25 == 0 or position == len(ids):
            print(f"{position}/{len(ids)} items", flush=True)
    found = checks(load_run(out_dir)[0])
    return {"full_q_no_longer_eligible": found["full_q_no_longer_eligible"],
            "blind_now_eligible": found["blind_now_eligible"]}


# --- Freeze ----------------------------------------------------------------------------------------


def main_set(items: list[dict], flags: dict[str, bool], decisions: Path, size: int) -> tuple[list[str], bool]:
    """The first `size` eligible items of the pool, in manifest order, and whether that set is final: it is
    full, or every pool item is screened (PREREG 4.5(d): then a short set is every eligible item)."""
    chosen = []
    for item in screen.pool(items, decisions):
        if len(chosen) == size:
            break
        if item["item_id"] not in flags:
            return chosen, False  # an unscreened pool item may not be skipped
        if flags[item["item_id"]]:
            chosen.append(item["item_id"])
    return chosen, True


def current_state(prereg: Path, manifest: Path, screen_dir: Path, decisions: Path, stage1: Path) -> dict:
    stage1_record = screen.check_screening(prereg, manifest, stage1)
    size, control_size = stage1_record["settings"]["main_size"], stage1_record["settings"]["control_size"]
    items = load_items(manifest)
    chosen, final = main_set(items, load_flags(screen_dir), decisions, size)
    if not final:
        raise SystemExit(f"the main set has {len(chosen)} of {size} items and pool items are unscreened; screen the "
                         "rest of the pool (`python -m kvsel_next count` shows what is left)")
    current, _ = load_decisions(decisions, items)
    return {
        "prereg_sha256": sha256_file(prereg),
        "stage1_record_sha256": sha256_file(stage1),
        "manifest": {Path(manifest).name: sha256_file(manifest)},
        "screen_flags_sha256": sha256_file(Path(screen_dir) / "eligible.jsonl"),
        "main_set_sha256": hashlib.sha256(json.dumps([[i, current[i]["digest"]] for i in chosen]).encode()).hexdigest(),
        "main_ids": chosen,
        "main_size_frozen_at_stage1": size,  # more than len(main_ids) means a shortfall (PREREG 4.5(d))
        "control_ids": chosen[:control_size],
        "code_sha256": screen.code_hash(),
        "settings": settings(),
    }


def freeze(prereg: Path, manifest: Path, screen_dir: Path, decisions: Path, stage1: Path, record: Path) -> dict:
    screen.refuse_unsettled(prereg)
    return screen.write_once(record, "next_comparative", current_state(prereg, manifest, screen_dir, decisions, stage1))


def check(prereg: Path, manifest: Path, screen_dir: Path, decisions: Path, stage1: Path, record: Path,
          prereg_frozen: bool = True) -> dict:
    """`prereg_frozen=False` (the report) lets PREREG.md carry dated amendments written after the run."""
    if not record.exists():
        raise SystemExit(f"no NeXT comparative freeze at {record}: run `python -m kvsel_next freeze` first")
    now = current_state(prereg, manifest, screen_dir, decisions, stage1)
    return screen.refuse_changes(record, now, frozenset() if prereg_frozen else frozenset({"prereg_sha256"}))
