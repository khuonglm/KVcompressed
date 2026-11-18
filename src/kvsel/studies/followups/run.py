"""Follow-up runs on the frozen main set of a finished confirmatory run (PREREG 2.4: outside the confirmatory study).

Parts (PREREG 2.4's list; implementation plan, section 5, stage P5; proposal, sections 6 and 10):
- budgets: PW-attn cells A, B_mis, C_mis, D_mis and the uniform grid (Q, QH_mis) at K = 288 and 72.
- lt: LT-early (layer 2, last suffix token, mean over heads, no pooling) cells A-D for mis at K = 144.
- random: random sets, seeds 0-4, answering Q and QH_mis at K = 144 (text-agnostic, so S_Q = S_mis).
- restore: oracle restoration on grounding items (control subset with an evidence box), m in {4, 8}.

Each part writes `<out>/<part>/items/<item_id>.json`, atomically; an item is prefilled once for all its
missing parts, so a stopped run resumes. Prints progress only.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ...engine import probe
from ...model import Bundle
from ...protocol import CODE_AMENDMENTS, Refusal, code_accepted, code_identity
from ...runner import NO_CONTEXT, ItemFile, Q, Unit, item_loop, open_image, require_images, run_unit
from ...runs import MainSet
from ...selectors import Budget, kept_set, pw_attn, top_k
from ..confirmatory.report import evidence_sets

PARTS = ("budgets", "lt", "random", "restore")
CONDITION = "mis"
K = 144
BUDGETS = (288, 72)
SEEDS = tuple(range(5))
RESTORE_M = (4, 8)
EVIDENCE_RULE = "primary"


def cells(selector: str, k: int) -> list[Unit]:
    """A, B_mis, C_mis, D_mis."""
    return [Unit(selector, k, False, Q, Q, followup=True), Unit(selector, k, False, CONDITION, CONDITION, followup=True),
            Unit(selector, k, False, Q, CONDITION, followup=True), Unit(selector, k, False, CONDITION, Q, followup=True)]


def part_units(part: str) -> list[Unit]:
    """Units of a part; restore's units depend on the item and come from `restore_units`."""
    if part == "budgets":
        units = []
        for k in BUDGETS:
            units += cells("pw", k)
            units += [Unit("uniform", k, False, NO_CONTEXT, answering, followup=True) for answering in (Q, CONDITION)]
        return units
    if part == "lt":
        return cells("lt", K)
    if part == "random":
        return [Unit("random", K, False, NO_CONTEXT, answering, seed=seed, followup=True)
                for seed in SEEDS for answering in (Q, CONDITION)]
    raise ValueError(f"unknown part {part!r}")


def _generator(*key) -> np.random.Generator:
    """Seeded per item and purpose, so a set never depends on run order."""
    digest = hashlib.sha256("|".join(map(str, key)).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def restoration_sets(item_id: str, scores_q: np.ndarray, scores_mis: np.ndarray, evidence: set[int], m: int,
                     k: int = K) -> dict | None:
    """Oracle restoration (proposal, section 6), or None if the item doesn't qualify for this m.

    Pushed out = entries S_Q keeps and S_mis doesn't. Both arms drop the same m lowest-scoring
    non-evidence entries of S_mis (by the QH_mis probe; ties drop the higher index first, the mirror of
    `top_k`). Treatment adds the m pushed-out evidence entries with the highest S_Q scores. Control
    adds m pushed-out non-evidence entries at random (seeds 0-4), so both arms restore entries the
    text displaced and differ only in whether they are annotated evidence.
    """
    s_q, s_mis = set(top_k(scores_q, k).tolist()), set(top_k(scores_mis, k).tolist())
    pushed = s_q - s_mis
    evidence_out = sorted(pushed & evidence, key=lambda i: (-scores_q[i], i))
    other_out = sorted(pushed - evidence)
    removable = sorted(s_mis - evidence, key=lambda i: (scores_mis[i], -i))
    if len(evidence_out) < m or len(other_out) < m or len(removable) < m:
        return None
    base = s_mis - set(removable[:m])
    treatment = np.array(sorted(base | set(evidence_out[:m])))
    controls = []
    for seed in SEEDS:
        added = _generator(item_id, "restore", m, seed).choice(other_out, size=m, replace=False)
        controls.append(np.array(sorted(base | {int(i) for i in added})))
    return {"treatment": treatment, "controls": controls, "removed": sorted(removable[:m])}


def followup_kept_set(unit: Unit, item_id: str, layout, late: dict, layer2: dict) -> np.ndarray:
    """The unit's kept set; a random set is seeded by the item and the unit's seed."""
    rng = _generator(item_id, "random", unit.seed) if unit.selector == "random" else None
    return kept_set(layout, unit.selector, Budget(k=unit.k), attention=late.get(unit.selection),
                    layer2=layer2.get(unit.selection), rng=rng)


def confirm_selection(confirm_run: Path, record: Path, items: list[dict]) -> tuple[MainSet, bool]:
    """The main set exactly as the confirmatory run analysed it, in manifest order, and whether the code is the
    record's (through the amendment log): a follow-up may run on other code, which the summary says (PREREG 2.4)."""
    frozen = json.loads(Path(record).read_text())
    subsets = {}
    for path in (Path(confirm_run) / "items").glob("*.json"):
        data = json.loads(path.read_text())
        subsets[data["item_id"]] = data.get("subset")
    if None in subsets.values():
        raise Refusal(f"{confirm_run} has item files without a subset; it isn't a confirmatory run")
    if len(subsets) != frozen["main_items"]:
        raise Refusal(f"{confirm_run} has {len(subsets)} item files, but the frozen main set has "
                         f"{frozen['main_items']}; move files outside the main set out of the folder first")
    order = [item["item_id"] for item in items if item["item_id"] in subsets]
    if len(order) != len(subsets):
        raise Refusal("the confirmatory run has items that aren't in this manifest")
    control = [i for i in order if subsets[i] == "control"]
    if len(control) != frozen["control_items"]:
        raise Refusal(f"{len(control)} control items, but the frozen record has {frozen['control_items']}")
    accepted = code_accepted(frozen["code_sha256"], code_identity(), Path(record).parent / CODE_AMENDMENTS)
    return MainSet.of(order, control), accepted


def run_followup(bundle: Bundle, items: list[dict], manifest: Path, images_dir: Path, out_dir: Path,
                 main_set: MainSet, record_sha256: str, parts: tuple[str, ...] = PARTS) -> dict:
    """``record_sha256``: the confirmatory stage-3 record's, whose main set this runs on."""
    unknown = set(parts) - set(PARTS)
    if unknown:
        raise Refusal(f"unknown parts: {sorted(unknown)}")
    by_id = {item["item_id"]: item for item in items}
    require_images([by_id[i] for i in main_set.ids], images_dir)
    control = main_set.control
    evidence = {i: rules[EVIDENCE_RULE] for i, rules in evidence_sets(manifest, control).items() if rules[EVIDENCE_RULE]}
    for part in parts:
        (out_dir / part / "items").mkdir(parents=True, exist_ok=True)

    def item_parts(item_id):
        return [p for p in parts if p != "restore" or item_id in evidence]

    def files(item):
        item_id = item["item_id"]
        return [ItemFile(out_dir / part / "items" / f"{item_id}.json", main_set.subset(item_id),
                         None if part == "restore" else tuple(u.record() for u in part_units(part)))
                for part in item_parts(item_id)]

    def produce(item, todo):
        item_id = item["item_id"]
        tokens, inputs = bundle.family.contexts(bundle, item, open_image(images_dir, item))
        prefix, layout = bundle.family.prefill(bundle, tokens[Q].prefix_ids, inputs)
        late, layer2 = {}, {}
        for context in (Q, CONDITION):
            late[context], layer2[context] = probe(bundle.decoder, prefix, torch.tensor([tokens[context].suffix_ids]),
                                                   layer2_last=True)
        kept_144 = {context: kept_set(layout, "pw", Budget(k=K), attention=late[context]).tolist()
                    for context in (Q, CONDITION)}
        results = []
        for target in todo:
            part, extra = target.path.parents[1].name, {}
            if part == "restore":
                rows, extra = _restore_rows(bundle, item, prefix, layout, tokens, late, evidence[item_id])
            else:
                rows = [run_unit(bundle, item, unit, prefix, followup_kept_set(unit, item_id, layout, late, layer2), tokens)
                        for unit in part_units(part)]
            results.append(({"item_id": item_id, "image_id": item["image_id"], "dataset": item["dataset"],
                             "subset": main_set.subset(item_id), "part": part, "kept_144": kept_144, **extra, "rows": rows}, None))
        return results

    qualifying = {m: 0 for m in RESTORE_M}
    chosen = [by_id[i] for i in main_set.ids]
    for position, item, results in item_loop(chosen, record_sha256, files, produce):
        done = dict(zip(item_parts(item["item_id"]), results, strict=True))
        if "restore" in done:
            for m in RESTORE_M:
                qualifying[m] += done["restore"]["qualifying"][str(m)]
        if position % 25 == 0 or position == len(chosen):
            print(f"{position}/{len(chosen)} items", flush=True)
    counts = {part: len(list((out_dir / part / "items").glob("*.json"))) for part in parts}
    summary = {"items": len(main_set.ids), "files_per_part": counts}
    if "restore" in parts:
        summary["grounding_items"] = len(evidence)
        summary["restore_qualifying"] = {str(m): n for m, n in qualifying.items()}
    return summary


def _restore_rows(bundle, item, prefix, layout, tokens, late, evidence: set[int]) -> tuple[list[dict], dict]:
    item_id = item["item_id"]
    scores_q, scores_mis = pw_attn(layout, late[Q]), pw_attn(layout, late[CONDITION])
    reference = Unit("restore", K, False, CONDITION, CONDITION, arm="reference", followup=True)
    rows = [run_unit(bundle, item, reference, prefix, kept_set(layout, "pw", Budget(k=K), attention=late[CONDITION]), tokens)]
    qualifying = {}
    for m in RESTORE_M:
        sets = restoration_sets(item_id, scores_q, scores_mis, evidence, m)
        qualifying[str(m)] = sets is not None
        if sets is None:
            continue
        unit = Unit("restore", K, False, CONDITION, CONDITION, m=m, arm="treatment", followup=True)
        rows.append(run_unit(bundle, item, unit, prefix, sets["treatment"], tokens))
        for seed, kept in zip(SEEDS, sets["controls"], strict=True):
            unit = Unit("restore", K, False, CONDITION, CONDITION, seed=seed, m=m, arm="control", followup=True)
            rows.append(run_unit(bundle, item, unit, prefix, kept, tokens))
    return rows, {"qualifying": qualifying}
