"""Follow-up runs on the frozen main set of a finished confirmatory run.

Parts (implementation plan, section 5, stage P5; proposal, sections 6 and 10):
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
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..confirm import code_hash_all
from ..confirm_analysis import evidence_sets
from ..model import Bundle, pixel_values, prefill
from ..runner import (
    NO_CONTEXT,
    Q,
    contexts_for,
    open_image,
    run_unit,
    write_json_atomic,
)
from ..selectors import pw_attn, top_k, uniform_grid

PARTS = ("budgets", "lt", "random", "restore")
CONDITION = "mis"
K = 144
BUDGETS = (288, 72)
SEEDS = tuple(range(5))
RESTORE_M = (4, 8)
EVIDENCE_RULE = "primary"
NUM_VISUAL = 576


@dataclass(frozen=True)
class FUnit:
    """Like `runner.Unit`, plus the follow-up fields. `runner.run_unit` accepts it unchanged."""

    selector: str  # pw, lt, uniform, random, restore
    k: int
    reserve: bool
    selection: str  # Q, mis or none
    answering: str  # Q or mis
    seed: int | None = None
    m: int | None = None
    arm: str | None = None  # restore: reference, treatment or control


def cells(selector: str, k: int) -> list[FUnit]:
    """A, B_mis, C_mis, D_mis."""
    return [FUnit(selector, k, False, Q, Q), FUnit(selector, k, False, CONDITION, CONDITION),
            FUnit(selector, k, False, Q, CONDITION), FUnit(selector, k, False, CONDITION, Q)]


def part_units(part: str) -> list[FUnit]:
    """Units of a part; restore's units depend on the item and come from `restore_units`."""
    if part == "budgets":
        units = []
        for k in BUDGETS:
            units += cells("pw", k)
            units += [FUnit("uniform", k, False, NO_CONTEXT, answering) for answering in (Q, CONDITION)]
        return units
    if part == "lt":
        return cells("lt", K)
    if part == "random":
        return [FUnit("random", K, False, NO_CONTEXT, answering, seed=seed) for seed in SEEDS for answering in (Q, CONDITION)]
    raise ValueError(f"unknown part {part!r}")


def _generator(*key) -> np.random.Generator:
    """Seeded per item and purpose, so a set never depends on run order."""
    digest = hashlib.sha256("|".join(map(str, key)).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def random_set(item_id: str, seed: int, k: int = K) -> np.ndarray:
    return np.sort(_generator(item_id, "random", seed).choice(NUM_VISUAL, size=k, replace=False))


def lt_early(layer2_last: np.ndarray) -> np.ndarray:
    """LT-early scores: layer-2 attention from the last suffix token, mean over heads; no pooling."""
    return np.asarray(layer2_last, dtype=np.float64)


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


def probe_two(decoder, prefix, suffix_ids: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """One probe pass giving both PW-attn's input (as `engine.probe`, same operations in the same
    order) and LT-early's input (layer 2, last suffix token, mean over heads)."""
    _, _, attentions = decoder.step(suffix_ids, prefix.entries, prefix.logical_start, prefix.length,
                                    output_attentions=True)
    start, stop = prefix.num_nonvisual, prefix.length
    total = None
    for attention in attentions[2:]:
        layer = attention[0, :, :, start:stop].float().mean(dim=0)
        total = layer if total is None else total + layer
    late = (total / (len(attentions) - 2)).float().cpu().numpy()
    layer2 = attentions[2][0, :, -1, start:stop].float().mean(dim=0).cpu().numpy()
    return late, layer2


def kept_for(unit: FUnit, item_id: str, late: dict, layer2: dict) -> np.ndarray:
    if unit.selector == "pw":
        return top_k(pw_attn(late[unit.selection]), unit.k)
    if unit.selector == "lt":
        return top_k(lt_early(layer2[unit.selection]), unit.k)
    if unit.selector == "uniform":
        return uniform_grid(unit.k)
    if unit.selector == "random":
        return random_set(item_id, unit.seed, unit.k)
    raise ValueError(f"unknown selector {unit.selector!r}")


def confirm_selection(confirm_run: Path, record: Path, items: list[dict]) -> dict:
    """The main set exactly as the confirmatory run analysed it, in manifest order."""
    frozen = json.loads(Path(record).read_text())
    subsets = {}
    for path in (Path(confirm_run) / "items").glob("*.json"):
        data = json.loads(path.read_text())
        subsets[data["item_id"]] = data.get("subset")
    if None in subsets.values():
        raise SystemExit(f"{confirm_run} has item files without a subset; it isn't a confirmatory run")
    if len(subsets) != frozen["main_items"]:
        raise SystemExit(f"{confirm_run} has {len(subsets)} item files, but the frozen main set has "
                         f"{frozen['main_items']}; move files outside the main set out of the folder first")
    order = [item["item_id"] for item in items if item["item_id"] in subsets]
    if len(order) != len(subsets):
        raise SystemExit("the confirmatory run has items that aren't in this manifest")
    control = [i for i in order if subsets[i] == "control"]
    if len(control) != frozen["control_items"]:
        raise SystemExit(f"{len(control)} control items, but the frozen record has {frozen['control_items']}")
    return {"main": order, "control": control, "code_unchanged": frozen["code_sha256"] == code_hash_all()}


def run_followup(bundle: Bundle, items: list[dict], manifest: Path, images_dir: Path, out_dir: Path,
                 selection: dict, parts: tuple[str, ...] = PARTS) -> dict:
    unknown = set(parts) - set(PARTS)
    if unknown:
        raise SystemExit(f"unknown parts: {sorted(unknown)}")
    by_id = {item["item_id"]: item for item in items}
    missing = [by_id[i]["image_file"] for i in selection["main"] if not (images_dir / by_id[i]["image_file"]).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} images are missing from {images_dir} (first: {missing[0]})")
    control = set(selection["control"])
    evidence = {i: rules[EVIDENCE_RULE] for i, rules in evidence_sets(manifest, control).items() if rules[EVIDENCE_RULE]}
    for part in parts:
        (out_dir / part / "items").mkdir(parents=True, exist_ok=True)
    qualifying = {m: 0 for m in RESTORE_M}
    for position, item_id in enumerate(selection["main"], 1):
        todo = [p for p in parts if not (out_dir / p / "items" / f"{item_id}.json").exists()
                and (p != "restore" or item_id in evidence)]
        if todo:
            item = by_id[item_id]
            tokens = contexts_for(item, bundle.tokenizer, bundle.family)
            prefix = prefill(bundle, tokens[Q].prefix_ids, pixel_values(bundle, open_image(images_dir, item)))
            late, layer2 = {}, {}
            for context in (Q, CONDITION):
                late[context], layer2[context] = probe_two(bundle.decoder, prefix, torch.tensor([tokens[context].suffix_ids]))
            kept_144 = {context: top_k(pw_attn(late[context]), K).tolist() for context in (Q, CONDITION)}
            for part in todo:
                extra = {}
                if part == "restore":
                    rows, extra = _restore_rows(bundle, item, prefix, tokens, late, evidence[item_id])
                else:
                    rows = [run_unit(bundle, item, unit, prefix, kept_for(unit, item_id, late, layer2), tokens)
                            for unit in part_units(part)]
                result = {"item_id": item_id, "image_id": item["image_id"], "dataset": item["dataset"],
                          "subset": "control" if item_id in control else "main", "part": part,
                          "kept_144": kept_144, **extra, "rows": rows}
                write_json_atomic(out_dir / part / "items" / f"{item_id}.json", result)
        if "restore" in parts and item_id in evidence:
            done = json.loads((out_dir / "restore" / "items" / f"{item_id}.json").read_text())
            for m in RESTORE_M:
                qualifying[m] += done["qualifying"][str(m)]
        if position % 25 == 0 or position == len(selection["main"]):
            print(f"{position}/{len(selection['main'])} items", flush=True)
    counts = {part: len(list((out_dir / part / "items").glob("*.json"))) for part in parts}
    summary = {"items": len(selection["main"]), "files_per_part": counts}
    if "restore" in parts:
        summary["grounding_items"] = len(evidence)
        summary["restore_qualifying"] = {str(m): n for m, n in qualifying.items()}
    return summary


def _restore_rows(bundle, item, prefix, tokens, late, evidence: set[int]) -> tuple[list[dict], dict]:
    item_id = item["item_id"]
    scores_q, scores_mis = pw_attn(late[Q]), pw_attn(late[CONDITION])
    reference = FUnit("restore", K, False, CONDITION, CONDITION, arm="reference")
    rows = [run_unit(bundle, item, reference, prefix, top_k(scores_mis, K), tokens)]
    qualifying = {}
    for m in RESTORE_M:
        sets = restoration_sets(item_id, scores_q, scores_mis, evidence, m)
        qualifying[str(m)] = sets is not None
        if sets is None:
            continue
        unit = FUnit("restore", K, False, CONDITION, CONDITION, m=m, arm="treatment")
        rows.append(run_unit(bundle, item, unit, prefix, sets["treatment"], tokens))
        for seed, kept in zip(SEEDS, sets["controls"], strict=True):
            unit = FUnit("restore", K, False, CONDITION, CONDITION, seed=seed, m=m, arm="control")
            rows.append(run_unit(bundle, item, unit, prefix, kept, tokens))
    return rows, {"qualifying": qualifying}
