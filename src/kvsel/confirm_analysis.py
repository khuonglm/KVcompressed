"""The pre-registered confirmatory analysis (PREREG.md, section 2.3), run once per dataset's run.

`dataset_results` computes every pre-registered quantity from one run directory; `confirm_report`
applies the Holm families across datasets and writes `results.json` and `report.md`. A run whose item
files carry no `subset` field (a development pilot) is analysed as a rehearsal: every item counts as a
control item, and the report says it supports no claim.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import (
    DELTA,
    RESAMPLES,
    cluster_bootstrap,
    column_name,
    estimands,
    holm,
    margin,
    read_interval,
)
from .geometry import MAPPING_RULES, box_to_square, evidence_patches
from .runner import eligible

SURFACE = "capitalized"
PRIMARY = "pw"
SECONDARY = "qspan"
DATASETS = ("gqa", "textvqa")
MARGIN_VARIANTS = ("primary", "logsum", "content_only", "length_normalised")
FAMILIES = {
    "H1": ["S_H^(mis)"],
    "H2": ["T_match", "T_irr"],
    "H3_H4": ["E_harm^(mis)", "S_Q^(mis)"],
}


def margin_variant(scores: dict, variant: str) -> float:
    """The primary margin and its pre-registered sensitivity forms. Each score list ends with </s>."""
    if variant == "primary":
        return margin(scores, SURFACE)
    if variant == "logsum":
        return margin(scores, "logsum")
    true, alt = scores["true"][SURFACE], scores["alt"][SURFACE]
    if variant == "content_only":
        return sum(true[:-1]) - sum(alt[:-1])
    if variant == "length_normalised":
        return float(np.mean(true) - np.mean(alt))
    raise ValueError(f"unknown margin variant {variant!r}")


def load_run(run_dir: Path) -> tuple[list[dict], dict[str, dict]]:
    rows, items = [], {}
    paths = sorted((Path(run_dir) / "items").glob("*.json"))
    if not paths:
        raise SystemExit(f"no item files under {run_dir}")
    for path in paths:
        item = json.loads(path.read_text())
        items[item["item_id"]] = {"image_id": item["image_id"], "subset": item.get("subset")}
        rows.extend({"item_id": item["item_id"], "image_id": item["image_id"], **row} for row in item["rows"])
    return rows, items


def wide_margins(rows: list[dict], variant: str) -> tuple[pd.DataFrame, pd.Series]:
    frame = pd.DataFrame([{"item_id": r["item_id"], "column": column_name(r),
                           "value": margin_variant(r["scores"], variant)} for r in rows])
    wide = frame.pivot(index="item_id", columns="column", values="value")
    images = pd.Series({r["item_id"]: r["image_id"] for r in rows}).reindex(wide.index)
    return wide, images


def has_cells(wide: pd.DataFrame, selector: str, condition: str) -> bool:
    needed = [f"{selector}|A", f"{selector}|B_{condition}", f"{selector}|C_{condition}", f"{selector}|D_{condition}",
              f"full|{condition}", "full|Q"]
    return all(column in wide for column in needed)


def kept_sets(rows: list[dict], selector: str = PRIMARY) -> dict[tuple[str, str], set[int]]:
    """Kept visual indices per (item, selection context), for the unreserved selector at its budget."""
    return {(r["item_id"], r["selection"]): set(r["kept"]) for r in rows
            if r["selector"] == selector and not r["reserve"] and r["kept"] is not None}


def jaccard(a: set[int], b: set[int]) -> float:
    return len(a & b) / len(a | b) if a | b else 1.0


def clustered_ols(y: np.ndarray, x: np.ndarray, clusters: np.ndarray) -> dict:
    """Slope of y on x with an intercept; CR1 cluster-robust SE; normal-approximation interval and p."""
    design = np.column_stack([np.ones_like(x), x])
    bread = np.linalg.inv(design.T @ design)
    beta = bread @ design.T @ y
    residuals = y - design @ beta
    meat = np.zeros((2, 2))
    groups = pd.Series(range(len(y))).groupby(clusters).indices
    for index in groups.values():
        score = design[index].T @ residuals[index]
        meat += np.outer(score, score)
    n, g = len(y), len(groups)
    covariance = bread @ meat @ bread * (g / (g - 1)) * ((n - 1) / (n - 2))
    se = float(math.sqrt(covariance[1, 1]))
    z = beta[1] / se if se > 0 else float("inf")
    return {"slope": float(beta[1]), "intercept": float(beta[0]), "se": se,
            "ci95": (float(beta[1] - 1.96 * se), float(beta[1] + 1.96 * se)),
            "p_two_sided": float(math.erfc(abs(z) / math.sqrt(2))), "items": n, "images": g}


def evidence_sets(manifest: Path, item_ids: set[str]) -> dict[str, dict[str, set[int]]]:
    """Evidence patches per item under each mapping rule; items without a box are left out."""
    sets = {}
    for line in Path(manifest).read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if item["item_id"] not in item_ids or not item.get("evidence_box"):
            continue
        boxes = item["evidence_box"] if isinstance(item["evidence_box"], list) else [item["evidence_box"]]
        size = item["image_size"]
        squares = [box_to_square(box, size["width"], size["height"]) for box in boxes]
        sets[item["item_id"]] = {rule: {p for s in squares for p in evidence_patches(s, rule)} for rule in MAPPING_RULES}
    return sets


def _boot(values: pd.Series, images: pd.Series, resamples: int) -> dict:
    values = values.dropna()
    if values.empty:
        return {"estimate": None, "items": 0}
    result = cluster_bootstrap(values, images.reindex(values.index), resamples)
    result["reading"] = read_interval(*result["ci95"])
    return result


def dataset_results(run_dir: Path, manifest: Path, resamples: int = RESAMPLES, expected_items: int | None = None) -> dict:
    rows, items = load_run(run_dir)
    rehearsal = any(meta["subset"] is None for meta in items.values())
    control = {i for i, meta in items.items() if rehearsal or meta["subset"] == "control"}
    wide, images = wide_margins(rows, "primary")
    out: dict = {
        "run": str(run_dir), "rehearsal": rehearsal, "surface": SURFACE,
        "items": len(items), "images": int(images.nunique()), "control_items": len(control),
    }
    if expected_items is not None and len(items) != expected_items:
        fix = ("rerun `kvsel confirm` to finish it" if len(items) < expected_items
               else "move item files outside the frozen main set out of the run folder")
        raise SystemExit(f"{run_dir} has {len(items)} items, but the frozen main set has {expected_items}; {fix}")

    # Checks: Full_Q still eligible, Blind_Q never eligible, truncation.
    full_q = [r for r in rows if r["selector"] == "full" and r["cell"] == "Q"]
    blind = [r for r in rows if r["selector"] == "blind"]
    out["checks"] = {
        "full_q_no_longer_eligible": sum(not eligible(r, SURFACE) for r in full_q),
        "blind_rows": len(blind),
        "blind_now_eligible": sum(eligible(r, SURFACE) for r in blind),
        "blind_mean_margin": float(np.mean([margin(r["scores"], SURFACE) for r in blind])) if blind else None,
        "truncated_share": float(np.mean([r["truncated"] for r in rows])),
    }

    # Pre-registered estimands. Control-only quantities are NaN on main-only items, so they drop out.
    tests: dict[str, dict] = {}
    exploratory: dict[str, dict] = {}
    per_condition = {c: estimands(wide, PRIMARY, c) for c in ("mis", "match", "irr") if has_cells(wide, PRIMARY, c)}
    if "mis" not in per_condition:
        raise SystemExit(f"{run_dir} lacks the misleading-condition cells for {PRIMARY}; is it a confirmatory run?")
    mis = per_condition["mis"]
    tests["S_H^(mis)"] = _boot(mis["S_H"], images, resamples)
    tests["E_harm^(mis)"] = _boot(mis["E_harm"], images, resamples)
    tests["S_Q^(mis)"] = _boot(mis["S_Q"], images, resamples)
    for other in ("match", "irr"):
        if other in per_condition:
            control_index = mis["S_H"].index.isin(control)
            difference = (mis["S_H"] - per_condition[other]["S_H"]).where(control_index)
            tests[f"T_{other}"] = _boot(difference, images, resamples)
            exploratory[f"S_H^({other})"] = _boot(per_condition[other]["S_H"].where(control_index), images, resamples)
    exploratory["I^(mis)"] = _boot(mis["I"], images, resamples)
    if has_cells(wide, SECONDARY, "mis"):
        exploratory["S_H^(mis), question-span"] = _boot(estimands(wide, SECONDARY, "mis")["S_H"], images, resamples)
    out["tests"], out["exploratory"] = tests, exploratory

    primary = tests["S_H^(mis)"]
    low, high = primary["ci90"]
    out["equivalence"] = {"delta": DELTA, "ci90": primary["ci90"], "equivalent": bool(-DELTA < low and high < DELTA)}

    out["sensitivity"] = {}
    for variant in MARGIN_VARIANTS[1:]:
        other_wide, other_images = wide_margins(rows, variant)
        out["sensitivity"][variant] = _boot(estimands(other_wide, PRIMARY, "mis")["S_H"], other_images, resamples)

    # Behaviour: outcome rates per run, and paired B_mis vs C_mis differences.
    outcomes = pd.DataFrame([{"item_id": r["item_id"], "column": column_name(r), "outcome": r["outcome"]} for r in rows])
    rates = outcomes.groupby("column")["outcome"].value_counts(normalize=True).unstack(fill_value=0.0)
    out["behaviour_rates"] = {column: {k: float(rates.loc[column].get(k, 0.0)) for k in ("correct", "alt", "other")}
                              for column in rates.index}
    pivot = outcomes.pivot(index="item_id", columns="column", values="outcome")
    out["behaviour_paired"] = {}
    for result in ("correct", "alt"):
        difference = (pivot[f"{PRIMARY}|C_mis"] == result).astype(float) - (pivot[f"{PRIMARY}|B_mis"] == result).astype(float)
        out["behaviour_paired"][f"{result}: C_mis - B_mis"] = _boot(difference, images, resamples)

    # Overlap and H5.
    kept = kept_sets(rows)
    overlap = {}
    for condition in ("mis", "match", "irr"):
        values = [jaccard(s, kept[(item, "Q")]) for (item, context), s in kept.items()
                  if context == condition and (item, "Q") in kept]
        if values:
            overlap[condition] = {"mean_jaccard": float(np.mean(values)), "share_changed": float(np.mean(np.array(values) < 1))}
    out["overlap"] = overlap
    distance = pd.Series({item: 1 - jaccard(kept[(item, "mis")], kept[(item, "Q")])
                          for item in wide.index if (item, "mis") in kept and (item, "Q") in kept})
    frame = pd.DataFrame({"y": mis["S_H"], "x": distance, "image": images}).dropna()
    out["H5"] = clustered_ols(frame["y"].to_numpy(), frame["x"].to_numpy(), frame["image"].to_numpy()) \
        if len(frame) > 2 and frame["x"].std() > 0 else {"slope": None, "items": len(frame)}

    # H6: annotated-region retention on the grounding subset (control items with an evidence box).
    evidence = evidence_sets(manifest, control)
    out["H6"] = {}
    for rule in MAPPING_RULES:
        values = {}
        for item, rule_sets in evidence.items():
            region = rule_sets[rule]
            if region and (item, "Q") in kept and (item, "mis") in kept:
                values[item] = (len(kept[(item, "Q")] & region) - len(kept[(item, "mis")] & region)) / len(region)
        out["H6"][rule] = _boot(pd.Series(values, dtype=float), images, resamples)
    out["grounding_items"] = len(evidence)

    # Mitigation (EQ1): each variant's margin gain over B_mis, and costs on neutral and matching answering.
    base = wide[f"{PRIMARY}|B_mis"]
    pairs = {
        "isolation (C_mis)": ("C_mis", "B_mis", ""),
        "reserve (B_mis + r)": ("B_mis", "B_mis", "+r"),
        "both (C_mis + r)": ("C_mis", "B_mis", "+r"),
        "reserve cost, neutral (A + r − A)": ("A", "A", "+r"),
        "reserve cost, matching (B_match + r − B_match)": ("B_match", "B_match", "+r"),
        "isolation cost, matching (C_match − B_match)": ("C_match", "B_match", ""),
    }
    out["mitigation"] = {}
    for name, (treated, baseline, reserve) in pairs.items():
        treated_column, baseline_column = f"{PRIMARY}{reserve}|{treated}", f"{PRIMARY}|{baseline}"
        if treated_column in wide and baseline_column in wide:
            out["mitigation"][name] = _boot(wide[treated_column] - wide[baseline_column], images, resamples)
    if "uniform|mis" in wide:
        out["mitigation"]["uniform (mis)"] = _boot(wide["uniform|mis"] - base, images, resamples)
    out["mean_margins"] = {column: float(wide[column].mean()) for column in sorted(wide.columns)}
    return out


def apply_families(results: dict[str, dict]) -> dict:
    """Holm within each family across datasets. Final only when every pre-registered dataset is present."""
    complete = all(dataset in results for dataset in DATASETS)
    decisions = {}
    for family, names in FAMILIES.items():
        p_values = {f"{name} [{dataset}]": results[dataset]["tests"][name]["p_two_sided"]
                    for dataset in results for name in names if results[dataset]["tests"].get(name, {}).get("estimate") is not None}
        decisions[family] = {"complete": complete, "p_values": p_values,
                             "reject": holm(p_values) if complete else None}
    return decisions


def _fmt(result: dict, nats: bool = True, digits: int = 3) -> str:
    """One table row. The δ-based reading applies to margins in nats only."""
    if result.get("estimate") is None:
        return "— | — | — | — | — |"
    low, high = result["ci95"]
    reading = result.get("reading", "") if nats else ""
    return (f"{result['estimate']:+.{digits}f} | [{low:+.{digits}f}, {high:+.{digits}f}] | {result['p_two_sided']:.4f} | "
            f"{reading} | {result['items']} / {result['images']} |")


def _table(title: str, block: dict, unit: str, nats: bool) -> list[str]:
    reading = "Reading" if nats else "—"
    return [f"### {title}", "", f"| Estimand | Mean ({unit}) | 95% interval | Two-sided p | {reading} | Items / images |",
            "| --- | ---: | --- | ---: | --- | ---: |"] + [f"| {name} | {_fmt(result, nats)}" for name, result in block.items()] + [""]


def render(results: dict[str, dict], families: dict) -> str:
    rehearsal = any(r["rehearsal"] for r in results.values())
    lines = ["# Confirmatory report" if not rehearsal else "# Confirmatory analysis: REHEARSAL", ""]
    if rehearsal:
        lines += ["**Rehearsal on development data. Nothing here supports a claim.**", ""]
    lines += [f"Surface rule `{SURFACE}`; primary selector PW-attn, K = 144; image-clustered bootstrap, two-sided tests.", ""]
    lines += ["## Holm families", "", "| Family | Test | Two-sided p | Holm decision |", "| --- | --- | ---: | --- |"]
    for family, entry in families.items():
        for test, p in entry["p_values"].items():
            decision = ("reject H0" if entry["reject"][test] else "not rejected") if entry["complete"] else "pending (family incomplete)"
            lines.append(f"| {family} | {test} | {p:.4f} | {decision} |")
    if not all(f["complete"] for f in families.values()):
        lines += ["", "Holm decisions become final when every pre-registered dataset has run (PREREG.md 2.3)."]
    lines.append("")
    for dataset, r in results.items():
        lines += [f"## {dataset}", "",
                  (f"{r['items']} items on {r['images']} images; {r['control_items']} control items; "
                  f"{r['grounding_items']} grounding items."), ""]
        c = r["checks"]
        lines += [f"Checks: Full_Q no longer eligible on {c['full_q_no_longer_eligible']} items; Blind_Q eligible on "
                  f"{c['blind_now_eligible']} of {c['blind_rows']} (should be 0); mean Blind_Q margin "
                  + (f"{c['blind_mean_margin']:+.3f}" if c["blind_mean_margin"] is not None else "n/a")
                  + f"; truncated answers {c['truncated_share']:.2%}.", ""]
        lines += _table("Pre-registered tests", r["tests"], "nats", True)
        e = r["equivalence"]
        lines += [f"Equivalence (δ = {e['delta']}): 90% interval [{e['ci90'][0]:+.3f}, {e['ci90'][1]:+.3f}], "
                  + ("inside ±δ: practically small." if e["equivalent"] else "not inside ±δ."), ""]
        lines += _table("Exploratory (intervals only)", r["exploratory"], "nats", True)
        lines += _table("Sensitivity of S_H^(mis) to the margin form", r["sensitivity"], "nats", True)
        lines += _table("Behaviour, paired difference in the share of answers", r["behaviour_paired"], "share", False)
        lines += _table("H6: ERR(S_Q) − ERR(S_mis), by evidence mapping rule", r["H6"], "share of region", False)
        lines += _table("Mitigation (EQ1): margin gain over B_mis unless noted", r["mitigation"], "nats", True)
        h5 = r["H5"]
        if h5.get("slope") is not None:
            lines += ["### H5: S_H^(mis) on 1 − Jaccard(S_Q, S_mis)", "",
                      (f"Slope {h5['slope']:+.3f} (cluster-robust SE {h5['se']:.3f}), 95% interval "
                      f"[{h5['ci95'][0]:+.3f}, {h5['ci95'][1]:+.3f}], two-sided p {h5['p_two_sided']:.4f}; "
                      f"{h5['items']} items, {h5['images']} images."), ""]
        lines += ["### Kept-set overlap with S_Q (PW-attn)", "", "| Condition | Mean Jaccard | Share changed |", "| --- | ---: | ---: |"]
        lines += [f"| {k} | {v['mean_jaccard']:.3f} | {v['share_changed']:.3f} |" for k, v in r["overlap"].items()] + [""]
        lines += ["### Behaviour rates", "", "| Run | Correct | Alternative | Other |", "| --- | ---: | ---: | ---: |"]
        lines += [f"| {k} | {v['correct']:.3f} | {v['alt']:.3f} | {v['other']:.3f} |" for k, v in r["behaviour_rates"].items()] + [""]
    return "\n".join(lines)


def _plain(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple | set):
        return list(value)
    raise TypeError(f"not serialisable: {type(value)}")


CODE_AMENDMENTS = "code_amendments.json"


def code_accepted(frozen: str, current: str, amendments: Path) -> bool:
    """The frozen code, or code reached from it through dated, outcome-blind amendments.

    Each entry of the amendments file, in order, is {"from", "to", "date", "reason", "evidence"}: code
    `to` replaced code `from` and reproduced the frozen results exactly (PREREG, section 1.9).
    """
    reached = {frozen}
    for entry in json.loads(amendments.read_text()) if amendments.exists() else []:
        if entry["from"] in reached:
            reached.add(entry["to"])
    return current in reached


def confirm_report(runs: dict[str, tuple[Path, Path]], out_dir: Path, resamples: int = RESAMPLES,
                   records: dict[str, Path] | None = None) -> dict:
    """runs: dataset -> (run directory, manifest). records: dataset -> stage-3 record, to check completeness."""
    unknown = set(runs) - set(DATASETS)
    if unknown:
        raise SystemExit(f"unknown datasets {sorted(unknown)}; expected {DATASETS}")
    results = {}
    for dataset, (run_dir, manifest) in runs.items():
        expected = None
        if records and dataset in records:
            from .confirm import code_hash_all

            record = json.loads(Path(records[dataset]).read_text())
            if not code_accepted(record["code_sha256"], code_hash_all(), Path(records[dataset]).parent / CODE_AMENDMENTS):
                raise SystemExit(f"the code changed since {records[dataset]} was frozen; analyse with the frozen code, "
                                 "or document the fix as a dated amendment and rerun every affected condition")
            expected = record["main_items"]
        results[dataset] = dataset_results(run_dir, manifest, resamples, expected)
    families = apply_families(results)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps({"datasets": results, "families": families}, indent=2, default=_plain))
    (out_dir / "report.md").write_text(render(results, families))
    return {"datasets": results, "families": families}
