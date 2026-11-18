"""Confirmatory study, the report (PREREG 2.3), run once per dataset's run.

`dataset_results` computes every pre-registered quantity from one run directory; `confirm_report`
applies the Holm families across datasets and writes `results.json` and `report.md`. A run whose item
files carry no `subset` field (a development pilot) is analysed as a rehearsal: every item counts as a
control item, and the report says it supports no claim.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ...analysis.blocks import rates_table, table, write_report
from ...analysis.statistics import (
    DELTA,
    RESAMPLES,
    SURFACE,
    bootstrap,
    clustered_ols,
    estimands,
    holm,
)
from ...analysis.tables import (
    PRIMARY,
    behaviour_rates,
    checks,
    has_cells,
    kept_sets,
    load_run,
    overlap,
    paired_behaviour,
    wide_margins,
)
from ...models.llava15 import MAPPING_RULES, box_to_square, evidence_patches
from ...protocol import Refusal, check_code
from ...selectors import jaccard

SECONDARY = "qspan"
DATASETS = ("gqa", "textvqa")
MARGIN_VARIANTS = ("primary", "logsum", "content_only", "length_normalised")
FAMILIES = {  # PREREG 2.3, multiplicity
    "H1": ["S_H^(mis)"],
    "H2": ["T_match", "T_irr"],
    "H3_H4": ["E_harm^(mis)", "S_Q^(mis)"],
}


def evidence_sets(manifest: Path, item_ids: set[str]) -> dict[str, dict[str, set[int]]]:
    """Evidence patches per item under each mapping rule (PREREG 2.3, H6); items without a box are left out."""
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


def dataset_results(run_dir: Path, manifest: Path, resamples: int = RESAMPLES, expected_items: int | None = None) -> dict:
    """Every pre-registered quantity of PREREG 2.3 from one run directory."""
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
        raise Refusal(f"{run_dir} has {len(items)} items, but the frozen main set has {expected_items}; {fix}")

    out["checks"] = checks(rows)

    # Pre-registered estimands. Control-only quantities are NaN on main-only items, so they drop out.
    tests: dict[str, dict] = {}
    exploratory: dict[str, dict] = {}
    per_condition = {c: estimands(wide, PRIMARY, c) for c in ("mis", "match", "irr") if has_cells(wide, PRIMARY, c)}
    if "mis" not in per_condition:
        raise Refusal(f"{run_dir} lacks the misleading-condition cells for {PRIMARY}; is it a confirmatory run?")
    mis = per_condition["mis"]
    tests["S_H^(mis)"] = bootstrap(mis["S_H"], images, resamples)
    tests["E_harm^(mis)"] = bootstrap(mis["E_harm"], images, resamples)
    tests["S_Q^(mis)"] = bootstrap(mis["S_Q"], images, resamples)
    for other in ("match", "irr"):
        if other in per_condition:
            control_index = mis["S_H"].index.isin(control)
            difference = (mis["S_H"] - per_condition[other]["S_H"]).where(control_index)
            tests[f"T_{other}"] = bootstrap(difference, images, resamples)
            exploratory[f"S_H^({other})"] = bootstrap(per_condition[other]["S_H"].where(control_index), images, resamples)
    exploratory["I^(mis)"] = bootstrap(mis["I"], images, resamples)
    if has_cells(wide, SECONDARY, "mis"):
        exploratory["S_H^(mis), question-span"] = bootstrap(estimands(wide, SECONDARY, "mis")["S_H"], images, resamples)
    out["tests"], out["exploratory"] = tests, exploratory

    primary = tests["S_H^(mis)"]
    low, high = primary["ci90"]
    out["equivalence"] = {"delta": DELTA, "ci90": primary["ci90"], "equivalent": bool(-DELTA <= low and high <= DELTA)}

    out["sensitivity"] = {}
    for variant in MARGIN_VARIANTS[1:]:
        other_wide, other_images = wide_margins(rows, variant)
        out["sensitivity"][variant] = bootstrap(estimands(other_wide, PRIMARY, "mis")["S_H"], other_images, resamples)

    # Behaviour: outcome rates per run, and paired B_mis vs C_mis differences.
    out["behaviour_rates"] = behaviour_rates(rows)
    out["behaviour_paired"] = {f"{result}: C_mis - B_mis": bootstrap(difference, images, resamples)
                               for result, difference in paired_behaviour(rows, "mis").items()}

    # Overlap and H5.
    kept = kept_sets(rows)
    out["overlap"] = overlap(rows, ("mis", "match", "irr"))
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
        out["H6"][rule] = bootstrap(pd.Series(values, dtype=float), images, resamples)
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
            out["mitigation"][name] = bootstrap(wide[treated_column] - wide[baseline_column], images, resamples)
    if "uniform|mis" in wide:
        out["mitigation"]["uniform (mis)"] = bootstrap(wide["uniform|mis"] - base, images, resamples)
    out["mean_margins"] = {column: float(wide[column].mean()) for column in sorted(wide.columns)}
    return out


def apply_families(results: dict[str, dict]) -> dict:
    """PREREG 2.3: Holm within each family across datasets. Final only when every pre-registered dataset is present."""
    complete = all(dataset in results for dataset in DATASETS)
    decisions = {}
    for family, names in FAMILIES.items():
        p_values = {f"{name} [{dataset}]": results[dataset]["tests"][name]["p_two_sided"]
                    for dataset in results for name in names if results[dataset]["tests"].get(name, {}).get("estimate") is not None}
        decisions[family] = {"complete": complete, "p_values": p_values,
                             "reject": holm(p_values) if complete else None}
    return decisions


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
        lines += table("Pre-registered tests", r["tests"], "nats", True)
        e = r["equivalence"]
        lines += [f"Equivalence (δ = {e['delta']}): 90% interval [{e['ci90'][0]:+.3f}, {e['ci90'][1]:+.3f}], "
                  + ("inside ±δ: practically small." if e["equivalent"] else "not inside ±δ."), ""]
        lines += table("Exploratory (intervals only)", r["exploratory"], "nats", True)
        lines += table("Sensitivity of S_H^(mis) to the margin form", r["sensitivity"], "nats", True)
        lines += table("Behaviour, paired difference in the share of answers", r["behaviour_paired"], "share", False)
        lines += table("H6: ERR(S_Q) − ERR(S_mis), by evidence mapping rule", r["H6"], "share of region", False)
        lines += table("Mitigation (EQ1): margin gain over B_mis unless noted", r["mitigation"], "nats", True)
        h5 = r["H5"]
        if h5.get("slope") is not None:
            lines += ["### H5: S_H^(mis) on 1 − Jaccard(S_Q, S_mis)", "",
                      (f"Slope {h5['slope']:+.3f} (cluster-robust SE {h5['se']:.3f}), 95% interval "
                      f"[{h5['ci95'][0]:+.3f}, {h5['ci95'][1]:+.3f}], two-sided p {h5['p_two_sided']:.4f}; "
                      f"{h5['items']} items, {h5['images']} images."), ""]
        lines += ["### Kept-set overlap with S_Q (PW-attn)", "", "| Condition | Mean Jaccard | Share changed |", "| --- | ---: | ---: |"]
        lines += [f"| {k} | {v['mean_jaccard']:.3f} | {v['share_changed']:.3f} |" for k, v in r["overlap"].items()] + [""]
        lines += ["### Behaviour rates", "", *rates_table(r["behaviour_rates"]), ""]
    return "\n".join(lines)


def confirm_report(runs: dict[str, tuple[Path, Path]], out_dir: Path, resamples: int = RESAMPLES,
                   records: dict[str, Path] | None = None) -> dict:
    """runs: dataset -> (run directory, manifest). records: dataset -> stage-3 record, to check completeness."""
    unknown = set(runs) - set(DATASETS)
    if unknown:
        raise Refusal(f"unknown datasets {sorted(unknown)}; expected {DATASETS}")
    results = {}
    for dataset, (run_dir, manifest) in runs.items():
        expected = None
        if records and dataset in records:
            expected = check_code(records[dataset])["main_items"]
        results[dataset] = dataset_results(run_dir, manifest, resamples, expected)
    families = apply_families(results)
    write_report(out_dir, {"datasets": results, "families": families}, render(results, families))
    return {"datasets": results, "families": families}
