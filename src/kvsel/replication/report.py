"""The replication's pre-registered analysis (PREREG section 3.3-3.4), run once on the finished run folder.

Confirmatory: R1, R2, R3 and R5 as one Holm family of 6, and R4 (equivalence at K = 288). Everything else is
intervals only. The original study's estimates are printed beside the replication's, never pooled.
The freeze check (`protocol.check`, code amendments included) runs before this, in the `report` command.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..analysis import DELTA, RESAMPLES, column_name, estimands, holm
from ..confirm_analysis import (
    PRIMARY,
    SURFACE,
    _boot,
    _fmt,
    _plain,
    _table,
    has_cells,
    jaccard,
    kept_sets,
    load_run,
    wide_margins,
)
from .protocol import ALPHA, CONDITIONS, CONTROL_CONDITIONS, EQUIVALENCE, FAMILY, checks

ALL_CONDITIONS = CONDITIONS + CONTROL_CONDITIONS


def budget_results(rows: list[dict], resamples: int) -> dict:
    """Every estimand at one budget, and paired behaviour differences (shares). Match cells exist on control items only, so match terms are control-only."""
    wide, images = wide_margins(rows, "primary")
    per = {c: estimands(wide, PRIMARY, c) for c in ALL_CONDITIONS if has_cells(wide, PRIMARY, c)}
    block = {f"{name}^({c})": _boot(series, images, resamples)
             for c, values in per.items() for name, series in values.items()}
    for other in ("irr", "match"):
        if other in per:
            block[f"T_{other}"] = _boot(per["mis"]["S_H"] - per[other]["S_H"], images, resamples)
    outcomes = pd.DataFrame([{"item_id": r["item_id"], "column": column_name(r), "outcome": r["outcome"]} for r in rows])
    pivot = outcomes.pivot(index="item_id", columns="column", values="outcome")
    paired = {}
    for c in per:
        for result in ("correct", "alt"):
            difference = ((pivot[f"{PRIMARY}|C_{c}"] == result).astype(float)
                          - (pivot[f"{PRIMARY}|B_{c}"] == result).astype(float)).where(pivot[f"{PRIMARY}|B_{c}"].notna())
            paired[f"{result}: C_{c} − B_{c}"] = _boot(difference, images, resamples)
    return {"margins": block, "behaviour_paired": paired}


def behaviour_rates(rows: list[dict]) -> dict:
    outcomes = pd.DataFrame([{"column": column_name(r), "outcome": r["outcome"]} for r in rows])
    rates = outcomes.groupby("column")["outcome"].value_counts(normalize=True).unstack(fill_value=0.0)
    return {column: {k: float(rates.loc[column].get(k, 0.0)) for k in ("correct", "alt", "other")} for column in rates.index}


def overlap(rows: list[dict]) -> dict:
    kept = kept_sets(rows)
    out = {}
    for c in ALL_CONDITIONS:
        values = [jaccard(s, kept[(item, "Q")]) for (item, context), s in kept.items()
                  if context == c and (item, "Q") in kept]
        if values:
            out[c] = {"mean_jaccard": float(np.mean(values)), "share_changed": float(np.mean(np.array(values) < 1))}
    return out


def original_estimates(confirm_results: Path | None, followup_results: Path | None) -> dict:
    """The original study's values by budget: K = 144 from the confirmatory report, 72 and 288 from the follow-ups."""
    original: dict[str, dict] = {}
    if confirm_results:
        gqa = json.loads(Path(confirm_results).read_text())["datasets"]["gqa"]
        original["144"] = {**gqa["exploratory"], **gqa["tests"]}
    if followup_results:
        for k, block in json.loads(Path(followup_results).read_text())["parts"]["budgets"]["by_k"].items():
            if k != "144":
                original[k] = block
    return original


def reading(rejected: bool, estimate: float, original: float | None) -> str:
    """PREREG 3.3: a rejection opposite in sign to the original estimate is a failed replication."""
    if not rejected:
        return "not rejected"
    if original is None:
        return "rejected (no original estimate to compare)"
    return "replicated" if np.sign(estimate) == np.sign(original) else "failed replication: opposite to the original"


def replication_results(run_dir: Path, record_path: Path, resamples: int = RESAMPLES,
                        original_confirm: Path | None = None, original_followup: Path | None = None) -> dict:
    record = json.loads(Path(record_path).read_text())
    rows, items = load_run(run_dir)
    expected = set(record["replication_ids"])
    missing, extra = expected - set(items), set(items) - expected
    if missing:
        raise SystemExit(f"{run_dir} is missing {len(missing)} replication items; rerun the replication run to finish it")
    if extra:
        raise SystemExit(f"{run_dir} has {len(extra)} extra item files outside the frozen set; move them out")
    budgets, rates, overlaps = {}, {}, {}
    for k in sorted({r["k"] for r in rows if r["selector"] == PRIMARY}):
        at_k = [r for r in rows if r["k"] == k or r["selector"] in ("full", "blind")]
        budgets[str(k)] = budget_results(at_k, resamples)
        rates[str(k)] = behaviour_rates(at_k)
        overlaps[str(k)] = overlap(at_k)
    tests = {f"{h}: {name}, K = {k}": budgets[k]["margins"][name] for h, name, k in FAMILY}
    reject = holm({t: r["p_two_sided"] for t, r in tests.items()}, ALPHA)
    original = original_estimates(original_confirm, original_followup)
    readings = {test: reading(reject[test], tests[test]["estimate"], original.get(k, {}).get(name, {}).get("estimate"))
                for test, (_, name, k) in zip(tests, FAMILY, strict=True)}
    _, name, k = EQUIVALENCE
    low, high = budgets[k]["margins"][name]["ci90"]
    return {
        "run": str(run_dir), "surface": SURFACE, "items": len(items), "images": len({r["image_id"] for r in rows}),
        "control_items": len(record["control_ids"]), "checks": checks(rows),
        "family": {"alpha": ALPHA, "tests": tests, "reject": reject, "reading": readings},
        "equivalence": {"test": f"R4: {name}, K = {k}", "delta": DELTA, "ci90": [low, high],
                        "equivalent": bool(-DELTA <= low and high <= DELTA)},
        "budgets": budgets, "behaviour_rates": rates, "overlap": overlaps, "original": original,
    }


def render(results: dict) -> str:
    lines = ["# GQA replication report", "",
             (f"Surface rule `{results['surface']}`; PW-attn; image-clustered bootstrap, two-sided tests. "
              f"{results['items']} items on {results['images']} images; {results['control_items']} control items."), "",
             "## Confirmatory: Holm family of 6 (α = 0.05), final", "",
             "| Test | Mean (nats) | 95% interval | Two-sided p | Reading | Items / images | Holm, against the original |",
             "| --- | ---: | --- | ---: | --- | ---: | --- |"]
    family = results["family"]
    for test, result in family["tests"].items():
        lines.append(f"| {test} | {_fmt(result)} {family['reading'][test]} |")
    eq = results["equivalence"]
    lines += ["", (f"{eq['test']}: 90% interval [{eq['ci90'][0]:+.3f}, {eq['ci90'][1]:+.3f}], "
              f"{'inside' if eq['equivalent'] else 'not inside'} ±{eq['delta']} (equivalence "
              f"{'supported' if eq['equivalent'] else 'not supported'})."), ""]
    checked = results["checks"]
    lines += [(f"Checks: Full_Q no longer eligible on {checked['full_q_no_longer_eligible']} items; Blind_Q eligible on "
              f"{checked['blind_now_eligible']} of {checked['blind_rows']} (should be 0); truncated answers "
              f"{checked['truncated_share']:.2%}."), "", "## Every estimand by budget (exploratory unless listed above)", ""]
    for k, block in results["budgets"].items():
        lines += _table(f"K = {k}", block["margins"], "nats", nats=True)
        lines += _table(f"K = {k}: behaviour, paired C − B", block["behaviour_paired"], "share", nats=False)
        lines += ["Mean Jaccard with S_Q: " + ", ".join(f"{c} {v['mean_jaccard']:.3f}" for c, v in results["overlap"][k].items()), ""]
    if results["original"]:
        lines += ["## Original study (exploratory values) beside the replication", "",
                  "| Estimand | K | Original | Original items | Replication |", "| --- | ---: | --- | ---: | --- |"]
        for k, block in results["budgets"].items():
            for name, result in block["margins"].items():
                old = results["original"].get(k, {}).get(name)
                if old and old.get("estimate") is not None and result.get("estimate") is not None:
                    lines.append(f"| {name} | {k} | {_interval(old)} | {old.get('items', '—')} | {_interval(result)} |")
        lines.append("")
    lines += ["## Behaviour rates", ""]
    for k, rates in results["behaviour_rates"].items():
        lines += [f"### K = {k}", "", "| Run | Correct | Alternative | Other |", "| --- | ---: | ---: | ---: |"]
        lines += [f"| {c} | {r['correct']:.3f} | {r['alt']:.3f} | {r['other']:.3f} |" for c, r in rates.items()] + [""]
    return "\n".join(lines)


def _interval(result: dict) -> str:
    low, high = result["ci95"]
    return f"{result['estimate']:+.3f} [{low:+.3f}, {high:+.3f}]"


def replication_report(run_dir: Path, record: Path, out_dir: Path, resamples: int = RESAMPLES,
                       original_confirm: Path | None = None, original_followup: Path | None = None) -> dict:
    results = replication_results(run_dir, record, resamples, original_confirm, original_followup)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=_plain))
    (out_dir / "report.md").write_text(render(results))
    return results
