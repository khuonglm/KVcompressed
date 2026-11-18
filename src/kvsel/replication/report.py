"""The replication's pre-registered analysis (PREREG section 3.3-3.4), run once on the finished run folder.

Confirmatory: R1, R2, R3 and R5 as one Holm family of 6, and R4 (equivalence at K = 288). Everything else is
intervals only. The original study's estimates are printed beside the replication's, never pooled.
The freeze check (`protocol.check`, code amendments included) runs before this, in the `report` command.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..analysis.blocks import (
    budget_results,
    check_complete,
    holm_family,
    interval,
    rates_table,
    row,
    source_file,
    table,
    write_report,
)
from ..analysis.statistics import DELTA, RESAMPLES, SURFACE
from ..analysis.tables import PRIMARY, behaviour_rates, checks, load_run, overlap
from .protocol import ALPHA, CONDITIONS, CONTROL_CONDITIONS, EQUIVALENCE, FAMILY

ALL_CONDITIONS = CONDITIONS + CONTROL_CONDITIONS


def original_estimates(confirm_results: Path | None, followup_results: Path | None) -> dict:
    """The original study's values by budget: K = 144 from the confirmatory report, 72 and 288 from the follow-ups."""
    original: dict[str, dict] = {}
    if confirm_results:
        gqa = source_file(confirm_results)["data"]["datasets"]["gqa"]
        original["144"] = {**gqa["exploratory"], **gqa["tests"]}
    if followup_results:
        for k, block in source_file(followup_results)["data"]["parts"]["budgets"]["by_k"].items():
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
    check_complete(run_dir, set(items), set(record["replication_ids"]), "replication items", "the replication run")
    budgets, rates, overlaps = {}, {}, {}
    for k in sorted({r["k"] for r in rows if r["selector"] == PRIMARY}):
        at_k = [r for r in rows if r["k"] == k or r["selector"] in ("full", "blind")]
        budgets[str(k)] = budget_results(at_k, ALL_CONDITIONS, resamples)
        rates[str(k)] = behaviour_rates(at_k)
        overlaps[str(k)] = overlap(at_k, ALL_CONDITIONS)
    tests = {f"{h}: {name}, K = {k}": budgets[k]["margins"][name] for h, name, k in FAMILY}
    family = holm_family(tests, ALPHA)
    original = original_estimates(original_confirm, original_followup)
    readings = {test: reading(family["reject"][test], tests[test]["estimate"], original.get(k, {}).get(name, {}).get("estimate"))
                for test, (_, name, k) in zip(tests, FAMILY, strict=True)}
    _, name, k = EQUIVALENCE
    low, high = budgets[k]["margins"][name]["ci90"]
    return {
        "run": str(run_dir), "surface": SURFACE, "items": len(items), "images": len({r["image_id"] for r in rows}),
        "control_items": len(record["control_ids"]), "checks": checks(rows),
        "family": {**family, "reading": readings},
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
        lines.append(f"| {test} | {row(result)} {family['reading'][test]} |")
    eq = results["equivalence"]
    lines += ["", (f"{eq['test']}: 90% interval [{eq['ci90'][0]:+.3f}, {eq['ci90'][1]:+.3f}], "
              f"{'inside' if eq['equivalent'] else 'not inside'} ±{eq['delta']} (equivalence "
              f"{'supported' if eq['equivalent'] else 'not supported'})."), ""]
    checked = results["checks"]
    lines += [(f"Checks: Full_Q no longer eligible on {checked['full_q_no_longer_eligible']} items; Blind_Q eligible on "
              f"{checked['blind_now_eligible']} of {checked['blind_rows']} (should be 0); truncated answers "
              f"{checked['truncated_share']:.2%}."), "", "## Every estimand by budget (exploratory unless listed above)", ""]
    for k, block in results["budgets"].items():
        lines += table(f"K = {k}", block["margins"], "nats", nats=True)
        lines += table(f"K = {k}: behaviour, paired C − B", block["behaviour_paired"], "share", nats=False)
        lines += ["Mean Jaccard with S_Q: " + ", ".join(f"{c} {v['mean_jaccard']:.3f}" for c, v in results["overlap"][k].items()), ""]
    if results["original"]:
        lines += ["## Original study (exploratory values) beside the replication", "",
                  "| Estimand | K | Original | Original items | Replication |", "| --- | ---: | --- | ---: | --- |"]
        for k, block in results["budgets"].items():
            for name, result in block["margins"].items():
                old = results["original"].get(k, {}).get(name)
                if old and old.get("estimate") is not None and result.get("estimate") is not None:
                    lines.append(f"| {name} | {k} | {interval(old)} | {old.get('items', '—')} | {interval(result)} |")
        lines.append("")
    lines += ["## Behaviour rates", ""]
    for k, rates in results["behaviour_rates"].items():
        lines += [f"### K = {k}", "", *rates_table(rates), ""]
    return "\n".join(lines)


def replication_report(run_dir: Path, record: Path, out_dir: Path, resamples: int = RESAMPLES,
                       original_confirm: Path | None = None, original_followup: Path | None = None) -> dict:
    results = replication_results(run_dir, record, resamples, original_confirm, original_followup)
    write_report(out_dir, results, render(results))
    return results
