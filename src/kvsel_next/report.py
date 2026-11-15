"""NeXT's pre-registered analysis (PREREG 4.3), run once on the finished run folder.

Confirmatory: N1-N4 at 25% as one Holm family of 4. Everything else is intervals only. LLaVA-1.5's
replication estimates (section 3) are printed beside NeXT's at the matching budget, never pooled.
Equivalence is not tested (4.3). The freeze check runs before this, in the `report` command.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from kvsel import NUM_VISUAL
from kvsel.analysis import RESAMPLES, holm
from kvsel.confirm_analysis import SURFACE, _fmt, _plain, _table, checks, load_run
from kvsel.replication.report import _interval, behaviour_rates, budget_results, overlap

from .selection import SHARES



def share_label(share: float) -> str:
    return f"{share:.1%}"


# LLaVA-1.5's K for each share of feature tokens (PREREG 4.3), keyed by the report's share labels.
K_OF = {share_label(share): str(round(share * NUM_VISUAL)) for share in SHARES}  # 72, 144, 288


def results(run_dir: Path, record_path: Path, llava15: Path, resamples: int = RESAMPLES) -> dict:
    record = json.loads(Path(record_path).read_text())
    settings = record["settings"]
    rows, items = load_run(run_dir)
    expected = set(record["main_ids"])
    missing, extra = expected - set(items), set(items) - expected
    if missing:
        raise SystemExit(f"{run_dir} is missing {len(missing)} main-set items; rerun `python -m kvsel_next run` to finish it")
    if extra:
        raise SystemExit(f"{run_dir} has {len(extra)} extra item files outside the frozen set; move them out")
    shares, rates, overlaps = {}, {}, {}
    for share in settings["shares"]:
        at_share = [r for r in rows if r["share"] == share or r["selector"] in ("full", "blind")]
        shares[share_label(share)] = budget_results(at_share, resamples)
        rates[share_label(share)] = behaviour_rates(at_share)
        overlaps[share_label(share)] = overlap(at_share)
    tests = {f"{h}: {name}, {share_label(share)}": shares[share_label(share)]["margins"][name]
             for h, name, share in settings["family"]}
    layouts = [r["layout"] for r in rows if r["selector"] == "full" and r["cell"] == "Q"]
    by_k = json.loads(Path(llava15).read_text())["budgets"]
    beside = {label: by_k.get(k, {}).get("margins", {}) for label, k in K_OF.items()}
    return {
        "run": str(run_dir), "surface": SURFACE, "items": len(items), "images": len({r["image_id"] for r in rows}),
        "control_items": len(record["control_ids"]),
        "main_size_frozen_at_stage1": record["main_size_frozen_at_stage1"],
        "checks": {**checks(rows), "visual_tokens_min": min(x["total"] for x in layouts),
                   "visual_tokens_max": max(x["total"] for x in layouts),
                   "grids": dict(Counter(f"{x['grid'][0]}x{x['grid'][1]}" for x in layouts).most_common())},
        "family": {"alpha": settings["alpha"], "tests": tests,
                   "reject": holm({t: r["p_two_sided"] for t, r in tests.items()}, settings["alpha"])},
        "shares": shares, "behaviour_rates": rates, "overlap": overlaps, "llava15": beside,
    }


def render(found: dict) -> str:
    family = found["family"]
    lines = ["# LLaVA-NeXT-7B on GQA: report", "",
             (f"Surface rule `{found['surface']}`; PW-attn; image-clustered bootstrap, two-sided tests. "
              f"{found['items']} items on {found['images']} images; {found['control_items']} control items. "
              "LLaVA-1.5's comparative outcomes on many of these items were seen before NeXT's hypotheses were chosen (PREREG 4)."), "",
             *([f"**Shortfall:** {found['items']} eligible items, not {found['main_size_frozen_at_stage1']}: the whole pool was "
                "screened, so the main set is every eligible item (PREREG 4.5(d)).", ""]
               if found["items"] < found["main_size_frozen_at_stage1"] else []),
             f"## Confirmatory: Holm family of {len(family['tests'])} (α = {family['alpha']}), final", "",
             "| Test | Mean (nats) | 95% interval | Two-sided p | Reading | Items / images | Holm |",
             "| --- | ---: | --- | ---: | --- | ---: | --- |"]
    lines += [f"| {test} | {_fmt(result)} {'rejected' if family['reject'][test] else 'not rejected'} |"
              for test, result in family["tests"].items()]
    checked = found["checks"]
    lines += ["", "Equivalence: not tested (PREREG 4.3).", "",
              (f"Checks: Full_Q no longer eligible on {checked['full_q_no_longer_eligible']} items; Blind_Q eligible on "
               f"{checked['blind_now_eligible']} of {checked['blind_rows']} (should be 0); truncated answers "
               f"{checked['truncated_share']:.2%}; visual tokens {checked['visual_tokens_min']}-{checked['visual_tokens_max']}; "
               "high-resolution grids " + ", ".join(f"{g} ({n})" for g, n in checked["grids"].items()) + "."),
              "", "## Every estimand by share (intervals only unless listed above)", ""]
    for share, block in found["shares"].items():
        lines += _table(share, block["margins"], "nats", nats=True)
        lines += _table(f"{share}: behaviour, paired C − B", block["behaviour_paired"], "share", nats=False)
        lines += ["Mean Jaccard with S_Q: " + ", ".join(f"{c} {v['mean_jaccard']:.3f}" for c, v in found["overlap"][share].items()), ""]
    lines += ["## LLaVA-1.5 (replication, section 3) beside NeXT: descriptive, never pooled", "",
              "The item sets and absolute budgets differ; sign agreement is reported, not tested.", "",
              "| Estimand | NeXT share | LLaVA-1.5 K | LLaVA-1.5 | NeXT | Same sign |", "| --- | ---: | ---: | --- | --- | --- |"]
    for share, block in found["shares"].items():
        for name, result in block["margins"].items():
            old = found["llava15"][share].get(name, {})
            both = old.get("estimate") is not None and result.get("estimate") is not None
            lines.append(f"| {name} | {share} | {K_OF[share]} | {_interval(old) if old.get('estimate') is not None else '—'} | "
                         f"{_interval(result) if result.get('estimate') is not None else '—'} | "
                         f"{('yes' if np.sign(old['estimate']) == np.sign(result['estimate']) else 'no') if both else '—'} |")
    lines.append("")
    lines += ["## Behaviour rates", ""]
    for share, rates in found["behaviour_rates"].items():
        lines += [f"### {share}", "", "| Run | Correct | Alternative | Other |", "| --- | ---: | ---: | ---: |"]
        lines += [f"| {c} | {r['correct']:.3f} | {r['alt']:.3f} | {r['other']:.3f} |" for c, r in rates.items()] + [""]
    return "\n".join(lines)


def report(run_dir: Path, record: Path, out_dir: Path, llava15: Path, resamples: int = RESAMPLES) -> dict:
    found = results(run_dir, record, llava15, resamples)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(found, indent=2, default=_plain))
    (out_dir / "report.md").write_text(render(found))
    return found
