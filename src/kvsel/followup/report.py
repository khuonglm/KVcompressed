"""Descriptive analysis of the follow-up runs. Not confirmatory: intervals and p-values only, no claims.

Margins use the primary form (capitalized surface rule). Cells at K = 144 for PW-attn, the full-cache
anchors and the uniform grid come from the confirmatory run of the same items. Same image-clustered
bootstrap as the confirmatory analysis.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..analysis import RESAMPLES
from ..confirm_analysis import (
    _boot,
    _table,
    evidence_sets,
    load_run,
    margin_variant,
    wide_margins,
)
from .run import BUDGETS, EVIDENCE_RULE, PARTS, RESTORE_M, SEEDS, K


def column(row: dict) -> str:
    if row["selector"] in ("pw", "lt", "uniform"):
        return f"{row['selector']}@{row['k']}|{row['cell']}"
    if row["selector"] == "random":
        return f"random{row['seed']}|{row['cell']}"
    if row["selector"] == "restore":
        if row["arm"] == "reference":
            return "restore|reference"
        suffix = "treatment" if row["arm"] == "treatment" else f"control{row['seed']}"
        return f"restore{row['m']}|{suffix}"
    raise ValueError(f"unknown follow-up selector {row['selector']!r}")


def load_part(run_dir: Path) -> tuple[pd.DataFrame, pd.Series, dict, list[dict]]:
    frame, images, files, rows = [], {}, {}, []
    for path in sorted((Path(run_dir) / "items").glob("*.json")):
        data = json.loads(path.read_text())
        files[data["item_id"]] = data
        images[data["item_id"]] = data["image_id"]
        for row in data["rows"]:
            rows.append({"item_id": data["item_id"], **row})
            frame.append({"item_id": data["item_id"], "column": column(row),
                          "value": margin_variant(row["scores"], "primary")})
    if not frame:
        return pd.DataFrame(), pd.Series(dtype=object), files, rows
    wide = pd.DataFrame(frame).pivot(index="item_id", columns="column", values="value")
    return wide, pd.Series(images).reindex(wide.index), files, rows


def quartet(wide: pd.DataFrame, prefix: str) -> dict[str, pd.Series]:
    """S_H, S_Q, I and E_harm for one selector and budget; Full anchors from the confirmatory run."""
    a, b, c, d = (wide[f"{prefix}|{cell}"] for cell in ("A", "B_mis", "C_mis", "D_mis"))
    s_h, s_q = c - b, a - d
    return {"S_H^(mis)": s_h, "S_Q^(mis)": s_q, "I^(mis)": s_h - s_q,
            "E_harm^(mis)": (wide["full|mis"] - b) - (wide["full|Q"] - a)}


def behaviour(rows: list[dict]) -> dict[str, dict]:
    frame = pd.DataFrame([{"column": column(r), "outcome": r["outcome"]} for r in rows])
    if frame.empty:
        return {}
    rates = frame.groupby("column")["outcome"].value_counts(normalize=True).unstack(fill_value=0.0)
    return {name: {key: float(rates.loc[name].get(key, 0.0)) for key in ("correct", "alt", "other")}
            for name in rates.index}


def followup_results(followup_dir: Path, confirm_run: Path, manifest: Path, resamples: int = RESAMPLES) -> dict:
    confirm_rows, confirm_items = load_run(confirm_run)
    base, images = wide_margins(confirm_rows, "primary")
    confirm_kept = {(r["item_id"], r["cell"]): r["kept"] for r in confirm_rows
                    if r["selector"] == "pw" and not r["reserve"] and r["cell"] in ("A", "B_mis")}
    control = {i for i, meta in confirm_items.items() if meta["subset"] == "control"}
    grounding = {i for i, rules in evidence_sets(manifest, control).items() if rules[EVIDENCE_RULE]}
    expected = {"budgets": len(base), "lt": len(base), "random": len(base), "restore": len(grounding)}

    def boot(series: pd.Series) -> dict:
        return _boot(series, images, resamples)

    results: dict = {"confirm_run": str(confirm_run), "parts": {}}
    kept_match = {"Q": [], "mis": []}
    for part in PARTS:
        run_dir = Path(followup_dir) / part
        if not (run_dir / "items").is_dir():
            continue
        wide, _, files, rows = load_part(run_dir)
        for item_id, data in files.items():
            kept_match["Q"].append(data["kept_144"]["Q"] == confirm_kept.get((item_id, "A")))
            kept_match["mis"].append(data["kept_144"]["mis"] == confirm_kept.get((item_id, "B_mis")))
        joined = base.join(wide, how="inner")
        entry = {"items": len(files), "expected": expected[part], "complete": len(files) == expected[part],
                 "behaviour": behaviour(rows)}
        if part == "budgets":
            entry["by_k"] = {}
            for k in (BUDGETS[0], K, BUDGETS[1]):
                prefix = "pw" if k == K else f"pw@{k}"
                uniform = "uniform|mis" if k == K else f"uniform@{k}|mis"
                block = {name: boot(series) for name, series in quartet(joined, prefix).items()}
                block["uniform − B_mis"] = boot(joined[uniform] - joined[f"{prefix}|B_mis"])
                entry["by_k"][str(k)] = block
        elif part == "lt":
            block = {name: boot(series) for name, series in quartet(joined, f"lt@{K}").items()}
            block["LT B_mis − PW B_mis"] = boot(joined[f"lt@{K}|B_mis"] - joined["pw|B_mis"])
            entry["estimates"] = block
        elif part == "random":
            random_q = joined[[f"random{s}|Q" for s in SEEDS]].mean(axis=1)
            random_mis = joined[[f"random{s}|mis" for s in SEEDS]].mean(axis=1)
            entry["estimates"] = {
                "PW A − random (Q)": boot(joined["pw|A"] - random_q),
                "PW B_mis − random (mis)": boot(joined["pw|B_mis"] - random_mis),
                "PW C_mis − random (mis)": boot(joined["pw|C_mis"] - random_mis),
                "random: mis − Q": boot(random_mis - random_q),
            }
        elif part == "restore":
            entry["qualifying"] = {str(m): sum(bool(d["qualifying"][str(m)]) for d in files.values()) for m in RESTORE_M}
            difference = joined["restore|reference"] - joined["pw|B_mis"]
            entry["reference_vs_confirm"] = {"max_abs_diff": float(difference.abs().max()) if len(difference) else None,
                                             "items": int(difference.notna().sum())}
            entry["by_m"] = {}
            for m in RESTORE_M:
                treatment = f"restore{m}|treatment"
                if treatment not in joined:
                    entry["by_m"][str(m)] = {}
                    continue
                controls = joined[[f"restore{m}|control{s}" for s in SEEDS]].mean(axis=1)
                reference = joined["restore|reference"]
                entry["by_m"][str(m)] = {
                    "treatment − control (EQ3)": boot(joined[treatment] - controls),
                    "treatment − reference": boot(joined[treatment] - reference),
                    "control − reference": boot(controls - reference),
                }
        results["parts"][part] = entry
    results["kept_144_matches_confirm"] = {context: (sum(v) / len(v) if v else None) for context, v in kept_match.items()}
    return results


def render(results: dict) -> str:
    lines = ["# Follow-ups (not confirmatory)", "",
             ("PREREG.md 2.4 lists these outside the confirmatory study. Intervals and p-values are descriptive; "
              "nothing here supports a confirmatory claim. Primary margin, PW-attn K = 144 cells and Full anchors "
              "from the confirmatory run."), ""]
    match = results["kept_144_matches_confirm"]
    lines += [(f"Consistency: K = 144 kept sets identical to the confirmatory run's for "
               f"{_share(match['Q'])} of S_Q and {_share(match['mis'])} of S_mis."), ""]
    titles = {"budgets": "Budgets: PW-attn at K = 288, 144 and 72", "lt": "LT-early, K = 144",
              "random": "Random selection, seeds 0-4 averaged per item, K = 144",
              "restore": "Oracle restoration on grounding items (EQ3)"}
    for part, entry in results["parts"].items():
        status = "complete" if entry["complete"] else f"INCOMPLETE: {entry['items']} of {entry['expected']} items"
        lines += [f"## {titles[part]}", "", f"{entry['items']} items ({status}).", ""]
        if part == "budgets":
            for k, block in entry["by_k"].items():
                lines += _table(f"K = {k}", block, "nats", True)
        elif part == "restore":
            q = entry["qualifying"]
            ref = entry["reference_vs_confirm"]
            lines += [(f"Qualifying items: m = 4: {q['4']}, m = 8: {q['8']}. Reference (B_mis rerun) vs the "
                      f"confirmatory B_mis: largest difference {_num(ref['max_abs_diff'])} nats over {ref['items']} items."), ""]
            for m, block in entry["by_m"].items():
                lines += _table(f"m = {m}", block, "nats", True) if block else [f"### m = {m}", "", "No qualifying items.", ""]
        else:
            lines += _table("Estimates", entry["estimates"], "nats", True)
        lines += ["### Behaviour rates", "", "| Run | Correct | Alternative | Other |", "| --- | ---: | ---: | ---: |"]
        lines += [f"| {name} | {r['correct']:.3f} | {r['alt']:.3f} | {r['other']:.3f} |" for name, r in entry["behaviour"].items()]
        lines.append("")
    return "\n".join(lines)


def _share(value) -> str:
    return "—" if value is None else f"{value:.1%}"


def _num(value) -> str:
    return "—" if value is None else f"{value:.2e}"


def followup_report(followup_dir: Path, confirm_run: Path, manifest: Path, out_dir: Path,
                    resamples: int = RESAMPLES) -> dict:
    results = followup_results(followup_dir, confirm_run, manifest, resamples)
    if not results["parts"]:
        raise SystemExit(f"no follow-up parts under {followup_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    (out_dir / "report.md").write_text(render(results))
    return {part: {"items": e["items"], "complete": e["complete"]} for part, e in results["parts"].items()}
