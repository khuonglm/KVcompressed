"""Estimands, image-clustered bootstrap and the development report. Reads raw item files only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DELTA = 0.10
RESAMPLES = 10_000
_BATCH = 500


def margin(scores: dict, surface: str = "lower") -> float:
    """M, in nats. ``logsum`` adds probability over the distinct surface forms of each candidate."""
    if surface == "logsum":
        def mass(by_form: dict[str, list[float]]) -> float:
            distinct = {tuple(values) for values in by_form.values()}
            return float(np.logaddexp.reduce([sum(values) for values in distinct]))

        return mass(scores["true"]) - mass(scores["alt"])
    return sum(scores["true"][surface]) - sum(scores["alt"][surface])


def column_name(row: dict) -> str:
    if row["selector"] in ("full", "uniform"):
        return f"{row['selector']}|{row['cell']}"
    reserve = "+r" if row["reserve"] else ""
    return f"{row['selector']}{reserve}|{row['cell']}"


def load_rows(run_dir: Path) -> list[dict]:
    rows = []
    for path in sorted((Path(run_dir) / "items").glob("*.json")):
        item = json.loads(path.read_text())
        for row in item["rows"]:
            rows.append({"item_id": item["item_id"], "image_id": item["image_id"], "contexts": item["contexts"], **row})
    return rows


def margins_wide(rows: list[dict], surface: str) -> tuple[pd.DataFrame, pd.Series]:
    """Items by run. Returns (margins, image id per item)."""
    frame = pd.DataFrame(
        [{"item_id": r["item_id"], "column": column_name(r), "margin": margin(r["scores"], surface)} for r in rows]
    )
    wide = frame.pivot(index="item_id", columns="column", values="margin")
    images = pd.Series({r["item_id"]: r["image_id"] for r in rows}).reindex(wide.index)
    return wide, images


def estimands(wide: pd.DataFrame, selector: str, condition: str) -> dict[str, pd.Series]:
    """Per-item estimands from section 5 of the proposal."""
    a = wide[f"{selector}|A"]
    b = wide[f"{selector}|B_{condition}"]
    c = wide[f"{selector}|C_{condition}"]
    d = wide[f"{selector}|D_{condition}"]
    s_h = c - b
    s_q = a - d
    return {
        "S_H": s_h,
        "S_Q": s_q,
        "I": s_h - s_q,
        "E_harm": (wide[f"full|{condition}"] - b) - (wide["full|Q"] - a),
    }


def cluster_bootstrap(values: pd.Series, clusters: pd.Series, resamples: int = RESAMPLES, seed: int = 0) -> dict:
    """Resample images with replacement, keep all their items, recompute the mean."""
    frame = pd.DataFrame({"value": values, "cluster": clusters}).dropna()
    grouped = frame.groupby("cluster")["value"]
    sums, counts = grouped.sum().to_numpy(), grouped.count().to_numpy()
    n_clusters = len(sums)
    generator = np.random.default_rng(seed)
    estimates = np.empty(resamples)
    probabilities = np.full(n_clusters, 1 / n_clusters)
    for start in range(0, resamples, _BATCH):
        size = min(_BATCH, resamples - start)
        weights = generator.multinomial(n_clusters, probabilities, size=size)
        estimates[start : start + size] = (weights @ sums) / (weights @ counts)
    return {
        "estimate": float(frame["value"].mean()),
        "ci95": (float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))),
        "ci90": (float(np.percentile(estimates, 5)), float(np.percentile(estimates, 95))),
        "se": float(estimates.std(ddof=1)),
        "p_one_sided": float((1 + np.sum(estimates <= 0)) / (resamples + 1)),
        # Every confirmatory test is two-sided (plan, decision record of 9 November 2026).
        "p_two_sided": float(min(1.0, 2 * (1 + min(np.sum(estimates <= 0), np.sum(estimates >= 0))) / (resamples + 1))),
        "items": len(frame),
        "images": n_clusters,
    }


def holm(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Step-down: reject in ascending p order until the first failure."""
    decisions = {name: False for name in p_values}
    ordered = sorted(p_values, key=p_values.get)
    for rank, name in enumerate(ordered):
        if p_values[name] > alpha / (len(ordered) - rank):
            break
        decisions[name] = True
    return decisions


def read_interval(low: float, high: float, delta: float = DELTA) -> str:
    """The interval patterns in section 10 of the proposal. An interval can match two of them."""
    small = -delta <= low and high <= delta
    if low > 0:
        return "positive, practically small" if small else "positive"
    if high < 0:
        return "opposite direction, practically small" if small else "opposite direction"
    return "practically small" if small else "inconclusive"


def _estimate_line(name: str, result: dict) -> str:
    low, high = result["ci95"]
    return (
        f"| {name} | {result['estimate']:+.3f} | [{low:+.3f}, {high:+.3f}] | {result['se']:.3f} | "
        f"{result['p_two_sided']:.4f} | {read_interval(low, high)} | {result['items']} / {result['images']} |"
    )


def _estimand_section(wide, images, selector, conditions, surface, resamples) -> list[str]:
    lines = [
        f"### {selector}, surface `{surface}`", "",
        "| Estimand | Mean (nats) | 95% interval | SE | Two-sided p | Reading | Items / images |",
        "| --- | ---: | --- | ---: | ---: | --- | ---: |",
    ]
    by_condition = {c: estimands(wide, selector, c) for c in conditions}
    for condition in conditions:
        for name, values in by_condition[condition].items():
            lines.append(_estimate_line(f"{name}^({condition})", cluster_bootstrap(values, images, resamples)))
    if "mis" in by_condition:
        for other in conditions:
            if other != "mis":
                difference = by_condition["mis"]["S_H"] - by_condition[other]["S_H"]
                lines.append(_estimate_line(f"T_{other}", cluster_bootstrap(difference, images, resamples)))
    return lines + [""]


def _overlap_section(rows: list[dict], selector: str, k: int) -> list[str]:
    kept = {}
    for row in rows:
        if row["selector"] == selector and not row["reserve"] and row["k"] == k:
            kept[(row["item_id"], row["selection"])] = set(row["kept"])
    lines = [f"### Kept-set overlap with S_Q, {selector} (rung 1)", "", "| Condition | Mean Jaccard | Share with any change |", "| --- | ---: | ---: |"]
    for condition in ("mis", "match", "irr"):
        values = [
            len(s & kept[(item, "Q")]) / len(s | kept[(item, "Q")])
            for (item, context), s in kept.items() if context == condition
        ]
        if values:
            lines.append(f"| {condition} | {np.mean(values):.3f} | {np.mean(np.array(values) < 1):.3f} |")
    return lines + [""]


def _margin_section(wide: pd.DataFrame) -> list[str]:
    """Margin levels per run, so differences can be read against where each condition sits."""
    lines = ["### Mean margin by run", "", "| Run | Mean M (nats) | Median | Share M > 0 |", "| --- | ---: | ---: | ---: |"]
    for column in sorted(wide.columns):
        values = wide[column].dropna()
        lines.append(f"| {column} | {values.mean():+.3f} | {values.median():+.3f} | {(values > 0).mean():.3f} |")
    return lines + [""]


def _behaviour_section(rows: list[dict]) -> list[str]:
    frame = pd.DataFrame([{"run": column_name(r), "outcome": r["outcome"], "truncated": r["truncated"]} for r in rows])
    table = frame.groupby("run")["outcome"].value_counts(normalize=True).unstack(fill_value=0)
    for column in ("correct", "alt", "other"):
        if column not in table:
            table[column] = 0.0
    lines = ["### Behaviour", "", "| Run | Correct | Alternative | Other |", "| --- | ---: | ---: | ---: |"]
    for run, row in table.iterrows():
        lines.append(f"| {run} | {row['correct']:.3f} | {row['alt']:.3f} | {row['other']:.3f} |")
    return lines + ["", f"Truncated answers: {frame['truncated'].mean():.3%}", ""]


def _window_section(rows: list[dict]) -> list[str]:
    contexts = {}
    for row in rows:
        contexts.setdefault(row["item_id"], row["contexts"])
    lines = ["### Window composition", "", "| Context | Question | Description | Instruction | Mean suffix tokens |", "| --- | ---: | ---: | ---: | ---: |"]
    names = sorted({name for c in contexts.values() for name in c})
    for name in names:
        present = [c[name] for c in contexts.values() if name in c]
        lines.append(
            f"| {name} | {np.mean([p['window']['question'] for p in present]):.3f} | "
            f"{np.mean([p['window']['description'] for p in present]):.3f} | "
            f"{np.mean([p['window']['instruction'] for p in present]):.3f} | "
            f"{np.mean([p['suffix_tokens'] for p in present]):.1f} |"
        )
    return lines + [""]


def _surface_section(rows: list[dict]) -> list[str]:
    """Which casing greedy answers use, as input to the surface-form decision."""
    forms = {"lower": 0, "capitalized": 0, "upper": 0, "other": 0}
    for row in rows:
        if row["cell"] != "Q" or row["selector"] != "full":
            continue
        text = row["greedy_text"]
        if text == text.lower():
            forms["lower"] += 1
        elif text == text[:1].upper() + text[1:].lower():
            forms["capitalized"] += 1
        elif text == text.upper():
            forms["upper"] += 1
        else:
            forms["other"] += 1
    total = max(1, sum(forms.values()))
    lines = ["### Greedy answer casing, Full_Q", "", "| Form | Share |", "| --- | ---: |"]
    lines += [f"| {form} | {count / total:.3f} |" for form, count in forms.items()]
    return lines + ["", "Digits and other caseless answers count as lower; one-letter capitals as capitalized.", ""]


def keep_verified(rows: list[dict], decisions_path: Path) -> list[dict]:
    """Rows of items whose latest hand-verification decision is a pass."""
    from .verify import passed_items

    passed = passed_items(decisions_path)
    return [row for row in rows if row["item_id"] in passed]


def report(run_dir: Path, surface: str = "lower", resamples: int = RESAMPLES, verified: Path | None = None) -> str:
    rows = load_rows(run_dir)
    total_items = len({row["item_id"] for row in rows})
    if verified is not None:
        rows = keep_verified(rows, verified)
    if not rows:
        raise SystemExit(f"no item files under {run_dir}" if verified is None else f"no items passed in {verified}")
    wide, images = margins_wide(rows, surface)
    selectors = sorted({r["selector"] for r in rows} - {"full", "uniform"})
    conditions = [c for c in ("mis", "match", "irr") if any(r["answering"] == c for r in rows)]
    lines = [
        f"# Development report: {Path(run_dir).name}", "",
        "**Development items only. Nothing here supports a claim.**", "",
        f"{wide.shape[0]} items on {images.nunique()} images."
        + (f" Hand-verified passes only, out of {total_items} items run." if verified is not None else " Not filtered by hand verification."),
        "",
    ]
    if selectors:
        for selector in selectors:
            lines += _estimand_section(wide, images, selector, conditions, surface, resamples)
            stored = set(rows[0]["scores"]["true"])
            for sensitivity in [form for form in ("lower", "capitalized", "upper") if form in stored] + ["logsum"]:
                if sensitivity != surface:
                    other, other_images = margins_wide(rows, sensitivity)
                    values = estimands(other, selector, conditions[0])["S_H"]
                    result = cluster_bootstrap(values, other_images, resamples)
                    lines += [f"Sensitivity, S_H^({conditions[0]}) with `{sensitivity}`: {result['estimate']:+.3f} "
                              f"[{result['ci95'][0]:+.3f}, {result['ci95'][1]:+.3f}]", ""]
            lines += _overlap_section(rows, selector, next(r["k"] for r in rows if r["selector"] == selector))
    lines += _margin_section(wide)
    lines += _behaviour_section(rows)
    lines += _surface_section(rows)
    lines += _window_section(rows)
    return "\n".join(lines)
