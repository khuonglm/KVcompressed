"""Margin, estimands, image-clustered bootstrap, Holm and interval readings."""

from __future__ import annotations

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

