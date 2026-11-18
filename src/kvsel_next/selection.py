"""PW-attn kept sets for a share of an image's feature tokens. Pure NumPy; no model access.

Scores are 3 x 3-pooled within each spatial map (the base tile, the high-resolution map), never across
a map boundary or a newline. Newline tokens are always kept and are not counted in the budget.
"""

from __future__ import annotations

import numpy as np

from kvsel.models.layout import Layout
from kvsel.prompts import WINDOW_SIZE  # last suffix tokens averaged, as LLaVA-1.5's PW-attn
from kvsel.selectors import top_k

SHARES = (0.125, 0.25, 0.5)


def pool(grid: np.ndarray) -> np.ndarray:
    """3 x 3 mean, stride 1, on a map of any shape. Edge cells average over in-map neighbours only."""
    rows, cols = grid.shape
    totals = np.zeros((rows, cols))
    counts = np.zeros((rows, cols))
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            src = slice(max(0, -dr), rows - max(0, dr)), slice(max(0, -dc), cols - max(0, dc))
            dst = slice(max(0, dr), rows - max(0, -dr)), slice(max(0, dc), cols - max(0, -dc))
            totals[dst] += grid[src]
            counts[dst] += 1
    return totals / counts


def pooled_scores(layout: Layout, scores: np.ndarray) -> np.ndarray:
    """Pooled scores over the visual span; newline entries are NaN."""
    scores = np.asarray(scores, dtype=np.float64)
    pooled = np.full(layout.total, np.nan)
    for index in range(len(layout.maps)):
        at = layout.positions(index)
        pooled[at] = pool(scores[at])
    return pooled


def budget(layout: Layout, share: float) -> int:
    """share x feature tokens, which must be whole: one map axis is never unpadded, so features are multiples of the base map's side."""
    k = share * (layout.total - len(layout.newlines))
    if k != int(k):
        raise ValueError(f"share {share} of {layout.total - len(layout.newlines)} feature tokens is not whole")
    return int(k)


def kept_set(layout: Layout, scores: np.ndarray, share: float) -> np.ndarray:
    """The top k feature tokens by pooled score (ties to the lower index) plus every newline, sorted."""
    features = np.setdiff1d(np.arange(layout.total), layout.newlines)
    if not np.isfinite(np.asarray(scores)[features]).all():
        raise ValueError("feature scores must be finite")  # pooling would spread a NaN to its neighbours
    pooled = pooled_scores(layout, scores)
    chosen = features[top_k(pooled[features], budget(layout, share))]
    return np.union1d(chosen, np.array(layout.newlines, dtype=chosen.dtype))


def kept_sets(layout: Layout, attention: np.ndarray) -> dict[float, np.ndarray]:
    """Every share's kept set from one probe: ``attention`` is the engine's (suffix tokens, visual) array."""
    scores = np.asarray(attention[-WINDOW_SIZE:], dtype=np.float64).mean(axis=0)
    return {share: kept_set(layout, scores, share) for share in SHARES}
