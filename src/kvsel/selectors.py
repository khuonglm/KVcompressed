"""Scoring and kept-set construction. Pure NumPy; no model access.

Indices are 0..575 in row-major order on the 24 x 24 grid.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from . import GRID, NUM_VISUAL

COVERAGE_START = (11, 11)
RESERVE_FRACTION = 0.10


def pool_3x3(values: np.ndarray) -> np.ndarray:
    """3 x 3 mean, stride 1. Edge cells average over in-image neighbours only."""
    grid = np.asarray(values, dtype=np.float64).reshape(GRID, GRID)
    totals = np.zeros((GRID, GRID))
    counts = np.zeros((GRID, GRID))
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            src_r = slice(max(0, -dr), GRID - max(0, dr))
            src_c = slice(max(0, -dc), GRID - max(0, dc))
            dst_r = slice(max(0, dr), GRID - max(0, -dr))
            dst_c = slice(max(0, dc), GRID - max(0, -dc))
            totals[dst_r, dst_c] += grid[src_r, src_c]
            counts[dst_r, dst_c] += 1
    return (totals / counts).ravel()


def window_scores(attention: np.ndarray, window: np.ndarray | slice) -> np.ndarray:
    """Mean attention over the query tokens in ``window``, then 3 x 3 pooled."""
    return pool_3x3(attention[window].astype(np.float64).mean(axis=0))


def pw_attn(attention: np.ndarray, window_size: int = 16) -> np.ndarray:
    """PW-attn: attention from the last ``window_size`` tokens of the probe suffix."""
    return window_scores(attention, slice(-window_size, None))


def question_span_attn(attention: np.ndarray, question_span: tuple[int, int]) -> np.ndarray:
    """Question-span attention: the same question tokens in every probe, however contextualized."""
    return window_scores(attention, slice(*question_span))


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k highest scores; ties go to the lower index. Sorted ascending."""
    order = np.lexsort((np.arange(len(scores)), -np.asarray(scores)))
    return np.sort(order[:k])


def uniform_grid(k: int) -> np.ndarray:
    """Nested regular lattices that hit K exactly, for K in {288, 144, 72}."""
    rows, cols = np.divmod(np.arange(NUM_VISUAL), GRID)
    if k == 288:
        keep = (rows + cols) % 2 == 0
    elif k == 144:
        keep = (rows % 2 == 0) & (cols % 2 == 0)
    elif k == 72:
        keep = (rows % 2 == 0) & (cols % 2 == 0) & ((rows // 2 + cols // 2) % 2 == 0)
    else:
        raise ValueError(f"uniform grid is defined for K in 288, 144, 72; got {k}")
    return np.flatnonzero(keep)


@lru_cache(maxsize=1)
def coverage_order() -> tuple[int, ...]:
    """Farthest-point order from (11, 11). Exact integer distances; row-major ties."""
    rows, cols = np.divmod(np.arange(NUM_VISUAL), GRID)
    first = COVERAGE_START[0] * GRID + COVERAGE_START[1]
    nearest = (rows - rows[first]) ** 2 + (cols - cols[first]) ** 2
    order = [first]
    for _ in range(NUM_VISUAL - 1):
        chosen = int(np.argmax(nearest))  # argmax returns the first maximum: row-major tie-break
        order.append(chosen)
        nearest = np.minimum(nearest, (rows - rows[chosen]) ** 2 + (cols - cols[chosen]) ** 2)
    return tuple(order)


def reserve_size(k: int, fraction: float = RESERVE_FRACTION) -> int:
    return round(fraction * k)


def with_reserve(scores: np.ndarray, k: int, fraction: float = RESERVE_FRACTION) -> np.ndarray:
    """Top K - k_r by score, then coverage-order entries not yet kept, until exactly K are kept."""
    kept = set(top_k(scores, k - reserve_size(k, fraction)).tolist())
    for index in coverage_order():
        if len(kept) == k:
            break
        kept.add(index)
    return np.array(sorted(kept))


def coverage_radius(kept: np.ndarray) -> float:
    """Largest distance from any grid cell to its nearest kept cell."""
    rows, cols = np.divmod(np.arange(NUM_VISUAL), GRID)
    kept_rows, kept_cols = np.divmod(np.asarray(kept), GRID)
    squared = (rows[:, None] - kept_rows[None]) ** 2 + (cols[:, None] - kept_cols[None]) ** 2
    return float(np.sqrt(squared.min(axis=1).max()))


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    first, second = set(np.asarray(a).tolist()), set(np.asarray(b).tolist())
    return len(first & second) / len(first | second)
