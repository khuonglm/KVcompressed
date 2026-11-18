"""Selection on a layout: scores, budgets and kept sets for every selector. Pure NumPy; no model access.

PREREG 2.1: PW-attn's 16-token window, question-span attention, the uniform grid and the coverage reserve.
PREREG 4.2: scores are 3 x 3-pooled within each map of the layout, never across a map boundary or a newline;
top-k ties go to the lower index; newlines are always kept and are not counted in k; a share must give a whole k.
Kept sets are visual-span indices.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .models.layout import Layout

WINDOW_SIZE = 16  # PW-attn: the last suffix tokens of the probe
RESERVE_FRACTION = 0.10


@dataclass(frozen=True)
class Budget:
    """An absolute K or a share of an image's feature tokens; exactly one is set."""

    k: int | None = None
    share: float | None = None

    def __post_init__(self):
        if (self.k is None) == (self.share is None):
            raise ValueError("a budget is either an absolute k or a share")

    def resolve(self, layout: Layout) -> int:
        """K for this layout. A share must give a whole number of feature tokens."""
        features = layout.total - len(layout.newlines)
        if self.k is not None:
            if not 0 <= self.k <= features:
                raise ValueError(f"k {self.k} is outside 0-{features} feature tokens")
            return self.k
        k = self.share * features
        # shortcut: exact only for binary-fraction shares (12.5%, 25%, 50%); use fractions.Fraction if another is
        # ever pre-registered, or 0.1 x 70 = 7.000000000000001 would be refused.
        if k != int(k):
            raise ValueError(f"share {self.share} of {features} feature tokens is not whole")
        return int(k)


# --- Scores ---------------------------------------------------------------------------------------------------


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
    """Scores pooled within each map; newline entries are NaN."""
    scores = np.asarray(scores, dtype=np.float64)
    pooled = np.full(layout.total, np.nan)
    for index in range(len(layout.maps)):
        at = layout.positions(index)
        pooled[at] = pool(scores[at])
    return pooled


def window_scores(layout: Layout, attention: np.ndarray, window: slice) -> np.ndarray:
    """Mean attention over the query tokens in ``window``, then pooled within each map."""
    return pooled_scores(layout, attention[window].astype(np.float64).mean(axis=0))


def pw_attn(layout: Layout, attention: np.ndarray, window_size: int = WINDOW_SIZE) -> np.ndarray:
    """PW-attn: attention from the last ``window_size`` tokens of the probe suffix."""
    return window_scores(layout, attention, slice(-window_size, None))


def question_span_attn(layout: Layout, attention: np.ndarray, question_span: tuple[int, int]) -> np.ndarray:
    """Question-span attention: the same question tokens in every probe, however contextualized."""
    return window_scores(layout, attention, slice(*question_span))


def lt_early(layer2_last: np.ndarray) -> np.ndarray:
    """LT-early scores: layer-2 attention from the last suffix token, mean over heads; no pooling."""
    return np.asarray(layer2_last, dtype=np.float64)


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k highest scores; ties go to the lower index. Sorted ascending."""
    order = np.lexsort((np.arange(len(scores)), -np.asarray(scores)))
    return np.sort(order[:k])


# --- Fixed patterns, defined only on a single map without newlines: LLaVA-1.5's layout ---------------------------


def _single_map(layout: Layout, what: str) -> tuple[np.ndarray, np.ndarray]:
    """(rows, cols) of every position of a layout with one map and no newlines; other layouts are refused."""
    if len(layout.maps) != 1 or layout.newlines:
        raise ValueError(f"the {what} is defined only on a single map without newlines")
    return np.divmod(np.arange(layout.total), layout.maps[0].cols)


def uniform_grid(layout: Layout, k: int) -> np.ndarray:
    """Nested regular lattices keeping 1/2, 1/4 or 1/8 of the map: a checkerboard, every other cell of every
    other row, and every other one of those."""
    rows, cols = _single_map(layout, "uniform grid")
    lattices = {
        2: (rows + cols) % 2 == 0,
        4: (rows % 2 == 0) & (cols % 2 == 0),
        8: (rows % 2 == 0) & (cols % 2 == 0) & ((rows // 2 + cols // 2) % 2 == 0),
    }
    sizes = {int(keep.sum()): fraction for fraction, keep in lattices.items()}  # K = 288, 144, 72 on LLaVA-1.5's
    if k not in sizes:
        raise ValueError(f"uniform grid is defined for K in {', '.join(map(str, sizes))}; got {k}")
    return np.flatnonzero(lattices[sizes[k]])


@lru_cache(maxsize=4)
def coverage_order(layout: Layout) -> tuple[int, ...]:
    """Farthest-point order from the map's centre cell (rounded up and left). Exact integer distances; row-major
    ties."""
    rows, cols = _single_map(layout, "coverage order")
    first = int(rows[-1] // 2 * layout.maps[0].cols + cols[-1] // 2)
    nearest = (rows - rows[first]) ** 2 + (cols - cols[first]) ** 2
    order = [first]
    for _ in range(layout.total - 1):
        chosen = int(np.argmax(nearest))  # argmax returns the first maximum: row-major tie-break
        order.append(chosen)
        nearest = np.minimum(nearest, (rows - rows[chosen]) ** 2 + (cols - cols[chosen]) ** 2)
    return tuple(order)


def reserve_size(k: int, fraction: float = RESERVE_FRACTION) -> int:
    return round(fraction * k)


def with_reserve(layout: Layout, scores: np.ndarray, k: int, fraction: float = RESERVE_FRACTION) -> np.ndarray:
    """Top K - k_r by score, then coverage-order entries not yet kept, until exactly K are kept."""
    kept = set(top_k(scores, k - reserve_size(k, fraction)).tolist())
    for index in coverage_order(layout):
        if len(kept) == k:
            break
        kept.add(index)
    return np.array(sorted(kept))


# --- The one entry point ----------------------------------------------------------------------------------------

SCORED = ("pw", "qspan", "lt")
NEEDS = {"pw": ("attention",), "qspan": ("attention", "question_span"), "lt": ("layer2",), "random": ("rng",)}


def kept_set(layout: Layout, selector: str, budget: Budget, *, attention: np.ndarray | None = None,
             question_span: tuple[int, int] | None = None, layer2: np.ndarray | None = None,
             rng: np.random.Generator | None = None, reserve: bool = False) -> np.ndarray:
    """Visual positions a unit keeps: k feature tokens chosen by ``selector``, plus every newline, sorted.

    full: every position. pw, qspan: the probe's ``attention`` (suffix tokens, visual); lt: ``layer2``, the
    layer-2 attention of the last suffix token. uniform: the fixed lattice. random: ``rng``'s draw.
    ``reserve`` swaps the last 10% of a scored set for the
    coverage order.
    """
    if selector == "full":
        return np.arange(layout.total)
    given = {"attention": attention, "question_span": question_span, "layer2": layer2, "rng": rng}
    missing = [name for name in NEEDS.get(selector, ()) if given[name] is None]
    if missing:
        raise ValueError(f"selector {selector!r} needs {', '.join(missing)}")
    k = budget.resolve(layout)
    features = layout.features
    if selector in SCORED:
        if selector == "pw":
            scores = pw_attn(layout, attention)
        elif selector == "qspan":
            scores = question_span_attn(layout, attention, question_span)
        else:
            scores = lt_early(layer2)
        if not np.isfinite(scores[features]).all():
            raise ValueError("feature scores must be finite")  # pooling would spread a NaN to its neighbours
        chosen = with_reserve(layout, scores, k) if reserve else features[top_k(scores[features], k)]
    elif reserve:
        raise ValueError(f"the coverage reserve applies to scored selectors, not {selector!r}")
    elif selector == "uniform":
        chosen = uniform_grid(layout, k)
    elif selector == "random":
        chosen = np.sort(rng.choice(features, size=k, replace=False))
    else:
        raise ValueError(f"unknown selector {selector!r}")
    return np.union1d(chosen, np.array(layout.newlines, dtype=np.int64))


def jaccard(a, b) -> float:
    """Overlap of two kept sets; two empty sets overlap fully."""
    first, second = set(a), set(b)
    return len(first & second) / len(first | second) if first | second else 1.0
