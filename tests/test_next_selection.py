"""NeXT selection by share: pooling within each map, budgets, newlines, nesting. Pure NumPy. See tests/README.md."""

import numpy as np
import pytest

from kvsel.selectors import pool_3x3
from kvsel.models.llava15 import LAYOUT as SINGLE_MAP
from kvsel.models.llava_next import anyres_layout
from kvsel_next.selection import SHARES, kept_set, kept_sets, pooled_scores

WIDE, TALL = anyres_layout(24, 24, 44), anyres_layout(24, 44, 24)
MAPS = {"base": 0, "high": 1}


def index(layout, map_, row, col):
    return int(layout.positions(MAPS[map_])[row, col])


def spiked(layout, at, value=100.0):
    scores = np.zeros(layout.total)
    scores[at] = value
    return scores


def test_shares():
    assert SHARES == (0.125, 0.25, 0.5)


def test_single_24x24_map_pools_as_kvsel():
    values = np.random.default_rng(0).random(576)
    assert np.allclose(pooled_scores(SINGLE_MAP, values), pool_3x3(values))


def test_a_24x24_high_resolution_map_pools_as_kvsel():
    layout = anyres_layout(24, 24, 24)
    values = np.random.default_rng(1).random(layout.total)
    high = layout.positions(1).ravel()
    assert np.allclose(pooled_scores(layout, values)[high], pool_3x3(values[high]))


@pytest.mark.parametrize("layout", [WIDE, TALL])
def test_spike_at_the_end_of_the_base_map_stays_in_it(layout):
    pooled = pooled_scores(layout, spiked(layout, 575))  # base (23, 23), flat-adjacent to high (0, 0)
    assert pooled[index(layout, "high", 0, 0)] == 0
    assert pooled[575] == pytest.approx(100 / 4)  # corner: 4 in-bounds cells
    assert pooled[574] == pytest.approx(100 / 6)  # edge: 6 in-bounds cells


@pytest.mark.parametrize("layout", [WIDE, TALL])
def test_spike_at_the_start_of_the_high_resolution_map_stays_in_it(layout):
    pooled = pooled_scores(layout, spiked(layout, index(layout, "high", 0, 0)))
    assert pooled[575] == 0
    assert pooled[index(layout, "high", 0, 0)] == pytest.approx(100 / 4)


@pytest.mark.parametrize("layout", [WIDE, TALL])
def test_spike_before_a_newline_does_not_reach_the_next_row_start(layout):
    rows, cols = layout.maps[1].rows, layout.maps[1].cols
    spike = index(layout, "high", 3, cols - 1)
    pooled = pooled_scores(layout, spiked(layout, spike))
    assert pooled[index(layout, "high", 4, 0)] == 0  # flat index spike + 2, across the newline
    assert pooled[index(layout, "high", 4, cols - 1)] == pytest.approx(100 / 6)  # spatial neighbour below


@pytest.mark.parametrize("layout", [WIDE, TALL])
def test_newline_scores_reach_nothing(layout):
    pooled = pooled_scores(layout, spiked(layout, list(layout.newlines)))
    features = np.setdiff1d(np.arange(layout.total), layout.newlines)
    assert np.all(pooled[features] == 0)


# k = share x feature tokens, worked by hand. Features = 576 + rows x cols.
BUDGETS = [
    (WIDE, (204, 408, 816)),  # 1632 features
    (TALL, (204, 408, 816)),  # 1632
    (anyres_layout(24, 48, 48), (360, 720, 1440)),  # 2880
    (anyres_layout(24, 24, 70), (282, 564, 1128)),  # 2256
]


@pytest.mark.parametrize("layout,expected", BUDGETS)
def test_budget_newlines_and_nesting(layout, expected):
    scores = np.random.default_rng(2).random(layout.total)
    scores[list(layout.newlines)] = -1.0  # newlines kept even when they score lowest...
    sets = kept_sets(layout, np.tile(scores, (16, 1)))
    newlines = set(layout.newlines)
    for share, k in zip(SHARES, expected):
        kept = sets[share]
        assert newlines <= set(kept.tolist())
        assert len(kept) - len(newlines) == k
    assert set(sets[0.125]) <= set(sets[0.25]) <= set(sets[0.5])


def test_a_share_that_is_not_a_whole_k_is_refused():
    with pytest.raises(ValueError, match="not whole"):
        kept_set(anyres_layout(24, 2, 2), np.zeros(582), 0.125)  # 580 features: 72.5; no real image has this


@pytest.mark.parametrize("layout", [WIDE, TALL])
def test_newlines_are_never_among_the_k_selected(layout):
    scores = spiked(layout, list(layout.newlines), 1e9)  # ...and never selected when they score highest
    kept = kept_set(layout, scores, 0.125)
    assert len(kept) == 204 + len(layout.newlines)  # 1632 features at 12.5%


def test_kept_set_is_sorted_int_indices_within_the_span():
    kept = kept_set(WIDE, np.random.default_rng(3).random(WIDE.total), 0.25)
    assert kept.dtype.kind == "i" and np.all(np.diff(kept) > 0)
    assert kept[0] >= 0 and kept[-1] < WIDE.total


def test_non_finite_feature_scores_are_refused():
    scores = np.zeros(WIDE.total)
    scores[600] = np.nan
    with pytest.raises(ValueError, match="finite"):
        kept_set(WIDE, scores, 0.25)
    scores[600] = 0
    scores[list(WIDE.newlines)] = np.nan  # newline scores are never used
    assert len(kept_set(WIDE, scores, 0.25)) == 408 + 24


def test_ties_go_to_the_lower_index():
    kept = kept_set(SINGLE_MAP, np.zeros(576), 0.125)
    assert kept.tolist() == list(range(72))
