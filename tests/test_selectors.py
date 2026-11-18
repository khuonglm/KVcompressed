"""E6: selection on both families' layouts, through the one kept-set entry point. Pure NumPy. See tests/README.md."""

import functools
import hashlib
import json

import numpy as np
import pytest

from kvsel.models.family import Tokens
from kvsel.models.llava15 import LAYOUT
from kvsel.models.llava_next import anyres_layout
from kvsel.runner import Unit, kept_for
from kvsel.selectors import (
    Budget,
    coverage_order,
    jaccard,
    kept_set,
    pooled_scores,
    pw_attn,
    question_span_attn,
    reserve_size,
    top_k,
    uniform_grid,
    with_reserve,
)
from kvsel.studies.followups import run as followup
from kvsel.studies.next.comparative import SHARES
from kvsel.validation import KINDS, concentrated_set, validation_set

WIDE, TALL = anyres_layout(24, 24, 44), anyres_layout(24, 44, 24)
KS = (288, 144, 72)


def pw_per_share(layout, attention):
    """Every share's PW-attn kept set from one probe."""
    return {share: kept_set(layout, "pw", Budget(share=share), attention=attention) for share in SHARES}


def pool_one_map(values):
    return pooled_scores(LAYOUT, values)


# --- Kept sets are the ones the code gave before selection moved onto layouts ---------------------------------

# Digests captured from the dispatchers before ticket 05 (runner.kept_for, the follow-ups' kept_for and random_set,
# validate.kept_sets, the NeXT package's selection.kept_sets and validate.sparse_set), on the inputs below; the
# validation sets are now `kvsel.validation.validation_set`.
KNOWN = {
    "llava15 pw reserve=False": "812e06081421f64d",
    "llava15 pw reserve=True": "55b088d8bbe0a1ed",
    "llava15 qspan reserve=False": "e8d417d1e12a28ed",
    "llava15 qspan reserve=True": "d6df704593095b77",
    "llava15 uniform": "a38ecabcfb6a820f",
    "followup lt": "67ca8fe83a8e435c",
    "followup pw": "812e06081421f64d",
    "followup random": "c5d5e745c5d8bdd2",
    "validate": "2d2979bafea2db2d",
    "next pw wide": "9b8cf57bedcad722",
    "next sparse wide": "4d87978ee4420175",
    "next pw tall": "7fcbbb550a1f500a",
    "next sparse tall": "1ef28692af645182",
    "next pw square": "cc4d2747161c8458",
    "next sparse square": "78af5eca946fa228",
    "next pw long": "51faae1499598911",
    "next sparse long": "4dc0f87ff56affbb",
    "next pw four_three": "67f139b3f7d64f55",
    "next sparse four_three": "517ce69b3e7c468c",
}
NEXT_GRIDS = {"wide": (24, 44), "tall": (44, 24), "square": (48, 48), "long": (24, 70), "four_three": (36, 48)}
ATTENTION = np.random.default_rng(7).random((40, 576)).astype(np.float32)
LAYER2 = np.random.default_rng(8).random(576).astype(np.float32)


def digest(sets) -> str:
    return hashlib.sha256(json.dumps([np.asarray(s).tolist() for s in sets]).encode()).hexdigest()[:16]


def llava15_sets(selector, reserve):
    tokens = {"Q": Tokens("p", [], [], (10, 13))}
    return [kept_for(Unit(selector, k, reserve, "Q", "Q"), LAYOUT, {"Q": ATTENTION}, tokens) for k in KS]


@functools.cache
def computed() -> dict[str, str]:
    out = {f"llava15 {s} reserve={r}": digest(llava15_sets(s, r)) for s in ("pw", "qspan") for r in (False, True)}
    out["llava15 uniform"] = digest(kept_set(LAYOUT, "uniform", Budget(k=k)) for k in KS)
    out["followup lt"] = digest(followup.followup_kept_set(Unit("lt", k, False, "Q", "Q", followup=True), "i1", LAYOUT, {}, {"Q": LAYER2})
                                for k in KS)
    out["followup pw"] = digest(followup.followup_kept_set(Unit("pw", k, False, "Q", "Q", followup=True), "i1", LAYOUT, {"Q": ATTENTION}, {})
                                for k in KS)
    out["followup random"] = digest(
        followup.followup_kept_set(Unit("random", k, False, "none", "Q", seed=s, followup=True), item, LAYOUT, {}, {})
        for item in ("i1", "18182570") for s in range(5) for k in KS)
    out["validate"] = digest(validation_set(LAYOUT, Budget(k=k), kind, s, ATTENTION) for k in KS for s in range(4)
                             for kind in KINDS)
    for name, (rows, cols) in NEXT_GRIDS.items():
        layout = anyres_layout(24, rows, cols)
        attention = np.random.default_rng(rows * 100 + cols).random((30, layout.total)).astype(np.float32)
        out[f"next pw {name}"] = digest(pw_per_share(layout, attention).values())
        out[f"next sparse {name}"] = digest(validation_set(layout, Budget(share=share), kind, seed) for share in SHARES
                                            for kind in ("random", "concentrated") for seed in range(3))
    return out


@pytest.mark.parametrize("case", KNOWN)
def test_kept_sets_are_unchanged(case):
    assert computed()[case] == KNOWN[case]


# --- Scores and top-k -----------------------------------------------------------------------------------------


def test_top_k_known_answer_and_ties():
    scores = np.array([0.1, 0.9, 0.5, 0.9, 0.5, 0.2])
    assert top_k(scores, 2).tolist() == [1, 3]
    assert top_k(scores, 3).tolist() == [1, 2, 3]  # the 0.5 tie goes to the lower index


def test_pool_interior_and_edges():
    values = np.zeros(576)
    values[0] = 9.0
    pooled = pool_one_map(values).reshape(24, 24)
    assert pooled[0, 0] == pytest.approx(9 / 4)  # corner: 4 in-image neighbours
    assert pooled[0, 1] == pytest.approx(9 / 6)  # edge: 6
    assert pooled[1, 1] == pytest.approx(9 / 9)  # interior: 9
    assert pooled[2, 2] == 0


def test_pool_of_a_constant_is_that_constant():
    assert np.allclose(pool_one_map(np.full(576, 2.5)), 2.5)


def test_pw_attn_uses_only_the_last_window_tokens():
    attention = np.zeros((40, 576))
    attention[:24, 0] = 1.0  # outside the window: must have no effect
    attention[24:, 300] = 1.0
    scores = pw_attn(LAYOUT, attention, window_size=16)
    assert scores[0] == 0 and scores[300] == pytest.approx(1 / 9)


def test_question_span_uses_only_the_span():
    attention = np.zeros((40, 576))
    attention[10:13, 300] = 1.0
    attention[13:, 0] = 1.0
    assert question_span_attn(LAYOUT, attention, (10, 13))[300] == pytest.approx(1 / 9)
    assert question_span_attn(LAYOUT, attention, (10, 13))[0] == 0


def test_lt_early_scores_are_unpooled():
    layer2 = np.zeros(576)
    layer2[5] = 1.0
    assert kept_set(LAYOUT, "lt", Budget(k=1), layer2=layer2).tolist() == [5]


def test_full_keeps_every_position():
    assert kept_set(LAYOUT, "full", Budget(k=576)).tolist() == list(range(576))
    assert kept_set(WIDE, "full", Budget(share=1.0)).tolist() == list(range(WIDE.total))


# --- Fixed patterns on LLaVA-1.5's layout ----------------------------------------------------------------------


@pytest.mark.parametrize("k", KS)
def test_uniform_grid_hits_k_and_nests(k):
    kept = uniform_grid(LAYOUT, k)
    assert len(kept) == k == len(set(kept.tolist()))
    if k < 288:
        assert set(kept.tolist()) <= set(uniform_grid(LAYOUT, k * 2).tolist())


def test_coverage_order_is_a_permutation_with_known_start():
    order = coverage_order(LAYOUT)
    assert sorted(order) == list(range(576))
    assert order[0] == 11 * 24 + 11
    assert order[1] == 23 * 24 + 23  # (23, 23) is the farthest cell from (11, 11)


def coverage_radius(kept) -> float:
    """Largest distance from any grid cell to its nearest kept cell."""
    rows, cols = np.divmod(np.arange(576), 24)
    kept_rows, kept_cols = np.divmod(np.asarray(kept), 24)
    return float(np.sqrt(((rows[:, None] - kept_rows) ** 2 + (cols[:, None] - kept_cols) ** 2).min(axis=1).max()))


def test_coverage_prefixes_are_spread():
    order = np.array(coverage_order(LAYOUT))
    generator = np.random.default_rng(0)
    for size in (7, 14, 29):
        random_radii = [coverage_radius(generator.choice(576, size, replace=False)) for _ in range(50)]
        assert coverage_radius(order[:size]) < np.mean(random_radii)


@pytest.mark.parametrize("k,expected", [(288, 29), (144, 14), (72, 7)])
def test_reserve_sizes(k, expected):
    assert reserve_size(k) == expected


@pytest.mark.parametrize("k", KS)
def test_reserve_keeps_exactly_k_unique_and_top_scores(k):
    scores = np.random.default_rng(1).random(576)
    kept = with_reserve(LAYOUT, scores, k)
    assert len(kept) == k == len(set(kept.tolist()))
    assert set(top_k(scores, k - reserve_size(k)).tolist()) <= set(kept.tolist())


def test_reserve_skips_entries_already_kept():
    order = coverage_order(LAYOUT)
    scores = np.zeros(576)
    scores[list(order[:130])] = 1.0  # the top 130 already contain the reserve's first choices
    kept = with_reserve(LAYOUT, scores, 144)
    assert len(kept) == 144 and set(order[:144]) == set(kept.tolist())


@pytest.mark.parametrize("call", [
    lambda: uniform_grid(WIDE, 144),
    lambda: coverage_order(WIDE),
    lambda: kept_set(WIDE, "pw", Budget(k=144), attention=np.zeros((16, WIDE.total)), reserve=True),
])
def test_fixed_patterns_refuse_other_layouts(call):
    with pytest.raises(ValueError, match="single map without newlines"):
        call()


@pytest.mark.parametrize("selector,missing", [("pw", "attention"), ("qspan", "question_span"), ("lt", "layer2"),
                                              ("random", "rng")])
def test_a_selector_without_its_input_is_refused(selector, missing):
    with pytest.raises(ValueError, match=f"needs.*{missing}"):
        kept_set(LAYOUT, selector, Budget(k=72), attention=np.zeros((16, 576)) if selector == "qspan" else None)


def test_concentrated_is_a_validation_kind_not_a_selector():
    with pytest.raises(ValueError, match="unknown selector"):
        kept_set(WIDE, "concentrated", Budget(k=72))
    kept = concentrated_set(WIDE, 72)
    assert kept.tolist() == sorted(set(WIDE.features[:72].tolist()) | set(WIDE.newlines))


def test_reserve_applies_only_to_scored_selectors():
    with pytest.raises(ValueError, match="reserve"):
        kept_set(LAYOUT, "uniform", Budget(k=144), reserve=True)


# --- Budgets ---------------------------------------------------------------------------------------------------


def test_a_budget_is_k_or_a_share():
    with pytest.raises(ValueError):
        Budget()
    with pytest.raises(ValueError):
        Budget(k=144, share=0.25)
    assert Budget(k=144).resolve(LAYOUT) == Budget(share=0.25).resolve(LAYOUT) == 144


def test_a_share_that_is_not_a_whole_k_is_refused():
    with pytest.raises(ValueError, match="not whole"):
        Budget(share=0.125).resolve(anyres_layout(24, 2, 2))  # 580 features: 72.5; no real image has this


def test_k_beyond_the_feature_tokens_is_refused():
    with pytest.raises(ValueError, match="outside"):
        Budget(k=577).resolve(LAYOUT)


# --- NeXT's layout: pooling within each map, newlines kept and outside k ---------------------------------------

MAPS = {"base": 0, "high": 1}


def index(layout, map_, row, col):
    return int(layout.positions(MAPS[map_])[row, col])


def spiked(layout, at, value=100.0):
    scores = np.zeros(layout.total)
    scores[at] = value
    return scores


def test_a_24x24_high_resolution_map_pools_as_the_single_map():
    layout = anyres_layout(24, 24, 24)
    values = np.random.default_rng(1).random(layout.total)
    high = layout.positions(1).ravel()
    assert np.array_equal(pooled_scores(layout, values)[high], pool_one_map(values[high]))


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
    cols = layout.maps[1].cols
    pooled = pooled_scores(layout, spiked(layout, index(layout, "high", 3, cols - 1)))
    assert pooled[index(layout, "high", 4, 0)] == 0  # flat index spike + 2, across the newline
    assert pooled[index(layout, "high", 4, cols - 1)] == pytest.approx(100 / 6)  # spatial neighbour below


@pytest.mark.parametrize("layout", [WIDE, TALL])
def test_newline_scores_reach_nothing(layout):
    pooled = pooled_scores(layout, spiked(layout, list(layout.newlines)))
    assert np.all(pooled[layout.features] == 0)


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
    sets = pw_per_share(layout, np.tile(scores, (16, 1)))
    newlines = set(layout.newlines)
    for share, k in zip(SHARES, expected, strict=True):
        assert newlines <= set(sets[share].tolist())
        assert len(sets[share]) - len(newlines) == k
    assert set(sets[0.125]) <= set(sets[0.25]) <= set(sets[0.5])


@pytest.mark.parametrize("layout", [WIDE, TALL])
def test_newlines_are_never_among_the_k_selected(layout):
    attention = np.tile(spiked(layout, list(layout.newlines), 1e9), (16, 1))  # ...nor selected when they score highest
    kept = kept_set(layout, "pw", Budget(share=0.125), attention=attention)
    assert len(kept) == 204 + len(layout.newlines)  # 1632 features at 12.5%


@pytest.mark.parametrize("selector", ["random", "uniform"])
def test_every_selector_keeps_the_newlines(selector):
    layout = WIDE if selector != "uniform" else LAYOUT
    kept = kept_set(layout, selector, Budget(k=72), rng=np.random.default_rng(0))
    assert set(layout.newlines) <= set(kept.tolist()) and len(kept) == 72 + len(layout.newlines)


def test_kept_set_is_sorted_int_indices_within_the_span():
    kept = kept_set(WIDE, "pw", Budget(share=0.25), attention=np.random.default_rng(3).random((16, WIDE.total)))
    assert kept.dtype.kind == "i" and np.all(np.diff(kept) > 0)
    assert kept[0] >= 0 and kept[-1] < WIDE.total


def test_non_finite_feature_scores_are_refused():
    attention = np.zeros((16, WIDE.total))
    attention[:, 600] = np.nan
    with pytest.raises(ValueError, match="finite"):
        kept_set(WIDE, "pw", Budget(share=0.25), attention=attention)
    attention[:, 600] = 0
    attention[:, list(WIDE.newlines)] = np.nan  # newline scores are never used
    assert len(kept_set(WIDE, "pw", Budget(share=0.25), attention=attention)) == 408 + 24


def test_ties_go_to_the_lower_index():
    assert kept_set(LAYOUT, "pw", Budget(share=0.125), attention=np.zeros((16, 576))).tolist() == list(range(72))


# --- Overlap ---------------------------------------------------------------------------------------------------


def test_jaccard():
    assert jaccard(np.array([1, 2]), np.array([1, 2])) == 1
    assert jaccard(np.array([1, 2]), np.array([2, 3])) == pytest.approx(1 / 3)
    assert jaccard({1, 2}, {2, 3}) == pytest.approx(1 / 3)
    assert jaccard(set(), set()) == 1.0  # two empty sets overlap fully
