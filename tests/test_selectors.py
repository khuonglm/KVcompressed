"""E6: selectors on hand-built attention. See tests/README.md."""

import numpy as np
import pytest

from kvsel.selectors import (
    coverage_order, coverage_radius, jaccard, pool_3x3, pw_attn, question_span_attn, reserve_size,
    top_k, uniform_grid, with_reserve,
)


def test_top_k_known_answer_and_ties():
    scores = np.array([0.1, 0.9, 0.5, 0.9, 0.5, 0.2])
    assert top_k(scores, 2).tolist() == [1, 3]
    assert top_k(scores, 3).tolist() == [1, 2, 3]  # the 0.5 tie goes to the lower index


def test_pool_interior_and_edges():
    values = np.zeros(576)
    values[0] = 9.0
    pooled = pool_3x3(values).reshape(24, 24)
    assert pooled[0, 0] == pytest.approx(9 / 4)  # corner: 4 in-image neighbours
    assert pooled[0, 1] == pytest.approx(9 / 6)  # edge: 6
    assert pooled[1, 1] == pytest.approx(9 / 9)  # interior: 9
    assert pooled[2, 2] == 0


def test_pool_of_a_constant_is_that_constant():
    assert np.allclose(pool_3x3(np.full(576, 2.5)), 2.5)


def test_pw_attn_uses_only_the_last_window_tokens():
    attention = np.zeros((40, 576))
    attention[:24, 0] = 1.0  # outside the window: must have no effect
    attention[24:, 300] = 1.0
    scores = pw_attn(attention, window_size=16)
    assert scores[0] == 0 and scores[300] == pytest.approx(1 / 9)


def test_question_span_uses_only_the_span():
    attention = np.zeros((40, 576))
    attention[10:13, 300] = 1.0
    attention[13:, 0] = 1.0
    assert question_span_attn(attention, (10, 13))[300] == pytest.approx(1 / 9)
    assert question_span_attn(attention, (10, 13))[0] == 0


@pytest.mark.parametrize("k", [288, 144, 72])
def test_uniform_grid_hits_k_and_nests(k):
    kept = uniform_grid(k)
    assert len(kept) == k == len(set(kept.tolist()))
    if k < 288:
        assert set(kept.tolist()) <= set(uniform_grid(k * 2).tolist())


def test_coverage_order_is_a_permutation_with_known_start():
    order = coverage_order()
    assert sorted(order) == list(range(576))
    assert order[0] == 11 * 24 + 11
    assert order[1] == 23 * 24 + 23  # (23, 23) is the farthest cell from (11, 11)


def test_coverage_prefixes_are_spread():
    order = np.array(coverage_order())
    generator = np.random.default_rng(0)
    for size in (7, 14, 29):
        random_radii = [coverage_radius(generator.choice(576, size, replace=False)) for _ in range(50)]
        assert coverage_radius(order[:size]) < np.mean(random_radii)


@pytest.mark.parametrize("k,expected", [(288, 29), (144, 14), (72, 7)])
def test_reserve_sizes(k, expected):
    assert reserve_size(k) == expected


@pytest.mark.parametrize("k", [288, 144, 72])
def test_reserve_keeps_exactly_k_unique_and_top_scores(k):
    scores = np.random.default_rng(1).random(576)
    kept = with_reserve(scores, k)
    assert len(kept) == k == len(set(kept.tolist()))
    assert set(top_k(scores, k - reserve_size(k)).tolist()) <= set(kept.tolist())


def test_reserve_skips_entries_already_kept():
    order = coverage_order()
    scores = np.zeros(576)
    scores[list(order[:130])] = 1.0  # the top 130 already contain the reserve's first choices
    kept = with_reserve(scores, 144)
    assert len(kept) == 144 and set(order[:144]) == set(kept.tolist())


def test_jaccard():
    assert jaccard(np.array([1, 2]), np.array([1, 2])) == 1
    assert jaccard(np.array([1, 2]), np.array([2, 3])) == pytest.approx(1 / 3)
