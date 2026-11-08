"""Analysis on simulated data with known answers."""

import numpy as np
import pandas as pd
import pytest

from kvsel.analysis import cluster_bootstrap, estimands, holm, margin, read_interval


def scores(true_lower, alt_lower, true_cap=None, alt_cap=None):
    return {
        "true": {"lower": true_lower, "capitalized": true_cap if true_cap is not None else true_lower},
        "alt": {"lower": alt_lower, "capitalized": alt_cap if alt_cap is not None else alt_lower},
    }


def test_margin_sums_tokens_including_eos():
    assert margin(scores([-1.0, -0.5], [-3.0])) == pytest.approx(1.5)


def test_margin_uses_the_requested_surface():
    assert margin(scores([-1.0], [-3.0], [-4.0], [-2.0]), "capitalized") == pytest.approx(-2.0)


def test_logsum_adds_probability_mass_over_distinct_forms():
    value = margin(scores([np.log(0.2)], [np.log(0.1)], [np.log(0.3)], [np.log(0.1)]), "logsum")
    assert value == pytest.approx(np.log(0.5) - np.log(0.1))  # the alt forms are identical: counted once


def wide_frame():
    return pd.DataFrame({
        "pw|A": [1.0, 2.0], "pw|B_mis": [0.0, 1.0], "pw|C_mis": [0.5, 1.5], "pw|D_mis": [0.8, 1.9],
        "full|Q": [3.0, 3.0], "full|mis": [2.0, 2.5],
    }, index=["a", "b"])


def test_estimands_match_hand_calculation():
    result = estimands(wide_frame(), "pw", "mis")
    assert result["S_H"].tolist() == pytest.approx([0.5, 0.5])
    assert result["S_Q"].tolist() == pytest.approx([0.2, 0.1])
    assert result["I"].tolist() == pytest.approx([0.3, 0.4])
    assert result["E_harm"].tolist() == pytest.approx([0.0, 0.5])


def test_e_harm_decomposes_exactly():
    wide = wide_frame()
    result = estimands(wide, "pw", "mis")
    remainder = (wide["full|mis"] - wide["pw|C_mis"]) - (wide["full|Q"] - wide["pw|A"])
    assert (result["E_harm"] - (result["S_H"] + remainder)).abs().max() == pytest.approx(0)


def simulated(n_images, per_image, effect, image_sd, noise_sd, seed):
    generator = np.random.default_rng(seed)
    images = np.repeat(np.arange(n_images), per_image)
    values = effect + generator.normal(0, image_sd, n_images)[images] + generator.normal(0, noise_sd, len(images))
    return pd.Series(values), pd.Series(images)


def test_bootstrap_recovers_the_mean_and_detects_a_large_effect():
    values, images = simulated(200, 3, 1.0, 0.5, 0.5, 0)
    result = cluster_bootstrap(values, images, resamples=2000)
    assert result["estimate"] == pytest.approx(values.mean())
    assert result["ci95"][0] < 1.0 < result["ci95"][1]
    assert result["p_one_sided"] == pytest.approx(1 / 2001)
    assert result["p_two_sided"] == pytest.approx(2 / 2001)
    assert result["images"] == 200 and result["items"] == 600


def test_p_value_formula_on_a_null_is_not_extreme():
    values, images = simulated(200, 3, 0.0, 0.5, 0.5, 1)
    result = cluster_bootstrap(values, images, resamples=2000)
    assert 0.001 < result["p_one_sided"] < 0.999 and 0.002 < result["p_two_sided"] <= 1


def test_two_sided_p_detects_an_effect_in_either_direction():
    up, images = simulated(200, 3, 1.0, 0.5, 0.5, 0)
    assert cluster_bootstrap(up, images, resamples=2000)["p_two_sided"] == pytest.approx(2 / 2001)
    assert cluster_bootstrap(-up, images, resamples=2000)["p_two_sided"] == pytest.approx(2 / 2001)


def test_bootstrap_is_deterministic_for_a_seed():
    values, images = simulated(50, 2, 0.3, 0.2, 0.2, 2)
    assert cluster_bootstrap(values, images, 500) == cluster_bootstrap(values, images, 500)


def test_clustering_widens_the_interval_for_correlated_items():
    values, images = simulated(100, 5, 0.0, 1.0, 0.1, 3)
    clustered = cluster_bootstrap(values, images, resamples=2000)["se"]
    naive = cluster_bootstrap(values, pd.Series(np.arange(len(values))), resamples=2000)["se"]
    assert clustered > 1.5 * naive


def test_coverage_is_near_nominal():
    hits = 0
    for seed in range(200):
        values, images = simulated(80, 3, 0.2, 0.4, 0.4, 100 + seed)
        low, high = cluster_bootstrap(values, images, resamples=400)["ci95"]
        hits += low <= 0.2 <= high
    assert 0.88 <= hits / 200 <= 0.99


def test_holm_hand_worked():
    assert holm({"a": 0.01, "b": 0.04, "c": 0.03}) == {"a": True, "b": False, "c": False}
    assert holm({"a": 0.01, "b": 0.02}) == {"a": True, "b": True}
    assert holm({"a": 0.03, "b": 0.04}) == {"a": False, "b": False}


@pytest.mark.parametrize("low,high,expected", [
    (0.2, 0.5, "positive"),
    (-0.5, -0.2, "opposite direction"),
    (-0.05, 0.05, "practically small"),
    (-0.1, 0.4, "inconclusive"),
    (0.01, 0.06, "positive, practically small"),
    (-0.06, -0.01, "opposite direction, practically small"),
])
def test_interval_reading(low, high, expected):
    assert read_interval(low, high) == expected
