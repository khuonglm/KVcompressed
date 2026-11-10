"""The confirmatory analysis on simulated runs with planted effects."""

import json
from dataclasses import asdict

import numpy as np
import pytest

from kvsel.confirm_analysis import (
    apply_families,
    clustered_ols,
    confirm_report,
    dataset_results,
    jaccard,
    margin_variant,
)
from kvsel.runner import cell_name, pilot_units

FORMS = ("lower", "capitalized", "upper")


def scores(m):
    return {"true": {f: [m, 0.0] for f in FORMS}, "alt": {f: [0.0, 0.0] for f in FORMS}}


def make_run(tmp_path, name="run", n=120, n_control=40, effect=0.3, noise=0.05, seed=0, subset=True,
             blind_margin=-1.0, shift=10, alt_rate=0.0, slope=0.0):
    """Items two per image. S_H^(mis) = effect + noise; every other estimand is 0 by construction."""
    generator = np.random.default_rng(seed)
    run = tmp_path / name
    (run / "items").mkdir(parents=True)
    manifest = []
    for i in range(n):
        item_id, image_id = f"it{i}", f"img{i // 2}"
        control = i < n_control
        a, b_mis, b_match = 2 + generator.normal(0, 0.5), 1 + generator.normal(0, 0.5), 2 + generator.normal(0, 0.5)
        item_shift = shift if not slope else int(generator.integers(0, 60))
        distance = 1 - len(set(range(144)) & set(range(item_shift, item_shift + 144))) / len(set(range(144)) | set(range(item_shift, item_shift + 144)))
        values = {
            "A": a, "B_mis": b_mis, "C_mis": b_mis + effect + slope * distance + generator.normal(0, noise), "D_mis": a,
            "B_match": b_match, "C_match": b_match, "D_match": a, "B_irr": b_match, "C_irr": b_match, "D_irr": a,
            "Q": a + 1, "mis": b_mis + 1, "match": b_match + 1, "irr": b_match + 1,
        }
        units = pilot_units() if control else pilot_units(("mis",))
        rows = []
        for unit in units:
            cell = cell_name(unit)
            kept = None
            if unit.selector != "full":
                start = item_shift if unit.selection == "mis" else 0
                kept = list(range(start, start + 144))
            outcome = "correct"
            if cell == "B_mis" and generator.random() < alt_rate:
                outcome = "alt"
            rows.append({**asdict(unit), "cell": cell, "kept": kept, "scores": scores(values[cell]),
                         "outcome": outcome, "truncated": False, "greedy_text": "Red"})
        rows.append({"selector": "blind", "k": 576, "reserve": False, "selection": "none", "answering": "Q",
                     "cell": "Q", "kept": None, "scores": scores(blind_margin), "outcome": "other", "truncated": False})
        record = {"item_id": item_id, "image_id": image_id, "rows": rows}
        if subset:
            record["subset"] = "control" if control else "main"
        (run / "items" / f"{item_id}.json").write_text(json.dumps(record))
        manifest.append({"item_id": item_id, "image_size": {"width": 336, "height": 336},
                         "evidence_box": {"x": 0, "y": 0, "w": 112, "h": 28}})  # rows 0-1, columns 0-7
    path = tmp_path / f"{name}.jsonl"
    path.write_text("".join(json.dumps(m) + "\n" for m in manifest))
    return run, path


def test_margin_variants():
    s = {"true": {"capitalized": [-1.0, -0.5], "lower": [-3.0, -0.5]}, "alt": {"capitalized": [-2.0, -1.0], "lower": [-2.0, -1.0]}}
    assert margin_variant(s, "primary") == pytest.approx(1.5)
    assert margin_variant(s, "content_only") == pytest.approx(1.0)
    assert margin_variant(s, "length_normalised") == pytest.approx(0.75)
    # identical surface forms count once: alt's two forms are the same string of scores
    assert margin_variant(s, "logsum") == pytest.approx(np.logaddexp(-1.5, -3.5) - (-3.0))


def test_recovers_planted_effects(tmp_path):
    run, manifest = make_run(tmp_path, effect=0.3)
    r = dataset_results(run, manifest, resamples=2000)
    assert not r["rehearsal"] and r["items"] == 120 and r["control_items"] == 40 and r["images"] == 60
    tests = r["tests"]
    assert tests["S_H^(mis)"]["estimate"] == pytest.approx(0.3, abs=0.02)
    assert tests["S_H^(mis)"]["ci95"][0] > 0.25 and tests["S_H^(mis)"]["items"] == 120
    assert tests["T_match"]["estimate"] == pytest.approx(0.3, abs=0.03) and tests["T_match"]["items"] == 40
    assert tests["T_irr"]["items"] == 40
    assert tests["E_harm^(mis)"]["estimate"] == pytest.approx(0.0, abs=1e-9)
    assert tests["S_Q^(mis)"]["estimate"] == pytest.approx(0.0, abs=1e-9)
    assert r["exploratory"]["S_H^(match)"]["items"] == 40 and r["exploratory"]["S_H^(match)"]["estimate"] == pytest.approx(0)
    assert not r["equivalence"]["equivalent"]
    assert r["sensitivity"]["content_only"]["estimate"] == pytest.approx(0.3, abs=0.02)
    assert r["sensitivity"]["length_normalised"]["estimate"] == pytest.approx(0.15, abs=0.01)
    assert r["mitigation"]["isolation (C_mis)"]["estimate"] == pytest.approx(tests["S_H^(mis)"]["estimate"])
    assert r["checks"] == {"full_q_no_longer_eligible": 0, "blind_rows": 120, "blind_now_eligible": 0,
                           "blind_mean_margin": -1.0, "truncated_share": 0.0}


def test_null_effect_is_equivalent(tmp_path):
    run, manifest = make_run(tmp_path, effect=0.0, noise=0.05)
    r = dataset_results(run, manifest, resamples=2000)
    low, high = r["tests"]["S_H^(mis)"]["ci95"]
    assert low < 0 < high and r["equivalence"]["equivalent"]


def test_overlap_retention_and_regression(tmp_path):
    run, manifest = make_run(tmp_path, shift=10)
    r = dataset_results(run, manifest, resamples=500)
    expected = jaccard(set(range(144)), set(range(10, 154)))
    assert r["overlap"]["mis"]["mean_jaccard"] == pytest.approx(expected)
    # Evidence rows 0-1, columns 0-7: patches 0-7 and 24-31. S_Q keeps all 16; S_mis (10-153) drops 0-7.
    assert r["grounding_items"] == 40 and r["H6"]["primary"]["items"] == 40
    assert r["H6"]["primary"]["estimate"] == pytest.approx((16 - 8) / 16)
    assert r["H5"]["slope"] is None  # every item has the same Jaccard: no slope to estimate


def test_behaviour_paired_difference(tmp_path):
    run, manifest = make_run(tmp_path, alt_rate=0.5, seed=3)
    r = dataset_results(run, manifest, resamples=500)
    share_alt_b = r["behaviour_rates"]["pw|B_mis"]["alt"]
    assert r["behaviour_paired"]["alt: C_mis - B_mis"]["estimate"] == pytest.approx(-share_alt_b)
    assert r["behaviour_paired"]["correct: C_mis - B_mis"]["estimate"] == pytest.approx(share_alt_b)


def test_checks_flag_blind_and_full_q_problems(tmp_path):
    run, manifest = make_run(tmp_path, n=10, n_control=4, blind_margin=1.0)
    for path in (run / "items").glob("*.json"):
        record = json.loads(path.read_text())
        for row in record["rows"]:
            if row["selector"] == "blind":
                row["outcome"] = "correct"
        path.write_text(json.dumps(record))
    r = dataset_results(run, manifest, resamples=200)
    assert r["checks"]["blind_now_eligible"] == 10 and r["checks"]["blind_mean_margin"] == 1.0


def test_incomplete_run_is_refused(tmp_path):
    run, manifest = make_run(tmp_path, n=10, n_control=4)
    with pytest.raises(SystemExit, match="frozen main set has 1000"):
        dataset_results(run, manifest, resamples=100, expected_items=1000)


def test_development_runs_are_rehearsals(tmp_path):
    run, manifest = make_run(tmp_path, n=10, n_control=4, subset=False)
    r = dataset_results(run, manifest, resamples=100)
    assert r["rehearsal"] and r["control_items"] == 10


def test_holm_is_pending_until_both_datasets_ran(tmp_path):
    gqa = dataset_results(*make_run(tmp_path, "g", effect=0.3), resamples=1000)
    families = apply_families({"gqa": gqa})
    assert not families["H1"]["complete"] and families["H1"]["reject"] is None
    assert set(families["H1"]["p_values"]) == {"S_H^(mis) [gqa]"}
    textvqa = dataset_results(*make_run(tmp_path, "t", effect=0.0, seed=1), resamples=1000)
    families = apply_families({"gqa": gqa, "textvqa": textvqa})
    assert families["H1"]["complete"]
    assert families["H1"]["reject"] == {"S_H^(mis) [gqa]": True, "S_H^(mis) [textvqa]": False}
    assert len(families["H2"]["p_values"]) == 4 and len(families["H3_H4"]["p_values"]) == 4


def test_clustered_ols():
    generator = np.random.default_rng(0)
    x = generator.uniform(0, 1, 400)
    clusters = np.repeat(np.arange(200), 2)
    y = 0.5 + 2.0 * x + generator.normal(0, 0.1, 400) + np.repeat(generator.normal(0, 0.1, 200), 2)
    result = clustered_ols(y, x, clusters)
    slope, intercept = np.polyfit(x, y, 1)
    assert result["slope"] == pytest.approx(slope) and result["intercept"] == pytest.approx(intercept)
    assert result["ci95"][0] < 2.0 < result["ci95"][1] and result["images"] == 200 and result["p_two_sided"] < 1e-6
    # With singleton clusters the CR1 variance equals the HC1 sandwich.
    single = clustered_ols(y, x, np.arange(400))
    design = np.column_stack([np.ones(400), x])
    bread = np.linalg.inv(design.T @ design)
    residuals = y - design @ np.array([single["intercept"], single["slope"]])
    hc = bread @ (design.T * residuals**2) @ design @ bread * (400 / 399) * (399 / 398)
    assert single["se"] == pytest.approx(np.sqrt(hc[1, 1]))


def test_report_files(tmp_path):
    run, manifest = make_run(tmp_path, n=20, n_control=6)
    out = confirm_report({"gqa": (run, manifest)}, tmp_path / "out", resamples=200)
    saved = json.loads((tmp_path / "out" / "results.json").read_text())
    assert saved["datasets"]["gqa"]["items"] == 20 and saved["families"]["H1"]["complete"] is False
    text = (tmp_path / "out" / "report.md").read_text()
    assert text.startswith("# Confirmatory report") and "pending (family incomplete)" in text and "## gqa" in text
    assert out["datasets"]["gqa"]["tests"]["S_H^(mis)"]["items"] == 20
    with pytest.raises(SystemExit, match="unknown datasets"):
        confirm_report({"vqa": (run, manifest)}, tmp_path / "x", resamples=10)


def test_h5_recovers_a_planted_slope(tmp_path):
    run, manifest = make_run(tmp_path, n=300, n_control=40, effect=0.0, slope=2.0, noise=0.05)
    h5 = dataset_results(run, manifest, resamples=200)["H5"]
    assert h5["slope"] == pytest.approx(2.0, abs=0.1) and h5["ci95"][0] > 1.5 and h5["images"] == 150


def test_record_checks_code_and_completeness(tmp_path):
    from kvsel.confirm import code_hash_all

    run, manifest = make_run(tmp_path, n=10, n_control=4)
    record = tmp_path / "record.json"
    record.write_text(json.dumps({"code_sha256": code_hash_all(), "main_items": 10}))
    confirm_report({"gqa": (run, manifest)}, tmp_path / "ok", resamples=50, records={"gqa": record})
    record.write_text(json.dumps({"code_sha256": "0" * 64, "main_items": 10}))
    with pytest.raises(SystemExit, match="code changed"):
        confirm_report({"gqa": (run, manifest)}, tmp_path / "bad", resamples=50, records={"gqa": record})


def test_run_without_misleading_cells_is_refused(tmp_path):
    run = tmp_path / "empty"
    (run / "items").mkdir(parents=True)
    (run / "items" / "a.json").write_text(json.dumps({"item_id": "a", "image_id": "i", "rows": [
        {"selector": "full", "reserve": False, "cell": "Q", "kept": None, "selection": "none", "answering": "Q",
         "scores": scores(1.0), "outcome": "correct", "truncated": False}]}))
    with pytest.raises(SystemExit, match="lacks the misleading-condition cells"):
        dataset_results(run, tmp_path / "m.jsonl", resamples=10)
