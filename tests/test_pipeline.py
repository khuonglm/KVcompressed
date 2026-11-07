"""End to end on a tiny random LLaVA with the real tokenizer: same classes and calls as the GPU run."""

import json

import numpy as np
import pytest
from PIL import Image

from kvsel.analysis import report
from kvsel.geometry import pad_to_square, PAD_COLOR
from kvsel.model import bundle_from_model, load_image_processor, load_tokenizer, pixel_values, prefill
from kvsel.prompts import tokenize_prompt
from kvsel.runner import Unit, cell_name, pilot_units, require_development, run_item, run_items, run_screen
from kvsel.selectors import uniform_grid
from kvsel.validate import noise_floor, run_e2 as e2, run_e3_e4 as e3_e4, run_e5 as e5, run_e7 as e7, validate

pytestmark = pytest.mark.network


def make_item(index):
    return {
        "item_id": f"q{index}", "image_id": f"img{index // 2}", "image_file": f"img{index // 2}.jpg",
        "dataset": "gqa", "split": "dev", "question": "What color is the bus?", "a_true": "red", "a_alt": "blue",
        "descriptions": {
            "mis": "The bus in this image is blue.", "match": "The bus in this image is red.",
            "irr": "The road in this image is gray.",
        },
    }


@pytest.fixture(scope="module")
def bundle(tiny_llava):
    return bundle_from_model(tiny_llava, load_tokenizer(), load_image_processor())


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    directory = tmp_path_factory.mktemp("images")
    generator = np.random.default_rng(0)
    items = [make_item(i) for i in range(4)]
    for item in items:
        pixels = generator.integers(0, 255, (120, 200, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(directory / item["image_file"])
    return items, directory


@pytest.fixture(scope="module")
def loaded(dataset):
    items, directory = dataset
    return [(item, Image.open(directory / item["image_file"]).convert("RGB")) for item in items]


def test_prefix_through_real_classes_has_expected_shape(bundle, loaded):
    tokens = tokenize_prompt(bundle.tokenizer, "What color is the bus?")
    prefix = prefill(bundle, tokens.prefix_ids, pixel_values(bundle, loaded[0][1]))
    assert prefix.num_nonvisual == 5 and prefix.num_visual == 576
    assert prefix.entries[0][0].shape[2] == 581 and len(prefix.entries) == 4


def test_image_is_padded_without_cropping():
    padded = pad_to_square(Image.new("RGB", (200, 100), (255, 255, 255)))
    assert padded.size == (200, 200)
    assert padded.getpixel((0, 0)) == PAD_COLOR and padded.getpixel((100, 100)) == (255, 255, 255)


def test_noise_floor_and_equivalence_tests_on_real_classes(bundle, loaded):
    floor = noise_floor(bundle, loaded)
    assert floor["logit_max"] < 1e-3 and floor["margin_max_abs"] < 1e-2
    floor_mis = noise_floor(bundle, loaded, "mis")
    result = e2(bundle, loaded, {"Q": floor, "mis": floor_mis})
    assert set(result["by_suffix"]) == {"Q", "mis"}
    for suffix_result in result["by_suffix"].values():
        assert suffix_result["max_logit_diff"] < 1e-3 and suffix_result["compact_vs_dense_max"] == 0
    sparse, hashes = e3_e4(bundle, loaded[:1], floor)
    assert hashes["pass"]
    assert sparse["margin_mean_abs"] < 0.01 and sparse["greedy_mismatches_not_near_tie"] == 0
    assert e5(bundle, loaded[:1], floor)["max_logit_diff"] == 0
    assert e7(bundle, loaded[:1])["pass"]


def test_cell_names():
    assert cell_name(Unit("pw", 144, False, "Q", "Q")) == "A"
    assert cell_name(Unit("pw", 144, False, "mis", "mis")) == "B_mis"
    assert cell_name(Unit("pw", 144, False, "Q", "mis")) == "C_mis"
    assert cell_name(Unit("pw", 144, False, "mis", "Q")) == "D_mis"
    assert cell_name(Unit("full", 576, False, "none", "mis")) == "mis"


def test_b_and_c_share_answering_text_and_differ_only_in_selection(bundle, loaded):
    item, image = loaded[0]
    result, attention = run_item(bundle, item, image, pilot_units())
    rows = {(r["selector"], r["reserve"], r["cell"]): r for r in result["rows"]}
    assert rows[("pw", False, "B_mis")]["answering"] == rows[("pw", False, "C_mis")]["answering"] == "mis"
    # D_mis uses the same kept set as B_mis; C_mis the same as A.
    assert rows[("pw", False, "B_mis")]["kept"] == rows[("pw", False, "D_mis")]["kept"]
    assert rows[("pw", False, "C_mis")]["kept"] == rows[("pw", False, "A")]["kept"]
    assert rows[("uniform", False, "mis")]["kept"] == uniform_grid(144).tolist()
    assert set(attention) == {"Q", "mis", "match", "irr"}
    for row in result["rows"]:
        if row["selector"] != "full":
            assert len(row["kept"]) == 144 == len(set(row["kept"]))


def test_rows_store_every_candidate_form(bundle, loaded):
    item, image = loaded[0]
    result, _ = run_item(bundle, item, image, [Unit("full", 576, False, "none", "Q")])
    scores = result["rows"][0]["scores"]
    assert set(scores) == {"true", "alt"} and set(scores["true"]) == {"lower", "capitalized"}


def test_confirmatory_items_are_refused():
    with pytest.raises(SystemExit):
        require_development([{**make_item(0), "split": "confirm"}])


def test_run_resumes_and_report_renders(bundle, dataset, tmp_path):
    items, directory = dataset
    run_items(bundle, items[:2], directory, tmp_path, pilot_units())
    first = tmp_path / "items" / "q0.json"
    stamp = first.stat().st_mtime_ns
    run_items(bundle, items[:2], directory, tmp_path, pilot_units())
    assert first.stat().st_mtime_ns == stamp
    assert json.loads(first.read_text())["item_id"] == "q0"
    text = report(tmp_path, "lower", resamples=100)
    for expected in ("S_H^(mis)", "E_harm^(mis)", "T_match", "T_irr", "Mean Jaccard", "Window composition"):
        assert expected in text


def test_screen_runs_only_full_q(bundle, dataset, tmp_path):
    items, directory = dataset
    run_screen(bundle, items[:1], directory, tmp_path)
    rows = json.loads((tmp_path / "items" / "q0.json").read_text())["rows"]
    assert [row["cell"] for row in rows] == ["Q"] and rows[0]["selector"] == "full"


def test_whole_validate_on_tiny_model_produces_a_report(bundle, dataset):
    items, directory = dataset
    result = validate(bundle, items, directory, noise_items=2, sparse_items=1, repeat_items=1)
    assert {"E1", "E2", "E3", "E4", "E5", "E7", "noise_floor", "noise_floor_mis", "noise_floor_native_head", "all_pass"} <= set(result)
    assert result["E1"]["pass"] and result["E4"]["pass"] and result["E7"]["pass"]


def test_diagnose_e2_reports_both_suffixes_and_exact_compact_equality(bundle, loaded):
    from kvsel.validate import diagnose_e2

    report = diagnose_e2(bundle, loaded[:2])
    assert set(report) == {"Q", "mis"}
    for result in report.values():
        assert result["compact_vs_dense_max"] == 0
        assert result["full_vs_compact_max"] < 1e-3
        assert len(result["worst_items"]) == 2 and result["worst_items"][0]["max_at_input_token"] is not None
