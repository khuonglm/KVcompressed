"""S1, R1 and E6 on synthetic inputs and the tiny random LLaVA."""

import json

import numpy as np
import pytest
from PIL import Image

from kvsel import NUM_VISUAL
from kvsel.sanity import build_s1_items, coverage_report, random_radii, s1_score, standard_prompt
from kvsel.selectors import coverage_order, coverage_radius


def test_standard_prompt_has_no_description_or_authority():
    prompt = standard_prompt("What color is the bus?")
    assert prompt.startswith("USER: <image>") and prompt.endswith(
        "\nWhat color is the bus?\nAnswer the question using a single word or phrase. ASSISTANT:")
    assert "description" not in prompt.lower() and "inaccurate" not in prompt


def test_s1_items_are_random_unfiltered_and_on_development_images(tmp_path):
    dev = tmp_path / "dev.jsonl"
    dev.write_text("".join(json.dumps({"image_id": image}) + "\n" for image in ("a", "b")))
    gqa = {str(i): {"imageId": "abc"[i % 3], "question": f"q{i}?", "answer": "yes"} for i in range(30)}
    (tmp_path / "gqa.json").write_text(json.dumps(gqa))
    items = build_s1_items("gqa", [tmp_path / "gqa.json"], dev, n=5)
    assert len(items) == 5 and {item["image_id"] for item in items} <= {"a", "b"}
    assert all(item["split"] == "dev" and item["image_file"] == f"{item['image_id']}.jpg" for item in items)
    assert items == build_s1_items("gqa", [tmp_path / "gqa.json"], dev, n=5)  # fixed seed
    textvqa = {"data": [{"image_id": "a", "question_id": 1, "question": "what?", "answers": ["x"] * 10},
                        {"image_id": "z", "question_id": 2, "question": "what?", "answers": ["y"] * 10}]}
    (tmp_path / "tv.json").write_text(json.dumps(textvqa))
    assert [i["item_id"] for i in build_s1_items("textvqa", [tmp_path / "tv.json"], dev)] == ["s1_textvqa_1"]


def test_s1_scores():
    assert s1_score("gqa", "Grey", ["gray"]) == 1.0 and s1_score("gqa", "red", ["blue"]) == 0.0
    answers = ["canon"] * 2 + ["nikon"] * 8
    assert s1_score("textvqa", "Canon", answers) == pytest.approx(2 / 3)
    assert s1_score("textvqa", "nikon", answers) == 1.0


def test_coverage_report_compares_reserve_with_random_sets(tmp_path):
    attention = tmp_path / "attention"
    attention.mkdir()
    generator = np.random.default_rng(0)
    for name in ("a", "b"):
        np.savez(attention / f"{name}.npz", Q=generator.random((20, NUM_VISUAL)).astype(np.float32))
    result = coverage_report(attention, tmp_path / "out")
    assert result["attention_files"] == 2
    entry = result["budgets"]["144"]
    assert entry["k_r"] == 14 and entry["reserve_radius"] == coverage_radius(np.array(coverage_order()[:14]))
    assert entry["kept_radius_pw_Q"]["reserve_mean"] <= entry["kept_radius_pw_Q"]["no_reserve_mean"] + 1e-9
    assert 0.0 <= entry["share_random_at_or_below_reserve"] <= 1.0
    svg = (tmp_path / "out" / "coverage_k144.svg").read_text()
    assert svg.startswith("<svg") and svg.count("<rect") > 3 * NUM_VISUAL
    assert coverage_report(None, tmp_path / "plain")["attention_files"] == 0


def test_random_radii_are_reproducible():
    assert np.array_equal(random_radii(7, sets=20), random_radii(7, sets=20))


@pytest.mark.network
def test_s1_and_timing_run_on_the_tiny_model(tiny_llava, tmp_path):
    from kvsel.model import bundle_from_model, load_image_processor, load_tokenizer
    from kvsel.sanity import run_s1, run_timing

    bundle = bundle_from_model(tiny_llava, load_tokenizer(), load_image_processor())
    Image.fromarray(np.random.default_rng(0).integers(0, 255, (60, 80, 3), dtype=np.uint8)).save(tmp_path / "img.jpg")
    s1_item = {"item_id": "s", "image_id": "img", "image_file": "img.jpg", "dataset": "gqa", "split": "dev",
               "question": "What color is the bus?", "answers": ["red"]}
    result = run_s1(bundle, [s1_item], tmp_path)
    assert result["items"] == 1 and result["rows"][0]["score"] in (0.0, 1.0)

    item = {**s1_item, "a_true": "red", "a_alt": "blue", "descriptions": {
        "mis": "The bus in this image is blue.", "match": "The bus in this image is red.",
        "irr": "The road in this image is gray."}}
    timing = run_timing(bundle, [item], tmp_path, {"screen_gqa": 100, "confirmatory_items": 10})
    assert timing["items"] == 1 and timing["per_item_seconds"]["pilot"]["n"] == 1
    assert {"prefill", "probe_short", "probe_long", "unit_full_short", "unit_pw_long", "unit_pw+r_short"} <= set(timing["stages_seconds"])
    assert timing["projected_hours"]["screen_gqa"] >= 0 and "peak_reserved_gib" not in timing  # CPU
    with pytest.raises(SystemExit):
        run_timing(bundle, [{**item, "split": "confirm"}], tmp_path, {})
