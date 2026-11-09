"""Stage-1 freeze and the confirmatory eligibility screen, without the real model."""

import json
from types import SimpleNamespace

import pytest

from kvsel import screening
from kvsel.families import LLAVA
from kvsel.freeze import check_screening, freeze_screening, screening_section

PREREG = """# Pre-registration

## 1. Screening rules
Eligibility: Full_Q correct with a positive margin.

## 2. Comparative experiment
Selector PW-attn.
"""


def manifest(path, split="confirm", n=3):
    path.write_text("".join(json.dumps({"item_id": f"i{i}", "split": split, "image_file": f"{i}.jpg"}) + "\n"
                            for i in range(n)))
    return path


@pytest.fixture
def frozen(tmp_path):
    prereg = tmp_path / "PREREG.md"
    prereg.write_text(PREREG)
    items = manifest(tmp_path / "gqa_confirm.jsonl")
    record = tmp_path / "freeze" / "screening.json"
    freeze_screening(prereg, [items], record)
    return prereg, items, record


def test_screening_section_stops_at_the_next_heading(tmp_path):
    (tmp_path / "p.md").write_text(PREREG)
    assert screening_section(tmp_path / "p.md") == "## 1. Screening rules\nEligibility: Full_Q correct with a positive margin.\n"


def test_freeze_records_and_check_passes(frozen):
    prereg, items, record = frozen
    payload = json.loads(record.read_text())
    assert payload["stage"] == "screening" and payload["settings"]["surface"] == "capitalized"
    assert payload["settings"]["template"] == LLAVA.template_version == "v3.1"
    assert check_screening(prereg, items, record)["code_sha256"] == payload["code_sha256"]
    assert freeze_screening(prereg, [items], record) == payload  # same state: nothing rewritten


def test_comparative_section_may_still_change(frozen):
    prereg, items, record = frozen
    prereg.write_text(PREREG.replace("Selector PW-attn.", "Selector PW-attn, K = 144."))
    check_screening(prereg, items, record)


def test_changed_screening_rule_is_refused(frozen):
    prereg, items, record = frozen
    prereg.write_text(PREREG.replace("positive margin", "margin above 1"))
    with pytest.raises(SystemExit, match="Screening rules"):
        check_screening(prereg, items, record)
    with pytest.raises(SystemExit, match="already exists"):
        freeze_screening(prereg, [items], record)


def test_changed_manifest_is_refused(frozen):
    prereg, items, record = frozen
    manifest(items, n=4)
    with pytest.raises(SystemExit, match="manifest"):
        check_screening(prereg, items, record)


def test_unfrozen_or_development_manifests_are_refused(frozen, tmp_path):
    prereg, _, record = frozen
    with pytest.raises(SystemExit, match="was not frozen"):
        check_screening(prereg, manifest(tmp_path / "other.jsonl"), record)
    with pytest.raises(SystemExit, match="not a confirmatory manifest"):
        freeze_screening(prereg, [manifest(tmp_path / "dev.jsonl", split="dev")], tmp_path / "r.json")
    with pytest.raises(SystemExit, match="no screening freeze"):
        check_screening(prereg, manifest(tmp_path / "x.jsonl"), tmp_path / "missing.json")


def fake_screen(monkeypatch, tmp_path, frozen_record, items, outcomes, blind=None):
    bundle = SimpleNamespace(family=LLAVA, decoder=SimpleNamespace(head_precision="fp32"))
    calls = []
    blind = blind or {}

    def run_item(bundle, item, image, units):
        is_blank = image is not None  # open_image is patched to return None for real images
        calls.append(item["item_id"] + ("/blank" if is_blank else ""))
        result = blind.get(item["item_id"], "other") if is_blank else outcomes[item["item_id"]]
        row = {"outcome": result, "greedy_text": "SECRET",
               "scores": {"true": {"capitalized": [-0.1]}, "alt": {"capitalized": [-2.0]}}}
        return {"rows": [row]}, {}

    monkeypatch.setattr(screening, "run_item", run_item)
    monkeypatch.setattr(screening, "open_image", lambda images, item: None)
    monkeypatch.setattr(screening, "blank_image", lambda bundle: "blank")
    for item in items:
        (tmp_path / item["image_file"]).touch()
    return screening.run_confirm_screen(bundle, items, tmp_path, tmp_path / "screen", frozen_record), calls


def test_screen_writes_only_eligibility_and_resumes(monkeypatch, tmp_path, frozen):
    _, items_path, record = frozen
    record = json.loads(record.read_text())
    items = [json.loads(line) for line in items_path.read_text().splitlines()]
    outcomes = {"i0": "correct", "i1": "alt", "i2": "correct"}
    funnel, calls = fake_screen(monkeypatch, tmp_path, record, items, outcomes)
    assert (funnel["screened"], funnel["eligible"], funnel["not_eligible"]) == (3, 2, 1)
    assert calls == ["i0", "i0/blank", "i1", "i2", "i2/blank"]  # blind only after Full_Q passes
    out = tmp_path / "screen"
    text = (out / "eligible.jsonl").read_text() + "".join(p.read_text() for p in (out / "items").glob("*.json"))
    assert "SECRET" not in text and "scores" not in text and "outcome" not in text
    assert [json.loads(line) for line in (out / "eligible.jsonl").read_text().splitlines()] == [
        {"item_id": "i0", "eligible": True}, {"item_id": "i1", "eligible": False}, {"item_id": "i2", "eligible": True}]
    assert screening.eligible_ids(out) == {"i0", "i2"}
    assert fake_screen(monkeypatch, tmp_path, record, items, outcomes)[1] == []  # rerun skips finished items


def test_items_answerable_with_a_blank_image_are_not_eligible(monkeypatch, tmp_path, frozen):
    _, items_path, record = frozen
    items = [json.loads(line) for line in items_path.read_text().splitlines()]
    outcomes = {"i0": "correct", "i1": "correct", "i2": "alt"}
    funnel, _ = fake_screen(monkeypatch, tmp_path, json.loads(record.read_text()), items, outcomes, blind={"i0": "correct"})
    assert (funnel["full_q_eligible"], funnel["removed_image_independent"], funnel["eligible"]) == (2, 1, 1)
    flags = {r["item_id"]: r for r in map(json.loads, (tmp_path / "screen" / "eligible.jsonl").read_text().splitlines())}
    assert [flags[i]["eligible"] for i in ("i0", "i1", "i2")] == [False, True, False]
    assert json.loads((tmp_path / "screen" / "items" / "i0.json").read_text()) == {
        "item_id": "i0", "full_q_eligible": True, "blind_eligible": True, "eligible": False}


def test_amendment_keeps_the_old_record_and_names_it(frozen, tmp_path):
    prereg, items, record = frozen
    old = record.read_text()
    prereg.write_text(PREREG.replace("positive margin.", "positive margin, and not with a blank image."))
    with pytest.raises(SystemExit, match="needs both"):
        freeze_screening(prereg, [items], tmp_path / "a.json", amends=record)
    amended = freeze_screening(prereg, [items], tmp_path / "a.json", amends=record, reason="language prior")
    assert record.read_text() == old and amended["amends"]["reason"] == "language prior"
    check_screening(prereg, items, tmp_path / "a.json")
    with pytest.raises(SystemExit, match="Screening rules"):
        check_screening(prereg, items, record)  # the superseded record no longer matches


def test_screen_refuses_bad_inputs(monkeypatch, tmp_path, frozen):
    record = json.loads(frozen[2].read_text())
    with pytest.raises(SystemExit, match="not confirmatory"):
        fake_screen(monkeypatch, tmp_path, record, [{"item_id": "d", "split": "dev", "image_file": "d.jpg"}], {})
    bundle = SimpleNamespace(family=LLAVA, decoder=SimpleNamespace(head_precision="native"))
    with pytest.raises(SystemExit, match="head precision"):
        screening.run_confirm_screen(bundle, [], tmp_path, tmp_path / "s", record)
    bundle = SimpleNamespace(family=LLAVA, decoder=SimpleNamespace(head_precision="fp32"))
    with pytest.raises(SystemExit, match="images are missing"):
        screening.run_confirm_screen(bundle, [{"item_id": "a", "split": "confirm", "image_file": "none.jpg"}],
                                     tmp_path, tmp_path / "s", record)


@pytest.mark.network
def test_screen_runs_on_the_tiny_model(tiny_llava, tmp_path, frozen):
    import numpy as np
    from PIL import Image

    from kvsel.model import bundle_from_model, load_image_processor, load_tokenizer

    bundle = bundle_from_model(tiny_llava, load_tokenizer(), load_image_processor())
    Image.fromarray(np.random.default_rng(0).integers(0, 255, (60, 80, 3), dtype=np.uint8)).save(tmp_path / "img.jpg")
    item = {"item_id": "c1", "image_id": "img", "image_file": "img.jpg", "dataset": "gqa", "split": "confirm",
            "question": "What color is the bus?", "a_true": "red", "a_alt": "blue", "descriptions": {
                "mis": "The bus in this image is blue.", "match": "The bus in this image is red.",
                "irr": "The road in this image is gray."}}
    funnel = screening.run_confirm_screen(bundle, [item], tmp_path, tmp_path / "screen", json.loads(frozen[2].read_text()))
    assert funnel["screened"] == 1 and set(json.loads((tmp_path / "screen" / "items" / "c1.json").read_text())) == {"item_id", "full_q_eligible", "blind_eligible", "eligible"}
