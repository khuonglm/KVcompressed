"""Main-set rule, stage-3 freeze and the confirmatory run."""

import json

import numpy as np
import pytest
from PIL import Image

from kvsel import confirm
from kvsel.freeze import freeze_screening
from kvsel.verify import append_decision

PREREG = "# Pre-registration\n\n## 1. Screening rules\nrule\n\n## 2. Comparative experiment\nPW-attn.\n"


def make_item(i):
    return {"item_id": f"i{i}", "image_id": f"img{i}", "image_file": "img.jpg", "dataset": "gqa", "split": "confirm",
            "question": "What color is the bus?", "a_true": "red", "a_alt": "blue", "descriptions": {
                "mis": "The bus in this image is blue.", "match": "The bus in this image is red.",
                "irr": "The road in this image is gray."}}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(confirm, "MAIN_SIZE", 3)
    monkeypatch.setattr(confirm, "CONTROL_SIZE", 1)
    items = [make_item(i) for i in range(8)]
    manifest = tmp_path / "gqa_confirm.jsonl"
    manifest.write_text("".join(json.dumps(item) + "\n" for item in items))
    screen = tmp_path / "screen"
    screen.mkdir()
    flags = {0: True, 1: False, 2: True, 3: True, 4: True, 5: True, 6: False, 7: True}
    (screen / "eligible.jsonl").write_text("".join(json.dumps({"item_id": f"i{i}", "eligible": f}) + "\n"
                                                   for i, f in flags.items()))
    decisions = tmp_path / "decisions.jsonl"
    prereg = tmp_path / "PREREG.md"
    prereg.write_text(PREREG)
    stage1 = tmp_path / "freeze" / "screening.json"
    freeze_screening(prereg, [manifest], stage1)
    return items, manifest, screen, decisions, prereg, stage1


def decide(decisions, items, verdicts):
    for i, verdict in verdicts.items():
        append_decision(decisions, items[i], verdict, "tester")


def test_main_set_follows_manifest_order_and_skips_failures(setup):
    items, _, screen, decisions, *_ = setup
    decide(decisions, items, {0: "pass", 1: "pass", 2: "a_true_wrong", 3: "pass", 4: "pass", 5: "pass"})
    selection = confirm.main_set(items, screen, decisions)
    assert selection["main"] == ["i0", "i3", "i4"]  # i1 isn't eligible, i2 failed, stop at 3
    assert selection["control"] == ["i0"] and selection["failed_before_cutoff"] == 1


def test_main_set_refuses_a_gap_in_verification(setup):
    items, _, screen, decisions, *_ = setup
    decide(decisions, items, {0: "pass", 3: "pass", 4: "pass"})  # i2 is eligible and unverified
    with pytest.raises(SystemExit, match="verify more.*i2"):
        confirm.main_set(items, screen, decisions)


def test_short_main_set_is_accepted_only_when_the_pool_is_exhausted(setup):
    items, _, screen, decisions, *_ = setup
    decide(decisions, items, {0: "pass", 2: "bad_box", 3: "wording", 4: "wording", 5: "pass", 7: "wording"})
    assert confirm.main_set(items, screen, decisions)["main"] == ["i0", "i5"]
    (screen / "eligible.jsonl").write_text("".join(l + "\n" for l in (screen / "eligible.jsonl").read_text().splitlines()[:6]))
    with pytest.raises(SystemExit, match="screen the rest"):
        confirm.main_set(items, screen, decisions)


def test_comparative_freeze_and_its_check(setup, tmp_path):
    items, manifest, screen, decisions, prereg, stage1 = setup
    decide(decisions, items, {0: "pass", 2: "pass", 3: "pass"})
    record = tmp_path / "freeze" / "comparative_gqa.json"
    args = (prereg, manifest, screen, decisions, record, stage1)
    prereg.write_text(PREREG.replace("# Pre-registration", "# Pre-registration\n\n**Status: DRAFT**"))
    with pytest.raises(SystemExit, match="DRAFT"):
        confirm.freeze_comparative(*args)
    prereg.write_text(PREREG)
    payload = confirm.freeze_comparative(*args)
    assert payload["main_items"] == 3 and payload["control_items"] == 1
    assert confirm.check_comparative(*args)["main_set_sha256"] == payload["main_set_sha256"]
    decide(decisions, items, {2: "wording"})  # the set is now short and i4 is unverified
    with pytest.raises(SystemExit, match="verify more"):
        confirm.check_comparative(*args)
    decide(decisions, items, {4: "pass"})  # full again, but a different main set
    with pytest.raises(SystemExit, match="main_set_sha256"):
        confirm.check_comparative(*args)
    decide(decisions, items, {2: "pass"})  # back to the frozen set
    confirm.check_comparative(*args)
    prereg.write_text(PREREG.replace("PW-attn.", "PW-attn, K = 72."))
    with pytest.raises(SystemExit, match="prereg_sha256"):
        confirm.check_comparative(*args)
    with pytest.raises(SystemExit, match="already exists"):
        confirm.freeze_comparative(*args)


def test_unchanged_stage1_code():
    """The confirm code lives outside the screening modules, so the committed stage-1 record still holds."""
    from pathlib import Path

    from kvsel.freeze import code_hash

    record = Path(__file__).parents[1] / "freeze" / "screening_amend1.json"
    if not record.exists():
        pytest.skip("no stage-1 freeze in this checkout")
    assert json.loads(record.read_text())["code_sha256"] == code_hash()


@pytest.mark.network
def test_confirm_runs_on_the_tiny_model(tiny_llava, tmp_path, monkeypatch):
    from kvsel.analysis import load_rows, margins_wide
    from kvsel.model import bundle_from_model, load_image_processor, load_tokenizer

    monkeypatch.setattr(confirm, "CONTROL_SIZE", 1)
    bundle = bundle_from_model(tiny_llava, load_tokenizer(), load_image_processor())
    Image.fromarray(np.random.default_rng(0).integers(0, 255, (60, 80, 3), dtype=np.uint8)).save(tmp_path / "img.jpg")
    items = [make_item(0), make_item(1)]
    summary = confirm.run_confirm(bundle, items, tmp_path, tmp_path / "run", {"main": ["i0", "i1"], "control": ["i0"]})
    assert summary["items"] == 2 and summary["control"] == 1
    control = json.loads((tmp_path / "run" / "items" / "i0.json").read_text())
    main = json.loads((tmp_path / "run" / "items" / "i1.json").read_text())
    assert control["subset"] == "control" and main["subset"] == "main"
    assert len(control["rows"]) == 34 and len(main["rows"]) == 16  # units plus the blind row
    assert {r["cell"] for r in main["rows"]} >= {"A", "B_mis", "C_mis", "D_mis", "Q", "mis"}
    assert not any("match" in r["cell"] for r in main["rows"])
    blind = [r for r in main["rows"] if r["selector"] == "blind"]
    assert len(blind) == 1 and blind[0]["cell"] == "Q"
    wide, _ = margins_wide(load_rows(tmp_path / "run"), "capitalized")
    assert {"blind|Q", "full|Q", "pw|C_mis"} <= set(wide.columns)
    mtime = (tmp_path / "run" / "items" / "i0.json").stat().st_mtime
    confirm.run_confirm(bundle, items, tmp_path, tmp_path / "run", {"main": ["i0", "i1"], "control": ["i0"]})
    assert (tmp_path / "run" / "items" / "i0.json").stat().st_mtime == mtime  # finished items are skipped


def test_confirm_refuses_missing_images_before_running(tmp_path):
    from types import SimpleNamespace

    from kvsel.families import LLAVA

    bundle = SimpleNamespace(family=LLAVA)
    with pytest.raises(SystemExit, match="images are missing"):
        confirm.run_confirm(bundle, [make_item(0)], tmp_path, tmp_path / "run", {"main": ["i0"], "control": []})
