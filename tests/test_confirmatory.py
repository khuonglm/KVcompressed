"""The confirmatory study: stage 1 and the eligibility screen, the main-set rule, stage 3 and the run."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from kvsel import protocol, runner
from kvsel.models.llava15 import LLAVA
from kvsel.protocol import CODE_AMENDMENTS, FrozenInputs, Prereg, Refusal
from kvsel.runs import MainSet
from kvsel.studies.confirmatory import comparative, screening
from kvsel.studies.confirmatory.screening import SECTION, check_screening, freeze_screening
from kvsel.verification import append_decision

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
    assert Prereg(tmp_path / "p.md", SECTION).text() == "## 1. Screening rules\nEligibility: Full_Q correct with a positive margin.\n"


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
    with pytest.raises(Refusal, match="prereg_screening_sha256"):
        check_screening(prereg, items, record)
    with pytest.raises(Refusal, match="already exists"):
        freeze_screening(prereg, [items], record)


def test_changed_manifest_is_refused(frozen):
    prereg, items, record = frozen
    manifest(items, n=4)
    with pytest.raises(Refusal, match="manifests"):
        check_screening(prereg, items, record)


def test_every_frozen_manifest_is_checked_whichever_is_screened(tmp_path):
    """Stage 1 froze both manifests, so a changed TextVQA manifest blocks the GQA screen too (decided 18 November)."""
    prereg = tmp_path / "PREREG.md"
    prereg.write_text(PREREG)
    gqa, textvqa = manifest(tmp_path / "gqa_confirm.jsonl"), manifest(tmp_path / "textvqa_confirm.jsonl")
    record = tmp_path / "freeze" / "screening.json"
    freeze_screening(prereg, [gqa, textvqa], record)
    check_screening(prereg, gqa, record)
    manifest(textvqa, n=4)
    with pytest.raises(Refusal, match="manifests"):
        check_screening(prereg, gqa, record)
    textvqa.unlink()
    with pytest.raises(Refusal, match="missing beside"):
        check_screening(prereg, gqa, record)


def test_unfrozen_or_development_manifests_are_refused(frozen, tmp_path):
    prereg, _, record = frozen
    with pytest.raises(Refusal, match="was not frozen"):
        check_screening(prereg, manifest(tmp_path / "other.jsonl"), record)
    with pytest.raises(Refusal, match="not a confirmatory manifest"):
        freeze_screening(prereg, [manifest(tmp_path / "dev.jsonl", split="dev")], tmp_path / "r.json")
    with pytest.raises(Refusal, match="no screening freeze"):
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

    monkeypatch.setattr(runner, "run_item", run_item)
    monkeypatch.setattr(screening, "open_image", lambda images, item: None)
    for item in items:
        (tmp_path / item["image_file"]).touch()
    return screening.run_confirm_screen(bundle, items, tmp_path, tmp_path / "screen", frozen_record, "rec"), calls


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
    assert runner.eligible_ids(out, items, "capitalized") == ["i0", "i2"]  # read as the flags its kind says
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
        "item_id": "i0", "full_q_eligible": True, "blind_eligible": True, "eligible": False, "record_sha256": "rec"}


def test_amendment_keeps_the_old_record_and_names_it(frozen, tmp_path):
    prereg, items, record = frozen
    old = record.read_text()
    prereg.write_text(PREREG.replace("positive margin.", "positive margin, and not with a blank image."))
    with pytest.raises(Refusal, match="needs both"):
        freeze_screening(prereg, [items], tmp_path / "a.json", amends=record)
    amended = freeze_screening(prereg, [items], tmp_path / "a.json", amends=record, reason="language prior")
    assert record.read_text() == old and amended["amends"]["reason"] == "language prior"
    check_screening(prereg, items, tmp_path / "a.json")
    with pytest.raises(Refusal, match="prereg_screening_sha256"):
        check_screening(prereg, items, record)  # the superseded record no longer matches


def test_screen_refuses_bad_inputs(monkeypatch, tmp_path, frozen):
    record = json.loads(frozen[2].read_text())
    with pytest.raises(Refusal, match="not confirmatory"):
        fake_screen(monkeypatch, tmp_path, record, [{"item_id": "d", "split": "dev", "image_file": "d.jpg"}], {})
    bundle = SimpleNamespace(family=LLAVA, decoder=SimpleNamespace(head_precision="native"))
    with pytest.raises(Refusal, match="head precision"):
        screening.run_confirm_screen(bundle, [], tmp_path, tmp_path / "s", record, "rec")
    bundle = SimpleNamespace(family=LLAVA, decoder=SimpleNamespace(head_precision="fp32"))
    with pytest.raises(Refusal, match="images are missing"):
        screening.run_confirm_screen(bundle, [{"item_id": "a", "split": "confirm", "image_file": "none.jpg"}],
                                     tmp_path, tmp_path / "s", record, "rec")


@pytest.mark.network
def test_screen_runs_on_the_tiny_model(tiny_llava, tmp_path, frozen):
    from kvsel.model import bundle_from_model

    bundle = bundle_from_model(tiny_llava, LLAVA.load_tokenizer(), LLAVA.load_processor())
    Image.fromarray(np.random.default_rng(0).integers(0, 255, (60, 80, 3), dtype=np.uint8)).save(tmp_path / "img.jpg")
    item = {"item_id": "c1", "image_id": "img", "image_file": "img.jpg", "dataset": "gqa", "split": "confirm",
            "question": "What color is the bus?", "a_true": "red", "a_alt": "blue", "descriptions": {
                "mis": "The bus in this image is blue.", "match": "The bus in this image is red.",
                "irr": "The road in this image is gray."}}
    funnel = screening.run_confirm_screen(bundle, [item], tmp_path, tmp_path / "screen", json.loads(frozen[2].read_text()),
                                          "rec")
    assert funnel["screened"] == 1 and set(json.loads((tmp_path / "screen" / "items" / "c1.json").read_text())) == {
        "item_id", "full_q_eligible", "blind_eligible", "eligible", "record_sha256"}


def test_changed_code_passes_only_through_the_amendment_log(frozen, monkeypatch):
    prereg, items, record = frozen
    old = json.loads(record.read_text())["code_sha256"]
    monkeypatch.setattr(protocol, "code_identity", lambda: "new")
    with pytest.raises(Refusal, match="code_sha256"):
        check_screening(prereg, items, record)
    entry = {"from": old, "to": "new", "date": "2026-11-18", "reason": "r", "evidence": "e"}
    (record.parent / CODE_AMENDMENTS).write_text(json.dumps([entry]))
    assert check_screening(prereg, items, record)["code_sha256"] == old


# --- Stage 3 and the run ---------------------------------------------------------------------------------------

STAGE3_PREREG = "# Pre-registration\n\n## 1. Screening rules\nrule\n\n## 2. Comparative experiment\nPW-attn.\n"


def make_confirm_item(i):
    return {"item_id": f"i{i}", "image_id": f"img{i}", "image_file": "img.jpg", "dataset": "gqa", "split": "confirm",
            "question": "What color is the bus?", "a_true": "red", "a_alt": "blue", "descriptions": {
                "mis": "The bus in this image is blue.", "match": "The bus in this image is red.",
                "irr": "The road in this image is gray."}}


@pytest.fixture
def stage3(tmp_path, monkeypatch):
    monkeypatch.setattr(comparative, "MAIN_SIZE", 3)
    monkeypatch.setattr(comparative, "CONTROL_SIZE", 1)
    items = [make_confirm_item(i) for i in range(8)]
    manifest = tmp_path / "gqa_comparative.jsonl"
    manifest.write_text("".join(json.dumps(item) + "\n" for item in items))
    screen = tmp_path / "screen"
    screen.mkdir()
    flags = {0: True, 1: False, 2: True, 3: True, 4: True, 5: True, 6: False, 7: True}
    (screen / "eligible.jsonl").write_text("".join(json.dumps({"item_id": f"i{i}", "eligible": f}) + "\n"
                                                   for i, f in flags.items()))
    decisions = tmp_path / "decisions.jsonl"
    prereg = tmp_path / "PREREG.md"
    prereg.write_text(STAGE3_PREREG)
    stage1 = tmp_path / "freeze" / "screening.json"
    freeze_screening(prereg, [manifest], stage1)
    return items, manifest, screen, decisions, prereg, stage1


def decide(decisions, items, verdicts):
    for i, verdict in verdicts.items():
        append_decision(decisions, items[i], verdict, "tester")


def test_main_set_follows_manifest_order_and_skips_failures(stage3):
    items, _, screen, decisions, *_ = stage3
    decide(decisions, items, {0: "pass", 1: "pass", 2: "a_true_wrong", 3: "pass", 4: "pass", 5: "pass"})
    selection = comparative.main_set(items, screen, decisions)
    assert selection["main"] == ["i0", "i3", "i4"]  # i1 isn't eligible, i2 failed, stop at 3
    assert selection["control"] == ["i0"] and selection["failed_before_cutoff"] == 1


def test_main_set_refuses_a_gap_in_verification(stage3):
    items, _, screen, decisions, *_ = stage3
    decide(decisions, items, {0: "pass", 3: "pass", 4: "pass"})  # i2 is eligible and unverified
    with pytest.raises(Refusal, match="verify more.*i2"):
        comparative.main_set(items, screen, decisions)


def test_short_main_set_is_accepted_only_when_the_pool_is_exhausted(stage3):
    items, _, screen, decisions, *_ = stage3
    decide(decisions, items, {0: "pass", 2: "bad_box", 3: "wording", 4: "wording", 5: "pass", 7: "wording"})
    assert comparative.main_set(items, screen, decisions)["main"] == ["i0", "i5"]
    (screen / "eligible.jsonl").write_text("".join(l + "\n" for l in (screen / "eligible.jsonl").read_text().splitlines()[:6]))
    with pytest.raises(Refusal, match="screen the rest"):
        comparative.main_set(items, screen, decisions)


def test_comparative_freeze_and_its_check(stage3, tmp_path):
    items, manifest, screen, decisions, prereg, stage1 = stage3
    decide(decisions, items, {0: "pass", 2: "pass", 3: "pass"})
    record = tmp_path / "freeze" / "comparative_gqa.json"
    given = FrozenInputs(prereg, manifest, screen, decisions, stage1, record)
    prereg.write_text(STAGE3_PREREG.replace("PW-attn.", "DRAFT: still open.\nPW-attn."))
    with pytest.raises(Refusal, match="DRAFT"):
        comparative.freeze_comparative(given)
    prereg.write_text(STAGE3_PREREG)
    payload = comparative.freeze_comparative(given)
    assert payload["main_items"] == 3 and payload["control_items"] == 1
    assert comparative.check_comparative(given)["main_set_sha256"] == payload["main_set_sha256"]
    decide(decisions, items, {2: "wording"})  # the set is now short and i4 is unverified
    with pytest.raises(Refusal, match="verify more"):
        comparative.check_comparative(given)
    decide(decisions, items, {4: "pass"})  # full again, but a different main set
    with pytest.raises(Refusal, match="main_set_sha256"):
        comparative.check_comparative(given)
    decide(decisions, items, {2: "pass"})  # back to the frozen set
    comparative.check_comparative(given)
    prereg.write_text(STAGE3_PREREG.replace("PW-attn.", "PW-attn, K = 72."))
    with pytest.raises(Refusal, match="prereg_sha256"):
        comparative.check_comparative(given)
    with pytest.raises(Refusal, match="already exists"):
        comparative.freeze_comparative(given)


@pytest.mark.network
def test_confirm_runs_on_the_tiny_model(tiny_llava, tmp_path, monkeypatch):
    from kvsel.analysis.tables import load_run, wide_margins
    from kvsel.model import bundle_from_model

    monkeypatch.setattr(comparative, "CONTROL_SIZE", 1)
    bundle = bundle_from_model(tiny_llava, LLAVA.load_tokenizer(), LLAVA.load_processor())
    Image.fromarray(np.random.default_rng(0).integers(0, 255, (60, 80, 3), dtype=np.uint8)).save(tmp_path / "img.jpg")
    items = [make_confirm_item(0), make_confirm_item(1)]
    summary = comparative.run_confirm(bundle, items, tmp_path, tmp_path / "run", MainSet.of(["i0", "i1"], ["i0"]), "rec")
    assert summary["items"] == 2 and summary["control"] == 1
    control = json.loads((tmp_path / "run" / "items" / "i0.json").read_text())
    main = json.loads((tmp_path / "run" / "items" / "i1.json").read_text())
    assert control["subset"] == "control" and main["subset"] == "main"
    assert len(control["rows"]) == 34 and len(main["rows"]) == 16  # units plus the blind row
    assert {r["cell"] for r in main["rows"]} >= {"A", "B_mis", "C_mis", "D_mis", "Q", "mis"}
    assert not any("match" in r["cell"] for r in main["rows"])
    blind = [r for r in main["rows"] if r["selector"] == "blind"]
    assert len(blind) == 1 and blind[0]["cell"] == "Q"
    wide, _ = wide_margins(load_run(tmp_path / "run")[0], "primary")
    assert {"blind|Q", "full|Q", "pw|C_mis"} <= set(wide.columns)
    mtime = (tmp_path / "run" / "items" / "i0.json").stat().st_mtime
    comparative.run_confirm(bundle, items, tmp_path, tmp_path / "run", MainSet.of(["i0", "i1"], ["i0"]), "rec")
    assert (tmp_path / "run" / "items" / "i0.json").stat().st_mtime == mtime  # finished items are skipped


def test_confirm_refuses_missing_images_before_running(tmp_path):
    bundle = SimpleNamespace(family=LLAVA)
    with pytest.raises(Refusal, match="images are missing"):
        comparative.run_confirm(bundle, [make_confirm_item(0)], tmp_path, tmp_path / "run", MainSet.of(["i0"], []), "rec")
