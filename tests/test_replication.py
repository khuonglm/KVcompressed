"""GQA replication (PREREG section 3): selection of fresh items and the pool count."""

import json

import numpy as np
import pytest

from kvsel import protocol
from kvsel.cli import main
from kvsel.protocol import FrozenInputs, Refusal, code_identity, sha256_file
from kvsel.runs import MainSet
from kvsel.studies.confirmatory import comparative
from kvsel.studies.replication import run as replication
from kvsel.studies.replication import select
from kvsel.verification import PASS, VALID_DECISIONS, append_decision, load_decisions

FAIL = min(VALID_DECISIONS - {PASS})


def item(i, image=None):
    return {"item_id": f"i{i}", "image_id": image or f"img{i}", "image_file": "img.jpg", "dataset": "gqa",
            "split": "confirm", "question": "What color is the bus?", "a_true": "red", "a_alt": "blue",
            "descriptions": {"mis": "The bus is blue.", "match": "The bus is red.", "irr": "The road is gray."}}


@pytest.fixture
def sizes(monkeypatch):
    monkeypatch.setattr(select, "REPLICATION_SIZE", 3)
    monkeypatch.setattr(select, "CONTROL_SIZE", 1)


def choose(items, flags, verdicts, main, excluded=frozenset(), tmp_path=None):
    path = tmp_path / "d.jsonl"
    for i, verdict in verdicts.items():
        append_decision(path, items[i], verdict, "tester")
    decisions, _ = load_decisions(path, items)
    return select.replication_set(items, flags, decisions, main, set(excluded))


def test_starts_after_the_main_set_and_skips_failures(sizes, tmp_path):
    items = [item(i) for i in range(8)]
    flags = {f"i{i}": i != 4 for i in range(8)}
    verdicts = {0: PASS, 1: PASS, 2: PASS, 3: FAIL, 5: PASS, 6: PASS, 7: PASS}
    result = choose(items, flags, verdicts, ["i0", "i1"], tmp_path=tmp_path)
    assert result["replication"] == ["i2", "i5", "i6"]
    assert result["control"] == ["i2"]
    assert result["failed"] == 1
    assert result["next_unverified"] is None


def test_drops_shared_images_and_earlier_run_items(sizes, tmp_path):
    items = [item(0, "a"), item(1, "a"), item(2), item(3), item(4), item(5)]
    flags = {f"i{i}": True for i in range(6)}
    result = choose(items, flags, {i: PASS for i in range(6)}, ["i0"], excluded={"i2"}, tmp_path=tmp_path)
    assert result["replication"] == ["i3", "i4", "i5"]
    assert result["excluded"] == {"i1": "image in the main set", "i2": "has a file in an earlier run"}


def test_stops_at_the_first_unverified_item(sizes, tmp_path):
    items = [item(i) for i in range(6)]
    flags = {f"i{i}": True for i in range(6)}
    result = choose(items, flags, {0: PASS, 1: PASS, 3: PASS, 4: PASS}, ["i0"], tmp_path=tmp_path)
    assert result["replication"] == ["i1"]
    assert result["next_unverified"] == "i2"


def test_a_stale_decision_counts_as_unverified(sizes, tmp_path):
    items = [item(i) for i in range(5)]
    flags = {f"i{i}": True for i in range(5)}
    path = tmp_path / "d.jsonl"
    for i in range(5):
        append_decision(path, items[i], PASS, "tester")
    items[2] = {**items[2], "a_alt": "green"}  # edited after verification
    decisions, stale = load_decisions(path, items)
    result = select.replication_set(items, flags, decisions, ["i0"], set())
    assert stale == 1
    assert result["replication"] == ["i1"]
    assert result["next_unverified"] == "i2"


def test_count_prints_the_pool(sizes, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(comparative, "MAIN_SIZE", 2)
    monkeypatch.setattr(comparative, "CONTROL_SIZE", 1)
    items = [item(i) for i in range(8)]
    manifest = tmp_path / "m.jsonl"
    manifest.write_text("".join(json.dumps(i) + "\n" for i in items))
    screen = tmp_path / "screen"
    screen.mkdir()
    (screen / "eligible.jsonl").write_text("".join(json.dumps({"item_id": f"i{i}", "eligible": True}) + "\n"
                                                   for i in range(8)))
    decisions = tmp_path / "d.jsonl"
    for i in range(8):
        append_decision(decisions, items[i], PASS, "tester")
    run = tmp_path / "old_run" / "items"
    run.mkdir(parents=True)
    (run / "i3.json").write_text("{}")
    main(["replication", "count", "--items", str(manifest), "--screen", str(screen), "--decisions", str(decisions),
          "--exclude-runs", str(tmp_path / "old_run")])
    out = json.loads(capsys.readouterr().out)
    assert out["main_set"] == 2
    assert out["replication"] == 3
    assert out["control"] == 1
    assert out["excluded"] == {"has a file in an earlier run": 1}
    assert out["next_unverified"] is None


# --- Freeze, run and report (ticket 03) ---------------------------------------------------------------

PREREG = ("# Pre-registration\n\n## 1. Screening rules\nrule\n\n## 2. Comparative experiment\nPW-attn.\n\n"
          "## 3. GQA replication on fresh items\n\nDRAFT (date). Review this section.\n\n"
          "R1. The freeze refuses while this section is marked DRAFT.\n")


def replicate(stage_function, record, args):
    """Freeze or check the replication with the world's inputs."""
    given = FrozenInputs(args["prereg"], args["manifest"], args["screen"], args["decisions"], args["stage1"], record)
    return stage_function(given, args["gqa_record"], args["exclude_runs"])


@pytest.fixture
def world(tmp_path, monkeypatch, sizes):
    """Main set i0, i1 (GQA record frozen); replication i2, i3, i4 with control i2; i5 is spare."""
    from kvsel.studies.confirmatory.screening import freeze_screening

    monkeypatch.setattr(comparative, "MAIN_SIZE", 2)
    monkeypatch.setattr(comparative, "CONTROL_SIZE", 1)
    items = [item(i) for i in range(6)]
    manifest = tmp_path / "gqa_confirm.jsonl"
    manifest.write_text("".join(json.dumps(i) + "\n" for i in items))
    screen = tmp_path / "screen"
    screen.mkdir()
    (screen / "eligible.jsonl").write_text("".join(json.dumps({"item_id": f"i{i}", "eligible": True}) + "\n"
                                                   for i in range(6)))
    decisions = tmp_path / "decisions.jsonl"
    for i in range(6):
        append_decision(decisions, items[i], PASS, "tester")
    prereg = tmp_path / "PREREG.md"
    prereg.write_text(PREREG)
    stage1 = tmp_path / "freeze" / "screening.json"
    freeze_screening(prereg, [manifest], stage1)
    gqa_record = tmp_path / "freeze" / "comparative_gqa.json"
    comparative.freeze_comparative(FrozenInputs(prereg, manifest, screen, decisions, stage1, gqa_record))
    args = {"prereg": prereg, "manifest": manifest, "screen": screen, "decisions": decisions, "stage1": stage1,
                "gqa_record": gqa_record, "exclude_runs": []}
    return items, args, tmp_path / "freeze" / "replication_gqa.json"


def undraft(prereg):
    prereg.write_text(prereg.read_text().replace("DRAFT (date). Review this section.\n\n", ""))


def test_freeze_refuses_a_draft_section(world):
    _, args, record = world
    with pytest.raises(Refusal, match="DRAFT"):
        replicate(replication.freeze_replication, record, args)
    assert not record.exists()


def test_freeze_records_the_replication_set_and_its_check(world):
    items, args, record = world
    undraft(args["prereg"])
    frozen = replicate(replication.freeze_replication, record, args)
    assert frozen["replication_ids"] == ["i2", "i3", "i4"]
    assert frozen["control_ids"] == ["i2"]
    assert frozen["code_sha256"] == code_identity()
    assert replicate(replication.freeze_replication, record, args) == frozen  # same state: no change, no error
    assert replicate(replication.check_replication, record, args)["replication_ids"] == ["i2", "i3", "i4"]
    append_decision(args["decisions"], items[3], FAIL, "tester")  # a decision changed after the freeze
    with pytest.raises(Refusal, match="replication_set_sha256"):
        replicate(replication.check_replication, record, args)
    with pytest.raises(Refusal, match="already exists"):
        replicate(replication.freeze_replication, record, args)


def test_freeze_refuses_a_short_set_and_a_changed_main_set(world, tmp_path):
    items, args, record = world
    undraft(args["prereg"])
    old = tmp_path / "old_run" / "items"
    old.mkdir(parents=True)
    for item_id in ("i4", "i5"):
        (old / f"{item_id}.json").write_text("{}")
    with pytest.raises(Refusal, match="replication set has 2 of 3"):
        replicate(replication.freeze_replication, record, {**args, "exclude_runs": [tmp_path / "old_run"]})
    append_decision(args["decisions"], items[1], FAIL, "tester")  # main set changes: i1 now failed
    with pytest.raises(Refusal, match="main set"):
        replicate(replication.freeze_replication, record, args)
    assert not record.exists()


def test_check_refuses_changes_and_accepts_only_amended_code(world, tmp_path, monkeypatch):
    from kvsel.protocol import CODE_AMENDMENTS

    _, args, record = world
    undraft(args["prereg"])
    frozen = replicate(replication.freeze_replication, record, args)
    old_run = tmp_path / "old_run"
    old_run.mkdir()
    with pytest.raises(Refusal, match="exclude_runs"):  # a different exclusion list, even an empty folder
        replicate(replication.check_replication, record, {**args, "exclude_runs": [old_run]})
    monkeypatch.setattr(protocol, "code_identity", lambda: "new code")
    with pytest.raises(Refusal, match="code_sha256"):
        replicate(replication.check_replication, record, args)
    (record.parent / CODE_AMENDMENTS).write_text(json.dumps([{"from": frozen["code_sha256"], "to": "new code",
                                                               "date": "2026-11-20", "reason": "r", "evidence": "e"}]))
    assert replicate(replication.check_replication, record, args)["replication_ids"] == ["i2", "i3", "i4"]
    args["prereg"].write_text(args["prereg"].read_text() + "\nAn edit after the freeze.\n")
    with pytest.raises(Refusal, match="prereg_sha256"):
        replicate(replication.check_replication, record, args)


def test_selection_matches_the_frozen_main_set_rule(sizes, tmp_path, monkeypatch):
    """With no shared images or earlier runs, the replication rule picks what `comparative.main_set` would."""
    monkeypatch.setattr(comparative, "MAIN_SIZE", 3)
    monkeypatch.setattr(comparative, "CONTROL_SIZE", 1)
    items = [item(i) for i in range(10)]
    eligible = {i: i not in (3, 6) for i in range(10)}
    screen = tmp_path / "screen"
    screen.mkdir()
    (screen / "eligible.jsonl").write_text("".join(json.dumps({"item_id": f"i{i}", "eligible": e}) + "\n"
                                                   for i, e in eligible.items()))
    verdicts = {i: FAIL if i in (2, 5) else PASS for i in range(10)}
    path = tmp_path / "d.jsonl"
    for i, verdict in verdicts.items():
        append_decision(path, items[i], verdict, "tester")
    decisions, _ = load_decisions(path, items)
    flags = {f"i{i}": e for i, e in eligible.items()}
    chosen = select.replication_set(items, flags, decisions, ["i0"], set())
    frozen = comparative.main_set(items[1:], screen, path)
    assert chosen["replication"] == frozen["main"] == ["i1", "i4", "i7"]
    assert chosen["failed"] == frozen["failed_before_cutoff"] == 2


def synthetic_run(tmp_path, ids, control, mis=None, irr=None, noise=0.05, seed=0):
    """Item files as the run writes them, two items per image. S_H^(c) at budget K is `c[K]` plus noise;
    every other estimand is 0 by construction. Defaults: S_H^(mis) = 0.3 at every K, S_H^(irr) = 0."""
    from dataclasses import asdict

    from kvsel.runner import cell_name
    from kvsel.studies.replication.run import units

    mis = mis or {72: 0.3, 144: 0.3, 288: 0.3}
    irr = irr or {72: 0.0, 144: 0.0, 288: 0.0}
    forms = ("lower", "capitalized", "upper")

    def scores(m):
        return {"true": {f: [m, 0.0] for f in forms}, "alt": {f: [0.0, 0.0] for f in forms}}

    generator = np.random.default_rng(seed)
    run = tmp_path / "run"
    (run / "items").mkdir(parents=True)
    for n, item_id in enumerate(ids):
        a, b = 2 + generator.normal(0, 0.5), 1 + generator.normal(0, 0.5)
        rows = []
        for unit in units(control=item_id in control):
            cell = cell_name(unit)
            effect = {"C_mis": mis, "C_irr": irr}.get(cell)
            value = (b + effect[unit.k] + generator.normal(0, noise) if effect
                     else a if cell in ("A", "D_mis", "D_irr", "D_match") else a + 1 if cell == "Q" else
                     b + 1 if cell in ("mis", "irr", "match") else b)
            rows.append({**asdict(unit), "cell": cell, "kept": None, "scores": scores(value), "outcome": "correct",
                         "truncated": False})
        rows.append({"selector": "blind", "k": 576, "reserve": False, "selection": "none", "answering": "Q",
                     "cell": "Q", "kept": None, "scores": scores(-1.0), "outcome": "other", "truncated": False})
        (run / "items" / f"{item_id}.json").write_text(json.dumps({
            "item_id": item_id, "image_id": f"img{n // 2}", "subset": "control" if item_id in control else "main",
            "rows": rows}))
    return run


def report_on(tmp_path, n=80, **planted):
    from kvsel.studies.replication import report

    ids = [f"r{i}" for i in range(n)]
    tmp_path.mkdir(exist_ok=True)
    record = tmp_path / "replication_gqa.json"
    record.write_text(json.dumps({"replication_ids": ids, "control_ids": ids[:20],
                                  "code_sha256": code_identity()}))
    run = synthetic_run(tmp_path, ids, set(ids[:20]), **planted)
    return report.replication_report(run, record, tmp_path / "out", resamples=1000)


def test_report_recovers_planted_effects_at_every_budget(tmp_path):
    out = report_on(tmp_path, mis={72: 0.5, 144: 0.3, 288: 0.0}, irr={72: 0.2, 144: 0.0, 288: 0.0})
    budgets = out["budgets"]
    assert 0.47 < budgets["72"]["margins"]["S_H^(mis)"]["estimate"] < 0.53
    assert 0.27 < budgets["144"]["margins"]["S_H^(mis)"]["estimate"] < 0.33
    assert abs(budgets["288"]["margins"]["S_H^(mis)"]["estimate"]) < 0.03
    assert 0.17 < budgets["72"]["margins"]["S_H^(irr)"]["estimate"] < 0.23
    assert 0.27 < budgets["72"]["margins"]["T_irr"]["estimate"] < 0.33 and budgets["72"]["margins"]["T_irr"]["items"] == 80
    assert budgets["144"]["margins"]["T_match"]["items"] == 20  # control subset only
    assert abs(budgets["72"]["margins"]["E_harm^(mis)"]["estimate"]) < 1e-9
    assert out["checks"] == {"full_q_no_longer_eligible": 0, "blind_rows": 80, "blind_now_eligible": 0,
                             "blind_mean_margin": -1.0, "truncated_share": 0.0}
    assert (tmp_path / "out" / "results.json").exists() and "Holm" in (tmp_path / "out" / "report.md").read_text()


def test_holm_family_of_six_and_equivalence(tmp_path):
    out = report_on(tmp_path, mis={72: 0.5, 144: 0.3, 288: 0.0}, irr={72: 0.2, 144: 0.0, 288: 0.0})
    family = out["family"]
    assert len(family["tests"]) == 6
    assert family["reject"] == {"R1: S_H^(mis), K = 72": True, "R1: S_H^(mis), K = 144": True,
                                "R2: S_H^(irr), K = 72": True, "R2: S_H^(irr), K = 144": False,
                                "R3: E_harm^(mis), K = 72": False, "R5: T_irr, K = 72": True}
    assert out["equivalence"]["equivalent"] is True
    large = report_on(tmp_path / "large", mis={72: 0.5, 144: 0.3, 288: 0.2})
    assert large["equivalence"]["equivalent"] is False


def test_report_prints_the_original_estimates_beside(tmp_path):
    from kvsel.studies.replication import report

    original = {"S_H^(mis)": {"estimate": 0.111, "ci95": [0.08, 0.144], "p_two_sided": 0.0002, "items": 1000,
                              "images": 708}}
    confirm_results = tmp_path / "confirm.json"
    confirm_results.write_text(json.dumps({"datasets": {"gqa": {"tests": original, "exploratory": {}}}}))
    followup_results = tmp_path / "followup.json"
    followup_results.write_text(json.dumps({"parts": {"budgets": {"by_k": {"72": {"S_H^(mis)": {
        **original["S_H^(mis)"], "estimate": -0.219}}}}}}))
    ids = [f"r{i}" for i in range(20)]
    record = tmp_path / "replication_gqa.json"
    record.write_text(json.dumps({"replication_ids": ids, "control_ids": ids[:5]}))
    out = report.replication_report(synthetic_run(tmp_path, ids, set(ids[:5])), record, tmp_path / "out", resamples=1000,
                                    original_confirm=confirm_results, original_followup=followup_results)
    assert out["original"]["144"]["S_H^(mis)"]["estimate"] == 0.111
    assert out["original"]["72"]["S_H^(mis)"]["estimate"] == -0.219
    text = (tmp_path / "out" / "report.md").read_text()
    assert "-0.219" in text
    # PREREG 3.3: the planted +0.3 is rejected at both budgets; at K = 72 the original was negative
    assert out["family"]["reading"]["R1: S_H^(mis), K = 144"] == "replicated"
    assert out["family"]["reading"]["R1: S_H^(mis), K = 72"] == "failed replication: opposite to the original"
    assert out["family"]["reading"]["R2: S_H^(irr), K = 144"] == "not rejected"
    assert out["family"]["reading"]["R5: T_irr, K = 72"] == "rejected (no original estimate to compare)"
    assert "failed replication" in text


def test_report_refuses_missing_or_extra_items(tmp_path):
    from kvsel.studies.replication import report

    ids = [f"r{i}" for i in range(6)]
    record = tmp_path / "replication_gqa.json"
    record.write_text(json.dumps({"replication_ids": ids + ["r6"], "control_ids": ids[:2],
                                  "code_sha256": code_identity()}))
    run = synthetic_run(tmp_path, ids, set(ids[:2]))
    with pytest.raises(Refusal, match="missing 1"):
        report.replication_report(run, record, tmp_path / "out", resamples=50)
    record.write_text(json.dumps({"replication_ids": ids[:5], "control_ids": ids[:2],
                                  "code_sha256": code_identity()}))
    with pytest.raises(Refusal, match="1 extra"):
        report.replication_report(run, record, tmp_path / "out", resamples=50)


@pytest.mark.network
def test_run_on_the_tiny_model(tiny_llava, tmp_path, capsys):
    from PIL import Image

    from kvsel.model import bundle_from_model
    from kvsel.models.llava15 import LLAVA
    from kvsel.studies.replication.run import run_replication, units

    bundle = bundle_from_model(tiny_llava, LLAVA.load_tokenizer(), LLAVA.load_processor())
    Image.fromarray(np.random.default_rng(0).integers(0, 255, (60, 80, 3), dtype=np.uint8)).save(tmp_path / "img.jpg")
    items = [item(0), item(1)]
    selection = MainSet.of(["i0", "i1"], ["i0"])
    summary = run_replication(bundle, items, tmp_path, tmp_path / "run", selection, "rec")
    assert capsys.readouterr().out == "2/2 items\n"  # progress only, no per-item outcome
    assert summary["items"] == 2 and summary["control"] == 1 and summary["blind_now_eligible"] in (0, 1, 2)
    control = json.loads((tmp_path / "run" / "items" / "i0.json").read_text())
    main = json.loads((tmp_path / "run" / "items" / "i1.json").read_text())
    assert main["subset"] == "main" and control["subset"] == "control"
    for data, control_item in ((main, False), (control, True)):
        expected = [(u.selector, u.k, u.selection, u.answering) for u in units(control=control_item)] + [("blind", 576, "none", "Q")]
        assert [(r["selector"], r["k"], r["selection"], r["answering"]) for r in data["rows"]] == expected
    assert not any("match" in r["cell"] for r in main["rows"])
    assert {r["cell"] for r in control["rows"]} >= {"B_match", "C_match", "D_match", "match", "B_irr"}
    kept = {(r["k"], r["selection"]): set(r["kept"]) for r in main["rows"] if r["selector"] == "pw"}
    for context in ("Q", "mis", "irr"):  # one probe per context: smaller budgets keep a subset of larger ones
        assert kept[(72, context)] <= kept[(144, context)] <= kept[(288, context)]
        assert [len(kept[(k, context)]) for k in (72, 144, 288)] == [72, 144, 288]
    mtime = (tmp_path / "run" / "items" / "i0.json").stat().st_mtime
    run_replication(bundle, items, tmp_path, tmp_path / "run", selection, "rec")
    assert (tmp_path / "run" / "items" / "i0.json").stat().st_mtime == mtime  # finished items are skipped


def test_commands_freeze_run_and_report(world, tmp_path, monkeypatch, capsys):
    from kvsel import cli
    from kvsel.studies.replication import run as run_module

    items, args, record = world
    undraft(args["prereg"])
    common = ["--prereg", str(args["prereg"]), "--items", str(args["manifest"]), "--screen", str(args["screen"]),
              "--decisions", str(args["decisions"]), "--stage1", str(args["stage1"]),
              "--gqa-record", str(args["gqa_record"]), "--record", str(record)]
    main(["replication", "freeze", *common])
    assert json.loads(capsys.readouterr().out)["replication"] == 3
    seen = {}
    monkeypatch.setattr(cli, "bundle", lambda args, family: "bundle")
    (tmp_path / "img.jpg").touch()  # the run checks its images before loading the model
    monkeypatch.setattr(run_module, "run_replication",
                        lambda bundle, items, images, out, selection, record_sha256: seen.update(
                            ids=list(selection.ids), out=out, record=record_sha256) or {"items": 3})
    main(["replication", "run", *common, "--images", str(tmp_path), "--out", str(tmp_path / "run")])
    assert seen == {"ids": ["i2", "i3", "i4"], "out": tmp_path / "run", "record": sha256_file(record)}
    run = synthetic_run(tmp_path / "synthetic", ["i2", "i3", "i4"], {"i2"})
    main(["replication", "report", *common, "--run", str(run), "--resamples", "50", "--out", str(tmp_path / "rep")])
    assert "K = 144" in (tmp_path / "rep" / "report.md").read_text()
    args["prereg"].write_text(args["prereg"].read_text() + "\nA dated amendment after the run.\n")
    main(["replication", "report", *common, "--run", str(run), "--resamples", "50", "--out", str(tmp_path / "rep")])  # PREREG may grow
    append_decision(args["decisions"], items[3], FAIL, "tester")
    with pytest.raises(SystemExit, match="replication_set_sha256"):  # everything else is still checked
        main(["replication", "report", *common, "--run", str(run), "--resamples", "50", "--out", str(tmp_path / "rep")])
