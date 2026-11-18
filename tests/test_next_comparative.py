"""NeXT stage 3: freeze, run and the pre-registered report (PREREG 4.2-4.4)."""

import json

import numpy as np
import pytest
from PIL import Image

from kvsel.verification import append_decision
from kvsel_next import __main__ as cli
from kvsel.runner import Unit, cell_name
from kvsel_next import comparative, screen
from kvsel_next.selection import SHARES

PREREG = (
    "## 3. GQA replication\n\nOld text.\n\n"
    "## 4. Second model\n\nDRAFT (date): delete after review.\n\n"
    "### 4.1 Screening rules\n\n- Model: NeXT.\n\n"
    "### 4.2 Configuration\n\n- Shares.\n"
)
ELIGIBLE = {"q0": True, "q2": False, "q3": True, "q4": True, "q5": True}  # q1 failed verification
MAIN, CONTROL = ["q0", "q3", "q4"], ["q0"]


def make_item(index):
    return {
        "item_id": f"q{index}", "image_id": f"img{index // 2}", "image_file": f"img{index}.png", "dataset": "gqa",
        "split": "confirm", "question": "What color is the bus?", "a_true": "red", "a_alt": "blue",
        "descriptions": {"mis": "The bus in this image is blue.", "match": "The bus in this image is red.",
                         "irr": "The road in this image is gray."},
    }


def write_flags(screen_dir, flags):
    (screen_dir / "items").mkdir(parents=True, exist_ok=True)
    for item_id, ok in flags.items():
        (screen_dir / "items" / f"{item_id}.json").write_text(json.dumps(
            {"item_id": item_id, "full_q_eligible": ok, "blind_eligible": False if ok else None, "eligible": ok}))
    (screen_dir / "eligible.jsonl").write_text(
        "".join(json.dumps({"item_id": i, "eligible": ok}) + "\n" for i, ok in flags.items()))


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A stage-1 record with a main set of 3 and a control subset of 1, and a finished screen."""
    monkeypatch.setitem(screen.SCREENING, "main_size", 3)
    monkeypatch.setitem(screen.SCREENING, "control_size", 1)
    prereg = tmp_path / "PREREG.md"
    prereg.write_text(PREREG.replace("DRAFT (date): delete after review.\n\n", ""))
    items = [make_item(i) for i in range(6)]
    manifest = tmp_path / "gqa_confirm.jsonl"
    manifest.write_text("".join(json.dumps(item) + "\n" for item in items))
    decisions = tmp_path / "decisions.jsonl"
    for item in items:
        append_decision(decisions, item, "wording" if item["item_id"] == "q1" else "pass", "tester")
    generator = np.random.default_rng(0)
    for i in range(6):
        Image.fromarray(generator.integers(0, 255, (300, 560, 3), dtype=np.uint8)).save(tmp_path / f"img{i}.png")
    stage1 = tmp_path / "freeze" / "next_screening.json"
    screen.freeze_screening(prereg, manifest, stage1)
    write_flags(tmp_path / "screen", ELIGIBLE)
    monkeypatch.setattr(cli, "load_bundle", lambda *_: pytest.fail("loaded the model"))
    return {"tmp": tmp_path, "prereg": prereg, "items": items, "manifest": manifest, "decisions": decisions,
            "stage1": stage1, "record": tmp_path / "freeze" / "next_gqa.json", "screen": tmp_path / "screen"}


def common(world):
    return ["--prereg", str(world["prereg"]), "--items", str(world["manifest"]), "--screen", str(world["screen"]),
            "--decisions", str(world["decisions"]), "--stage1", str(world["stage1"]), "--record", str(world["record"])]


def run_args(world):
    return ["run", *common(world), "--images", str(world["tmp"]), "--out", str(world["tmp"] / "run"), "--device", "cpu"]


LLAVA15 = {"budgets": {k: {"margins": {"S_H^(mis)": {"estimate": est, "ci95": [est - 0.03, est + 0.03],
                                                     "p_two_sided": 0.0002, "items": 1000, "images": 708}}}
                       for k, est in (("72", 0.2), ("144", 0.111), ("288", -0.05))}}


def report_args(world, run=None):
    llava15 = world["tmp"] / "replication_results.json"
    llava15.write_text(json.dumps(LLAVA15))
    return ["report", *common(world), "--run", str(run or world["tmp"] / "run"), "--resamples", "200",
            "--llava15", str(llava15), "--out", str(world["tmp"] / "report")]


# --- freeze ----------------------------------------------------------------------------------------


def test_freeze_records_the_main_set_control_subset_and_hashes(world, capsys):
    cli.main(["freeze", *common(world)])
    assert json.loads(capsys.readouterr().out)["main"] == 3
    record = json.loads(world["record"].read_text())
    assert record["main_ids"] == MAIN and record["control_ids"] == CONTROL
    for key in ("prereg_sha256", "stage1_record_sha256", "screen_flags_sha256", "main_set_sha256", "code_sha256"):
        assert len(record[key]) == 64, key
    assert set(record["manifest"]) == {"gqa_confirm.jsonl"}
    assert record["settings"]["shares"] == [0.125, 0.25, 0.5] and record["settings"]["equivalence"] is None


def test_unit_lists_match_prereg_42():
    """Written out from section 4.2: per share, A then B, C, D per condition; then the Full anchors."""
    def listed(units):
        return [(u.selector, u.share, f"{u.selection}>{u.answering}") for u in units]

    per_share = ["Q>Q", "mis>mis", "Q>mis", "mis>Q", "irr>irr", "Q>irr", "irr>Q"]
    control_extra = ["match>match", "Q>match", "match>Q"]
    main = [("pw", s, c) for s in (0.125, 0.25, 0.5) for c in per_share]
    main += [("full", 1.0, f"none>{a}") for a in ("Q", "mis", "irr")]
    control = [("pw", s, c) for s in (0.125, 0.25, 0.5) for c in per_share + control_extra]
    control += [("full", 1.0, f"none>{a}") for a in ("Q", "mis", "irr", "match")]
    assert listed(comparative.units(control=False)) == main
    assert listed(comparative.units(control=True)) == control
    assert comparative.BLIND == comparative.NextUnit("full", 1.0, "none", "Q")


def test_freeze_refuses_a_draft_a_blank_and_a_different_record(world):
    settled = world["prereg"].read_text()
    world["prereg"].write_text(PREREG)
    with pytest.raises(SystemExit, match="DRAFT"):
        cli.main(["freeze", *common(world)])
    world["prereg"].write_text(settled.replace("- Shares.", "- Shares: **[to fill before stage 3]**."))
    with pytest.raises(SystemExit, match="blank"):
        cli.main(["freeze", *common(world)])
    world["prereg"].write_text(PREREG.replace("DRAFT (date): delete after review.\n\n", "Mentions DRAFT mid-line.\n\n"))
    cli.main(["freeze", *common(world)])
    first = world["record"].read_text()
    cli.main(["freeze", *common(world)])  # unchanged: the same record
    assert world["record"].read_text() == first
    world["prereg"].write_text(world["prereg"].read_text() + "\nMore text.\n")
    with pytest.raises(SystemExit, match="already exists"):
        cli.main(["freeze", *common(world)])


def test_a_short_set_after_the_whole_pool_is_every_eligible_item(world):
    """PREREG 4.5(d): stage 1 froze a size of 3; only q0 and q3 are eligible and every pool item is screened."""
    write_flags(world["screen"], {**ELIGIBLE, "q4": False, "q5": False})
    cli.main(["freeze", *common(world)])
    record = json.loads(world["record"].read_text())
    assert record["main_ids"] == ["q0", "q3"] and record["control_ids"] == CONTROL
    assert record["main_size_frozen_at_stage1"] == 3


def test_an_unscreened_pool_item_ends_the_main_set(world):
    write_flags(world["screen"], {"q0": True, "q2": False, "q4": True, "q5": True})  # q3 never screened
    with pytest.raises(SystemExit, match="1 of 3"):
        cli.main(["freeze", *common(world)])


# --- run refusals ----------------------------------------------------------------------------------


@pytest.mark.parametrize("change", ["prereg", "decision", "flags", "code", "stage1"])
def test_run_refuses_any_change_before_loading_the_model(world, monkeypatch, change):
    cli.main(["freeze", *common(world)])
    if change == "prereg":
        world["prereg"].write_text(world["prereg"].read_text() + "\nA dated amendment.\n")
    elif change == "decision":
        append_decision(world["decisions"], world["items"][3], "wording", "tester")
    elif change == "flags":
        write_flags(world["screen"], {**ELIGIBLE, "q5": False})
    elif change == "code":
        monkeypatch.setattr(screen, "code_hash", lambda: "0" * 64)
    else:
        world["stage1"].write_text(world["stage1"].read_text() + " ")
    with pytest.raises(SystemExit, match="changed since|main set has"):
        cli.main(run_args(world))


def test_amended_code_is_accepted(world, monkeypatch, capsys):
    cli.main(["freeze", *common(world)])
    frozen = json.loads(world["record"].read_text())["code_sha256"]
    (world["record"].parent / "code_amendments.json").write_text(json.dumps([{"from": frozen, "to": "1" * 64}]))
    monkeypatch.setattr(screen, "code_hash", lambda: "1" * 64)
    assert comparative.check(world["prereg"], world["manifest"], world["screen"], world["decisions"], world["stage1"],
                             world["record"])["main_ids"] == MAIN


def test_settings_changed_since_the_thin_path_freeze_are_caught(world):
    cli.main(["freeze", *common(world)])
    record = json.loads(world["record"].read_text())
    thin = [u for u in record["settings"]["main_units"] if u["share"] in (0.25, 1.0) and "irr" not in u.values()]
    record["settings"] |= {"main_units": thin, "control_units": thin}
    world["record"].write_text(json.dumps(record))
    with pytest.raises(SystemExit, match="settings"):
        cli.main(run_args(world))


def test_run_refuses_missing_images_before_loading_the_model(world):
    cli.main(["freeze", *common(world)])
    (world["tmp"] / "img3.png").unlink()
    with pytest.raises(SystemExit, match="1 images are missing"):
        cli.main(run_args(world))


# --- end to end on the tiny model ------------------------------------------------------------------


def plant(run, effect):
    """Adds `effect` nats to every C_mis margin, so S_H^(mis) moves by exactly `effect`."""
    for path in (run / "items").glob("*.json"):
        data = json.loads(path.read_text())
        for row in data["rows"]:
            if row["selector"] == "pw" and row["cell"] == "C_mis":
                row["scores"]["true"]["capitalized"][0] += effect
        path.write_text(json.dumps(data))


@pytest.mark.network
def test_freeze_run_resume_and_report_on_the_tiny_model(world, monkeypatch, capsys, tiny_llava_next):
    from kvsel.model import bundle_from_model
    from kvsel.models.llava_next import NEXT

    processor = NEXT.load_processor()
    bundle = bundle_from_model(tiny_llava_next, processor.tokenizer, processor, family=NEXT)
    monkeypatch.setattr(cli, "load_bundle", lambda device, model_dir: bundle)
    cli.main(["freeze", *common(world)])

    probes = []
    real_probe = comparative.probe
    monkeypatch.setattr(comparative, "probe", lambda *a: probes.append(1) or real_probe(*a))
    real, calls = comparative.run_item, []

    def stops_on_the_second_item(*args):
        if len(calls) == 2:  # the first item's units and its Blind_Q
            raise KeyboardInterrupt
        calls.append(args[1]["item_id"])
        return real(*args)

    monkeypatch.setattr(comparative, "run_item", stops_on_the_second_item)
    with pytest.raises(KeyboardInterrupt):
        cli.main(run_args(world))
    run = world["tmp"] / "run"
    assert [p.stem for p in (run / "items").glob("*.json")] == ["q0"] and not list(run.glob("items/*.tmp"))
    first = (run / "items" / "q0.json").stat().st_mtime
    monkeypatch.setattr(comparative, "run_item", real)
    capsys.readouterr()
    cli.main(run_args(world))
    printed = capsys.readouterr().out.splitlines()
    assert printed[0] == "3/3 items"  # progress, then the summary's counts only
    summary = json.loads("\n".join(printed[1:]))
    assert set(summary) == {"full_q_no_longer_eligible", "blind_now_eligible"}
    assert (run / "items" / "q0.json").stat().st_mtime == first  # not redone
    assert len(probes) == 4 + 3 + 3  # once per context and item: q0 (control: Q, mis, irr, match), q3, q4

    data = {i: json.loads((run / "items" / f"{i}.json").read_text()) for i in MAIN}
    assert [data[i]["subset"] for i in MAIN] == ["control", "main", "main"]
    for item_id in MAIN:
        planned = comparative.units(control=item_id in CONTROL) + [comparative.BLIND]
        assert [(r["selector"], r["share"], r["selection"], r["answering"]) for r in data[item_id]["rows"]] == [
            ("blind" if n == len(planned) - 1 else u.selector, u.share, u.selection, u.answering)
            for n, u in enumerate(planned)]
    layout = {"base": 24, "grid": [24, 44], "total": 1656}  # 560 x 300, worked in test_next.ASPECTS
    kept = {}
    for row in data["q3"]["rows"]:
        assert row["layout"] == layout
        if row["selector"] == "pw":
            assert row["k"] == row["share"] * (1656 - 24) and len(row["kept"]) == row["k"] + 24
            kept[(row["share"], row["selection"])] = set(row["kept"])
        else:
            assert row["share"] == 1.0 and row["k"] == 1656
    for context in ("Q", "mis", "irr"):  # one probe gives nested kept sets
        assert kept[(0.125, context)] <= kept[(0.25, context)] <= kept[(0.5, context)]

    def n1():
        cli.main(report_args(world))
        return json.loads((world["tmp"] / "report" / "results.json").read_text())["family"]["tests"]["N1: S_H^(mis), 25.0%"]

    before = n1()
    plant(run, 0.7)
    after = n1()
    assert after["estimate"] == pytest.approx(before["estimate"] + 0.7)
    assert after["items"] == 3 and after["images"] == 3  # img0, img1, img2
    assert "visual tokens 1656-1656" in (world["tmp"] / "report" / "report.md").read_text()


def test_report_refuses_missing_or_extra_items_and_lets_prereg_grow(world, tmp_path):
    cli.main(["freeze", *common(world)])
    run = synthetic_run(tmp_path / "fake_run", MAIN[:2], set(CONTROL))
    with pytest.raises(SystemExit, match="missing 1"):
        cli.main(report_args(world, run))
    synthetic_run(run, [MAIN[2], "q5"], set())
    with pytest.raises(SystemExit, match="1 extra"):
        cli.main(report_args(world, run))
    (run / "items" / "q5.json").unlink()
    world["prereg"].write_text(world["prereg"].read_text() + "\nA dated amendment after the run.\n")
    cli.main(report_args(world, run))  # the report allows it; run does not
    with pytest.raises(SystemExit, match="prereg_sha256"):
        cli.main(run_args(world))


# --- the full report on planted effects ------------------------------------------------------------

S_H = {"mis": (0.5, 0.3, 0.1), "irr": (0.2, 0.0, 0.05), "match": (-0.1, -0.2, -0.3)}
S_Q = {"mis": (0.2, 0.1, 0.0), "irr": (0.05, 0.15, 0.25), "match": (0.3, 0.2, 0.1)}
E_HARM = {"mis": 0.2, "irr": 0.0, "match": 0.1}


def synthetic_run(run, ids, control, noise=0.05, seed=0):
    """Item files as the run writes them, two items per image, with these margins per item:
    A = a, B_c = b, C_c = b + S_H[c], D_c = a - S_Q[c], Full_Q = a + 1, Full_c = b + 1 + E_HARM[c].
    So S_H, S_Q, I = S_H - S_Q, E_harm and T_c = S_H^(mis) - S_H^(c) are known at every share; only C has noise."""
    forms = ("lower", "capitalized", "upper")

    def scores(m):
        return {"true": {f: [m, 0.0] for f in forms}, "alt": {f: [0.0, 0.0] for f in forms}}

    generator = np.random.default_rng(seed)
    (run / "items").mkdir(parents=True, exist_ok=True)
    for n, item_id in enumerate(ids):
        a, b = 2 + generator.normal(0, 0.5), 1 + generator.normal(0, 0.5)
        total = 1656 if n % 2 else 1676  # wide and tall, as test_next.ASPECTS
        rows = []
        planned = comparative.units(control=item_id in control) + [comparative.BLIND]
        for position, u in enumerate(planned):
            blind = position == len(planned) - 1
            unit = Unit(u.selector, 1, False, u.selection, u.answering)
            cell, at = cell_name(unit), SHARES.index(u.share) if u.share in SHARES else None
            if blind:
                value = -1.0
            elif u.selector == "full":
                value = a + 1 if cell == "Q" else b + 1 + E_HARM[cell]
            elif cell == "A":
                value = a
            elif cell.startswith("B_"):
                value = b
            elif cell.startswith("C_"):
                value = b + S_H[cell[2:]][at] + generator.normal(0, noise)
            else:
                value = a - S_Q[cell[2:]][at]
            rows.append({"selector": "blind" if blind else u.selector, "share": u.share, "k": 1, "reserve": False,
                         "selection": u.selection, "answering": u.answering, "cell": cell,
                         "kept": [0, 1] if u.selection == "Q" else [0, 2], "scores": scores(value),
                         "outcome": "other" if blind else "correct", "truncated": False,
                         "layout": {"base": 24, "total": total, "grid": [24, 44] if n % 2 else [44, 24]}})
        (run / "items" / f"{item_id}.json").write_text(json.dumps({
            "item_id": item_id, "image_id": f"img{n // 2}", "subset": "control" if item_id in control else "main",
            "rows": rows}))
    return run


@pytest.fixture
def planted(tmp_path):
    from kvsel_next import report

    ids = [f"r{i}" for i in range(80)]
    record = tmp_path / "next_gqa.json"
    record.write_text(json.dumps({"main_ids": ids, "control_ids": ids[:20], "main_size_frozen_at_stage1": 80,
                                  "settings": comparative.settings()}))
    run = synthetic_run(tmp_path / "run", ids, set(ids[:20]))
    llava15 = tmp_path / "replication_results.json"
    llava15.write_text(json.dumps(LLAVA15))
    return report.report(run, record, tmp_path / "out", llava15, resamples=1000), tmp_path / "out"


def test_every_estimand_at_every_share_is_recovered(planted):
    found, _ = planted
    for at, share in enumerate(("12.5%", "25.0%", "50.0%")):
        margins = found["shares"][share]["margins"]
        for c in ("mis", "irr", "match"):
            assert margins[f"S_H^({c})"]["estimate"] == pytest.approx(S_H[c][at], abs=0.03), (share, c)
            assert margins[f"S_Q^({c})"]["estimate"] == pytest.approx(S_Q[c][at], abs=1e-9), (share, c)
            assert margins[f"I^({c})"]["estimate"] == pytest.approx(S_H[c][at] - S_Q[c][at], abs=0.03), (share, c)
            assert margins[f"E_harm^({c})"]["estimate"] == pytest.approx(E_HARM[c], abs=1e-9), (share, c)
        for c in ("irr", "match"):
            assert margins[f"T_{c}"]["estimate"] == pytest.approx(S_H["mis"][at] - S_H[c][at], abs=0.04), (share, c)
        assert margins["T_irr"]["items"] == 80 and margins["T_match"]["items"] == 20  # match: control items only
        assert margins["S_H^(match)"]["items"] == 20
        assert set(found["shares"][share]["behaviour_paired"]) >= {"correct: C_mis − B_mis", "alt: C_match − B_match"}
        assert set(found["overlap"][share]) == {"mis", "irr", "match"}
    assert found["checks"]["blind_now_eligible"] == 0 and found["checks"]["full_q_no_longer_eligible"] == 0
    assert (found["checks"]["visual_tokens_min"], found["checks"]["visual_tokens_max"]) == (1656, 1676)
    assert found["checks"]["grids"] == {"24x44": 40, "44x24": 40}


def test_holm_over_four_at_25_percent(planted):
    found, _ = planted
    family = found["family"]
    names = ["N1: S_H^(mis), 25.0%", "N2: S_H^(irr), 25.0%", "N3: E_harm^(mis), 25.0%", "N4: T_irr, 25.0%"]
    assert list(family["tests"]) == names and family["alpha"] == 0.05
    p = {name: family["tests"][name]["p_two_sided"] for name in names}
    # By hand: N1 (0.3), N3 (0.2, noiseless) and N4 (0.3) sit at the bootstrap's floor of 2/1001; N2 (0.0) does not.
    assert p[names[0]] == p[names[2]] == p[names[3]] == pytest.approx(2 / 1001) and p[names[1]] > 0.05
    # Holm's step-down, thresholds written out: the three tied smallest p against 0.05/4, 0.05/3, 0.05/2 all pass;
    # N2 against 0.05 fails. Bonferroni (0.05/4 each) gives the same here, since the three sit at the floor.
    ordered = sorted(names, key=p.get)
    assert ordered[-1] == names[1]
    assert [p[n] <= t for n, t in zip(ordered, (0.05 / 4, 0.05 / 3, 0.05 / 2, 0.05))] == [True, True, True, False]
    assert family["reject"] == {names[0]: True, names[1]: False, names[2]: True, names[3]: True}
    assert "equivalence" not in found


def test_report_prints_llava15_beside_and_states_the_checks(planted):
    found, out = planted
    text = (out / "report.md").read_text()
    assert "Holm family of 4" in text and "Equivalence: not tested" in text
    for share, k, old, same in (("12.5%", 72, "+0.200 [+0.170, +0.230]", "yes"), ("25.0%", 144, "+0.111 [+0.081, +0.141]", "yes"),
                                ("50.0%", 288, "-0.050 [-0.080, -0.020]", "no")):  # NeXT's S_H^(mis): 0.5, 0.3, 0.1
        row = next(line for line in text.splitlines() if line.startswith(f"| S_H^(mis) | {share} | {k} |"))
        assert row.startswith(f"| S_H^(mis) | {share} | {k} | {old} | ") and row.endswith(f"| {same} |")
    assert "| S_H^(irr) | 25.0% | 144 | — | " in text  # absent from the replication file: shown, not dropped
    assert "never pooled" in text
    assert "Blind_Q eligible on 0 of 80" in text and "visual tokens 1656-1676" in text and "truncated answers 0.00%" in text
    assert "24x44 (40)" in text and "Shortfall" not in text


def test_report_states_a_shortfall(planted):
    from kvsel_next import report

    found, _ = planted
    text = report.render({**found, "main_size_frozen_at_stage1": 1000})
    assert "**Shortfall:** 80 eligible items, not 1000" in text
