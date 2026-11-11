"""Follow-ups (PREREG 2.4): unit lists, selectors, restoration sets, the run and the report."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from test_confirm_analysis import make_run, scores

from kvsel.confirm import code_hash_all
from kvsel.followup import run as fu
from kvsel.followup.report import followup_report, followup_results
from kvsel.runner import cell_name, pilot_units


def test_followup_code_is_outside_the_frozen_code_hash():
    from kvsel import confirm

    hashed = {p.name for p in Path(confirm.__file__).parent.glob("*.py")}
    assert "run.py" not in hashed and "report.py" not in hashed
    assert Path(fu.__file__).parent.name == "followup"


def test_part_units():
    budgets = fu.part_units("budgets")
    assert len(budgets) == 12 and {u.k for u in budgets} == {288, 72}
    assert [cell_name(u) for u in budgets[:4]] == ["A", "B_mis", "C_mis", "D_mis"]
    assert [cell_name(u) for u in budgets[4:6]] == ["Q", "mis"]
    lt = fu.part_units("lt")
    assert len(lt) == 4 and {u.selector for u in lt} == {"lt"} and {u.k for u in lt} == {144}
    random = fu.part_units("random")
    assert len(random) == 10 and sorted({u.seed for u in random}) == [0, 1, 2, 3, 4]
    assert {cell_name(u) for u in random} == {"Q", "mis"}
    with pytest.raises(ValueError):
        fu.part_units("restore")
    # the follow-up units carry the confirmatory unit fields, so rows look the same
    assert set(asdict(pilot_units()[0])) <= set(asdict(budgets[0]))


def test_random_sets_are_seeded_per_item():
    a = fu.random_set("i1", 0)
    assert len(a) == 144 == len(set(a.tolist())) and np.all(np.diff(a) > 0) and a.max() < 576
    assert np.array_equal(a, fu.random_set("i1", 0))
    assert not np.array_equal(a, fu.random_set("i1", 1))
    assert not np.array_equal(a, fu.random_set("i2", 0))


def restoration_case():
    """S_Q keeps 0..143, S_mis keeps 20..163; pushed out = 0..19; evidence = 0..9 plus 150..159."""
    scores_q = np.zeros(576)
    scores_q[:144] = np.linspace(2, 1, 144)  # 0 highest
    scores_mis = np.zeros(576)
    scores_mis[20:164] = np.linspace(2, 1, 144)  # 163 lowest
    evidence = set(range(10)) | set(range(150, 160))
    return scores_q, scores_mis, evidence


def test_restoration_sets():
    scores_q, scores_mis, evidence = restoration_case()
    sets = fu.restoration_sets("i1", scores_q, scores_mis, evidence, 4)
    s_mis = set(range(20, 164))
    # removed: the 4 lowest-scoring non-evidence entries of S_mis (163..160), never evidence 150..159
    assert sets["removed"] == [160, 161, 162, 163]
    assert set(sets["treatment"].tolist()) == (s_mis - {160, 161, 162, 163}) | {0, 1, 2, 3}
    for kept in sets["controls"]:
        added = set(kept.tolist()) - s_mis
        assert len(kept) == 144 and len(added) == 4 and added <= set(range(10, 20))
    assert len({tuple(k.tolist()) for k in sets["controls"]}) > 1
    again = fu.restoration_sets("i1", scores_q, scores_mis, evidence, 4)
    assert all(np.array_equal(x, y) for x, y in zip(sets["controls"], again["controls"], strict=True))
    assert len(sets["treatment"]) == 144


def test_restoration_needs_enough_pushed_out_entries():
    scores_q, scores_mis, evidence = restoration_case()
    assert fu.restoration_sets("i1", scores_q, scores_mis, set(range(3)), 4) is None  # 3 evidence entries out
    assert fu.restoration_sets("i1", scores_q, scores_mis, set(range(17)), 4) is None  # 3 non-evidence out
    assert fu.restoration_sets("i1", scores_q, scores_q, evidence, 4) is None  # nothing pushed out


def test_lt_early_scores_are_unpooled():
    values = np.zeros(576)
    values[5] = 1.0
    assert fu.top_k(fu.lt_early(values), 1).tolist() == [5]


def confirm_fixture(tmp_path, n=20, n_control=8):
    run, manifest = make_run(tmp_path, "confirm", n=n, n_control=n_control, effect=0.3)
    record = tmp_path / "record.json"
    record.write_text(json.dumps({"main_items": n, "control_items": n_control, "code_sha256": code_hash_all()}))
    items = [json.loads(line) for line in manifest.read_text().splitlines()]
    return run, manifest, record, items


def test_confirm_selection(tmp_path):
    run, _, record, items = confirm_fixture(tmp_path)
    selection = fu.confirm_selection(run, record, items)
    assert len(selection["main"]) == 20 and len(selection["control"]) == 8 and selection["code_unchanged"]
    (run / "items" / "extra.json").write_text(json.dumps({"item_id": "extra", "subset": "main"}))
    with pytest.raises(SystemExit, match="move files outside the main set"):
        fu.confirm_selection(run, record, items)


def confirm_margins(run):
    out = {}
    for path in (run / "items").glob("*.json"):
        data = json.loads(path.read_text())
        out[data["item_id"]] = {f"{r['selector']}|{r['cell']}": sum(r["scores"]["true"]["capitalized"])
                                for r in data["rows"] if not r["reserve"]}
        out[data["item_id"]]["kept"] = {r["cell"]: r["kept"] for r in data["rows"] if r["selector"] == "pw" and not r["reserve"]}
    return out


def row(unit, value, outcome="correct"):
    return {**asdict(unit), "cell": cell_name(unit), "kept": None, "scores": scores(value), "outcome": outcome}


def write_followups(tmp_path, run, items, grounding):
    """Planted: S_H = 0.5 and S_Q = 0 at K = 288 and 72; LT S_H = 0.2; random 0.25 below PW B_mis;
    restoration treatment 0.4 above its controls, which equal the reference."""
    confirm = confirm_margins(run)
    out = tmp_path / "followup"
    for item in items:
        item_id = item["item_id"]
        values = confirm[item_id]
        kept_144 = {"Q": values["kept"]["A"], "mis": values["kept"]["B_mis"]}
        b = values["pw|B_mis"]
        parts = {
            "budgets": [row(u, {"A": 2.0, "B_mis": 1.0, "C_mis": 1.5, "D_mis": 2.0, "Q": 0.0, "mis": 0.0}[cell_name(u)])
                        for u in fu.part_units("budgets")],
            "lt": [row(u, {"A": 2.0, "B_mis": 1.0, "C_mis": 1.2, "D_mis": 2.0}[cell_name(u)]) for u in fu.part_units("lt")],
            "random": [row(u, values["pw|A"] - 0.1 if u.answering == "Q" else b - 0.25) for u in fu.part_units("random")],
        }
        if item_id in grounding:
            reference = fu.FUnit("restore", 144, False, "mis", "mis", arm="reference")
            rows = [row(reference, b)]
            for m in fu.RESTORE_M:
                rows.append(row(fu.FUnit("restore", 144, False, "mis", "mis", m=m, arm="treatment"), b + 0.4))
                rows += [row(fu.FUnit("restore", 144, False, "mis", "mis", seed=s, m=m, arm="control"), b) for s in fu.SEEDS]
            parts["restore"] = rows
        for part, rows in parts.items():
            (out / part / "items").mkdir(parents=True, exist_ok=True)
            extra = {"qualifying": {"4": True, "8": True}} if part == "restore" else {}
            payload = {"item_id": item_id, "image_id": f"img{int(item_id[2:]) // 2}", "dataset": "gqa", "part": part,
                       "kept_144": kept_144, **extra, "rows": rows}
            (out / part / "items" / f"{item_id}.json").write_text(json.dumps(payload))
    return out


def test_report_recovers_planted_followup_effects(tmp_path):
    run, manifest, _, items = confirm_fixture(tmp_path)
    grounding = {f"it{i}" for i in range(8)}  # every control item has an evidence box
    out = write_followups(tmp_path, run, items, grounding)
    results = followup_results(out, run, manifest, resamples=500)
    parts = results["parts"]
    assert all(entry["complete"] for entry in parts.values())
    for k in ("288", "72"):
        assert parts["budgets"]["by_k"][k]["S_H^(mis)"]["estimate"] == pytest.approx(0.5)
        assert parts["budgets"]["by_k"][k]["S_Q^(mis)"]["estimate"] == pytest.approx(0.0)
    assert parts["budgets"]["by_k"]["144"]["S_H^(mis)"]["estimate"] == pytest.approx(0.3, abs=0.05)  # the confirmatory run
    assert parts["lt"]["estimates"]["S_H^(mis)"]["estimate"] == pytest.approx(0.2)
    assert parts["random"]["estimates"]["PW B_mis − random (mis)"]["estimate"] == pytest.approx(0.25)
    assert parts["random"]["estimates"]["PW A − random (Q)"]["estimate"] == pytest.approx(0.1)
    restore = parts["restore"]
    assert restore["items"] == 8 and restore["qualifying"] == {"4": 8, "8": 8}
    assert restore["reference_vs_confirm"]["max_abs_diff"] == pytest.approx(0.0)
    for m in ("4", "8"):
        assert restore["by_m"][m]["treatment − control (EQ3)"]["estimate"] == pytest.approx(0.4)
        assert restore["by_m"][m]["control − reference"]["estimate"] == pytest.approx(0.0)
    assert results["kept_144_matches_confirm"] == {"Q": 1.0, "mis": 1.0}
    assert parts["budgets"]["behaviour"]["pw@288|A"]["correct"] == 1.0


def test_report_marks_incomplete_parts_and_writes_files(tmp_path):
    run, manifest, _, items = confirm_fixture(tmp_path)
    out = write_followups(tmp_path, run, items, {f"it{i}" for i in range(8)})
    (out / "lt" / "items" / "it3.json").unlink()
    summary = followup_report(out, run, manifest, tmp_path / "report", resamples=200)
    assert summary["lt"] == {"items": 19, "complete": False}
    text = (tmp_path / "report" / "report.md").read_text()
    assert "not confirmatory" in text and "INCOMPLETE: 19 of 20" in text
    json.loads((tmp_path / "report" / "results.json").read_text())


def test_cli_parses():
    from kvsel.followup.__main__ import build_parser

    args = build_parser().parse_args(["run", "--items", "m.jsonl", "--images", "img", "--confirm-run", "r",
                                      "--record", "rec.json", "--out", "o", "--parts", "budgets", "restore"])
    assert args.parts == ["budgets", "restore"] and args.head_precision is None
    args = build_parser().parse_args(["report", "--run", "o", "--confirm-run", "r", "--items", "m.jsonl", "--out", "x"])
    assert args.resamples == 10000


@pytest.mark.network
def test_followup_runs_on_the_tiny_model(tiny_llava, tmp_path):
    from test_confirm import make_item

    from kvsel.engine import probe
    from kvsel.model import (
        bundle_from_model,
        load_image_processor,
        load_tokenizer,
        pixel_values,
        prefill,
    )
    from kvsel.runner import Q, contexts_for

    bundle = bundle_from_model(tiny_llava, load_tokenizer(), load_image_processor())
    Image.fromarray(np.random.default_rng(0).integers(0, 255, (60, 80, 3), dtype=np.uint8)).save(tmp_path / "img.jpg")
    items = [make_item(0), make_item(1)]
    for item in items:
        item["image_size"] = {"width": 80, "height": 60}
        item["evidence_box"] = {"x": 0, "y": 0, "w": 80, "h": 60}  # the whole image: plenty of evidence
    manifest = tmp_path / "m.jsonl"
    manifest.write_text("".join(json.dumps(i) + "\n" for i in items))

    # the follow-up probe gives exactly the confirmatory probe's PW-attn input
    tokens = contexts_for(items[0], bundle.tokenizer, bundle.family)
    prefix = prefill(bundle, tokens[Q].prefix_ids, pixel_values(bundle, Image.open(tmp_path / "img.jpg").convert("RGB")))
    import torch

    suffix = torch.tensor([tokens["mis"].suffix_ids])
    late, layer2 = fu.probe_two(bundle.decoder, prefix, suffix)
    assert np.array_equal(late, probe(bundle.decoder, prefix, suffix)) and layer2.shape == (576,)

    selection = {"main": ["i0", "i1"], "control": ["i0"]}
    summary = fu.run_followup(bundle, items, manifest, tmp_path, tmp_path / "fu", selection)
    assert summary["files_per_part"] == {"budgets": 2, "lt": 2, "random": 2, "restore": 1}
    budgets = json.loads((tmp_path / "fu" / "budgets" / "items" / "i1.json").read_text())
    assert len(budgets["rows"]) == 12 and budgets["subset"] == "main" and len(budgets["kept_144"]["mis"]) == 144
    assert {len(r["kept"]) for r in budgets["rows"]} == {288, 72}
    restore = json.loads((tmp_path / "fu" / "restore" / "items" / "i0.json").read_text())
    expected = 1 + sum(6 for m in fu.RESTORE_M if restore["qualifying"][str(m)])
    assert len(restore["rows"]) == expected and {len(r["kept"]) for r in restore["rows"]} == {144}
    mtime = (tmp_path / "fu" / "lt" / "items" / "i0.json").stat().st_mtime
    fu.run_followup(bundle, items, manifest, tmp_path, tmp_path / "fu", selection)
    assert (tmp_path / "fu" / "lt" / "items" / "i0.json").stat().st_mtime == mtime  # finished items are skipped
