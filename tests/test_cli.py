"""The command line itself: argument names must not collide with the dispatcher."""

import json
import sys
from pathlib import Path

import pytest

from kvsel.cli import main


def run_cli(monkeypatch, *arguments):
    monkeypatch.setattr(sys, "argv", ["kvsel", *arguments])
    main()


def test_report_reaches_the_report_code_through_the_command_line(monkeypatch, tmp_path):
    with pytest.raises(SystemExit, match="no item files"):
        run_cli(monkeypatch, "report", str(tmp_path))


def test_report_writes_its_output_file(monkeypatch, tmp_path, capsys):
    from kvsel import analysis

    monkeypatch.setattr(analysis, "report", lambda run, surface, resamples, verified: f"report for {run} {surface}")
    target = tmp_path / "out" / "report.md"
    run_cli(monkeypatch, "report", "somewhere", "--surface", "capitalized", "--out", str(target))
    assert target.read_text() == "report for somewhere capitalized"


@pytest.mark.parametrize("command", [
    "build-gqa", "build-textvqa", "fetch-model", "fetch-images", "check-serialization", "build-s1", "s1", "timing", "coverage",
    "validate", "diagnose-e2", "check-tokenization", "screen", "pilot", "prompt-test", "verify", "report",
    "freeze-screening", "screen-confirm", "freeze-comparative", "confirm", "blind-rate", "confirm-report",
])
def test_every_command_parses_its_help(monkeypatch, command):
    with pytest.raises(SystemExit) as stopped:
        run_cli(monkeypatch, command, "--help")
    assert stopped.value.code == 0


@pytest.mark.parametrize("arguments,expected", [([], "llava"), (["--model", "qwen"], "qwen")])
def test_check_serialization_dispatches_with_the_chosen_family(monkeypatch, tmp_path, capsys, arguments, expected):
    items = tmp_path / "items.jsonl"
    items.write_text("")
    seen = {}
    monkeypatch.setattr("kvsel.model.load_tokenizer", lambda model_dir, family: seen.setdefault("family", family.name))
    run_cli(monkeypatch, "check-serialization", "--items", str(items), *arguments)
    assert '"pass": true' in capsys.readouterr().out and seen["family"] == expected


def test_fetch_model_uses_the_family_repository_and_directory(monkeypatch):
    calls = {}
    monkeypatch.setattr("kvsel.model_files.fetch_model", lambda out, workers, model_id: calls.update(out=out, model_id=model_id) or [])
    run_cli(monkeypatch, "fetch-model", "--model", "qwen")
    assert calls == {"out": __import__("pathlib").Path("models/qwen2.5-vl-7b-instruct"), "model_id": "Qwen/Qwen2.5-VL-7B-Instruct"}


def test_screen_prints_eligibility_and_casing(monkeypatch, tmp_path, capsys):
    import json

    items = tmp_path / "items.jsonl"
    items.write_text(json.dumps({"item_id": "q1", "split": "dev"}) + "\n")
    out = tmp_path / "screen"

    def fake_screen(bundle, loaded_items, images, out_dir):
        assert [item["item_id"] for item in loaded_items] == ["q1"] and out_dir == out
        (out_dir / "items").mkdir(parents=True)
        row = {"selector": "full", "cell": "Q", "greedy_text": "NOKIA", "outcome": "correct",
               "scores": {"true": {f: [-0.1] for f in ("lower", "capitalized", "upper")},
                          "alt": {f: [-3.0] for f in ("lower", "capitalized", "upper")}}}
        (out_dir / "items" / "q1.json").write_text(json.dumps(
            {"item_id": "q1", "image_id": "i", "contexts": {}, "rows": [row]}))

    monkeypatch.setattr("kvsel.runner.run_screen", fake_screen)
    monkeypatch.setattr("kvsel.cli._bundle", lambda args: None)
    run_cli(monkeypatch, "screen", "--items", str(items), "--images", str(tmp_path), "--out", str(out))
    printed = capsys.readouterr().out
    assert "eligible under `upper`: 1 of 1" in printed
    assert "| upper | 1.000 |" in printed


def test_timing_and_s1_pass_their_arguments(monkeypatch, tmp_path):
    from kvsel import cli, sanity

    items = tmp_path / "items.jsonl"
    items.write_text("".join(json.dumps({"item_id": str(i), "split": "dev"}) + "\n" for i in range(30)))
    seen = {}
    monkeypatch.setattr(cli, "_bundle", lambda args: "bundle")
    monkeypatch.setattr(sanity, "run_timing", lambda b, it, images, volumes: seen.update(n=len(it), volumes=volumes) or {"ok": 1})
    monkeypatch.setattr(sanity, "run_s1", lambda b, it, images: seen.update(s1=len(it)) or {"items": len(it), "rows": []})
    run_cli(monkeypatch, "timing", "--items", str(items), "--images", str(tmp_path), "--limit", "5",
            "--out", str(tmp_path / "r1.json"))
    run_cli(monkeypatch, "s1", "--items", str(items), "--images", str(tmp_path), "--out", str(tmp_path / "s1.json"))
    assert seen["n"] == 5 and seen["s1"] == 30 and seen["volumes"]["confirmatory_items"] == 2000
    assert json.loads((tmp_path / "r1.json").read_text()) == {"ok": 1}


def test_screen_confirm_checks_the_freeze_before_loading_the_model(monkeypatch, tmp_path):
    from kvsel import cli

    monkeypatch.setattr(cli, "_bundle", lambda args: pytest.fail("the model must not load"))
    items = tmp_path / "gqa_confirm.jsonl"
    items.write_text(json.dumps({"item_id": "a", "split": "confirm"}) + "\n")
    with pytest.raises(SystemExit, match="no screening freeze"):
        run_cli(monkeypatch, "screen-confirm", "--items", str(items), "--images", str(tmp_path),
                "--record", str(tmp_path / "none.json"), "--out", str(tmp_path / "out"))


def test_freeze_screening_command_writes_the_record(monkeypatch, tmp_path, capsys):
    prereg = tmp_path / "PREREG.md"
    prereg.write_text("## 1. Screening rules\nrule\n")
    items = tmp_path / "gqa_confirm.jsonl"
    items.write_text(json.dumps({"item_id": "a", "split": "confirm"}) + "\n")
    record = tmp_path / "freeze" / "screening.json"
    run_cli(monkeypatch, "freeze-screening", "--prereg", str(prereg), "--manifests", str(items), "--record", str(record))
    assert json.loads(record.read_text())["manifests"] == {"gqa_confirm.jsonl": json.loads(capsys.readouterr().out)["manifests"]["gqa_confirm.jsonl"]}


def test_verify_tells_confirmatory_screens_from_development_screens(tmp_path):
    from kvsel.cli import _is_confirm_screen

    (tmp_path / "c" / "items").mkdir(parents=True)
    (tmp_path / "c" / "items" / "a.json").write_text(json.dumps({"item_id": "a", "eligible": True}))
    (tmp_path / "d" / "items").mkdir(parents=True)
    (tmp_path / "d" / "items" / "a.json").write_text(json.dumps({"item_id": "a", "rows": []}))
    assert _is_confirm_screen(tmp_path / "c") and not _is_confirm_screen(tmp_path / "d")


def test_confirm_checks_the_freeze_before_loading_the_model(monkeypatch, tmp_path):
    from kvsel import cli

    monkeypatch.setattr(cli, "_bundle", lambda args: pytest.fail("the model must not load"))
    with pytest.raises(SystemExit, match="no comparative freeze"):
        run_cli(monkeypatch, "confirm", "--items", str(tmp_path / "m.jsonl"), "--screen", str(tmp_path),
                "--decisions", str(tmp_path / "d.jsonl"), "--record", str(tmp_path / "none.json"),
                "--images", str(tmp_path), "--out", str(tmp_path / "out"))


def test_blind_rate_refuses_a_missing_or_empty_verification_file(monkeypatch, tmp_path):
    from kvsel import cli

    monkeypatch.setattr(cli, "_bundle", lambda args: pytest.fail("the model must not load"))
    items = tmp_path / "items.jsonl"
    items.write_text(json.dumps({"item_id": "a", "split": "dev"}) + "\n")
    common = ("blind-rate", "--items", str(items), "--images", str(tmp_path), "--out", str(tmp_path / "o.json"))
    with pytest.raises(SystemExit, match="does not exist"):
        run_cli(monkeypatch, *common, "--verified", str(tmp_path / "missing.jsonl"))
    (tmp_path / "d.jsonl").write_text("")
    with pytest.raises(SystemExit, match="no items to run"):
        run_cli(monkeypatch, *common, "--verified", str(tmp_path / "d.jsonl"))


def test_confirm_report_passes_runs_and_records(monkeypatch, tmp_path, capsys):
    from kvsel import confirm_analysis

    seen = {}

    def fake(runs, out, resamples, records):
        seen.update(runs=runs, records=records, resamples=resamples)
        out.mkdir(parents=True)
        (out / "report.md").write_text("report")

    monkeypatch.setattr(confirm_analysis, "confirm_report", fake)
    run_cli(monkeypatch, "confirm-report", "--gqa-run", "r", "--gqa-items", "m.jsonl", "--gqa-record", "f.json",
            "--resamples", "50", "--out", str(tmp_path / "o"))
    assert seen["runs"] == {"gqa": (Path("r"), Path("m.jsonl"))} and seen["records"] == {"gqa": Path("f.json")}
    assert seen["resamples"] == 50 and capsys.readouterr().out.strip() == "report"
    with pytest.raises(SystemExit, match="--textvqa-items"):
        run_cli(monkeypatch, "confirm-report", "--textvqa-run", "r", "--out", str(tmp_path / "p"))
    with pytest.raises(SystemExit, match="at least one run"):
        run_cli(monkeypatch, "confirm-report", "--out", str(tmp_path / "q"))
