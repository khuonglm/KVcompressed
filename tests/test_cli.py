"""The command line itself: argument names must not collide with the dispatcher."""

import json
import sys
from pathlib import Path

import pytest

from kvsel.cli import main


def run_cli(monkeypatch, *arguments):
    monkeypatch.setattr(sys, "argv", ["kvsel", *arguments])
    main()


@pytest.mark.parametrize("command", [
    "build-gqa", "build-textvqa", "fetch-model", "fetch-images", "validate", "screen", "pilot", "verify",
    "freeze-screening", "screen-confirm", "freeze-comparative", "confirm", "confirm-report",
])
def test_every_command_parses_its_help(monkeypatch, command):
    with pytest.raises(SystemExit) as stopped:
        run_cli(monkeypatch, command, "--help")
    assert stopped.value.code == 0


def test_fetch_model_uses_the_family_repository_and_directory(monkeypatch):
    calls = {}
    monkeypatch.setattr("kvsel.model_files.fetch_model", lambda out, workers, model_id: calls.update(out=out, model_id=model_id) or [])
    run_cli(monkeypatch, "fetch-model")
    assert calls == {"out": Path("models/llava-1.5-7b-hf"), "model_id": "llava-hf/llava-1.5-7b-hf"}


def test_screen_prints_eligibility(monkeypatch, tmp_path, capsys):
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
