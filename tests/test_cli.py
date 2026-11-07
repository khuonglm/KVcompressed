"""The command line itself: argument names must not collide with the dispatcher."""

import sys

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
    "build-gqa", "build-textvqa", "fetch-model", "fetch-images", "check-serialization",
    "validate", "diagnose-e2", "check-tokenization", "screen", "pilot", "verify", "report",
])
def test_every_command_parses_its_help(monkeypatch, command):
    with pytest.raises(SystemExit) as stopped:
        run_cli(monkeypatch, command, "--help")
    assert stopped.value.code == 0


def test_check_serialization_dispatches(monkeypatch, tmp_path, capsys):
    items = tmp_path / "items.jsonl"
    items.write_text("")
    monkeypatch.setattr("kvsel.model.load_tokenizer", lambda model_dir: None)
    run_cli(monkeypatch, "check-serialization", "--items", str(items))
    assert '"pass": true' in capsys.readouterr().out


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
