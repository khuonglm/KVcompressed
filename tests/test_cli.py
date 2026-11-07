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
    "validate", "diagnose-e2", "screen", "pilot", "verify", "report",
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
