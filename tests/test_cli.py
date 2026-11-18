"""The one command line: every command parses, the model loads only after the checks, and the exit codes."""

import json
from pathlib import Path

import pytest

from kvsel import cli
from kvsel.cli import build_parser, main


def commands() -> list[tuple[str, ...]]:
    """Every command, as the words that name it."""
    found = []

    def walk(parser, words):
        actions = [a for a in parser._actions if a.__class__.__name__ == "_SubParsersAction"]
        if not actions:
            found.append(tuple(words))
        for action in actions:
            for name, sub in action.choices.items():
                walk(sub, [*words, name])

    walk(build_parser(), [])
    return found


def test_the_groups_are_data_model_verify_validate_reproduce_and_one_per_study():
    assert {words[0] for words in commands()} == {"data", "model", "verify", "validate", "reproduce", "pilot", "confirmatory",
                                                  "followups", "replication", "next"}
    assert ("confirmatory", "freeze-screening") in commands() and ("next", "count") in commands()


@pytest.mark.parametrize("words", commands(), ids=" ".join)
def test_every_command_parses_its_help(words, capsys):
    with pytest.raises(SystemExit) as stopped:
        main([*words, "--help"])
    assert stopped.value.code == 0 and "usage: kvsel" in capsys.readouterr().out


def test_one_name_and_default_per_meaning():
    """The surface rule's default is the same wherever a command takes one."""
    surfaces = {action.default for words in commands() for action in _parser(words)._actions if action.dest == "surface"}
    assert surfaces == {"capitalized"}


def _parser(words):
    parser = build_parser()
    for word in words:
        parser = next(a for a in parser._actions if a.__class__.__name__ == "_SubParsersAction").choices[word]
    return parser


@pytest.mark.parametrize("model,out,model_id", [
    (None, "models/llava-1.5-7b-hf", "llava-hf/llava-1.5-7b-hf"),
    ("llava_next", "models/llava-v1.6-vicuna-7b-hf", "llava-hf/llava-v1.6-vicuna-7b-hf"),
])
def test_fetch_uses_the_family_repository_and_directory(monkeypatch, model, out, model_id):
    calls = {}
    monkeypatch.setattr("kvsel.models.family.Family.fetch",
                        lambda family, out, workers: calls.update(out=out, model_id=family.model_id) or [])
    main(["model", "fetch", *(["--model", model] if model else [])])
    assert calls == {"out": Path(out), "model_id": model_id}


def test_pilot_screen_prints_eligibility(monkeypatch, tmp_path, capsys):
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

    monkeypatch.setattr("kvsel.studies.pilot.run_screen", fake_screen)
    monkeypatch.setattr(cli, "bundle", lambda args, family: None)
    main(["pilot", "screen", "--items", str(items), "--images", str(tmp_path), "--out", str(out)])
    assert "eligible under `upper`: 1 of 1" in capsys.readouterr().out


def refused(argv, capsys) -> str:
    """Runs a command that must refuse: exit code 2, the reason on stderr."""
    with pytest.raises(SystemExit) as stopped:
        main(argv)
    assert stopped.value.code == 2
    return capsys.readouterr().err


def test_confirmatory_screen_checks_the_freeze_before_loading_the_model(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "bundle", lambda *_: pytest.fail("the model must not load"))
    items = tmp_path / "gqa_confirm.jsonl"
    items.write_text(json.dumps({"item_id": "a", "split": "confirm"}) + "\n")
    assert "no screening freeze" in refused(["confirmatory", "screen", "--items", str(items), "--images", str(tmp_path),
                                             "--record", str(tmp_path / "none.json"), "--out", str(tmp_path / "out")],
                                            capsys)


def test_confirmatory_freeze_screening_writes_the_record(tmp_path, capsys):
    prereg = tmp_path / "PREREG.md"
    prereg.write_text("## 1. Screening rules\nrule\n")
    items = tmp_path / "gqa_confirm.jsonl"
    items.write_text(json.dumps({"item_id": "a", "split": "confirm"}) + "\n")
    record = tmp_path / "freeze" / "screening.json"
    main(["confirmatory", "freeze-screening", "--prereg", str(prereg), "--manifests", str(items), "--record", str(record)])
    printed = json.loads(capsys.readouterr().out)
    assert json.loads(record.read_text())["manifests"] == printed["manifests"]


def test_confirmatory_run_checks_the_freeze_before_loading_the_model(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "bundle", lambda *_: pytest.fail("the model must not load"))
    assert "no comparative freeze" in refused(
        ["confirmatory", "run", "--items", str(tmp_path / "m.jsonl"), "--screen", str(tmp_path), "--decisions",
         str(tmp_path / "d.jsonl"), "--record", str(tmp_path / "none.json"), "--images", str(tmp_path),
         "--out", str(tmp_path / "out")], capsys)


def test_confirmatory_report_passes_runs_and_records(monkeypatch, tmp_path, capsys):
    seen = {}

    def fake(runs, out, resamples, records):
        seen.update(runs=runs, records=records, resamples=resamples)
        out.mkdir(parents=True)
        (out / "report.md").write_text("report")

    monkeypatch.setattr(cli.report, "confirm_report", fake)
    main(["confirmatory", "report", "--gqa-run", "r", "--gqa-items", "m.jsonl", "--gqa-record", "f.json",
          "--resamples", "50", "--out", str(tmp_path / "o")])
    assert seen["runs"] == {"gqa": (Path("r"), Path("m.jsonl"))} and seen["records"] == {"gqa": Path("f.json")}
    assert seen["resamples"] == 50 and capsys.readouterr().out.strip() == "report"
    assert "--textvqa-items" in refused(["confirmatory", "report", "--textvqa-run", "r", "--out", str(tmp_path / "p")],
                                        capsys)
    assert "at least one run" in refused(["confirmatory", "report", "--out", str(tmp_path / "q")], capsys)


def test_pilot_report_is_the_rehearsal_without_records(monkeypatch, tmp_path):
    seen = {}

    def fake(runs, out, resamples, records):
        seen.update(records=records)
        out.mkdir(parents=True)
        (out / "report.md").write_text("")

    monkeypatch.setattr(cli.report, "confirm_report", fake)
    main(["pilot", "report", "--gqa-run", "r", "--gqa-items", "m.jsonl", "--out", str(tmp_path / "o")])
    assert seen["records"] == {}
    with pytest.raises(SystemExit):
        main(["pilot", "report", "--gqa-record", "f.json", "--out", str(tmp_path / "o")])  # not an option here


def test_an_error_is_exit_code_1(monkeypatch):
    monkeypatch.setattr("kvsel.models.family.Family.fetch", lambda family, out, workers: ["model.safetensors"])
    with pytest.raises(SystemExit, match="still missing") as stopped:
        main(["model", "fetch"])
    assert str(stopped.value.code).startswith("still missing")  # a message exit: Python prints it, status 1
