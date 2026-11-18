"""Proving a code amendment: comparing reruns with stored reports, and the amendment log's pending entries."""

import json

import pytest

from kvsel import protocol
from kvsel.protocol import CODE_AMENDMENTS, Refusal, fill_evidence, latest_accepted, live_records, write_pending
from kvsel.reproduce import StoredReport, difference, reproduce_reports

# --- Comparing ------------------------------------------------------------------------------------------------


def test_difference_names_the_first_changed_value():
    assert difference({"a": {"b": [1.0, 2.0]}}, {"a": {"b": [1.0, 2.0]}}, False) is None
    assert difference({"a": {"b": [1.0, 2.5]}}, {"a": {"b": [1.0, 2.0]}}, False) == "$.a.b[1]: stored 2.5, now 2.0"
    assert difference({"a": 1}, {"b": 1}, False) == "$.a: missing"


def test_keys_mode_allows_new_keys_and_bytes_mode_does_not():
    assert difference({"a": 1}, {"a": 1, "reading": "x"}, True) is None
    assert difference({"a": 1}, {"a": 1, "reading": "x"}, False) == "$.reading: not in the stored file"


@pytest.mark.parametrize("compare", ["bytes", "keys"])
def test_a_report_reproduces_and_an_altered_stored_number_is_named(tmp_path, compare):
    stored = tmp_path / "stored" / "results.json"
    stored.parent.mkdir()
    made = {"tests": {"S_H": {"estimate": 0.25}}}

    def run(argv):
        out = tmp_path / argv[argv.index("--out") + 1]
        out.mkdir(parents=True, exist_ok=True)
        (out / "results.json").write_text(json.dumps(made))

    report = StoredReport("study", ("study", "report"), stored, compare)
    stored.write_text(json.dumps(made))
    assert reproduce_reports([report], tmp_path / "out", run) == []
    stored.write_text(json.dumps({"tests": {"S_H": {"estimate": 0.26}}}))
    (found,) = reproduce_reports([report], tmp_path / "again", run)
    assert "$.tests.S_H.estimate: stored 0.26, now 0.25" in found


def test_a_missing_stored_report_is_refused(tmp_path):
    with pytest.raises(Refusal, match="is missing"):
        reproduce_reports([StoredReport("x", (), tmp_path / "none.json", "bytes")], tmp_path, lambda argv: None)


# --- The amendment log ----------------------------------------------------------------------------------------


@pytest.fixture
def freeze_dir(tmp_path):
    folder = tmp_path / "freeze"
    folder.mkdir()
    (folder / "stage.json").write_text(json.dumps({"stage": "s", "code_sha256": "a" * 64}))
    (folder / "old.json").write_text(json.dumps({"stage": "s", "code_sha256": "0" * 64}))
    (folder / "new.json").write_text(json.dumps({"stage": "s", "code_sha256": "0" * 64,
                                                 "amends": {"record": "freeze/old.json"}}))
    (folder / CODE_AMENDMENTS).write_text(json.dumps([{"from": "a" * 64, "to": "b" * 64, "evidence": "proven"}]))
    return folder


def test_superseded_records_are_not_live(freeze_dir):
    assert sorted(live_records(freeze_dir)) == ["new.json", "stage.json"]


def test_pending_entries_start_from_each_records_latest_accepted_code(freeze_dir, monkeypatch):
    monkeypatch.setattr(protocol, "code_identity", lambda: "c" * 64)
    pending = write_pending(freeze_dir, "reorganization", "2026-11-18")
    assert sorted((e["from"][0], e["to"][0]) for e in pending) == [("0", "c"), ("b", "c")]
    assert latest_accepted("a" * 64, json.loads((freeze_dir / CODE_AMENDMENTS).read_text())) == "c" * 64
    assert write_pending(freeze_dir, "reorganization", "2026-11-18") == pending  # nothing to add twice


def test_evidence_fills_only_entries_for_the_current_code(freeze_dir, monkeypatch):
    monkeypatch.setattr(protocol, "code_identity", lambda: "c" * 64)
    write_pending(freeze_dir, "reorganization", "2026-11-18")
    monkeypatch.setattr(protocol, "code_identity", lambda: "d" * 64)  # the code changed after the entries
    with pytest.raises(Refusal, match="code changed"):
        fill_evidence(freeze_dir, "reproduced")
    assert len(write_pending(freeze_dir, "reorganization", "2026-11-19")) == 2  # rewritten for the new code
    assert fill_evidence(freeze_dir, "reproduced") == 2
    entries = json.loads((freeze_dir / CODE_AMENDMENTS).read_text())
    assert [e["evidence"] for e in entries] == ["proven", "reproduced", "reproduced"]
    assert fill_evidence(freeze_dir, "again") == 0


def test_a_pending_entry_counts_only_while_reproducing(tmp_path):
    from kvsel.protocol import accepting_pending, code_accepted

    log = tmp_path / CODE_AMENDMENTS
    log.write_text(json.dumps([{"from": "a", "to": "b", "evidence": "pending: run `kvsel reproduce`"}]))
    assert not code_accepted("a", "b", log)
    with accepting_pending():
        assert code_accepted("a", "b", log)
    assert not code_accepted("a", "b", log)
    log.write_text(json.dumps([{"from": "a", "to": "b", "evidence": "reproduced"}]))
    assert code_accepted("a", "b", log)


def test_reproduce_refuses_a_used_output_folder(tmp_path, capsys):
    from kvsel.cli import main

    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "old.json").write_text("{}")
    with pytest.raises(SystemExit, match="not empty") as stop:
        main(["reproduce", "--list", str(tmp_path / "none.json"), "--out", str(tmp_path / "out")])
    assert stop.value.code == 2
