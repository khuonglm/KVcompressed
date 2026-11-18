"""The freeze mechanism (`kvsel.protocol`), tested once on a declared test stage."""

import json

import pytest

from kvsel import protocol
from kvsel.protocol import Prereg, Refusal, Stage, check, code_identity, freeze

PREREG = """# Pre-registration

## 1. Rules
Eligibility: Full_Q correct.

## 2. Test study
Selector PW-attn.
"""


@pytest.fixture
def stage(tmp_path):
    """A test stage: the whole PREREG file, a manifest, a section to refuse while draft, settings."""
    (tmp_path / "PREREG.md").write_text(PREREG)
    (tmp_path / "items.jsonl").write_text('{"item_id": "i0"}\n')

    def declare(settings=None):
        return Stage(
            name="test",
            record=tmp_path / "freeze" / "test.json",
            inputs={
                "prereg_sha256": Prereg(tmp_path / "PREREG.md"),
                "manifest": protocol.manifests([tmp_path / "items.jsonl"]),
            },
            settings=settings or {"k": 144},
            draft=Prereg(tmp_path / "PREREG.md", "## 2."),
        )

    return declare


def test_freeze_writes_once_and_check_returns_the_record(stage):
    written = freeze(stage())
    assert written["stage"] == "test" and written["settings"] == {"k": 144}
    assert written["code_sha256"] == code_identity()
    assert json.loads(stage().record.read_text()) == written
    assert freeze(stage()) == written  # equal state: the existing record, not a rewrite
    assert check(stage()) == written


@pytest.mark.parametrize("line", ["DRAFT: not reviewed yet", "Budget: [to fill after validation]"])
def test_freeze_refuses_a_draft_line_or_a_blank(stage, line):
    prereg = stage().inputs["prereg_sha256"].path
    prereg.write_text(PREREG + line + "\n")
    with pytest.raises(Refusal, match="DRAFT|blank"):
        freeze(stage())
    assert not stage().record.exists()


def test_a_mention_of_draft_is_not_a_draft_line(stage):
    prereg = stage().inputs["prereg_sha256"].path
    prereg.write_text(PREREG + "The DRAFT marker was removed.\n")
    freeze(stage())


def test_freeze_refuses_a_different_record(stage):
    freeze(stage())
    with pytest.raises(Refusal, match="already exists with different contents"):
        freeze(stage({"k": 72}))


def test_check_names_every_changed_input(stage):
    freeze(stage())
    stage().inputs["prereg_sha256"].path.write_text(PREREG + "One more rule.\n")
    (stage().record.parent.parent / "items.jsonl").write_text('{"item_id": "i1"}\n')
    with pytest.raises(Refusal, match="prereg_sha256, manifest, settings$"):
        check(stage({"k": 72}))


def test_check_refuses_without_a_record(stage):
    with pytest.raises(Refusal, match="no test freeze"):
        check(stage())


def test_changed_code_passes_only_through_the_amendment_log(stage, monkeypatch):
    frozen = freeze(stage())
    monkeypatch.setattr(protocol, "code_identity", lambda: "new")
    with pytest.raises(Refusal, match="code_sha256"):
        check(stage())
    entry = {"from": frozen["code_sha256"], "to": "new", "date": "2026-11-18", "reason": "r", "evidence": "e"}
    (stage().record.parent / protocol.CODE_AMENDMENTS).write_text(json.dumps([entry]))
    assert check(stage()) == frozen
    assert freeze(stage()) == frozen  # freezing again on amended code returns the record


def test_an_interrupted_freeze_leaves_no_record(stage, monkeypatch):
    def interrupted(*_):
        raise KeyboardInterrupt

    monkeypatch.setattr("kvsel.files.os.replace", interrupted)
    with pytest.raises(KeyboardInterrupt):
        freeze(stage())
    assert list(stage().record.parent.iterdir()) == []


def test_the_code_hash_changes_when_any_source_file_changes(tmp_path, monkeypatch):
    package = tmp_path / "pkg"
    (package / "sub").mkdir(parents=True)
    (package / "protocol.py").write_text("a = 1\n")
    (package / "sub" / "deep.py").write_text("b = 2\n")
    (package / "sub" / "page.html").write_text("<p>a</p>\n")
    monkeypatch.setattr(protocol, "__file__", str(package / "protocol.py"))
    before = code_identity()
    (package / "sub" / "deep.py").write_text("b = 3\n")
    edited = code_identity()
    (package / "sub" / "deep.py").rename(package / "sub" / "moved.py")
    moved = code_identity()
    (package / "sub" / "page.html").write_text("<p>b</p>\n")
    resource = code_identity()
    assert len({before, edited, moved, resource}) == 4  # an edit, a move, a resource edit
    (package / ".DS_Store").write_text("finder")
    (package / "sub" / "notes.txt").write_text("not source")
    assert code_identity() == resource



def test_report_check_does_not_compare_prereg(stage):
    frozen = freeze(stage())
    stage().inputs["prereg_sha256"].path.write_text(PREREG + "\n### 2.1 Amendment 1 (19 November 2026): x\nY.\n")
    assert check(stage(), report=True) == frozen
    with pytest.raises(Refusal, match="prereg_sha256"):
        check(stage())  # a run's check accepts no PREREG change
    with pytest.raises(Refusal, match="settings"):
        check(stage({"k": 72}), report=True)  # everything else is still compared
