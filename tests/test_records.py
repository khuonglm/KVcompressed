"""Every live freeze record still fits the current code ("Codebase reorganization spec", the record-compatibility test).

A record's code hash must reach the current code through `freeze/code_amendments.json`, and its settings must equal
what the code declares; a stage-1 record's PREREG section must also be unchanged. A record that another record
amends is superseded, no command checks it, and it is left out. This replaces the six guard tests that proved
frozen code untouched.

The code part fails until amendment 4's log entries exist: `uv run kvsel reproduce --reason ... --images ...` on the
server writes them (PREREG 5). It is marked `pytest.mark.xfail(reason="amendment 4 pending", strict=True)`, so once
the entries are committed it passes, the strict mark fails the suite, and the mark is removed. The settings and PREREG
part must pass throughout.
"""

import hashlib
import json
from pathlib import Path

import pytest

from kvsel.protocol import CODE_AMENDMENTS, code_accepted

ROOT = Path(__file__).parents[1]
FREEZE = ROOT / "freeze"


def declared(stage: str) -> tuple[str, dict]:
    """The current code hash and settings for a record's stage, each in that stage's own scope."""
    if stage == "screening":
        from kvsel.protocol import code_identity
        from kvsel.studies.confirmatory.screening import SCREENING

        return code_identity(), SCREENING
    if stage == "comparative":
        from kvsel.protocol import code_identity
        from kvsel.studies.confirmatory.comparative import comparative_settings

        return code_identity(), comparative_settings()
    if stage == "replication":
        from kvsel.protocol import code_identity
        from kvsel.studies.replication.run import settings

        return code_identity(), settings()
    from kvsel.protocol import code_identity
    from kvsel.studies.next import comparative, screening

    return code_identity(), {"next_screening": screening.SCREENING, "next_comparative": comparative.settings()}[stage]


def prereg_section(stage: str) -> tuple[str, str] | None:
    """For a stage-1 record: the key holding its PREREG section hash, and that section's current text."""
    if stage == "screening":
        from kvsel.protocol import Prereg
        from kvsel.studies.confirmatory.screening import SECTION

        return "prereg_screening_sha256", Prereg(ROOT / "PREREG.md", SECTION).text()
    if stage == "next_screening":
        from kvsel.studies.next.screening import rules

        return "prereg_section_sha256", rules(ROOT / "PREREG.md").text()
    return None


def live_records() -> list[str]:
    records = {path.name: json.loads(path.read_text()) for path in sorted(FREEZE.glob("*.json"))
               if path.name != CODE_AMENDMENTS}
    superseded = {Path(r["amends"]["record"]).name for r in records.values() if "amends" in r}
    return [name for name in records if name not in superseded]


@pytest.mark.parametrize("name", live_records())
def test_record_settings_and_prereg_section_equal_what_the_code_declares(name):
    record = json.loads((FREEZE / name).read_text())
    assert record["settings"] == declared(record["stage"])[1], f"{name}: the frozen settings differ from the code's"
    section = prereg_section(record["stage"])
    if section:
        key, text = section
        assert record[key] == hashlib.sha256(text.encode()).hexdigest(), f"{name}: its PREREG section changed"


@pytest.mark.xfail(reason="amendment 4 pending", strict=True)
@pytest.mark.parametrize("name", live_records())
def test_record_code_reaches_the_current_code(name):
    record = json.loads((FREEZE / name).read_text())
    code = declared(record["stage"])[0]
    assert code_accepted(record["code_sha256"], code, FREEZE / CODE_AMENDMENTS), (
        f"{name}: its code {record['code_sha256'][:12]} doesn't reach the current {code[:12]} through {CODE_AMENDMENTS}")
