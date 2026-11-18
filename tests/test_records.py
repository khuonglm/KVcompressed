"""Every live freeze record still fits the current code ("Codebase reorganization spec", the record-compatibility test).

A record's code hash must reach the current code through `freeze/code_amendments.json`, and its settings must equal
what the code declares; a stage-1 record's PREREG section must also be unchanged. A record that another record
amends is superseded, no command checks it, and it is left out. This replaces the six guard tests that proved
frozen code untouched.

Code has started moving (ticket 02), so it fails until amendment 4 writes its log entries: it is marked
`pytest.mark.xfail(reason="amendment 4 pending", strict=True)`, and amendment 4's ticket removes the mark.
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
        from kvsel.freeze import SCREENING, code_hash

        return code_hash(), SCREENING
    if stage == "comparative":
        from kvsel.confirm import code_hash_all, comparative_settings

        return code_hash_all(), comparative_settings()
    if stage == "replication":
        from kvsel.replication.protocol import code_hash_tree, settings

        return code_hash_tree(), settings()
    from kvsel_next import comparative, screen

    return screen.code_hash(), {"next_screening": screen.SCREENING, "next_comparative": comparative.settings()}[stage]


def prereg_section(stage: str) -> tuple[str, str] | None:
    """For a stage-1 record: the key holding its PREREG section hash, and that section's current text."""
    if stage == "screening":
        from kvsel.freeze import screening_section

        return "prereg_screening_sha256", screening_section(ROOT / "PREREG.md")
    if stage == "next_screening":
        from kvsel_next.screen import section_text

        return "prereg_section_sha256", section_text(ROOT / "PREREG.md")
    return None


def live_records() -> list[str]:
    records = {path.name: json.loads(path.read_text()) for path in sorted(FREEZE.glob("*.json"))
               if path.name != CODE_AMENDMENTS}
    superseded = {Path(r["amends"]["record"]).name for r in records.values() if "amends" in r}
    return [name for name in records if name not in superseded]


@pytest.mark.xfail(reason="amendment 4 pending", strict=True)
@pytest.mark.parametrize("name", live_records())
def test_record_fits_the_current_code(name):
    record = json.loads((FREEZE / name).read_text())
    code, settings = declared(record["stage"])
    assert code_accepted(record["code_sha256"], code, FREEZE / CODE_AMENDMENTS), (
        f"{name}: its code {record['code_sha256'][:12]} doesn't reach the current {code[:12]} through {CODE_AMENDMENTS}")
    assert record["settings"] == settings, f"{name}: the frozen settings differ from what the code declares"
    section = prereg_section(record["stage"])
    if section:
        key, text = section
        assert record[key] == hashlib.sha256(text.encode()).hexdigest(), f"{name}: its PREREG section changed"
