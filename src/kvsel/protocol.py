"""The freeze mechanism shared by every stage of every study (PREREG 1.9, 3.5, 4.4).

A `Stage` declares its name, its record, its inputs, its settings and the PREREG section whose draft line
or "[to fill" blank it refuses. `freeze` writes the record once, atomically; `check` recomputes the state
and returns the record it verified. Code may differ from the frozen code only through the amendment log.
Every refusal raises `Refusal`; only the command line turns it into an exit.
"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .files import write_json

CODE_AMENDMENTS = "code_amendments.json"
PENDING = "pending"  # the evidence of an entry not yet shown to reproduce
_ACCEPTING_PENDING = contextvars.ContextVar("accepting_pending", default=False)
SOURCE_SUFFIXES = (".py", ".html")


class Refusal(Exception):
    """A stop for action: a frozen input changed, a section is still draft, or a record differs."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifests(paths: list[Path]) -> dict[str, str]:
    return {Path(p).name: sha256_file(p) for p in paths}


def selected_set(ids: list[str], decisions: dict[str, dict]) -> str:
    """The selected items with their verification digests, as the stage-3 records' `main_set_sha256`."""
    return hashlib.sha256(json.dumps([[i, decisions[i]["digest"]] for i in ids]).encode()).hexdigest()


def code_identity() -> str:
    """Every source file of the package (.py, and .html resources), subpackages included, by relative path.
    Other files, such as .DS_Store, are not source."""
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.suffix in SOURCE_SUFFIXES):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def code_accepted(frozen: str, current: str, amendments: Path) -> bool:
    """The frozen code, or code reached from it through dated, outcome-blind amendments.

    Each entry of the amendments file, in order, is {"from", "to", "date", "reason", "evidence"} (and an optional
    "scope"): code `to` replaced code `from` and reproduced the frozen results exactly (PREREG 1.9). An entry whose
    evidence is still pending counts only inside `accepting_pending`, which `kvsel reproduce` uses to prove it.
    """
    reached = {frozen}
    for entry in json.loads(amendments.read_text()) if amendments.exists() else []:
        proven = not entry["evidence"].startswith(PENDING) or _ACCEPTING_PENDING.get()
        if entry["from"] in reached and proven:
            reached.add(entry["to"])
    return current in reached


@contextlib.contextmanager
def accepting_pending():
    """Pending amendment entries count as accepted, only while reproduction reruns the stored reports."""
    token = _ACCEPTING_PENDING.set(True)
    try:
        yield
    finally:
        _ACCEPTING_PENDING.reset(token)


@dataclass(frozen=True)
class Prereg:
    """The whole PREREG file, or its section: from the line starting with `heading` up to the next line
    starting with any of `stops`, stripped, with one final newline (the stage-1 records' form)."""

    path: Path
    heading: str | None = None
    stops: tuple[str, ...] = ("## ",)

    def text(self) -> str:
        text = Path(self.path).read_text()
        if self.heading is None:
            return text
        lines = text.splitlines()
        start = next((i for i, line in enumerate(lines) if line.startswith(self.heading)), None)
        if start is None:
            raise Refusal(f"{self.path} has no line starting '{self.heading}'")
        end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith(self.stops)), len(lines))
        return "\n".join(lines[start:end]).strip() + "\n"

    def sha256(self) -> str:
        return hashlib.sha256(self.text().encode()).hexdigest()


@dataclass(frozen=True)
class FrozenInputs:
    """What a stage-3 freeze reads and records: PREREG, the manifest, the screen folder, the verification decisions,
    the stage-1 record, and the stage's own record."""

    prereg: Path
    manifest: Path
    screen_dir: Path
    decisions: Path
    stage1: Path
    record: Path


@dataclass(frozen=True)
class Stage:
    """`inputs` maps each record key to its current value; a `Prereg` value is stored as its sha256."""

    name: str
    record: Path
    inputs: dict
    settings: dict
    draft: Prereg | None = None

    def state(self) -> dict:
        values = {key: value.sha256() if isinstance(value, Prereg) else value for key, value in self.inputs.items()}
        return {**values, "code_sha256": code_identity(), "settings": self.settings}


def _refuse_draft(section: Prereg) -> None:
    """One draft rule (PREREG 3.5, 4.4): a line starting with DRAFT (the marker, not a mention), or a '[to fill' blank."""
    lines = section.text().splitlines()
    where = section.heading or str(section.path)
    if any(line.startswith("DRAFT") for line in lines):
        raise Refusal(f"'{where}' in {section.path} is still marked DRAFT; review it and delete that line first")
    blanks = [line.strip() for line in lines if "[to fill" in line]
    if blanks:
        raise Refusal(f"'{where}' in {section.path} still has {len(blanks)} blank(s) to fill, first: {blanks[0][:120]}")


def _changed(stage: Stage, frozen: dict, report: bool = False) -> list[str]:
    """The inputs that differ from the record. Code passes through the amendment log (PREREG 1.9); with
    `report`, PREREG inputs are not compared, since the file may have grown after the run (PREREG 3.5, 4.4)."""
    changed = []
    for key, now in stage.state().items():
        if key == "code_sha256":
            same = code_accepted(frozen.get(key), now, Path(stage.record).parent / CODE_AMENDMENTS)
        else:
            same = (report and isinstance(stage.inputs.get(key), Prereg)) or frozen.get(key) == now
        if not same:
            changed.append(key)
    return changed


def freeze(stage: Stage, extra: dict | None = None) -> dict:
    """Writes the record once. An equal state, code included through the amendment log, returns the existing
    record; a different one is refused. ``extra`` is stored after the state and never compared (a stage-1
    amendment's `amends`)."""
    if stage.draft is not None:
        _refuse_draft(stage.draft)
    record = Path(stage.record)
    if record.exists():
        frozen = json.loads(record.read_text())
        if not _changed(stage, frozen):
            return frozen
        raise Refusal(f"{record} already exists with different contents; {stage.name} is frozen")
    payload = {"stage": stage.name, "frozen_at": datetime.now(UTC).isoformat(timespec="seconds"), **stage.state()}
    payload |= extra or {}
    write_json(record, payload, indent=2)
    return payload


def check(stage: Stage, report: bool = False) -> dict:
    """The record, if the current state matches it; otherwise refuses naming every changed input.
    `report` is the report-time check, which doesn't compare PREREG."""
    record = Path(stage.record)
    if not record.exists():
        raise Refusal(f"no {stage.name} freeze at {record}: freeze it first")
    frozen = json.loads(record.read_text())
    changed = _changed(stage, frozen, report)
    if changed:
        raise Refusal(f"Refusing: changed since {record}: " + ", ".join(changed))
    return frozen


def check_code(record: Path) -> dict:
    """The record, if the current code is its frozen code or reached from it through the amendment log: the
    report-time check of a record whose other inputs the caller doesn't hold (PREREG 1.9)."""
    record = Path(record)
    if not record.exists():
        raise Refusal(f"no freeze record at {record}")
    frozen = json.loads(record.read_text())
    if not code_accepted(frozen["code_sha256"], code_identity(), record.parent / CODE_AMENDMENTS):
        raise Refusal(f"the code changed since {record} was frozen; analyse with the frozen code, or document the fix "
                      "as a dated amendment (PREREG 1.9) and rerun every affected condition")
    return frozen


# --- The amendment log ----------------------------------------------------------------------------------------

def live_records(freeze_dir: Path) -> dict[str, dict]:
    """Every freeze record by file name, except those another record amends (superseded: no command checks them)."""
    records = {path.name: json.loads(path.read_text()) for path in sorted(Path(freeze_dir).glob("*.json"))
               if path.name != CODE_AMENDMENTS}
    superseded = {Path(r["amends"]["record"]).name for r in records.values() if "amends" in r}
    return {name: record for name, record in records.items() if name not in superseded}


def latest_accepted(frozen: str, amendments: list[dict]) -> str:
    """The last code reached from ``frozen`` through the log, in log order."""
    latest = frozen
    for entry in amendments:
        if entry["from"] == latest:
            latest = entry["to"]
    return latest


def write_pending(freeze_dir: Path, reason: str, date: str) -> list[dict]:
    """One pending entry from each live record's latest accepted code to the current code, replacing pending
    entries written for other code. Returns the pending entries now in the log (PREREG 1.9)."""
    path = Path(freeze_dir) / CODE_AMENDMENTS
    current = code_identity()
    entries = [e for e in (json.loads(path.read_text()) if path.exists() else [])
               if not (e["evidence"].startswith(PENDING) and e["to"] != current)]
    for record in live_records(freeze_dir).values():
        latest = latest_accepted(record["code_sha256"], entries)
        if latest != current:
            entries.append({"from": latest, "to": current, "date": date, "scope": "package (code_identity)",
                            "reason": reason, "evidence": f"{PENDING}: run `kvsel reproduce`"})
    write_json(path, entries, indent=2)
    return [e for e in entries if e["evidence"].startswith(PENDING)]


def fill_evidence(freeze_dir: Path, evidence: str) -> int:
    """Writes ``evidence`` into every pending entry that leads to the current code; returns how many."""
    path = Path(freeze_dir) / CODE_AMENDMENTS
    entries = json.loads(path.read_text()) if path.exists() else []
    pending = [e for e in entries if e["evidence"].startswith(PENDING)]
    if not pending:
        return 0
    stale = [e["to"][:12] for e in pending if e["to"] != code_identity()]
    if stale:
        raise Refusal(f"pending entries lead to code {', '.join(stale)}, not the current code: the code changed after "
                      "they were written; rerun with --reason to rewrite them")
    for entry in pending:
        entry["evidence"] = evidence
    write_json(path, entries, indent=2)
    return len(pending)
