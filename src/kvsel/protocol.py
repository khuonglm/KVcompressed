"""The freeze mechanism shared by every stage of every study (PREREG 1.9, 3.5, 4.4).

A `Stage` declares its name, its record, its inputs, its settings and the PREREG section whose draft line
or "[to fill" blank it refuses. `freeze` writes the record once, atomically; `check` recomputes the state
and returns the record it verified. Code may differ from the frozen code only through the amendment log.
Every refusal raises `Refusal`; only the command line turns it into an exit.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

CODE_AMENDMENTS = "code_amendments.json"
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

    Each entry of the amendments file, in order, is {"from", "to", "date", "reason", "evidence"}: code
    `to` replaced code `from` and reproduced the frozen results exactly (PREREG, section 1.9).
    """
    reached = {frozen}
    for entry in json.loads(amendments.read_text()) if amendments.exists() else []:
        if entry["from"] in reached:
            reached.add(entry["to"])
    return current in reached


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


def _write_atomic(path: Path, payload: dict) -> None:
    """A temporary file in the same folder, renamed into place: an interrupted write leaves no record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(handle, "w") as out:
            out.write(json.dumps(payload, indent=2) + "\n")
            out.flush()
            os.fsync(out.fileno())  # the bytes are on disk before the name points at them
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


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


def freeze(stage: Stage) -> dict:
    """Writes the record once. An equal state, code included through the amendment log, returns the existing
    record; a different one is refused."""
    if stage.draft is not None:
        _refuse_draft(stage.draft)
    record = Path(stage.record)
    if record.exists():
        frozen = json.loads(record.read_text())
        if not _changed(stage, frozen):
            return frozen
        raise Refusal(f"{record} already exists with different contents; {stage.name} is frozen")
    payload = {"stage": stage.name, "frozen_at": datetime.now(UTC).isoformat(timespec="seconds"), **stage.state()}
    _write_atomic(record, payload)
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
