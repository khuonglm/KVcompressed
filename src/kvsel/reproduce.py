"""Proving a code amendment (PREREG 1.9) the same way every time: rerun stored work with the current code and compare.

The committed list (`stored_reports.json`) names each stored report (the command that made it, its results file, and
whether the comparison is byte for byte or on the stored keys only) and each stored run (study, folder, manifest and
record). Reports rerun into a fresh folder; a few items of each run rerun on the GPU and are compared row for row.
The first difference is named. The amendment log's pending entries get their evidence only when everything reproduces.
"""

from __future__ import annotations

import contextlib
import io
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .items.manifest import load_items
from .model import Bundle
from .protocol import Refusal, sha256_file
from .studies.confirmatory.comparative import run_confirm
from .studies.followups.run import PARTS, run_followup
from .studies.replication.run import run_replication

LIST = Path("stored_reports.json")
STUDIES = ("confirmatory", "followups", "replication")  # whose runs can be rerun item by item


@dataclass(frozen=True)
class StoredReport:
    name: str
    argv: tuple[str, ...]  # the `kvsel` command, without --out
    results: Path
    compare: str  # "bytes", or "keys": every key the stored file holds, equal (a later amendment may add keys)


@dataclass(frozen=True)
class StoredRun:
    study: str
    run: Path
    items: Path  # the manifest
    record: Path  # the record whose hash the run's item files carry
    confirm_run: Path | None = None  # follow-ups: the confirmatory run whose subsets they follow


def load_list(path: Path = LIST) -> tuple[list[StoredReport], list[StoredRun]]:
    data = json.loads(Path(path).read_text())
    reports = [StoredReport(r["name"], tuple(r["argv"]), Path(r["results"]), r["compare"]) for r in data["reports"]]
    runs = [StoredRun(r["study"], Path(r["run"]), Path(r["items"]), Path(r["record"]),
                      Path(r["confirm_run"]) if r.get("confirm_run") else None) for r in data["runs"]]
    for report in reports:
        if report.compare not in ("bytes", "keys"):
            raise Refusal(f"{path}: {report.name} compares by {report.compare!r}, not bytes or keys")
    for run in runs:
        if run.study not in STUDIES:
            raise Refusal(f"{path}: {run.run} is a {run.study!r} run; runs of {', '.join(STUDIES)} can rerun")
    return reports, runs


def difference(stored, new, keys_only: bool, where: str = "$") -> str | None:
    """The first place ``new`` differs from ``stored``; with ``keys_only``, keys ``new`` adds are not differences."""
    if isinstance(stored, dict) and isinstance(new, dict):
        for key, value in stored.items():
            if key not in new:
                return f"{where}.{key}: missing"
            found = difference(value, new[key], keys_only, f"{where}.{key}")
            if found:
                return found
        extra = [key for key in new if key not in stored]
        return f"{where}.{extra[0]}: not in the stored file" if extra and not keys_only else None
    if isinstance(stored, list) and isinstance(new, list) and len(stored) == len(new):
        for n, (a, b) in enumerate(zip(stored, new)):
            found = difference(a, b, keys_only, f"{where}[{n}]")
            if found:
                return found
        return None
    return None if stored == new else f"{where}: stored {json.dumps(stored)[:80]}, now {json.dumps(new)[:80]}"


def reproduce_reports(reports: list[StoredReport], out_dir: Path, run: Callable[[list[str]], None]) -> list[str]:
    """Reruns each report into ``out_dir/<name>`` with ``run`` (the command line) and compares; the differences,
    a report that refuses to run among them."""
    found = []
    for report in reports:
        if not report.results.exists():
            raise Refusal(f"{report.name}: the stored {report.results} is missing")
        out = Path(out_dir) / report.name
        try:
            with contextlib.redirect_stdout(io.StringIO()):  # the report's own printout
                run([*report.argv, "--out", str(out)])
        except Refusal as refusal:
            print(f"{report.name}: REFUSED", flush=True)
            found.append(f"{report.name} refused to run: {refusal}")
            continue
        new = out / report.results.name
        if report.compare == "bytes":
            same = report.results.read_bytes() == new.read_bytes()
            where = None if same else difference(json.loads(report.results.read_text()), json.loads(new.read_text()),
                                                 False) or "same values, other bytes"
        else:
            where = difference(json.loads(report.results.read_text()), json.loads(new.read_text()), True)
        print(f"{report.name}: {'reproduced' if where is None else 'DIFFERS'} ({report.results})", flush=True)
        if where:
            found.append(f"{report.results} vs {new}: {where}")
    return found


def _stored(folder: Path) -> dict[str, Path]:
    return {path.stem: path for path in sorted((Path(folder) / "items").glob("*.json"))}


def _spot_ids(files: dict[str, Path], items: list[dict], per_subset: int) -> tuple[list[str], list[str]]:
    """The first ``per_subset`` main and control items in manifest order: (all chosen, the control ones)."""
    subsets = {item_id: json.loads(path.read_text()).get("subset") for item_id, path in files.items()}
    order = [item["item_id"] for item in items if item["item_id"] in subsets]
    main = [i for i in order if subsets[i] == "main"][:per_subset]
    control = [i for i in order if subsets[i] == "control"][:per_subset]
    return [i for i in order if i in main or i in control], control


def _compare_files(stored: dict[str, Path], new_folder: Path, ids: list[str]) -> list[str]:
    """Each rerun item file against the stored one, without the record hash newer files carry; attention too."""
    found = []
    for item_id in ids:
        new = {k: v for k, v in json.loads((new_folder / "items" / f"{item_id}.json").read_text()).items()
               if k != "record_sha256"}
        old = {k: v for k, v in json.loads(stored[item_id].read_text()).items() if k != "record_sha256"}
        where = difference(old, new, False)
        if where:
            found.append(f"{stored[item_id]}: {where}")
        old_arrays = stored[item_id].parent.parent / "attention" / f"{item_id}.npz"
        if old_arrays.exists():
            with np.load(old_arrays) as a, np.load(new_folder / "attention" / f"{item_id}.npz") as b:
                if a.files != b.files or any(not np.array_equal(a[k], b[k]) for k in a.files):
                    found.append(f"{old_arrays}: the probe attention differs")
    return found


def reproduce_rows(bundle: Bundle, runs: list[StoredRun], images_dir: Path, out_dir: Path,
                   per_subset: int) -> tuple[list[str], int]:
    """Reruns the first ``per_subset`` main and control items of each stored run; (differences, items compared)."""
    found, compared = [], 0
    for stored in runs:
        items = load_items(stored.items)
        out = Path(out_dir) / f"{stored.study}_rows"
        record = sha256_file(stored.record)
        if stored.study == "followups":
            folders = {part: stored.run / part for part in PARTS if (stored.run / part / "items").is_dir()}
            ids, control = _spot_ids(_stored(stored.confirm_run), items, per_subset)
            run_followup(bundle, items, stored.items, images_dir, out, {"main": ids, "control": control}, record,
                         tuple(folders))
            for part, folder in folders.items():
                files = _stored(folder)
                found += _compare_files(files, out / part, [i for i in ids if i in files])
        else:
            files = _stored(stored.run)
            ids, control = _spot_ids(files, items, per_subset)
            if stored.study == "confirmatory":
                run_confirm(bundle, items, images_dir, out, {"main": ids, "control": control}, record)
            else:
                run_replication(bundle, items, images_dir, out, {"replication_ids": ids, "control_ids": control}, record)
            found += _compare_files(files, out, ids)
        compared += len(ids)
        print(f"{stored.study}: {len(ids)} items rerun ({stored.run})", flush=True)
    return found, compared
