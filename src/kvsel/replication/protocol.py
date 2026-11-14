"""The replication's units and its freeze (PREREG section 3.2 and 3.5).

`freeze` records hashes of PREREG.md, the stage-1 record, the GQA stage-3 record, the manifest, the screen
flags, the replication set with its verification digests, the exclusions, the settings and all code under
`src/kvsel/` (subfolders included). `check` recomputes that state and refuses any difference.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from ..confirm import load_flags, main_set
from ..freeze import check_screening, sha256_file
from ..runner import NO_CONTEXT, Q, Unit, load_items
from ..verify import load_decisions
from . import select

SECTION = "## 3. GQA replication"
BUDGETS = (72, 144, 288)
CONDITIONS = ("mis", "irr")
CONTROL_CONDITIONS = ("match",)


def units(control: bool) -> list[Unit]:
    """PW-attn cells A, B_c, C_c, D_c at every budget, then the full-cache anchors. The run adds Blind_Q."""
    conditions = CONDITIONS + (CONTROL_CONDITIONS if control else ())
    out = []
    for k in BUDGETS:
        out.append(Unit("pw", k, False, Q, Q))
        for c in conditions:
            out += [Unit("pw", k, False, c, c), Unit("pw", k, False, Q, c), Unit("pw", k, False, c, Q)]
    out += [Unit("full", 576, False, NO_CONTEXT, answering) for answering in (Q, *conditions)]
    return out


def settings() -> dict:
    return {"budgets": list(BUDGETS), "size": select.REPLICATION_SIZE, "control_size": select.CONTROL_SIZE,
            "main_units": [asdict(u) for u in units(False)], "control_units": [asdict(u) for u in units(True)]}


def code_hash_tree() -> str:
    """Every .py file under `src/kvsel/`, subpackages included, by relative path."""
    root = Path(__file__).parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def _set_hash(ids: list[str], decisions: dict[str, dict]) -> str:
    """The same form as the GQA stage-3 record's `main_set_sha256`."""
    return hashlib.sha256(json.dumps([[i, decisions[i]["digest"]] for i in ids]).encode()).hexdigest()


def _section_is_draft(prereg: Path) -> bool:
    text = Path(prereg).read_text()
    if SECTION not in text:
        raise SystemExit(f"{prereg} has no '{SECTION}' section")
    return "DRAFT" in text.split(SECTION, 1)[1].split("\n## ", 1)[0]


def current_state(prereg: Path, manifest: Path, screen: Path, decisions: Path, stage1: Path, gqa_record: Path,
                  exclude_runs: list[Path]) -> dict:
    check_screening(prereg, manifest, stage1)
    items = load_items(manifest)
    current, _ = load_decisions(decisions, items)
    main = main_set(items, screen, decisions)["main"]
    if _set_hash(main, current) != json.loads(Path(gqa_record).read_text())["main_set_sha256"]:
        raise SystemExit(f"the main set no longer matches {gqa_record}; a decision before its cutoff changed")
    chosen = select.replication_set(items, load_flags(screen), current, main, select.earlier_run_ids(exclude_runs))
    if len(chosen["replication"]) < select.REPLICATION_SIZE:
        where = f"verify {chosen['next_unverified']} next" if chosen["next_unverified"] else "the pool is exhausted"
        raise SystemExit(f"replication set has {len(chosen['replication'])} of {select.REPLICATION_SIZE}; {where}")
    return {
        "prereg_sha256": sha256_file(prereg),
        "stage1_record_sha256": sha256_file(stage1),
        "gqa_record_sha256": sha256_file(gqa_record),
        "manifest": {Path(manifest).name: sha256_file(manifest)},
        "screen_flags_sha256": sha256_file(Path(screen) / "eligible.jsonl"),
        "replication_set_sha256": _set_hash(chosen["replication"], current),
        "replication_ids": chosen["replication"],
        "control_ids": chosen["control"],
        "failed_before_cutoff": chosen["failed"],
        "excluded": chosen["excluded"],
        "code_sha256": code_hash_tree(),
        "settings": settings(),
    }


def freeze(prereg: Path, manifest: Path, screen: Path, decisions: Path, stage1: Path, gqa_record: Path,
           exclude_runs: list[Path], record: Path) -> dict:
    if _section_is_draft(prereg):
        raise SystemExit(f"'{SECTION}' in {prereg} is still marked DRAFT; review it and delete that line first")
    state = current_state(prereg, manifest, screen, decisions, stage1, gqa_record, exclude_runs)
    if record.exists():
        frozen = json.loads(record.read_text())
        if {key: frozen.get(key) for key in state} == state:
            return frozen
        raise SystemExit(f"{record} already exists with different contents; the replication is frozen")
    payload = {"stage": "replication", "frozen_at": datetime.now(UTC).isoformat(timespec="seconds"), **state}
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def check(prereg: Path, manifest: Path, screen: Path, decisions: Path, stage1: Path, gqa_record: Path,
          exclude_runs: list[Path], record: Path) -> dict:
    if not record.exists():
        raise SystemExit(f"no replication freeze at {record}: run `python -m kvsel.replication freeze` first")
    frozen = json.loads(record.read_text())
    now = current_state(prereg, manifest, screen, decisions, stage1, gqa_record, exclude_runs)
    changed = [key for key in now if frozen.get(key) != now[key]]
    if changed:
        raise SystemExit("Refusing to run: changed since the replication freeze: " + ", ".join(changed))
    return frozen
