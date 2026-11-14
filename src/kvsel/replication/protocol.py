"""The replication's units and its freeze (PREREG section 3.2 and 3.5).

`freeze` records hashes of PREREG.md, the stage-1 record, the GQA stage-3 record, the manifest, the screen
flags, the replication set with its verification digests, the earlier run folders and the exclusions, the
settings and all code under `src/kvsel/` (subfolders included). `check` recomputes that state and refuses any
difference; a code change passes only through `freeze/code_amendments.json` (PREREG 1.9).

This mirrors `confirm.current_state`/`freeze_comparative` instead of calling them: those are frozen code.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from ..analysis import DELTA, margin
from ..confirm import load_flags, main_set
from ..confirm_analysis import CODE_AMENDMENTS, SURFACE, code_accepted
from ..freeze import check_screening, sha256_file
from ..runner import NO_CONTEXT, Q, Unit, eligible, load_items
from ..verify import load_decisions
from . import select

SECTION = "## 3. GQA replication"
BUDGETS = (72, 144, 288)
CONDITIONS = ("mis", "irr")
CONTROL_CONDITIONS = ("match",)
BLIND = Unit("full", 576, False, NO_CONTEXT, Q)
ALPHA = 0.05
FAMILY = (("R1", "S_H^(mis)", "72"), ("R1", "S_H^(mis)", "144"), ("R2", "S_H^(irr)", "72"),
          ("R2", "S_H^(irr)", "144"), ("R3", "E_harm^(mis)", "72"), ("R5", "T_irr", "72"))
EQUIVALENCE = ("R4", "S_H^(mis)", "288")


def units(*, control: bool) -> list[Unit]:
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
            "main_units": [asdict(u) for u in units(control=False)],
            "control_units": [asdict(u) for u in units(control=True)], "blind_unit": asdict(BLIND),
            "surface": SURFACE, "family": [list(t) for t in FAMILY], "alpha": ALPHA,
            "equivalence": list(EQUIVALENCE), "delta": DELTA}


def checks(rows: list[dict]) -> dict:
    """The run's consistency checks (PREREG 3.2): Full_Q still eligible, Blind_Q eligible should be 0."""
    full_q = [r for r in rows if r["selector"] == "full" and r["cell"] == "Q"]
    blind = [r for r in rows if r["selector"] == "blind"]
    return {
        "full_q_no_longer_eligible": sum(not eligible(r, SURFACE) for r in full_q),
        "blind_rows": len(blind),
        "blind_now_eligible": sum(eligible(r, SURFACE) for r in blind),
        "blind_mean_margin": float(np.mean([margin(r["scores"], SURFACE) for r in blind])) if blind else None,
        "truncated_share": float(np.mean([r["truncated"] for r in rows])),
    }


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
    section = text.split(SECTION, 1)[1].split("\n## ", 1)[0]
    return any(line.startswith("DRAFT") for line in section.splitlines())  # the marker line, not a mention


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
        "exclude_runs": [str(run) for run in exclude_runs],
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
    changed = [key for key in now if key != "code_sha256" and frozen.get(key) != now[key]]
    if not code_accepted(frozen["code_sha256"], now["code_sha256"], Path(record).parent / CODE_AMENDMENTS):
        changed.append("code_sha256")
    if changed:
        raise SystemExit("Refusing to run: changed since the replication freeze: " + ", ".join(changed))
    return frozen
