"""NeXT stage 1 (PREREG 4.1, 4.4): the freeze, the eligibility screen and its count.

The screen writes each item's id and flags only, never answers or margins, so no NeXT outcome can steer
the stage-3 freeze. It mirrors `kvsel.freeze` and `kvsel.screening` instead of calling them: their
section reader, settings and 576-token Full_Q are LLaVA-1.5's.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image

import kvsel
import kvsel_next
from kvsel.confirm_analysis import CODE_AMENDMENTS, code_accepted
from kvsel.freeze import sha256_file
from kvsel.geometry import PAD_COLOR
from kvsel.model import Bundle
from kvsel.runner import NO_CONTEXT, Q, Unit, eligible, open_image, run_unit, write_json_atomic
from kvsel.verify import PASS, load_decisions

from .family import NEXT
from .model import prefill
from .model_files import REVISION
from .prompts import tokenize_prompt

RECORD = Path("freeze/next_screening.json")
SECTION = "## 4."
SUBSECTION = "### 4.1"
SCREENING = {
    "model": NEXT.name,
    "model_id": NEXT.model_id,
    "revision": REVISION,
    "anyres_cap": None,  # none implemented; a cap set in 4.1 needs code, which changes this hash
    "template": NEXT.template_version,
    "head_precision": NEXT.default_head_precision,
    "surface": "capitalized",
    "eligibility": "Full_Q greedy answer correct and margin(true - alt) > 0",
    "blind_filter": "not eligible if the same rule passes on a blank of the padding colour at the real image's size",
    "pool": "manifest items whose latest current-digest decision is pass, in manifest order",
    "main_size": 1000,
    "control_size": 300,
    "max_new_tokens": 16,
    "batch_size": 1,
}


def _section(prereg: Path, heading: str, stops: tuple[str, ...]) -> list[str]:
    """Lines from the one starting with ``heading`` up to the next starting with any of ``stops``, not included."""
    lines = Path(prereg).read_text().splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith(heading)), None)
    if start is None:
        raise SystemExit(f"{prereg} has no line starting '{heading}'")
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith(stops)), len(lines))
    return lines[start:end]


def section_text(prereg: Path) -> str:
    """Section 4.1: kvsel's reader stops only at '## ', which would take in 4.2 onwards."""
    return "\n".join(_section(prereg, SUBSECTION, ("### ", "## "))).strip() + "\n"


def is_draft(prereg: Path) -> bool:
    """Section 4 has a line starting with DRAFT: the marker, not a mention."""
    return any(line.startswith("DRAFT") for line in _section(prereg, SECTION, ("## ",)))


def code_hash() -> str:
    """Every .py file under `src/kvsel/` and `src/kvsel_next/`, subpackages included."""
    digest = hashlib.sha256()
    for package in (kvsel, kvsel_next):
        root = Path(package.__file__).parent
        for path in sorted(root.rglob("*.py")):
            name = f"{package.__name__}/{path.relative_to(root).as_posix()}"
            digest.update(name.encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def current_state(prereg: Path, manifest: Path) -> dict:
    return {
        "prereg_section_sha256": hashlib.sha256(section_text(prereg).encode()).hexdigest(),
        "manifest": {Path(manifest).name: sha256_file(manifest)},
        "code_sha256": code_hash(),
        "settings": SCREENING,
    }


def freeze_screening(prereg: Path, manifest: Path, record: Path = RECORD) -> dict:
    if is_draft(prereg):
        raise SystemExit(f"section 4 of {prereg} is still marked DRAFT; review it and delete that line first")
    splits = {json.loads(line).get("split") for line in Path(manifest).read_text().splitlines() if line.strip()}
    if splits != {"confirm"}:
        raise SystemExit(f"{manifest} is not a confirmatory manifest (splits: {sorted(map(str, splits))})")
    return write_once(record, "next_screening", current_state(prereg, manifest))


def write_once(record: Path, stage: str, state: dict) -> dict:
    """Writes a freeze record; an unchanged rerun returns the existing one, a different one is refused."""
    if record.exists():
        frozen = json.loads(record.read_text())
        if {key: frozen.get(key) for key in state} == state:
            return frozen
        raise SystemExit(f"{record} already exists with different contents; {stage} is frozen")
    payload = {"stage": stage, "frozen_at": datetime.now(UTC).isoformat(timespec="seconds"), **state}
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def refuse_changes(record: Path, now: dict, skipped: frozenset[str] = frozenset()) -> dict:
    """The frozen record if ``now`` matches it; otherwise exits naming what changed.

    Code may differ only through `freeze/code_amendments.json` (PREREG 1.9)."""
    frozen = json.loads(record.read_text())
    changed = [key for key in now if key not in skipped | {"code_sha256"} and frozen.get(key) != now[key]]
    if not code_accepted(frozen["code_sha256"], now["code_sha256"], Path(record).parent / CODE_AMENDMENTS):
        changed.append("code_sha256")
    if changed:
        raise SystemExit(f"Refusing: changed since {record}: " + ", ".join(changed))
    return frozen


def check_screening(prereg: Path, manifest: Path, record: Path = RECORD) -> dict:
    if not record.exists():
        raise SystemExit(f"no NeXT screening freeze at {record}: run `python -m kvsel_next freeze-screening` first")
    return refuse_changes(record, current_state(prereg, manifest))


def pool(items: list[dict], decisions: Path) -> list[dict]:
    """Items whose latest decision, bound to the current digest, is pass; manifest order."""
    current, _ = load_decisions(decisions, items)
    return [item for item in items if current.get(item["item_id"], {}).get("decision") == PASS]


def blank_like(image: Image.Image) -> Image.Image:
    """Blind_Q's image: the padding colour at the real image's size, so it gets the same AnyRes layout."""
    return Image.new("RGB", image.size, PAD_COLOR)


def passes(bundle: Bundle, item: dict, image: Image.Image) -> bool:
    """Full_Q on this image passes the eligibility rule."""
    tokens, inputs = tokenize_prompt(bundle.image_processor, image, item["question"])
    prefix, layout = prefill(bundle, tokens.prefix_ids, inputs)
    unit = Unit("full", layout.total, False, NO_CONTEXT, Q)
    return eligible(run_unit(bundle, item, unit, prefix, np.arange(layout.total), {Q: tokens}), SCREENING["surface"])


def screen_item(bundle: Bundle, item: dict, image: Image.Image) -> dict:
    """Flags only. Blind_Q runs only if Full_Q passes."""
    full_q = passes(bundle, item, image)
    blind = passes(bundle, item, blank_like(image)) if full_q else None
    return {"item_id": item["item_id"], "full_q_eligible": full_q, "blind_eligible": blind,
            "eligible": full_q and not blind}


def run_screen(bundle: Bundle, items: list[dict], images_dir: Path, out_dir: Path, settings: dict) -> dict:
    """`items` is the pool. Resumable; stops once the main set's size of items is eligible.

    `settings` is the frozen record's, which `check_screening` has shown equal to SCREENING."""
    if bundle.family.template_version != settings["template"]:
        raise SystemExit(f"template {bundle.family.template_version} differs from the frozen {settings['template']}")
    if bundle.decoder.head_precision != settings["head_precision"]:
        raise SystemExit(f"head precision {bundle.decoder.head_precision} differs from the frozen {settings['head_precision']}")
    (out_dir / "items").mkdir(parents=True, exist_ok=True)
    records, found = [], 0
    for position, item in enumerate(items, 1):
        if found == settings["main_size"]:
            break
        target = out_dir / "items" / f"{item['item_id']}.json"
        if target.exists():
            records.append(json.loads(target.read_text()))
        else:
            records.append(screen_item(bundle, item, open_image(images_dir, item)))
            write_json_atomic(target, records[-1])
        found += records[-1]["eligible"]
        if position % 50 == 0:
            print(f"{position}/{len(items)} pool items screened, {found} eligible", flush=True)
    temporary = out_dir / "eligible.jsonl.tmp"  # screened items, manifest order
    temporary.write_text("".join(json.dumps({"item_id": r["item_id"], "eligible": r["eligible"]}) + "\n" for r in records))
    os.replace(temporary, out_dir / "eligible.jsonl")
    return {"pool": len(items), "screened": len(records), "full_q_eligible": sum(r["full_q_eligible"] for r in records),
            "removed_image_independent": sum(bool(r["blind_eligible"]) for r in records),
            "eligible": found, "unscreened": len(items) - len(records)}


def count(items: list[dict], decisions: Path, screen_dir: Path) -> dict:
    """Sizes from the item files' flags, so it works mid-screen too; never loads the model."""
    files = [Path(screen_dir) / "items" / f"{item['item_id']}.json" for item in pool(items, decisions)]
    eligible_count = sum(json.loads(f.read_text())["eligible"] for f in files if f.exists())
    main = min(eligible_count, SCREENING["main_size"])
    return {"main_set": main, "control": min(main, SCREENING["control_size"]),
            "short_of_1000": main < SCREENING["main_size"], "unscreened_passed": sum(not f.exists() for f in files)}
