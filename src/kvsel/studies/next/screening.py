"""NeXT stage 1 (PREREG 4.1, 4.4): the freeze, the eligibility screen and its count.

The screen writes each item's id and flags only, never answers or margins, so no NeXT outcome can steer the stage-3
freeze. Both NeXT stages refuse while section 4 has a DRAFT line or a "[to fill" blank (4.4).
"""

from __future__ import annotations

import json
from pathlib import Path

from ...model import Bundle
from ...models.llava_next import NEXT
from ...protocol import Prereg, Refusal, Stage, check, freeze, manifests
from ...runner import (
    FLAGS,
    NO_CONTEXT,
    ItemFile,
    Q,
    Unit,
    item_loop,
    mark_screen,
    open_image,
    require_images,
    screen_flags,
    screen_item,
    write_flags,
)
from ...verification import PASS, load_decisions

RECORD = Path("freeze/next_screening.json")
SCREENING = {  # PREREG 4.1
    "model": NEXT.name,
    "model_id": NEXT.model_id,
    "revision": NEXT.revision,
    "anyres_cap": None,  # none implemented; a cap set in 4.1 needs code, which changes this hash
    "template": NEXT.template_version,
    "head_precision": NEXT.default_head_precision,
    "surface": "capitalized",
    "eligibility": "Full_Q greedy answer correct and margin(true - alt) > 0",
    "blind_filter": "not eligible if the same rule passes on a blank of the padding colour at the real image's size",
    "pool": "manifest items whose latest current-digest decision is pass, in manifest order",
    "main_size": 1000,
    "control_size": 300,
    # PREREG 4.5(d), settled 16 November 2026: as 1.6, a short main set is every eligible item and is reported.
    "shortfall": "fewer than main_size eligible after the whole pool is screened: the main set is every eligible "
                 "item, the control subset its first control_size, and the shortfall is reported",
    "max_new_tokens": 16,
    "batch_size": 1,
}




def section(prereg: Path) -> Prereg:
    """Section 4: its draft line and blanks block both NeXT freezes (PREREG 4.4)."""
    return Prereg(prereg, "## 4.")


def rules(prereg: Path) -> Prereg:
    """Section 4.1, up to 4.2 rather than the next '## '."""
    return Prereg(prereg, "### 4.1", stops=("### ", "## "))


def stage(prereg: Path, manifest: Path, record: Path) -> Stage:
    """NeXT stage 1 (PREREG 4.4): section 4.1's text, the manifest, the code and the settings."""
    inputs = {"prereg_section_sha256": rules(prereg), "manifest": manifests([manifest])}
    return Stage("next_screening", record, inputs, SCREENING, draft=section(prereg))


def freeze_screening(prereg: Path, manifest: Path, record: Path = RECORD) -> dict:
    splits = {json.loads(line).get("split") for line in Path(manifest).read_text().splitlines() if line.strip()}
    if splits != {"confirm"}:
        raise Refusal(f"{manifest} is not a confirmatory manifest (splits: {sorted(map(str, splits))})")
    return freeze(stage(prereg, manifest, record))


def check_screening(prereg: Path, manifest: Path, record: Path = RECORD, report: bool = False) -> dict:
    return check(stage(prereg, manifest, record), report)


def pool(items: list[dict], decisions: Path) -> list[dict]:
    """PREREG 4.1: the pool is the items whose latest decision, bound to the current digest, is pass; manifest order."""
    current, _ = load_decisions(decisions, items)
    return [item for item in items if current.get(item["item_id"], {}).get("decision") == PASS]


FULL_Q = Unit("full", None, False, NO_CONTEXT, Q, share=1.0)  # flags only: no row is stored


def run_screen(bundle: Bundle, items: list[dict], images_dir: Path, out_dir: Path, settings: dict,
               record_sha256: str) -> dict:
    """PREREG 4.1: `items` is the pool. Resumable; stops once the main set's size of items is eligible.

    `settings` is the frozen record's, which `check_screening` has shown equal to SCREENING; ``record_sha256`` is
    that record's hash."""
    if bundle.family.template_version != settings["template"]:
        raise Refusal(f"template {bundle.family.template_version} differs from the frozen {settings['template']}")
    if bundle.decoder.head_precision != settings["head_precision"]:
        raise Refusal(f"head precision {bundle.decoder.head_precision} differs from the frozen {settings['head_precision']}")
    require_images(items, images_dir)
    mark_screen(out_dir, FLAGS)

    def files(item):
        return [ItemFile(out_dir / "items" / f"{item['item_id']}.json")]

    def produce(item, missing):
        return [(screen_item(bundle, item, open_image(images_dir, item), settings["surface"], FULL_Q), None)]

    records, found = [], 0
    if settings["main_size"]:
        for position, _, (record,) in item_loop(items, record_sha256, files, produce):
            records.append(record)
            write_flags(out_dir, records)  # screened items, manifest order
            found += record["eligible"]
            if position % 50 == 0:
                print(f"{position}/{len(items)} pool items screened, {found} eligible", flush=True)
            if found == settings["main_size"]:
                break
    write_flags(out_dir, records)
    return {"pool": len(items), "screened": len(records), "full_q_eligible": sum(r["full_q_eligible"] for r in records),
            "removed_image_independent": sum(bool(r["blind_eligible"]) for r in records),
            "eligible": found, "unscreened": len(items) - len(records)}


def count(items: list[dict], decisions: Path, screen_dir: Path) -> dict:
    """PREREG 4.1 and 4.5(d): sizes from the screen's flags, the ones the stage-3 freeze reads, so it works mid-screen too; never loads the
    model."""
    flags = screen_flags(screen_dir)
    passed = [item["item_id"] for item in pool(items, decisions)]
    main = min(sum(flags.get(i, False) for i in passed), SCREENING["main_size"])
    return {"main_set": main, "control": min(main, SCREENING["control_size"]),
            "short_of_1000": main < SCREENING["main_size"], "unscreened_passed": sum(i not in flags for i in passed)}
