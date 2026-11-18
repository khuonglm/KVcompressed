"""Confirmatory study, stage 1 and the eligibility screen (PREREG 1).

Stage 1 freezes everything the screen depends on, before any confirmatory model output: PREREG section 1, the
confirmatory manifests, the code and the settings below. The screen then exposes only eligibility: each item runs
Full_Q, and, if that passes, Full_Q on a blank image (amendment 1). Nothing but the item id and its flags is written:
no answers, scores or margins, so no outcome can steer verification or the comparative experiment. Changing a
screening rule after the freeze needs a documented amendment and a fresh held-out partition, not a new record.
"""

from __future__ import annotations

import json
from pathlib import Path

from ...model import Bundle
from ...models.llava15 import LAYOUT, LLAVA
from ...protocol import Prereg, Refusal, Stage, check, freeze, manifests, sha256_file
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
    screen_item,
    write_flags,
    write_json_atomic,
)

# PREREG 1.1, 1.3 and 1.4: eligibility is Full_Q's greedy answer correct and its margin positive under this surface rule
# (greedy answers are mostly capitalized on both datasets).
SCREENING = {
    "model": LLAVA.name,
    "model_id": LLAVA.model_id,
    "template": LLAVA.template_version,
    "head_precision": LLAVA.default_head_precision,
    "surface": "capitalized",
    "eligibility": "Full_Q greedy answer correct and margin(true - alt) > 0",
    # Amendment 1 (10 November 2026): an item that passes the rule above with a blank image instead
    # of the real one is not eligible, since the model answers it without the image.
    "blind_filter": "not eligible if the same rule passes on a blank 336 x 336 image of the padding colour",
    "amendment": 1,
    "max_new_tokens": 16,
    "batch_size": 1,
}
SECTION = "## 1. Screening rules"
RECORD = Path("freeze/screening_amend2.json")  # supersedes amendment 1, which superseded screening.json
FULL_Q = Unit("full", LAYOUT.total, False, NO_CONTEXT, Q)


def stage(prereg: Path, manifest_paths: list[Path], record: Path) -> Stage:
    """Stage 1's declaration (PREREG, stage 1 in the preamble): section 1's text, every frozen manifest, the code and the settings."""
    inputs = {"prereg_screening_sha256": Prereg(prereg, SECTION), "manifests": manifests(manifest_paths)}
    return Stage("screening", record, inputs, SCREENING)


def _refuse_non_confirmatory(manifest: Path) -> None:
    splits = {json.loads(line).get("split") for line in Path(manifest).read_text().splitlines() if line.strip()}
    if splits != {"confirm"}:
        raise Refusal(f"{manifest} is not a confirmatory manifest (splits: {sorted(map(str, splits))})")


def freeze_screening(prereg: Path, manifest_paths: list[Path], record: Path = RECORD, amends: Path | None = None,
                     reason: str | None = None) -> dict:
    """Writes the stage-1 record. An amendment names the record it supersedes and why; that record stays (PREREG 1.7-1.9)."""
    if (amends is None) != (reason is None):
        raise Refusal("an amendment needs both --amends and --reason")
    if amends is not None and not Path(amends).exists():
        raise Refusal(f"{amends} does not exist")
    for manifest in manifest_paths:
        _refuse_non_confirmatory(manifest)
    extra = {"amends": {"record": str(amends), "sha256": sha256_file(amends), "reason": reason}} if amends else None
    return freeze(stage(prereg, manifest_paths, record), extra)


def check_screening(prereg: Path, manifest: Path, record: Path = RECORD, report: bool = False) -> dict:
    """The stage-1 record, if the screen may run on ``manifest``. Every manifest the record froze is checked, read
    from ``manifest``'s folder, so a changed one blocks every screen; a manifest the record never froze is refused.
    ``report``: the report-time check, which doesn't compare PREREG."""
    if not Path(record).exists():
        raise Refusal(f"no screening freeze at {record}: run `kvsel freeze-screening` first")
    frozen = json.loads(Path(record).read_text())
    if Path(manifest).name not in frozen["manifests"]:
        raise Refusal(f"{Path(manifest).name} was not frozen; frozen manifests: {sorted(frozen['manifests'])}")
    paths = [Path(manifest).parent / name for name in frozen["manifests"]]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise Refusal(f"frozen manifests are missing beside {manifest}: {', '.join(missing)}")
    return check(stage(prereg, paths, record), report)


def run_confirm_screen(bundle: Bundle, items: list[dict], images_dir: Path, out_dir: Path, frozen: dict,
                       record_sha256: str) -> dict:
    """The eligibility screen (PREREG 1.4: four flags per item, Blind_Q only after Full_Q passes; 1.5: a manifest
    prefix with `--limit`). ``frozen``: the stage-1 record, whose file hashes to ``record_sha256``."""
    settings = frozen["settings"]
    if bundle.family.template_version != settings["template"]:
        raise Refusal(f"template {bundle.family.template_version} differs from the frozen {settings['template']}")
    if bundle.decoder.head_precision != settings["head_precision"]:
        raise Refusal(f"head precision {bundle.decoder.head_precision} differs from the frozen {settings['head_precision']}")
    not_confirm = [item["item_id"] for item in items if item.get("split") != "confirm"]
    if not_confirm:
        raise Refusal(f"{len(not_confirm)} items are not confirmatory (first: {not_confirm[0]})")
    require_images(items, images_dir, "; fetch them or screen fewer items with --limit")

    mark_screen(out_dir, FLAGS)

    def files(item):
        return [ItemFile(out_dir / "items" / f"{item['item_id']}.json")]

    def produce(item, missing):
        return [(screen_item(bundle, item, open_image(images_dir, item), settings["surface"], FULL_Q), None)]

    records = []
    for position, _, (result,) in item_loop(items, record_sha256, files, produce):
        records.append(result)
        if position % 100 == 0 or position == len(items):
            print(f"{position}/{len(items)} items screened", flush=True)
    write_flags(out_dir, records)  # once, at the end: a freeze must not see a part-screened manifest

    eligible = sum(r["eligible"] for r in records)
    funnel = {
        "screened": len(items),
        "full_q_eligible": sum(r["full_q_eligible"] for r in records),
        "removed_image_independent": sum(bool(r["blind_eligible"]) for r in records),
        "eligible": eligible,
        "not_eligible": len(items) - eligible,
        "frozen_at": frozen["frozen_at"],
        "code_sha256": frozen["code_sha256"],
        "settings": settings,
    }
    write_json_atomic(out_dir / "funnel.json", funnel)
    return funnel
