"""The replication selection rule (PREREG section 3; spec, Implementation Decisions)."""

from __future__ import annotations

from pathlib import Path

from ..verify import PASS

REPLICATION_SIZE = 1000
CONTROL_SIZE = 300
IMAGE_IN_MAIN = "image in the main set"
EARLIER_RUN = "has a file in an earlier run"


def earlier_run_ids(run_dirs: list[Path]) -> set[str]:
    """Item ids with a result file anywhere under the given run folders (`.../items/<item_id>.json`)."""
    return {path.stem for run in run_dirs for path in Path(run).glob("**/items/*.json")}


def replication_set(items: list[dict], flags: dict[str, bool], decisions: dict[str, dict], main: list[str],
                    already_run: set[str]) -> dict:
    """Eligible items after the main set's last item, in manifest order, minus items sharing an image with the
    main set or already run; stops at the first unverified one (as the main-set rule does), passes over
    failures, and takes the first `REPLICATION_SIZE` passed. `decisions` must hold current digests only.

    Same order rule as the frozen `confirm.main_set`, which can't be called here: it raises instead of
    returning the next unverified item, and its screened-all check assumes the whole manifest."""
    ids = [item["item_id"] for item in items]
    by_id = {item["item_id"]: item for item in items}
    main_images = {by_id[i]["image_id"] for i in main}
    chosen, dropped, failed, next_unverified = [], {}, 0, None
    for item in items[ids.index(main[-1]) + 1:]:
        item_id = item["item_id"]
        if not flags.get(item_id, False):
            continue
        if item["image_id"] in main_images:
            dropped[item_id] = IMAGE_IN_MAIN
        elif item_id in already_run:
            dropped[item_id] = EARLIER_RUN
        elif item_id not in decisions:
            next_unverified = item_id
            break
        elif decisions[item_id]["decision"] == PASS:
            chosen.append(item_id)
            if len(chosen) == REPLICATION_SIZE:
                break
        else:
            failed += 1
    return {"replication": chosen, "control": chosen[:CONTROL_SIZE], "excluded": dropped, "failed": failed,
            "next_unverified": next_unverified}
