"""Running a frozen main set: each item once with its subset's units and the blind anchor, resumable, printing
progress only, then the consistency counts (PREREG 2.1-2.2, 3.2, 4.2). Every comparative study runs this way."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .analysis.tables import consistency_counts, load_run
from .model import Bundle
from .runner import ItemFile, Unit, item_loop, open_image, require_images, row_units, run_item


@dataclass(frozen=True)
class MainSet:
    """A study's main set in manifest order and its control subset (PREREG 1.6, 3.1, 4.1)."""

    ids: tuple[str, ...]
    control: frozenset[str]

    @classmethod
    def of(cls, ids, control) -> MainSet:
        return cls(tuple(ids), frozenset(control))

    def subset(self, item_id: str) -> str:
        return "control" if item_id in self.control else "main"


def run_main_set(bundle: Bundle, items: list[dict], images_dir: Path, out_dir: Path, main_set: MainSet,
                 units: Callable[..., list[Unit]], blind: Unit, record_sha256: str, keep_attention: bool = False,
                 contexts: bool = True) -> dict:
    """``units(control=...)``: the study's unit list for a main or a control item. ``record_sha256``: the record the
    run is checked against. ``keep_attention``: save each item's probe attention. Returns the counts only."""
    by_id = {item["item_id"]: item for item in items}
    require_images([by_id[i] for i in main_set.ids], images_dir)

    def item_units(item_id):
        return units(control=item_id in main_set.control)

    def files(item):
        return [ItemFile(Path(out_dir) / "items" / f"{item['item_id']}.json", main_set.subset(item["item_id"]),
                         row_units(item_units(item["item_id"]), blind))]

    def produce(item, missing):
        result, attention = run_item(bundle, item, open_image(images_dir, item), item_units(item["item_id"]),
                                     blind=blind, contexts=contexts)
        return [({**result, "subset": main_set.subset(item["item_id"])}, attention if keep_attention else None)]

    for position, _, _ in item_loop([by_id[i] for i in main_set.ids], record_sha256, files, produce):
        if position % 25 == 0 or position == len(main_set.ids):
            print(f"{position}/{len(main_set.ids)} items", flush=True)
    found = consistency_counts(load_run(out_dir)[0])  # flags and counts only, never margins per item
    return {"items": len(main_set.ids), "control": len(main_set.control),
            "full_q_no_longer_eligible": found["full_q_no_longer_eligible"], "blind_now_eligible": found["blind_now_eligible"]}
