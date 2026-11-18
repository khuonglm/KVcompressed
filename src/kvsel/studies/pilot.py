"""The development pilot: Full_Q screens and the full cell grid on development items, and LLaVA-1.5's validation.

Nothing here is frozen. Its rehearsal report is the confirmatory report run on a pilot folder, whose item files carry
no subset (`kvsel.studies.confirmatory.report`).
"""

from __future__ import annotations

from pathlib import Path

from ..model import Bundle
from ..models.llava15 import LAYOUT
from ..protocol import Refusal
from ..runner import DEVELOPMENT, NO_CONTEXT, ItemFile, Q, Unit, item_loop, mark_screen, open_image, row_units, run_item
from ..selectors import Budget
from ..validation import KINDS, Plan

ALL_CONDITIONS = ("mis", "match", "irr")


def pilot_units(conditions=ALL_CONDITIONS, k: int = 144, selectors=("pw", "qspan")) -> list[Unit]:
    """Cells A-D, Full anchors, uniform reference, and the reserve variants for the primary selector (PREREG 2.1,
    whose main and control unit lists are this grid at K = 144)."""
    units: list[Unit] = []
    for selector in selectors:
        units.append(Unit(selector, k, False, Q, Q))  # A
        for condition in conditions:
            units.append(Unit(selector, k, False, condition, condition))  # B_c
            units.append(Unit(selector, k, False, Q, condition))  # C_c
            units.append(Unit(selector, k, False, condition, Q))  # D_c
    for answering in (Q, *conditions):
        units.append(Unit("full", LAYOUT.total, False, NO_CONTEXT, answering))
        units.append(Unit("uniform", k, False, NO_CONTEXT, answering))
    units.append(Unit("pw", k, True, Q, Q))
    for condition in conditions:
        if condition in ("mis", "match"):
            units.append(Unit("pw", k, True, condition, condition))  # coverage only
            units.append(Unit("pw", k, True, Q, condition))  # isolation + coverage
    return units


VALIDATION = Plan(budgets=(Budget(k=288), Budget(k=144), Budget(k=72)), kinds=KINDS, units=tuple(pilot_units()),
                  placebo=Budget(k=144))  # tests/README.md, "Level 2: LLaVA pipeline"


def require_development(items: list[dict]) -> None:
    """Development commands take development items only: confirmatory items run only after the freeze stages of
    the PREREG preamble."""
    other = [item["item_id"] for item in items if item.get("split") != "dev"]
    if other:
        raise Refusal(f"Refusing to run: {len(other)} items are not development items (first: {other[0]}). "
                      "Confirmatory items run only through the frozen confirmatory commands.")


def run_items(bundle: Bundle, items: list[dict], images_dir: Path, out_dir: Path, units: list[Unit]) -> None:
    """Runs each item once, saving its probe attention. An item's file appears only when complete, so reruns resume."""
    require_development(items)

    def files(item):
        return [ItemFile(out_dir / "items" / f"{item['item_id']}.json", units=row_units(units))]

    def produce(item, missing):
        return [run_item(bundle, item, open_image(images_dir, item), units)]

    for position, _, _ in item_loop(items, None, files, produce):
        if position % 10 == 0 or position == len(items):
            print(f"{position}/{len(items)} items", flush=True)


def run_screen(bundle: Bundle, items: list[dict], images_dir: Path, out_dir: Path) -> None:
    """Full_Q only, on development items, in the same one-file-per-item form."""
    mark_screen(out_dir, DEVELOPMENT)
    run_items(bundle, items, images_dir, out_dir, [Unit("full", LAYOUT.total, False, NO_CONTEXT, Q)])
