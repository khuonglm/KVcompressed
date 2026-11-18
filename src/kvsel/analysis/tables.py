"""Run tables: one run folder's rows as margins, cells, kept sets, behaviour and consistency checks."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from ..protocol import Refusal
from ..runner import eligible
from ..selectors import jaccard
from .statistics import SURFACE, margin, margin_variant

PRIMARY = "pw"  # the primary selector, PW-attn (PREREG.md 2.3)
OUTCOMES = ("correct", "alt", "other")


def column_name(row: dict) -> str:
    if row["selector"] in ("full", "uniform"):
        return f"{row['selector']}|{row['cell']}"
    reserve = "+r" if row["reserve"] else ""
    return f"{row['selector']}{reserve}|{row['cell']}"


def load_run(run_dir: Path) -> tuple[list[dict], dict[str, dict]]:
    rows, items = [], {}
    paths = sorted((Path(run_dir) / "items").glob("*.json"))
    if not paths:
        raise Refusal(f"no item files under {run_dir}")
    for path in paths:
        item = json.loads(path.read_text())
        items[item["item_id"]] = {"image_id": item["image_id"], "subset": item.get("subset")}
        rows.extend({"item_id": item["item_id"], "image_id": item["image_id"], **row} for row in item["rows"])
    return rows, items


def wide_margins(rows: list[dict], variant: str) -> tuple[pd.DataFrame, pd.Series]:
    frame = pd.DataFrame([{"item_id": r["item_id"], "column": column_name(r),
                           "value": margin_variant(r["scores"], variant)} for r in rows])
    wide = frame.pivot(index="item_id", columns="column", values="value")
    images = pd.Series({r["item_id"]: r["image_id"] for r in rows}).reindex(wide.index)
    return wide, images


def has_cells(wide: pd.DataFrame, selector: str, condition: str) -> bool:
    needed = [f"{selector}|A", f"{selector}|B_{condition}", f"{selector}|C_{condition}", f"{selector}|D_{condition}",
              f"full|{condition}", "full|Q"]
    return all(column in wide for column in needed)


def kept_sets(rows: list[dict], selector: str = PRIMARY) -> dict[tuple[str, str], set[int]]:
    """Kept visual indices per (item, selection context), for the unreserved selector at its budget."""
    return {(r["item_id"], r["selection"]): set(r["kept"]) for r in rows
            if r["selector"] == selector and not r["reserve"] and r["kept"] is not None}


def overlap(rows: list[dict], conditions: tuple[str, ...]) -> dict[str, dict]:
    """Per condition with kept sets: mean Jaccard of each item's kept set with its S_Q, and the share that changed."""
    kept = kept_sets(rows)
    out = {}
    for condition in conditions:
        values = [jaccard(s, kept[(item, "Q")]) for (item, context), s in kept.items()
                  if context == condition and (item, "Q") in kept]
        if values:
            out[condition] = {"mean_jaccard": float(np.mean(values)), "share_changed": float(np.mean(np.array(values) < 1))}
    return out


def behaviour_rates(rows: list[dict], column: Callable[[dict], str] = column_name) -> dict[str, dict]:
    """The share of correct, alternative and other answers per column."""
    frame = pd.DataFrame([{"column": column(r), "outcome": r["outcome"]} for r in rows])
    if frame.empty:
        return {}
    rates = frame.groupby("column")["outcome"].value_counts(normalize=True).unstack(fill_value=0.0)
    return {name: {key: float(rates.loc[name].get(key, 0.0)) for key in OUTCOMES} for name in rates.index}


def paired_behaviour(rows: list[dict], condition: str) -> dict[str, pd.Series]:
    """Per item, C − B in "the answer is correct" and in "the answer is the alternative", for the primary selector.
    Items without the condition's B cell drop out."""
    outcomes = pd.DataFrame([{"item_id": r["item_id"], "column": column_name(r), "outcome": r["outcome"]} for r in rows])
    pivot = outcomes.pivot(index="item_id", columns="column", values="outcome")
    b, c = pivot[f"{PRIMARY}|B_{condition}"], pivot[f"{PRIMARY}|C_{condition}"]
    return {result: ((c == result).astype(float) - (b == result).astype(float)).where(b.notna())
            for result in ("correct", "alt")}


def consistency_counts(rows: list[dict]) -> dict:
    """Consistency with the screen: Full_Q still eligible, Blind_Q never eligible, truncation."""
    full_q = [r for r in rows if r["selector"] == "full" and r["cell"] == "Q"]
    blind = [r for r in rows if r["selector"] == "blind"]
    return {
        "full_q_no_longer_eligible": sum(not eligible(r, SURFACE) for r in full_q),
        "blind_rows": len(blind),
        "blind_now_eligible": sum(eligible(r, SURFACE) for r in blind),
        "blind_mean_margin": float(np.mean([margin(r["scores"], SURFACE) for r in blind])) if blind else None,
        "truncated_share": float(np.mean([r["truncated"] for r in rows])),
    }
