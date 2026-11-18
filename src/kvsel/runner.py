"""Expands an item into work units, runs them and stores one result file per item."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .engine import greedy, probe, replay, score_sequence
from .items.manifest import load_items  # noqa: F401  callers still import it from here until ticket 08
from .model import Bundle
from .models.family import Tokens
from .models.llava15 import NUM_VISUAL
from .normalize import outcome
from .prompts import surface_forms, window_composition
from .selectors import Budget, kept_set

ALL_CONDITIONS = ("mis", "match", "irr")
ATTENTION_SELECTORS = ("pw", "qspan")
Q = "Q"
NO_CONTEXT = "none"  # selection context of text-agnostic selectors and full-cache anchors


@dataclass(frozen=True)
class Unit:
    selector: str  # pw, qspan, uniform or full
    k: int
    reserve: bool
    selection: str  # Q, a condition, or none
    answering: str  # Q or a condition


def pilot_units(conditions=ALL_CONDITIONS, k: int = 144, selectors=("pw", "qspan")) -> list[Unit]:
    """Cells A-D, Full anchors, uniform reference, and the reserve variants for the primary selector."""
    units: list[Unit] = []
    for selector in selectors:
        units.append(Unit(selector, k, False, Q, Q))  # A
        for condition in conditions:
            units.append(Unit(selector, k, False, condition, condition))  # B_c
            units.append(Unit(selector, k, False, Q, condition))  # C_c
            units.append(Unit(selector, k, False, condition, Q))  # D_c
    for answering in (Q, *conditions):
        units.append(Unit("full", NUM_VISUAL, False, NO_CONTEXT, answering))
        units.append(Unit("uniform", k, False, NO_CONTEXT, answering))
    units.append(Unit("pw", k, True, Q, Q))
    for condition in conditions:
        if condition in ("mis", "match"):
            units.append(Unit("pw", k, True, condition, condition))  # coverage only
            units.append(Unit("pw", k, True, Q, condition))  # isolation + coverage
    return units


def cell_name(unit: Unit) -> str:
    if unit.selection in (NO_CONTEXT,):
        return unit.answering
    if unit.selection == Q and unit.answering == Q:
        return "A"
    if unit.selection == unit.answering:
        return f"B_{unit.answering}"
    if unit.selection == Q:
        return f"C_{unit.answering}"
    return f"D_{unit.selection}"


def kept_for(unit: Unit, layout, attention: dict[str, np.ndarray], tokens: dict[str, Tokens]) -> np.ndarray:
    selection = tokens.get(unit.selection)
    return kept_set(layout, unit.selector, Budget(k=unit.k), attention=attention.get(unit.selection),
                    question_span=selection and selection.question_span, reserve=unit.reserve)


def score_candidates(bundle: Bundle, state, prompt: str, a_true: str, a_alt: str) -> dict:
    """Per-token log-probabilities (content tokens, then EOS) for each candidate and surface form."""
    scores: dict[str, dict[str, list[float]]] = {}
    for name, answer in (("true", a_true), ("alt", a_alt)):
        by_form: dict[str, list[float]] = {}
        seen: dict[str, list[float]] = {}
        for form, surface in surface_forms(answer).items():
            if surface not in seen:
                ids = bundle.family.candidate_ids(bundle.tokenizer, prompt, surface)
                seen[surface] = score_sequence(bundle.decoder, state, ids)
            by_form[form] = seen[surface]
        scores[name] = by_form
    return scores


def run_unit(bundle: Bundle, item: dict, unit: Unit, prefix, kept: np.ndarray, tokens: dict[str, Tokens]) -> dict:
    answering = tokens[unit.answering]
    suffix = torch.tensor([answering.suffix_ids])
    state, _ = replay(bundle.decoder, prefix, torch.from_numpy(kept), suffix, "compact")
    decoded = greedy(bundle.decoder, state)
    text = bundle.tokenizer.decode(decoded.token_ids, skip_special_tokens=True).strip()
    return {
        **asdict(unit),
        "cell": cell_name(unit),
        "kept": kept.tolist() if unit.selector != "full" else None,
        "scores": score_candidates(bundle, state, answering.prompt, item["a_true"], item["a_alt"]),
        "greedy_ids": decoded.token_ids,
        "greedy_text": text,
        "outcome": "other" if decoded.truncated else outcome(text, item["a_true"], item["a_alt"]),
        "truncated": decoded.truncated,
        "top_two_gaps": decoded.top_two_gaps,
    }


def run_item(bundle: Bundle, item: dict, image: Image.Image, units: list[Unit]) -> tuple[dict, dict[str, np.ndarray]]:
    tokens, inputs = bundle.family.contexts(bundle, item, image)
    prefix, layout = bundle.family.prefill(bundle, tokens[Q].prefix_ids, inputs)
    selection_contexts = sorted({u.selection for u in units if u.selector in ATTENTION_SELECTORS})
    attention = {
        context: probe(bundle.decoder, prefix, torch.tensor([tokens[context].suffix_ids]))
        for context in selection_contexts
    }
    rows = [run_unit(bundle, item, unit, prefix, kept_for(unit, layout, attention, tokens), tokens) for unit in units]
    result = {
        "item_id": item["item_id"],
        "image_id": item["image_id"],
        "dataset": item["dataset"],
        "a_true": item["a_true"],
        "a_alt": item["a_alt"],
        "model": bundle.family.name,
        "template": bundle.family.template_version,
        "head_precision": bundle.decoder.head_precision,
        "contexts": {
            name: {
                "suffix_tokens": len(t.suffix_ids),
                "question_span": list(t.question_span),
                "window": window_composition(t),
            }
            for name, t in tokens.items()
        },
        "rows": rows,
    }
    return result, attention


def require_development(items: list[dict]) -> None:
    """Confirmatory items need the freeze guard, which does not exist yet."""
    other = [item["item_id"] for item in items if item.get("split") != "dev"]
    if other:
        raise SystemExit(
            f"Refusing to run: {len(other)} items are not development items (first: {other[0]}). "
            "Confirmatory runs are not available before the freeze tooling exists."
        )


def write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w") as handle:
        json.dump(payload, handle)
    os.replace(temporary, path)


def open_image(images_dir: Path, item: dict) -> Image.Image:
    with Image.open(images_dir / item["image_file"]) as image:
        return image.convert("RGB")


def run_items(bundle: Bundle, items: list[dict], images_dir: Path, out_dir: Path, units: list[Unit]) -> None:
    """Runs each item once. An item's file appears only when complete, so reruns resume safely."""
    require_development(items)
    (out_dir / "items").mkdir(parents=True, exist_ok=True)
    (out_dir / "attention").mkdir(exist_ok=True)
    for position, item in enumerate(items, 1):
        target = out_dir / "items" / f"{item['item_id']}.json"
        if target.exists():
            continue
        result, attention = run_item(bundle, item, open_image(images_dir, item), units)
        np.savez_compressed(out_dir / "attention" / f"{item['item_id']}.npz", **attention)
        write_json_atomic(target, result)
        if position % 10 == 0 or position == len(items):
            print(f"{position}/{len(items)} items", flush=True)


def eligible(screen_row: dict, surface: str) -> bool:
    """Full_Q gives a correct greedy answer and a positive margin under this surface rule."""
    scores = screen_row["scores"]
    margin = sum(scores["true"][surface]) - sum(scores["alt"][surface])
    return screen_row["outcome"] == "correct" and margin > 0


def run_screen(bundle: Bundle, items: list[dict], images_dir: Path, out_dir: Path) -> None:
    """Full_Q only, on development items, in the same one-file-per-item form."""
    run_items(bundle, items, images_dir, out_dir, [Unit("full", NUM_VISUAL, False, NO_CONTEXT, Q)])

