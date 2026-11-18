"""Units, runs and screens: one unit type, the per-unit and per-item runners, one resumable item loop, eligibility."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .analysis.statistics import margin
from .engine import greedy_releasing, probe, replay, score_sequence
from .files import write_atomic, write_json
from .model import Bundle
from .models.family import Tokens
from .normalize import outcome
from .prompts import surface_forms, window_composition
from .protocol import Refusal
from .selectors import Budget, kept_set

ATTENTION_SELECTORS = ("pw", "qspan")
Q = "Q"
NO_CONTEXT = "none"  # selection context of text-agnostic selectors and full-cache anchors

# --- Units ---------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Unit:
    """One cell to run on an item: a selector at a budget (an absolute k or a share), selection and answering contexts.

    ``followup`` units carry the follow-ups' seed, m and arm in their record and rows, even when unset (as frozen).
    """

    selector: str  # pw, qspan, lt, uniform, random, restore or full
    k: int | None
    reserve: bool
    selection: str  # Q, a condition, or none
    answering: str  # Q or a condition
    seed: int | None = None
    m: int | None = None
    arm: str | None = None  # restore: reference, treatment or control
    share: float | None = None
    followup: bool = False

    def __post_init__(self):
        Budget(self.k, self.share)  # exactly one is set

    @property
    def budget(self) -> Budget:
        return Budget(self.k, self.share)

    def record(self) -> dict:
        """The unit as freeze records and rows hold it."""
        if self.share is not None:
            return {"selector": self.selector, "share": self.share, "selection": self.selection, "answering": self.answering}
        fields = {"selector": self.selector, "k": self.k, "reserve": self.reserve, "selection": self.selection,
                  "answering": self.answering}
        return fields | ({"seed": self.seed, "m": self.m, "arm": self.arm} if self.followup else {})


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


def row_units(units: list[Unit], blind: Unit | None = None) -> tuple[dict, ...]:
    """Each row's unit fields, in row order, as `run_item` writes them: the blind row is labelled "blind"."""
    labels = tuple(unit.record() for unit in units)
    return labels + ((blind.record() | {"selector": "blind"},) if blind else ())


# --- Running --------------------------------------------------------------------------------------------------


def kept_for(unit: Unit, layout, attention: dict[str, np.ndarray], tokens: dict[str, Tokens]) -> np.ndarray:
    selection = tokens.get(unit.selection)
    return kept_set(layout, unit.selector, unit.budget, attention=attention.get(unit.selection),
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
    """One row. Candidates are scored before greedy decoding (neither changes the state), and greedy frees the replay
    state as it steps, so with a CPU-held prefix the device holds at most two cache copies."""
    answering = tokens[unit.answering]
    suffix = torch.tensor([answering.suffix_ids])
    box = [replay(bundle.decoder, prefix, torch.from_numpy(kept), suffix, "compact")[0]]
    scores = score_candidates(bundle, box[0], answering.prompt, item["a_true"], item["a_alt"])
    decoded = greedy_releasing(bundle.decoder, box)
    text = bundle.tokenizer.decode(decoded.token_ids, skip_special_tokens=True).strip()
    return {
        **unit.record(),
        "cell": cell_name(unit),
        "kept": kept.tolist() if unit.selector != "full" else None,
        "scores": scores,
        "greedy_ids": decoded.token_ids,
        "greedy_text": text,
        "outcome": "other" if decoded.truncated else outcome(text, item["a_true"], item["a_alt"]),
        "truncated": decoded.truncated,
        "top_two_gaps": decoded.top_two_gaps,
    }


def _rows(bundle: Bundle, item: dict, image: Image.Image, units: list[Unit]) -> tuple[dict, list[dict], dict]:
    """(tokens, rows, attention) from one prefill and one probe per selection context; the prefix is freed on return."""
    tokens, inputs = bundle.family.contexts(bundle, item, image)
    prefix, layout = bundle.family.prefill(bundle, tokens[Q].prefix_ids, inputs)
    selection_contexts = sorted({u.selection for u in units if u.selector in ATTENTION_SELECTORS})
    attention = {
        context: probe(bundle.decoder, prefix, torch.tensor([tokens[context].suffix_ids]))
        for context in selection_contexts
    }
    rows = []
    for unit in units:
        kept = kept_for(unit, layout, attention, tokens)
        if unit.share is None:
            rows.append(run_unit(bundle, item, unit, prefix, kept, tokens))
            continue
        k = layout.total if unit.selector == "full" else unit.budget.resolve(layout)
        row = run_unit(bundle, item, replace(unit, k=k, share=None), prefix, kept, tokens)
        rows.append({**row, "share": unit.share,
                     "layout": {"base": layout.maps[0].rows, "grid": list(layout.grid), "total": layout.total}})
    return tokens, rows, attention


def run_item(bundle: Bundle, item: dict, image: Image.Image, units: list[Unit], *, blind: Unit | None = None,
             contexts: bool = True) -> tuple[dict, dict[str, np.ndarray]]:
    """(item result, probe attention by selection context). ``blind``: the blind anchor, a Full unit run on the
    family's blank image and labelled "blind" in its row. ``contexts``: record each context's suffix and window."""
    tokens, rows, attention = _rows(bundle, item, image, units)
    if blind is not None:
        _, (row,), _ = _rows(bundle, item, bundle.family.blank_image(image), [blind])
        rows.append({**row, "selector": "blind"})
    result = {
        "item_id": item["item_id"],
        "image_id": item["image_id"],
        "dataset": item["dataset"],
        "a_true": item["a_true"],
        "a_alt": item["a_alt"],
        "model": bundle.family.name,
        "template": bundle.family.template_version,
        "head_precision": bundle.decoder.head_precision,
    }
    if contexts:
        result["contexts"] = {
            name: {"suffix_tokens": len(t.suffix_ids), "question_span": list(t.question_span),
                   "window": window_composition(t)}
            for name, t in tokens.items()
        }
    return {**result, "rows": rows}, attention


# --- The item loop -------------------------------------------------------------------------------------------


def require_frozen_numerics(bundle: Bundle, settings: dict) -> None:
    """Refuses a model whose prompt template or output head differs from the frozen screening settings."""
    if bundle.family.template_version != settings["template"]:
        raise Refusal(f"template {bundle.family.template_version} differs from the frozen {settings['template']}")
    if bundle.decoder.head_precision != settings["head_precision"]:
        raise Refusal(f"head precision {bundle.decoder.head_precision} differs from the frozen {settings['head_precision']}")


def require_images(items: list[dict], images_dir: Path, hint: str = "") -> None:
    """Refuses before any item runs if an item's image file is missing."""
    missing = [item["image_file"] for item in items if not (Path(images_dir) / item["image_file"]).exists()]
    if missing:
        raise Refusal(f"{len(missing)} images are missing from {images_dir} (first: {missing[0]}){hint}")


def open_image(images_dir: Path, item: dict) -> Image.Image:
    with Image.open(images_dir / item["image_file"]) as image:
        return image.convert("RGB")


@dataclass(frozen=True)
class ItemFile:
    """Where one item's result goes, and what an existing file there must match to be resumed."""

    path: Path
    subset: str | None = None  # None: the file has no subset
    units: tuple[dict, ...] | None = None  # `row_units` of its rows; None: rows not checked


def _differences(found: dict, expected: ItemFile, record_sha256: str | None) -> list[str]:
    differences = []
    if found.get("record_sha256") != record_sha256:
        differences.append(f"record {found.get('record_sha256')} (expected {record_sha256})")
    if found.get("subset") != expected.subset:
        differences.append(f"subset {found.get('subset')} (expected {expected.subset})")
    if expected.units is not None:
        keys = [list(unit) for unit in expected.units]
        rows = found.get("rows", [])
        units = [{key: row.get(key) for key in (keys[0] if keys else [])} for row in rows]
        missing = [unit for unit in expected.units if unit not in units]
        extra = [unit for unit in units if unit not in expected.units]
        if missing:
            differences.append(f"missing unit {missing[0]}")
        elif extra:
            differences.append(f"unexpected unit {extra[0]}")
        elif units != list(expected.units):
            differences.append("units in another order")
    return differences


def item_loop(items: list[dict], record_sha256: str | None, files: Callable[[dict], list[ItemFile]],
              produce: Callable[[dict, list[ItemFile]], list[tuple[dict, dict | None]]]) -> Iterator[tuple[int, dict, list[dict]]]:
    """Every screen and run: yields (position, item, the result of each of ``files(item)``), resuming finished items.

    A file that exists is used only if it was produced under ``record_sha256`` with the expected subset and units;
    otherwise the loop refuses, naming the item and what differs, and leaves the file in place. ``produce`` makes the
    missing files' results, each with the attention arrays to save beside it or None. Attention is written first and
    each item file atomically, so a file exists only when its item is complete.
    """
    for position, item in enumerate(items, 1):
        wanted, results, missing = files(item), {}, []
        for expected in wanted:
            if not expected.path.exists():
                missing.append(expected)
                continue
            found = json.loads(expected.path.read_text())
            differences = _differences(found, expected, record_sha256)
            if differences:
                raise Refusal(f"item {item['item_id']}: {expected.path} differs: {'; '.join(differences)}. "
                              "It is left in place; move it out of the folder to rerun the item")
            results[expected.path] = found
        if missing:
            for expected, (result, attention) in zip(missing, produce(item, missing), strict=True):
                expected.path.parent.mkdir(parents=True, exist_ok=True)
                if attention is not None:
                    arrays = expected.path.parent.parent / "attention"
                    arrays.mkdir(exist_ok=True)
                    np.savez_compressed(arrays / f"{expected.path.stem}.npz", **attention)
                if record_sha256 is not None:
                    result = {**result, "record_sha256": record_sha256}
                write_json(expected.path, result)
                results[expected.path] = result
        yield position, item, [results[expected.path] for expected in wanted]


# --- Screens and eligibility ---------------------------------------------------------------------------------

DEVELOPMENT = "development"  # full rows of Full_Q per item
FLAGS = "flags"  # confirmatory: eligibility flags only, and eligible.jsonl


def mark_screen(screen_dir: Path, kind: str) -> None:
    """Records the screen's kind in its folder; a folder of another kind is refused."""
    path = Path(screen_dir) / "kind.json"
    if path.exists() and (found := screen_kind(screen_dir)) != kind:
        raise Refusal(f"{screen_dir} holds a {found} screen, not a {kind} screen")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, {"kind": kind})


def screen_kind(screen_dir: Path) -> str:
    path = Path(screen_dir) / "kind.json"
    if not path.exists():
        raise Refusal(f"{path} is missing, so the screen's kind is unknown (screens written before kinds were recorded "
                      f'have none): write {{"kind": "{FLAGS}"}} or {{"kind": "{DEVELOPMENT}"}} there')
    return json.loads(path.read_text())["kind"]


def eligible(screen_row: dict, surface: str) -> bool:
    """PREREG 1.4, condition 1: Full_Q gives a correct greedy answer and a positive margin under this surface rule."""
    return screen_row["outcome"] == "correct" and margin(screen_row["scores"], surface) > 0


def screen_item(bundle: Bundle, item: dict, image: Image.Image, surface: str, full_q: Unit) -> dict:
    """Flags only. Eligible: Full_Q passes the rule with the real image and would not pass it with the blank one
    (PREREG amendment 1). The blank image runs only if Full_Q passes."""
    full = eligible(run_item(bundle, item, image, [full_q])[0]["rows"][0], surface)
    blind = None
    if full:
        blind = eligible(run_item(bundle, item, bundle.family.blank_image(image), [full_q])[0]["rows"][0], surface)
    return {"item_id": item["item_id"], "full_q_eligible": full, "blind_eligible": blind, "eligible": full and not blind}


def write_flags(screen_dir: Path, records: list[dict]) -> None:
    """eligible.jsonl: every screened item's flag, in screening (manifest) order. The one file counts and freezes
    read; the NeXT screen rewrites it after each item, so its count works mid-screen."""
    write_atomic(Path(screen_dir) / "eligible.jsonl",
                 "".join(json.dumps({"item_id": r["item_id"], "eligible": r["eligible"]}) + "\n" for r in records))


def screen_flags(screen_dir: Path) -> dict[str, bool]:
    """The eligibility flags of a confirmatory screen: the one source for counts, freezes and verification."""
    path = Path(screen_dir) / "eligible.jsonl"
    if not path.exists():
        raise Refusal(f"{path} is missing: the screen has not finished")
    return {r["item_id"]: r["eligible"] for r in map(json.loads, path.read_text().splitlines())}


def eligible_ids(screen_dir: Path, items: list[dict], surface: str) -> list[str]:
    """Ids eligible in a screen, in manifest order: a development screen's rows under ``surface``, or a confirmatory
    screen's flags."""
    if screen_kind(screen_dir) == FLAGS:
        flags = screen_flags(screen_dir)
        return [item["item_id"] for item in items if flags.get(item["item_id"])]
    chosen = []
    for item in items:
        path = Path(screen_dir) / "items" / f"{item['item_id']}.json"
        if path.exists() and eligible(json.loads(path.read_text())["rows"][0], surface):
            chosen.append(item["item_id"])
    return chosen
