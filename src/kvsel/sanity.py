"""Checks before the freeze: S1 (standard-prompt accuracy), R1 (timing and memory), E6 (coverage).

S1 detects implementation and prompt bugs only; it supports no claim. R1 replaces the guessed
runtime estimate. E6 plots the coverage reserve and compares its radius with random sets.
"""

from __future__ import annotations

import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np

from . import GRID, IMAGE_TOKEN, NUM_VISUAL
from .families import INSTRUCTION

S1_SEED = 0
S1_ITEMS = 200
BUDGETS = (288, 144, 72)
RANDOM_SETS = 1000


# ---------------------------------------------------------------- S1


def standard_prompt(question: str) -> str:
    """LLaVA-1.5's own short-answer VQA prompt, with no description and no authority line."""
    return f"USER: {IMAGE_TOKEN * NUM_VISUAL}\n{question}\n{INSTRUCTION} ASSISTANT:"


def build_s1_items(dataset: str, question_paths: list[Path], dev_manifest: Path,
                   n: int = S1_ITEMS, seed: int = S1_SEED) -> list[dict]:
    """n random questions of any type on development images: no builder filter is applied."""
    dev_images = {json.loads(line)["image_id"] for line in Path(dev_manifest).read_text().splitlines() if line.strip()}
    pool = []
    for path in question_paths:
        raw = json.loads(Path(path).read_text())
        if dataset == "gqa":
            for question_id, q in raw.items():
                if q["imageId"] in dev_images:
                    pool.append({"item_id": f"s1_gqa_{question_id}", "image_id": q["imageId"],
                                 "question": q["question"], "answers": [q["answer"]]})
        elif dataset == "textvqa":
            for q in raw["data"]:
                if q["image_id"] in dev_images:
                    pool.append({"item_id": f"s1_textvqa_{q['question_id']}", "image_id": q["image_id"],
                                 "question": q["question"], "answers": q["answers"]})
        else:
            raise ValueError(f"unknown dataset {dataset!r}")
    pool.sort(key=lambda item: item["item_id"])
    chosen = random.Random(seed).sample(pool, min(n, len(pool)))
    return [{**item, "dataset": dataset, "split": "dev", "image_file": f"{item['image_id']}.jpg"} for item in chosen]


def s1_score(dataset: str, answer: str, answers: list[str]) -> float:
    """GQA: exact match after normalisation. TextVQA: the standard VQA score, min(1, matches / 3)."""
    if dataset == "gqa":
        from .normalize import normalize

        return float(normalize(answer) == normalize(answers[0]))
    from .textvqa import normalize_text

    matches = sum(normalize_text(a) == normalize_text(answer) for a in answers)
    return min(1.0, matches / 3)


def run_s1(bundle, items: list[dict], images_dir: Path) -> dict:
    import torch

    from .engine import greedy, replay
    from .model import pixel_values, prefill
    from .runner import open_image, require_development

    require_development(items)
    rows = []
    for position, item in enumerate(items, 1):
        prompt = standard_prompt(item["question"])
        ids = bundle.tokenizer(prompt, add_special_tokens=True)["input_ids"]
        first_suffix = max(i for i, token in enumerate(ids) if token == bundle.family.image_token_id) + 1
        prefix = prefill(bundle, ids[:first_suffix], pixel_values(bundle, open_image(images_dir, item)))
        state, _ = replay(bundle.decoder, prefix, torch.arange(NUM_VISUAL), torch.tensor([ids[first_suffix:]]))
        decoded = greedy(bundle.decoder, state)
        text = bundle.tokenizer.decode(decoded.token_ids, skip_special_tokens=True).strip()
        rows.append({"item_id": item["item_id"], "question": item["question"], "answer": text,
                     "gold": Counter(item["answers"]).most_common(1)[0][0],
                     "score": s1_score(item["dataset"], text, item["answers"]), "truncated": decoded.truncated})
        if position % 20 == 0 or position == len(items):
            print(f"  {position}/{len(items)} items", flush=True)
    scores = [row["score"] for row in rows]
    return {"items": len(rows), "accuracy": float(np.mean(scores)) if scores else None,
            "truncated": sum(row["truncated"] for row in rows), "prompt": "llava-1.5 short answer", "rows": rows}


# ---------------------------------------------------------------- R1


class _Stage:
    """Wall time and peak GPU memory of one stage. Memory is None on CPU."""

    def __init__(self, device):
        import torch

        self.torch, self.cuda = torch, torch.device(device).type == "cuda"

    def __enter__(self):
        if self.cuda:
            self.torch.cuda.synchronize()
            self.torch.cuda.reset_peak_memory_stats()
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.cuda:
            self.torch.cuda.synchronize()
        self.seconds = time.perf_counter() - self.start
        gib = 2**30
        self.allocated = self.torch.cuda.max_memory_allocated() / gib if self.cuda else None
        self.reserved = self.torch.cuda.max_memory_reserved() / gib if self.cuda else None


def _summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {"n": len(array), "mean": float(array.mean()), "median": float(np.median(array)),
            "p90": float(np.percentile(array, 90))}


def run_timing(bundle, items: list[dict], images_dir: Path, volumes: dict[str, int]) -> dict:
    """Times every stage of the pilot workload per item: image, prefill, each probe, each unit.

    Short suffixes answer with Q; long ones with a description. Projections multiply measured
    per-item seconds by the given volumes.
    """
    import torch

    from .engine import probe
    from .model import pixel_values, prefill
    from .runner import (
        Q,
        contexts_for,
        kept_for,
        open_image,
        pilot_units,
        require_development,
        run_unit,
    )

    require_development(items)
    units = pilot_units()
    stages: dict[str, list[float]] = {}
    peaks: dict[str, list[float]] = {}
    totals, screens = [], []

    def record(name: str, stage: _Stage) -> None:
        stages.setdefault(name, []).append(stage.seconds)
        if stage.reserved is not None:
            peaks.setdefault(name, []).append(stage.reserved)

    for position, item in enumerate(items, 1):
        start = time.perf_counter()
        tokens = contexts_for(item, bundle.tokenizer, bundle.family)
        with _Stage(bundle.device) as stage:
            image = open_image(images_dir, item)
            inputs = pixel_values(bundle, image)
        record("image", stage)
        image_seconds = stage.seconds
        with _Stage(bundle.device) as stage:
            prefix = prefill(bundle, tokens[Q].prefix_ids, inputs)
        record("prefill", stage)
        prefill_seconds = stage.seconds
        attention = {}
        for context in sorted({u.selection for u in units if u.selector in ("pw", "qspan")}):
            with _Stage(bundle.device) as stage:
                attention[context] = probe(bundle.decoder, prefix, torch.tensor([tokens[context].suffix_ids]))
            record(f"probe_{'short' if context == Q else 'long'}", stage)
        for unit in units:
            with _Stage(bundle.device) as stage:
                run_unit(bundle, item, unit, prefix, kept_for(unit, attention, tokens), tokens)
            selector = f"{unit.selector}+r" if unit.reserve else unit.selector
            record(f"unit_{selector}_{'short' if unit.answering == Q else 'long'}", stage)
            if unit.selector == "full" and unit.answering == Q:
                screens.append(image_seconds + prefill_seconds + stage.seconds)
        totals.append(time.perf_counter() - start)
        print(f"  {position}/{len(items)} items", flush=True)

    per_item = {"screen": _summary(screens), "pilot": _summary(totals)}
    projection = {
        name: round(volume * per_item["screen" if name.startswith("screen") else "pilot"]["mean"] / 3600, 2)
        for name, volume in volumes.items()
    }
    result = {
        "items": len(items),
        "units_per_item": len(units),
        "device": str(bundle.device),
        "stages_seconds": {name: _summary(values) for name, values in sorted(stages.items())},
        "per_item_seconds": per_item,
        "projected_hours": projection,
        "volumes": volumes,
    }
    if peaks:
        result["peak_reserved_gib"] = {name: max(values) for name, values in sorted(peaks.items())}
        result["peak_reserved_gib_overall"] = max(max(values) for values in peaks.values())
    return result


# ---------------------------------------------------------------- E6

SCORE_FILL, RESERVE_FILL, DROPPED_FILL = "#2a78d6", "#eb6834", "#e6e5e0"
INK, MUTED = "#1a1a19", "#5f5e58"


def random_radii(size: int, sets: int = RANDOM_SETS, seed: int = 0) -> np.ndarray:
    from .selectors import coverage_radius

    generator = np.random.default_rng(seed)
    return np.array([coverage_radius(generator.choice(NUM_VISUAL, size, replace=False)) for _ in range(sets)])


def _grid_svg(title: str, score: set[int], reserve: set[int], x: float, y: float, cell: float = 7) -> str:
    parts = [f'<text x="{x}" y="{y - 8}" font-size="12" fill="{INK}">{title}</text>']
    for index in range(NUM_VISUAL):
        row, col = divmod(index, GRID)
        fill = RESERVE_FILL if index in reserve else SCORE_FILL if index in score else DROPPED_FILL
        parts.append(f'<rect x="{x + col * cell:.1f}" y="{y + row * cell:.1f}" width="{cell - 1}" '
                     f'height="{cell - 1}" rx="1" fill="{fill}"><title>cell ({row}, {col})</title></rect>')
    return "\n".join(parts)


def coverage_figure(k: int, example: np.ndarray | None, example_label: str) -> str:
    """Three grids: the reserve alone, a random set of the same size, one complete kept set."""
    from .selectors import coverage_order, reserve_size, top_k, with_reserve

    k_r = reserve_size(k)
    reserve = set(coverage_order()[:k_r])
    random_set = set(np.random.default_rng(0).choice(NUM_VISUAL, k_r, replace=False).tolist())
    panels = [(f"Reserve alone (k_r = {k_r})", set(), reserve), (f"Random, same size ({k_r})", random_set, set())]
    if example is not None:
        kept = set(with_reserve(example, k).tolist())
        by_score = set(top_k(example, k - k_r).tolist())
        panels.append((f"Kept set, K = {k}: {example_label}", by_score, kept - by_score))
    width, side = 7 * GRID, 7 * GRID
    total_width = 24 + len(panels) * (width + 36)
    body = [_grid_svg(title, score, res, 24 + i * (width + 36), 40) for i, (title, score, res) in enumerate(panels)]
    legend_y = 40 + side + 24
    legend = (f'<rect x="24" y="{legend_y - 10}" width="10" height="10" rx="2" fill="{SCORE_FILL}"/>'
              f'<text x="40" y="{legend_y}" font-size="12" fill="{MUTED}">kept by score / random</text>'
              f'<rect x="200" y="{legend_y - 10}" width="10" height="10" rx="2" fill="{RESERVE_FILL}"/>'
              f'<text x="216" y="{legend_y}" font-size="12" fill="{MUTED}">coverage reserve</text>'
              f'<rect x="340" y="{legend_y - 10}" width="10" height="10" rx="2" fill="{DROPPED_FILL}"/>'
              f'<text x="356" y="{legend_y}" font-size="12" fill="{MUTED}">dropped</text>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_width}" height="{legend_y + 16}" '
            f'font-family="system-ui, sans-serif" role="img" aria-label="Coverage at K = {k}">'
            f'<rect width="100%" height="100%" fill="#ffffff"/>\n'
            f'<text x="24" y="18" font-size="14" fill="{INK}">Coverage reserve at K = {k} (24 x 24 grid)</text>\n'
            + "\n".join(body) + legend + "</svg>\n")


def coverage_report(attention_dir: Path | None, out_dir: Path) -> dict:
    """Reserve radius against random sets of the same size; kept-set radius with and without reserve
    on PW-attn scores from development attention files (context Q). Writes one SVG per budget."""
    from .selectors import (
        coverage_order,
        coverage_radius,
        pw_attn,
        reserve_size,
        top_k,
        with_reserve,
    )

    files = sorted(Path(attention_dir).glob("*.npz")) if attention_dir else []
    scores = []
    for path in files:
        with np.load(path) as archive:
            if "Q" in archive:
                scores.append((path.stem, pw_attn(archive["Q"])))
    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict = {"attention_files": len(scores), "random_sets": RANDOM_SETS, "budgets": {}}
    for k in BUDGETS:
        k_r = reserve_size(k)
        reserve_radius = coverage_radius(np.array(coverage_order()[:k_r]))
        radii = random_radii(k_r)
        entry = {
            "k_r": k_r,
            "reserve_radius": reserve_radius,
            "random_radius": {"mean": float(radii.mean()), "min": float(radii.min()), "max": float(radii.max())},
            "share_random_at_or_below_reserve": float((radii <= reserve_radius).mean()),
        }
        if scores:
            without = [coverage_radius(top_k(s, k)) for _, s in scores]
            with_r = [coverage_radius(with_reserve(s, k)) for _, s in scores]
            entry["kept_radius_pw_Q"] = {"no_reserve_mean": float(np.mean(without)),
                                         "reserve_mean": float(np.mean(with_r)),
                                         "no_reserve_max": float(np.max(without)), "reserve_max": float(np.max(with_r))}
        example = scores[0] if scores else None
        svg = coverage_figure(k, example[1] if example else None, f"item {example[0]}" if example else "")
        (out_dir / f"coverage_k{k}.svg").write_text(svg)
        result["budgets"][str(k)] = entry
    (out_dir / "coverage.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def run_blind_rate(bundle, items: list[dict], images_dir: Path, surface: str = "capitalized") -> dict:
    """Development only: how often the Q prompt prefers a_true with a blank image instead of the real one.

    Reports rates only. It never splits S_H by blind status, which would choose an analysis by outcome.
    """
    from .runner import (
        NO_CONTEXT,
        Q,
        Unit,
        eligible,
        open_image,
        require_development,
        run_item,
    )
    from .screening import blank_image

    require_development(items)
    full_q = Unit("full", 576, False, NO_CONTEXT, Q)
    blank = blank_image(bundle)
    rows = []
    for position, item in enumerate(items, 1):
        real = run_item(bundle, item, open_image(images_dir, item), [full_q])[0]["rows"][0]
        blind = run_item(bundle, item, blank, [full_q])[0]["rows"][0]
        rows.append({
            "item_id": item["item_id"],
            "eligible": eligible(real, surface),
            "blind_margin": sum(blind["scores"]["true"][surface]) - sum(blind["scores"]["alt"][surface]),
            "blind_outcome": blind["outcome"],
            "blind_text": blind["greedy_text"],
        })
        if position % 25 == 0 or position == len(items):
            print(f"{position}/{len(items)} items", flush=True)
    chosen = [r for r in rows if r["eligible"]]

    def share(selected: list[dict], test) -> float | None:
        return round(sum(map(test, selected)) / len(selected), 3) if selected else None

    return {
        "items": len(rows),
        "eligible": len(chosen),
        "surface": surface,
        "eligible_image_independent": share(chosen, lambda r: r["blind_margin"] > 0),
        "eligible_blind_greedy_correct": share(chosen, lambda r: r["blind_outcome"] == "correct"),
        # amendment 1: removed if it would pass eligibility with the blank image too
        "eligible_removed_by_blind_rule": share(
            chosen, lambda r: r["blind_outcome"] == "correct" and r["blind_margin"] > 0),
        "all_image_independent": share(rows, lambda r: r["blind_margin"] > 0),
        "rows": rows,
    }
