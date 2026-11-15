"""`kvsel validate`'s checks on LLaVA-NeXT's variable visual span, ending in a mechanical GO or NO-GO.

Criteria are kvsel's, written down in tests/README.md ("LLaVA-NeXT, Level 2") before the GPU run, and
not changed after a failure. The NeXT prefix is shared by every context, so each item is prefilled once.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from kvsel.cache import entries_hash
from kvsel.engine import greedy, probe, replay
from kvsel.model import Bundle, full_forward_logits
from kvsel.prompts import candidate_ids, surface_forms
from kvsel.runner import NO_CONTEXT, Q, Unit, open_image, run_unit
from kvsel.validate import SURFACE, margin_full_forward, margin_incremental, sample_differences

from .family import NEXT
from .layout import Layout, derive_layout, visual_span
from .model import prefill
from .model_files import REVISION
from .prompts import contexts_for

CHECKS = ("E1", "E2", "E3", "E4", "E7")
SHARES = (0.125, 0.25, 0.5)
KINDS = ("random", "concentrated")  # no scored sets: selection is ticket 02, and E3 needs only some sparse sets
MARGIN_LIMIT = 0.01  # E3's mean |margin| limit, as kvsel validate
# For the projections. Pool and set sizes from PREREG 4.1 and 4.5(d); units per item from 4.2.
POOL, MAIN, CONTROL = 2335, 1000, 300
MAIN_PROBES, MAIN_UNITS = 3, 3 * 7 + 3  # probes Q, mis, irr; per share A and B, C, D for mis and irr; 3 Full
CONTROL_PROBES, CONTROL_UNITS = 1, 3 * 3 + 1  # probe match; per share B, C, D for match; Full_match


def kept_set(layout: Layout, share: float, kind: str, seed: int) -> np.ndarray:
    """k = round(share x feature tokens) feature tokens, random or the first k, plus every newline."""
    newlines = np.array(layout.newlines, dtype=np.int64)
    features = np.setdiff1d(np.arange(layout.total), newlines)
    k = round(share * len(features))
    chosen = np.random.default_rng(seed).choice(features, k, replace=False) if kind == "random" else features[:k]
    return np.sort(np.concatenate([chosen, newlines]))


def _reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _peak_gb(device: torch.device) -> float | None:
    return torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None


def _timed(device: torch.device, function):
    start = time.perf_counter()
    result = function()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return result, time.perf_counter() - start


def check_serialization(bundle: Bundle, items: list[dict], images_dir: Path) -> tuple[dict, dict]:
    """E1 on every item: processor expansion, shared prefix and question span, candidate tokens, layout vs span."""
    failures, totals, grids = [], [], Counter()
    for item in items:
        try:
            tokens, inputs = contexts_for(bundle.image_processor, item, open_image(images_dir, item))
            start, stop = visual_span(tokens[Q].prefix_ids, NEXT.image_token_id)
            layout = derive_layout(bundle.model.config, inputs["image_sizes"][0].tolist())
            if stop - start != layout.total:
                raise ValueError(f"span has {stop - start} tokens, layout {layout.total}")
            if len({tuple(t.suffix_ids[slice(*t.question_span)]) for t in tokens.values()}) != 1:
                raise ValueError("question tokens differ between contexts")
            for answer in (item["a_true"], item["a_alt"]):
                for surface in surface_forms(answer).values():
                    for t in tokens.values():
                        candidate_ids(bundle.tokenizer, t.prompt, surface, bundle.family)
        except ValueError as error:
            failures.append(f"{item['item_id']}: {error}")
            continue
        totals.append(layout.total)
        grids[f"{layout.grid[0]}x{layout.grid[1]}"] += 1
    layouts = {"visual_tokens_min": min(totals, default=None), "visual_tokens_max": max(totals, default=None),
               "grids": dict(grids.most_common())}
    return {"pass": not failures, "items": len(items), "failures": failures[:50]}, layouts


def replayed(bundle: Bundle, prefix, kept: torch.Tensor, suffix: torch.Tensor, mode: str, prompt: str, item: dict,
             surface: str, greedy_steps: int = 0):
    """(suffix logits, incremental margin, greedy or None) for one replay. Its state, a whole cache copy in
    dense mode, is freed on return, so validate never holds more copies than a run unit does."""
    state, logits = replay(bundle.decoder, prefix, kept, suffix, mode)
    margin = margin_incremental(bundle, state, prompt, item["a_true"], item["a_alt"], surface)
    decoded = greedy(bundle.decoder, state, max_new=greedy_steps, stop_at_eos=False) if greedy_steps else None
    return logits, margin, decoded


def check_item(bundle: Bundle, item: dict, image, position: int, sparse: bool, surface: str, found: dict) -> dict:
    """Noise floor and E2 on Q and mis; on sparse items also E3 and E4. Raw numbers go into ``found``."""
    decoder, device = bundle.decoder, bundle.device
    generator = np.random.default_rng(position)
    _reset_peak(device)
    tokens, inputs = contexts_for(bundle.image_processor, item, image)
    (prefix, layout), prefill_seconds = _timed(device, lambda: prefill(bundle, tokens[Q].prefix_ids, inputs))
    record = {"item_id": item["item_id"], "visual_tokens": layout.total, "grid": list(layout.grid),
              "prefill_peak_gb": _peak_gb(device), "prefill_seconds": prefill_seconds}
    vision = {"pixel_values": inputs["pixel_values"].to(device, next(bundle.model.parameters()).dtype),
              "image_sizes": inputs["image_sizes"].to(device)}
    keep_all = torch.arange(layout.total)
    before = entries_hash(prefix.entries)
    for name in (Q, "mis"):
        t, suffix = tokens[name], torch.tensor([tokens[name].suffix_ids])
        full = full_forward_logits(bundle, t.prefix_ids + t.suffix_ids, vision, len(t.prefix_ids))
        dense, margin_dense, _ = replayed(bundle, prefix, keep_all, suffix, "dense", t.prompt, item, surface)
        compact = replay(decoder, prefix, keep_all, suffix, "compact")[1]  # the state is dropped at once
        found[f"floor_samples_{name}"].append(sample_differences(full - dense, generator))
        found[f"floor_max_{name}"].append(float((full - dense).abs().max()))
        found[f"floor_margin_{name}"].append(abs(
            margin_full_forward(bundle, t, vision, item["a_true"], item["a_alt"], surface) - margin_dense
        ))
        found[f"full_vs_compact_{name}"].append(float((full - compact).abs().max()))
        found[f"compact_vs_dense_{name}"].append(float((compact - dense).abs().max()))
        if not sparse:
            continue
        _, seconds = _timed(device, lambda: probe(decoder, prefix, suffix))
        found["probe_seconds"].append(seconds)
        if entries_hash(prefix.entries) != before:
            found["hash_failures"].append(f"{item['item_id']}: changed by probe {name}")
        for share in SHARES:
            for kind in KINDS:
                kept = torch.from_numpy(kept_set(layout, share, kind, position))
                logits_c, margin_c, greedy_c = replayed(bundle, prefix, kept, suffix, "compact", t.prompt, item, surface, 10)
                logits_d, margin_d, greedy_d = replayed(bundle, prefix, kept, suffix, "dense", t.prompt, item, surface, 10)
                found["sparse_samples"].append(sample_differences(logits_c - logits_d, generator))
                found["sparse_margins"].append(abs(margin_c - margin_d))
                for step, (logit_c, logit_d) in enumerate(zip(greedy_c.step_logits, greedy_d.step_logits)):
                    found["sparse_samples"].append(sample_differences(logit_c - logit_d, generator))
                    if greedy_c.token_ids[step] != greedy_d.token_ids[step]:
                        found["mismatches"].append({
                            "item": item["item_id"], "context": name, "share": share, "kind": kind, "step": step,
                            "gap": min(greedy_c.top_two_gaps[step], greedy_d.top_two_gaps[step]),
                        })
                        break
                if entries_hash(prefix.entries) != before:
                    found["hash_failures"].append(f"{item['item_id']}: changed by replay")
    record["peak_gb"] = _peak_gb(device)  # includes validation-only full forward passes
    # The run's own peak: a Full unit on top of the prefix cache, on the longest suffix, so its time is an upper bound.
    longest = max(tokens, key=lambda name: len(tokens[name].suffix_ids))
    full = Unit("full", layout.total, False, NO_CONTEXT, longest)
    _reset_peak(device)
    _, record["unit_seconds"] = _timed(device, lambda: run_unit(bundle, item, full, prefix, keep_all.numpy(), tokens))
    unit_peak = _peak_gb(device)
    record["run_peak_gb"] = None if unit_peak is None else max(unit_peak, record["prefill_peak_gb"])
    return record


def fingerprint(bundle: Bundle, item: dict, image) -> str:
    """Everything a run would store for a few units, plus the prefix and probe, for the determinism check."""
    tokens, inputs = contexts_for(bundle.image_processor, item, image)
    prefix, layout = prefill(bundle, tokens[Q].prefix_ids, inputs)
    attention = probe(bundle.decoder, prefix, torch.tensor([tokens["mis"].suffix_ids]))
    kept = kept_set(layout, 0.25, "random", 0)
    rows = [run_unit(bundle, item, Unit("full", layout.total, False, NO_CONTEXT, Q), prefix, np.arange(layout.total), tokens),
            run_unit(bundle, item, Unit("random", len(kept), False, NO_CONTEXT, "mis"), prefix, kept, tokens)]
    return json.dumps({"prefix": entries_hash(prefix.entries),
                       "probe": hashlib.sha256(np.ascontiguousarray(attention).tobytes()).hexdigest(), "rows": rows},
                      sort_keys=True)


def floor_of(found: dict, name: str) -> dict:
    return {
        "logit_p99": float(np.percentile(np.concatenate(found[f"floor_samples_{name}"]), 99)),
        "logit_max": max(found[f"floor_max_{name}"]),
        "margin_mean_abs": float(np.mean(found[f"floor_margin_{name}"])),
        "margin_max_abs": max(found[f"floor_margin_{name}"]),
    }


def projection(prefill_seconds: float, probe_seconds: float, unit_seconds: float) -> dict:
    """Upper bounds: every unit costs a Full unit, and the screen runs Blind_Q on every pool item."""
    blind = prefill_seconds + unit_seconds
    screen = POOL * 2 * blind
    comparative = (MAIN * (prefill_seconds + MAIN_PROBES * probe_seconds + MAIN_UNITS * unit_seconds + blind)
                   + CONTROL * (CONTROL_PROBES * probe_seconds + CONTROL_UNITS * unit_seconds))
    return {"screen_hours_upper": screen / 3600, "comparative_hours_upper": comparative / 3600}


def validate(bundle: Bundle, items: list[dict], images_dir: Path, noise_items: int = 50, sparse_items: int = 20,
             repeat_items: int = 5, surface: str = SURFACE) -> dict:
    start = time.time()

    def stage(name: str) -> None:
        print(f"[{(time.time() - start) / 60:5.1f} min] {name}", flush=True)

    report: dict = {"model": NEXT.model_id, "revision": REVISION, "head_precision": bundle.decoder.head_precision,
                    "surface": surface, "device": str(bundle.device)}
    stage(f"E1: serialization and layouts, {len(items)} items")
    report["E1"], report["layouts"] = check_serialization(bundle, items, images_dir)

    found: dict = defaultdict(list)
    records = []
    for position, item in enumerate(items[:noise_items]):
        stage(f"item {position + 1}/{min(noise_items, len(items))}: noise floor, E2" + (", E3, E4" if position < sparse_items else ""))
        began = time.perf_counter()
        record = check_item(bundle, item, open_image(images_dir, item), position, position < sparse_items, surface, found)
        record["seconds"] = time.perf_counter() - began
        records.append(record)
        print(f"  {record['visual_tokens']} visual tokens, run peak {record['run_peak_gb']} GB, {record['seconds']:.1f} s",
              flush=True)
    floors = {name: floor_of(found, name) for name in (Q, "mis")}
    report["noise_floor"], report["noise_floor_mis"] = floors[Q], floors["mis"]

    by_suffix = {}
    for name, floor in floors.items():
        worst_full, worst_dense = max(found[f"full_vs_compact_{name}"]), max(found[f"compact_vs_dense_{name}"])
        by_suffix[name] = {"pass": worst_full <= floor["logit_max"] and worst_dense == 0, "max_logit_diff": worst_full,
                           "limit": floor["logit_max"], "compact_vs_dense_max": worst_dense}
    report["E2"] = {"pass": all(r["pass"] for r in by_suffix.values()), "by_suffix": by_suffix}

    near_tie = 2 * floors[Q]["logit_max"]
    mismatches = [m | {"near_tie": m["gap"] < near_tie} for m in found["mismatches"]]
    real = sum(not m["near_tie"] for m in mismatches)
    p99 = float(np.percentile(np.concatenate(found["sparse_samples"]), 99)) if found["sparse_samples"] else float("nan")
    margin = float(np.mean(found["sparse_margins"])) if found["sparse_margins"] else float("nan")
    report["E3"] = {
        "pass": p99 <= 2 * floors[Q]["logit_p99"] and margin < MARGIN_LIMIT and not real,
        "logit_p99": p99, "p99_limit": 2 * floors[Q]["logit_p99"], "margin_mean_abs": margin, "margin_limit": MARGIN_LIMIT,
        "greedy_mismatches": mismatches, "greedy_mismatches_not_near_tie": real, "comparisons": len(found["sparse_margins"]),
        "shares": list(SHARES), "kinds": list(KINDS),
    }
    report["E4"] = {"pass": not found["hash_failures"], "failures": found["hash_failures"][:20]}

    stage(f"E7: determinism, {min(repeat_items, len(items))} items")
    different = []
    for item in items[:repeat_items]:
        image = open_image(images_dir, item)
        if fingerprint(bundle, item, image) != fingerprint(bundle, item, image):
            different.append(item["item_id"])
    report["E7"] = {"pass": not different, "different_items": different}

    def worst(key):
        return max((r[key] for r in records if r[key] is not None), default=None)

    report["memory"] = {"run_peak_gb": worst("run_peak_gb"), "prefill_peak_gb": worst("prefill_peak_gb"),
                        "validate_peak_gb": worst("peak_gb"),
                        "at_visual_tokens": max(records, key=lambda r: r["run_peak_gb"] or 0)["visual_tokens"] if records else None}
    mean = {key: float(np.mean([r[key] for r in records])) for key in ("seconds", "prefill_seconds", "unit_seconds")}
    mean["probe_seconds"] = float(np.mean(found["probe_seconds"])) if found["probe_seconds"] else float("nan")
    report["time"] = {**mean, **projection(mean["prefill_seconds"], mean["probe_seconds"], mean["unit_seconds"])}
    report["items"] = records
    report["go"] = all(report[name]["pass"] for name in CHECKS)
    stage("done")
    return report
