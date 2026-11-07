"""Level 2 acceptance tests on the real model. Criteria come from tests/README.md."""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import torch

from .cache import entries_hash
from .engine import Decoder, greedy, probe, replay, score_sequence
from .model import Bundle, full_forward_logits, pixel_values, prefill
from .prompts import build_prompt, candidate_ids, surface_forms, window_composition
from .runner import Q, contexts_for, open_image, pilot_units, run_item
from .selectors import pw_attn, top_k

SAMPLE_PER_COMPARISON = 20_000
SURFACE = "lower"


def sample_differences(difference: torch.Tensor, generator: np.random.Generator) -> np.ndarray:
    values = difference.abs().flatten().cpu().numpy()
    if values.size <= SAMPLE_PER_COMPARISON:
        return values
    return values[generator.integers(0, values.size, SAMPLE_PER_COMPARISON)]


def margin_incremental(bundle: Bundle, state, prompt: str, a_true: str, a_alt: str) -> float:
    total = {}
    for name, answer in (("true", a_true), ("alt", a_alt)):
        ids = candidate_ids(bundle.tokenizer, prompt, surface_forms(answer)[SURFACE])
        total[name] = sum(score_sequence(bundle.decoder, state, ids))
    return total["true"] - total["alt"]


def margin_full_forward(bundle: Bundle, tokens, pixels, a_true: str, a_alt: str) -> float:
    total = {}
    base = tokens.prefix_ids + tokens.suffix_ids
    for name, answer in (("true", a_true), ("alt", a_alt)):
        ids = candidate_ids(bundle.tokenizer, tokens.prompt, surface_forms(answer)[SURFACE])
        logits = full_forward_logits(bundle, base + ids[:-1], pixels, len(base) - 1)
        log_probs = torch.log_softmax(logits[0], dim=-1)
        total[name] = sum(float(log_probs[i, token]) for i, token in enumerate(ids))
    return total["true"] - total["alt"]


def check_serialization(tokenizer, items: list[dict]) -> dict:
    """E1 on all development items. Tokenizer only."""
    failures: list[str] = []
    compositions = {}
    for item in items:
        tokens = contexts_for(item, tokenizer)
        if len({tuple(t.prefix_ids) for t in tokens.values()}) != 1:
            failures.append(f"{item['item_id']}: prefix differs between contexts")
        spans = {tuple(t.suffix_ids[t.question_span[0] : t.question_span[1]]) for t in tokens.values()}
        if len(spans) != 1:
            failures.append(f"{item['item_id']}: question tokens differ between contexts")
        for answer in (item["a_true"], item["a_alt"]):
            for surface in surface_forms(answer).values():
                for t in tokens.values():
                    try:
                        candidate_ids(tokenizer, t.prompt, surface)
                    except ValueError as error:
                        failures.append(f"{item['item_id']}: {error}")
        for name, t in tokens.items():
            composition = window_composition(t)
            key = f"{name}: q={composition['question']:.3f} d={composition['description']:.3f}"
            compositions[key] = compositions.get(key, 0) + 1
    return {"pass": not failures, "items": len(items), "failures": failures[:50], "window_composition": compositions}


def noise_floor(bundle: Bundle, loaded: list[tuple[dict, object]]) -> dict:
    """Full forward vs prefix-then-suffix with the full cache, on the Q suffix."""
    generator = np.random.default_rng(0)
    samples, maxima, margins = [], [], []
    keep_all = torch.arange(576)
    for item, image in loaded:
        tokens = contexts_for(item, bundle.tokenizer)[Q]
        pixels = pixel_values(bundle, image)
        ids = tokens.prefix_ids + tokens.suffix_ids
        full = full_forward_logits(bundle, ids, pixels, len(tokens.prefix_ids))
        prefix = prefill(bundle, tokens.prefix_ids, pixels)
        state, incremental = replay(bundle.decoder, prefix, keep_all, torch.tensor([tokens.suffix_ids]), "dense")
        difference = full - incremental
        samples.append(sample_differences(difference, generator))
        maxima.append(float(difference.abs().max()))
        margins.append(abs(
            margin_full_forward(bundle, tokens, pixels, item["a_true"], item["a_alt"])
            - margin_incremental(bundle, state, tokens.prompt, item["a_true"], item["a_alt"])
        ))
    pooled = np.concatenate(samples)
    return {
        "items": len(loaded),
        "logit_p99": float(np.percentile(pooled, 99)),
        "logit_max": float(max(maxima)),
        "margin_mean_abs": float(np.mean(margins)),
        "margin_max_abs": float(np.max(margins)),
    }


def run_e2(bundle: Bundle, loaded, floor: dict) -> dict:
    worst = 0.0
    keep_all = torch.arange(576)
    for item, image in loaded:
        tokens = contexts_for(item, bundle.tokenizer)["mis"]
        pixels = pixel_values(bundle, image)
        full = full_forward_logits(bundle, tokens.prefix_ids + tokens.suffix_ids, pixels, len(tokens.prefix_ids))
        prefix = prefill(bundle, tokens.prefix_ids, pixels)
        _, compact_logits = replay(bundle.decoder, prefix, keep_all, torch.tensor([tokens.suffix_ids]), "compact")
        worst = max(worst, float((full - compact_logits).abs().max()))
    return {"pass": worst <= floor["logit_max"], "max_logit_diff": worst, "limit": floor["logit_max"]}


def kept_sets(k: int, seed: int, attention: np.ndarray) -> dict[str, np.ndarray]:
    random_set = np.random.default_rng(seed).choice(576, k, replace=False)
    return {"random": np.sort(random_set), "concentrated": np.arange(k), "scored": top_k(pw_attn(attention), k)}


def run_e3_e4(bundle: Bundle, loaded, floor: dict, budgets=(288, 144, 72), progress=None) -> tuple[dict, dict]:
    generator = np.random.default_rng(1)
    samples, margins, mismatches = [], [], []
    hash_failures = []
    near_tie_limit = 2 * floor["logit_max"]
    for position, (item, image) in enumerate(loaded):
        if progress:
            progress(f"  E3 item {position + 1}/{len(loaded)}")
        tokens = contexts_for(item, bundle.tokenizer)
        prefix = prefill(bundle, tokens[Q].prefix_ids, pixel_values(bundle, image))
        before = entries_hash(prefix.entries)
        for name in (Q, "mis"):
            suffix = torch.tensor([tokens[name].suffix_ids])
            attention = probe(bundle.decoder, prefix, suffix)
            if entries_hash(prefix.entries) != before:
                hash_failures.append(f"{item['item_id']}: changed by probe {name}")
            for k in budgets:
                for kind, kept in kept_sets(k, position, attention).items():
                    kept_tensor = torch.from_numpy(kept)
                    state_c, logits_c = replay(bundle.decoder, prefix, kept_tensor, suffix, "compact")
                    state_d, logits_d = replay(bundle.decoder, prefix, kept_tensor, suffix, "dense")
                    samples.append(sample_differences(logits_c - logits_d, generator))
                    margins.append(abs(
                        margin_incremental(bundle, state_c, tokens[name].prompt, item["a_true"], item["a_alt"])
                        - margin_incremental(bundle, state_d, tokens[name].prompt, item["a_true"], item["a_alt"])
                    ))
                    greedy_c = greedy(bundle.decoder, state_c, max_new=10, stop_at_eos=False)
                    greedy_d = greedy(bundle.decoder, state_d, max_new=10, stop_at_eos=False)
                    for step, (logit_c, logit_d) in enumerate(zip(greedy_c.step_logits, greedy_d.step_logits)):
                        samples.append(sample_differences(logit_c - logit_d, generator))
                        if greedy_c.token_ids[step] != greedy_d.token_ids[step]:
                            gap = min(greedy_c.top_two_gaps[step], greedy_d.top_two_gaps[step])
                            mismatches.append({
                                "item": item["item_id"], "context": name, "k": k, "kind": kind, "step": step,
                                "gap": gap, "near_tie": gap < near_tie_limit,
                            })
                            break
                    if entries_hash(prefix.entries) != before:
                        hash_failures.append(f"{item['item_id']}: changed by replay")
    p99 = float(np.percentile(np.concatenate(samples), 99))
    mean_margin = float(np.mean(margins))
    real_mismatches = [m for m in mismatches if not m["near_tie"]]
    e3 = {
        "pass": p99 <= 2 * floor["logit_p99"] and mean_margin < 0.01 and not real_mismatches,
        "logit_p99": p99,
        "p99_limit": 2 * floor["logit_p99"],
        "margin_mean_abs": mean_margin,
        "margin_limit": 0.01,
        "greedy_mismatches": mismatches,
        "greedy_mismatches_not_near_tie": len(real_mismatches),
        "comparisons": len(margins),
    }
    e4 = {"pass": not hash_failures, "failures": hash_failures[:20]}
    return e3, e4


def run_e5(bundle: Bundle, loaded, floor: dict) -> dict:
    """Placebo: the same kept indices and suffix through the B and C code path."""
    worst, different_answers = 0.0, 0
    for position, (item, image) in enumerate(loaded):
        tokens = contexts_for(item, bundle.tokenizer)
        prefix = prefill(bundle, tokens[Q].prefix_ids, pixel_values(bundle, image))
        suffix = torch.tensor([tokens["mis"].suffix_ids])
        kept = torch.from_numpy(kept_sets(144, position, probe(bundle.decoder, prefix, suffix))["scored"])
        state_b, logits_b = replay(bundle.decoder, prefix, kept, suffix)
        state_c, logits_c = replay(bundle.decoder, prefix, kept.clone(), suffix)
        worst = max(worst, float((logits_b - logits_c).abs().max()))
        different_answers += greedy(bundle.decoder, state_b).token_ids != greedy(bundle.decoder, state_c).token_ids
    return {
        "pass": worst <= floor["logit_max"] and different_answers == 0,
        "max_logit_diff": worst,
        "different_answers": different_answers,
    }


def run_e7(bundle: Bundle, loaded) -> dict:
    units = pilot_units()
    different = []
    for item, image in loaded:
        first, _ = run_item(bundle, item, image, units)
        second, _ = run_item(bundle, item, image, units)
        if json.dumps(first, sort_keys=True) != json.dumps(second, sort_keys=True):
            different.append(item["item_id"])
    return {"pass": not different, "different_items": different}


def native_head(bundle: Bundle) -> Bundle:
    decoder = Decoder(bundle.decoder.language_model, bundle.decoder.lm_head, "native")
    return dataclasses.replace(bundle, decoder=decoder)


def validate(
    bundle: Bundle, items: list[dict], images_dir: Path, noise_items: int = 50, sparse_items: int = 20, repeat_items: int = 5,
) -> dict:
    start = time.time()

    def stage(name: str) -> None:
        print(f"[{(time.time() - start) / 60:5.1f} min] {name}", flush=True)

    report: dict = {"head_precision": bundle.decoder.head_precision}
    stage(f"E1: serialization, {len(items)} items")
    report["E1"] = check_serialization(bundle.tokenizer, items)
    loaded = [(item, open_image(images_dir, item)) for item in items[:noise_items]]
    stage(f"noise floor, fp32 head, {len(loaded)} items")
    floor = noise_floor(bundle, loaded)
    report["noise_floor"] = floor
    stage(f"noise floor, native head, {len(loaded)} items")
    report["noise_floor_native_head"] = noise_floor(native_head(bundle), loaded)
    stage(f"E2: full fidelity, {len(loaded)} items")
    report["E2"] = run_e2(bundle, loaded, floor)
    stage(f"E3 and E4: sparse equivalence and cache hashes, {len(loaded[:sparse_items])} items (the slowest stage)")
    report["E3"], report["E4"] = run_e3_e4(bundle, loaded[:sparse_items], floor, progress=stage)
    stage("E5: placebo")
    report["E5"] = run_e5(bundle, loaded[:sparse_items], floor)
    stage(f"E7: determinism, {len(loaded[:repeat_items])} items")
    report["E7"] = run_e7(bundle, loaded[:repeat_items])
    stage("done")
    report["all_pass"] = all(report[name]["pass"] for name in ("E1", "E2", "E3", "E4", "E5", "E7"))
    return report
