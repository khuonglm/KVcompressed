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
from .families import LLAVA, Family
from .model import Bundle, full_forward_logits, pixel_values, prefill
from .prompts import candidate_ids, surface_forms, window_composition
from .runner import Q, contexts_for, open_image, pilot_units, run_item
from .selectors import pw_attn, top_k

SAMPLE_PER_COMPARISON = 20_000
SURFACE = "capitalized"  # the form frozen for GQA; set per run with --surface


def sample_differences(difference: torch.Tensor, generator: np.random.Generator) -> np.ndarray:
    values = difference.abs().flatten().cpu().numpy()
    if values.size <= SAMPLE_PER_COMPARISON:
        return values
    return values[generator.integers(0, values.size, SAMPLE_PER_COMPARISON)]


def margin_incremental(bundle: Bundle, state, prompt: str, a_true: str, a_alt: str) -> float:
    total = {}
    for name, answer in (("true", a_true), ("alt", a_alt)):
        ids = candidate_ids(bundle.tokenizer, prompt, surface_forms(answer)[SURFACE], bundle.family)
        total[name] = sum(score_sequence(bundle.decoder, state, ids))
    return total["true"] - total["alt"]


def margin_full_forward(bundle: Bundle, tokens, pixels, a_true: str, a_alt: str) -> float:
    total = {}
    base = tokens.prefix_ids + tokens.suffix_ids
    for name, answer in (("true", a_true), ("alt", a_alt)):
        ids = candidate_ids(bundle.tokenizer, tokens.prompt, surface_forms(answer)[SURFACE], bundle.family)
        logits = full_forward_logits(bundle, base + ids[:-1], pixels, len(base) - 1)
        log_probs = torch.log_softmax(logits[0], dim=-1)
        total[name] = sum(float(log_probs[i, token]) for i, token in enumerate(ids))
    return total["true"] - total["alt"]


def check_serialization(tokenizer, items: list[dict], family: Family = LLAVA) -> dict:
    """E1 on all development items. Tokenizer only."""
    failures: list[str] = []
    compositions = {}
    for item in items:
        tokens = contexts_for(item, tokenizer, family)
        if len({tuple(t.prefix_ids) for t in tokens.values()}) != 1:
            failures.append(f"{item['item_id']}: prefix differs between contexts")
        spans = {tuple(t.suffix_ids[t.question_span[0] : t.question_span[1]]) for t in tokens.values()}
        if len(spans) != 1:
            failures.append(f"{item['item_id']}: question tokens differ between contexts")
        for answer in (item["a_true"], item["a_alt"]):
            for surface in surface_forms(answer).values():
                for t in tokens.values():
                    try:
                        candidate_ids(tokenizer, t.prompt, surface, family)
                    except ValueError as error:
                        failures.append(f"{item['item_id']}: {error}")
        for name, t in tokens.items():
            composition = window_composition(t)
            key = f"{name}: q={composition['question']:.3f} d={composition['description']:.3f}"
            compositions[key] = compositions.get(key, 0) + 1
    return {"pass": not failures, "items": len(items), "failures": failures[:50], "window_composition": compositions}


def noise_floor(bundle: Bundle, loaded: list[tuple[dict, object]], context: str = Q) -> dict:
    """Full forward vs prefix-then-suffix with the full cache, on one suffix."""
    generator = np.random.default_rng(0)
    samples, maxima, margins = [], [], []
    keep_all = torch.arange(576)
    for item, image in loaded:
        tokens = contexts_for(item, bundle.tokenizer, bundle.family)[context]
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


def run_e2(bundle: Bundle, loaded, floors: dict[str, dict]) -> dict:
    """Compact K = 576 on each suffix, against one full pass and that suffix's own noise floor.

    Compact must also equal dense with nothing dropped exactly: that is the part of this test the
    noise floor cannot show, since at K = 576 the two compute the same thing.
    """
    keep_all = torch.arange(576)
    by_suffix = {}
    for name, floor in floors.items():
        worst_full, worst_dense = 0.0, 0.0
        for item, image in loaded:
            tokens = contexts_for(item, bundle.tokenizer, bundle.family)[name]
            pixels = pixel_values(bundle, image)
            full = full_forward_logits(bundle, tokens.prefix_ids + tokens.suffix_ids, pixels, len(tokens.prefix_ids))
            prefix = prefill(bundle, tokens.prefix_ids, pixels)
            suffix = torch.tensor([tokens.suffix_ids])
            _, compact_logits = replay(bundle.decoder, prefix, keep_all, suffix, "compact")
            _, dense_logits = replay(bundle.decoder, prefix, keep_all, suffix, "dense")
            worst_full = max(worst_full, float((full - compact_logits).abs().max()))
            worst_dense = max(worst_dense, float((compact_logits - dense_logits).abs().max()))
        by_suffix[name] = {
            "pass": worst_full <= floor["logit_max"] and worst_dense == 0,
            "max_logit_diff": worst_full,
            "limit": floor["logit_max"],
            "compact_vs_dense_max": worst_dense,
        }
    return {"pass": all(result["pass"] for result in by_suffix.values()), "by_suffix": by_suffix}


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
        tokens = contexts_for(item, bundle.tokenizer, bundle.family)
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
        tokens = contexts_for(item, bundle.tokenizer, bundle.family)
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


def diagnose_e2(bundle: Bundle, loaded) -> dict:
    """Separates three comparisons per suffix: full vs dense, full vs compact, compact vs dense."""
    keep_all = torch.arange(576)
    report: dict = {}
    for name in (Q, "mis"):
        rows = []
        for item, image in loaded:
            tokens = contexts_for(item, bundle.tokenizer, bundle.family)[name]
            pixels = pixel_values(bundle, image)
            full = full_forward_logits(bundle, tokens.prefix_ids + tokens.suffix_ids, pixels, len(tokens.prefix_ids))
            prefix = prefill(bundle, tokens.prefix_ids, pixels)
            suffix = torch.tensor([tokens.suffix_ids])
            state_c, compact_logits = replay(bundle.decoder, prefix, keep_all, suffix, "compact")
            _, dense_logits = replay(bundle.decoder, prefix, keep_all, suffix, "dense")
            difference = (full - compact_logits).abs()[0]
            position, vocabulary = np.unravel_index(int(difference.argmax()), difference.shape)
            top = full[0].topk(10, dim=-1).indices
            rows.append({
                "item": item["item_id"],
                "suffix_tokens": len(tokens.suffix_ids),
                "full_vs_compact_max": float(difference.max()),
                "full_vs_dense_max": float((full - dense_logits).abs().max()),
                "compact_vs_dense_max": float((compact_logits - dense_logits).abs().max()),
                "max_top10_logits": float(difference.gather(1, top).max()),
                "max_at_suffix_position": int(position),
                "max_at_input_token": bundle.tokenizer.decode(tokens.suffix_ids[int(position)]),
                "max_at_vocabulary_token": bundle.tokenizer.decode(int(vocabulary)),
                "logit_there": float(full[0, int(position), int(vocabulary)]),
                "margin_full_vs_compact": abs(
                    margin_full_forward(bundle, tokens, pixels, item["a_true"], item["a_alt"])
                    - margin_incremental(bundle, state_c, tokens.prompt, item["a_true"], item["a_alt"])
                ),
            })
        maxima = [row["full_vs_compact_max"] for row in rows]
        report[name] = {
            "items": len(rows),
            "full_vs_compact_max": max(maxima),
            "full_vs_dense_max": max(row["full_vs_dense_max"] for row in rows),
            "compact_vs_dense_max": max(row["compact_vs_dense_max"] for row in rows),
            "per_item_max_median": float(np.median(maxima)),
            "per_item_max_p90": float(np.percentile(maxima, 90)),
            "max_among_top10_logits": max(row["max_top10_logits"] for row in rows),
            "margin_mean_abs": float(np.mean([row["margin_full_vs_compact"] for row in rows])),
            "margin_max_abs": max(row["margin_full_vs_compact"] for row in rows),
            "worst_items": sorted(rows, key=lambda row: -row["full_vs_compact_max"])[:5],
        }
    return report


def check_tokenization(tokenizer, items: list[dict], run_dir: Path, family: Family = LLAVA) -> dict:
    """Whether scoring a string uses the same tokens the model emits when it generates that string.

    Uses Full_Q rows. A mismatch means M scores a tokenization the model does not produce.
    """
    by_id = {item["item_id"]: item for item in items}
    agree, checked, examples = 0, 0, []
    for path in sorted((Path(run_dir) / "items").glob("*.json")):
        result = json.loads(path.read_text())
        item = by_id.get(result["item_id"])
        if item is None:
            continue
        prompt = contexts_for(item, tokenizer, family)[Q].prompt
        for row in result["rows"]:
            if row["selector"] != "full" or row["cell"] != "Q" or not row["greedy_text"] or row["truncated"]:
                continue
            checked += 1
            rescored = candidate_ids(tokenizer, prompt, row["greedy_text"], family)[:-1]
            if rescored == row["greedy_ids"]:
                agree += 1
            elif len(examples) < 10:
                examples.append({"item": item["item_id"], "text": row["greedy_text"],
                                 "generated": row["greedy_ids"], "rescored": rescored})
    return {"checked": checked, "agree": agree, "share": agree / checked if checked else None, "mismatches": examples}


def native_head(bundle: Bundle) -> Bundle:
    old = bundle.decoder
    decoder = Decoder(old.language_model, old.lm_head, "native", old.stop_ids, old.fp32_head_copy)
    return dataclasses.replace(bundle, decoder=decoder)


def validate(
    bundle: Bundle, items: list[dict], images_dir: Path, noise_items: int = 50, sparse_items: int = 20, repeat_items: int = 5,
) -> dict:
    start = time.time()

    def stage(name: str) -> None:
        print(f"[{(time.time() - start) / 60:5.1f} min] {name}", flush=True)

    report: dict = {"model": bundle.family.name, "head_precision": bundle.decoder.head_precision}
    stage(f"E1: serialization, {len(items)} items")
    report["E1"] = check_serialization(bundle.tokenizer, items, bundle.family)
    loaded = [(item, open_image(images_dir, item)) for item in items[:noise_items]]
    head = bundle.decoder.head_precision
    stage(f"noise floor, {head} head, {len(loaded)} items")
    floor = noise_floor(bundle, loaded)
    report["noise_floor"] = floor
    if bundle.decoder.head_precision == "native":
        report["noise_floor_native_head"] = floor  # the default head is already the native one
    else:
        stage(f"noise floor, native head, {len(loaded)} items")
        report["noise_floor_native_head"] = noise_floor(native_head(bundle), loaded)
    stage(f"noise floor, {head} head, QH_mis suffix, {len(loaded)} items")
    floor_mis = noise_floor(bundle, loaded, "mis")
    report["noise_floor_mis"] = floor_mis
    stage(f"E2: full fidelity, both suffixes, {len(loaded)} items")
    report["E2"] = run_e2(bundle, loaded, {Q: floor, "mis": floor_mis})
    stage(f"E3 and E4: sparse equivalence and cache hashes, {len(loaded[:sparse_items])} items (the slowest stage)")
    report["E3"], report["E4"] = run_e3_e4(bundle, loaded[:sparse_items], floor, progress=stage)
    stage("E5: placebo")
    report["E5"] = run_e5(bundle, loaded[:sparse_items], floor)
    stage(f"E7: determinism, {len(loaded[:repeat_items])} items")
    report["E7"] = run_e7(bundle, loaded[:repeat_items])
    stage("done")
    report["all_pass"] = all(report[name]["pass"] for name in ("E1", "E2", "E3", "E4", "E5", "E7"))
    return report
