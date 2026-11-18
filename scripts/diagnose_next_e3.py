"""Diagnostic only (19 November 2026): where NeXT's E3 logit p99 comes from. Changes no rule or verdict.

`kvsel validate --model llava_next` failed E3's logit criterion (p99 0.0380, limit 0.0321). This reruns E3 exactly as
`validation.sparse_equivalence` does (same items, order, kept sets and random draws), checks that the pooled p99 and
margin mean equal the validate report's, and splits them by kept-set kind, share, context and source (the suffix
pass or a greedy step).

With `--published-tokenizer --kinds random,concentrated --floor-items 50` it repeats the first validate's setup
(16 November 2026: NeXT's tokenizer as published, without the space token before the newline; random and
concentrated sets only) on the reorganized code. Reproducing that run's Q floor (p99 0.0216) and E3 (p99 0.0352)
exactly means the space token is the only cause of the change; anything else points at the code.

    uv run python scripts/diagnose_next_e3.py --items manifests/gqa_dev.jsonl --images data/gqa/images \
        --out results/next/diagnose_e3.json [--published-tokenizer] [--kinds random,concentrated] [--floor-items 50]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from kvsel.analysis.statistics import SURFACE
from kvsel.engine import probe
from kvsel.items.manifest import load_items
from kvsel.model import load_bundle
from kvsel.models.llava_next import NEXT
from kvsel.runner import Q, open_image
from kvsel.studies.next.comparative import VALIDATION
from kvsel.validation import MIS, _label, _prefilled, _replayed, noise_floor, sample_differences, validation_set

REPORTED = {"logit_p99": 0.03800487518310547, "margin_mean_abs": 0.004719372947364617}  # the NO-GO run
FIRST_RUN = {"floor_logit_p99": 0.021567828953266144, "floor_logit_max": 0.44490909576416016,
             "logit_p99": 0.03519105911254883, "margin_mean_abs": 0.0051291250977840265}  # 16 November, GO


def published(bundle, model_dir):
    """The bundle with NeXT's tokenizer as published (no legacy option); the slow image processor is kept."""
    import transformers

    path = NEXT.resolve(model_dir, weights=False)
    processor = transformers.LlavaNextProcessor.from_pretrained(path, revision=NEXT.revision,
                                                                image_processor=bundle.image_processor.image_processor)
    return dataclasses.replace(bundle, tokenizer=processor.tokenizer, image_processor=processor)


def p99(chunks: list[np.ndarray]) -> float:
    return float(np.percentile(np.concatenate(chunks), 99))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--sparse-items", type=int, default=20)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--published-tokenizer", action="store_true", help="NeXT's tokenizer without legacy=True")
    parser.add_argument("--kinds", default=",".join(VALIDATION.kinds), help="E3's kept-set kinds, comma-separated")
    parser.add_argument("--floor-items", type=int, default=0, help="also the Q noise floor on this many items")
    args = parser.parse_args()
    kinds = tuple(args.kinds.split(","))

    bundle = load_bundle("cuda", None, args.model_dir, NEXT)
    if args.published_tokenizer:
        bundle = published(bundle, args.model_dir)
    all_items = load_items(args.items)
    floor = None
    if args.floor_items:
        print(f"noise floor, Q suffix, {args.floor_items} items", flush=True)
        loaded = [(item, open_image(args.images, item)) for item in all_items[: args.floor_items]]
        floor = dataclasses.asdict(noise_floor(bundle, loaded, Q, SURFACE))
        del loaded
    items = all_items[: args.sparse_items]
    generator = np.random.default_rng(1)  # as sparse_equivalence
    pooled, margins = [], []
    groups: dict[str, dict[str, list]] = defaultdict(lambda: {"samples": [], "margins": []})
    for position, item in enumerate(items):
        image = open_image(args.images, item)
        tokens, _, prefix, layout = _prefilled(bundle, item, image)
        print(f"{position + 1}/{len(items)} {item['item_id']}: {layout.total} visual tokens", flush=True)
        for name in (Q, MIS):
            t, suffix = tokens[name], torch.tensor([tokens[name].suffix_ids])
            attention = probe(bundle.decoder, prefix, suffix)
            for budget in VALIDATION.budgets:
                for kind in kinds:
                    kept = torch.from_numpy(validation_set(layout, budget, kind, position, attention))
                    logits_c, margin_c, greedy_c = _replayed(bundle, prefix, kept, suffix, "compact", t.prompt, item,
                                                             SURFACE)
                    logits_d, margin_d, greedy_d = _replayed(bundle, prefix, kept, suffix, "dense", t.prompt, item,
                                                             SURFACE)
                    keys = (f"kind={kind}", _label(budget), f"context={name}", f"tokens={layout.total}",
                            f"kind={kind} {_label(budget)}")
                    sample = sample_differences(logits_c - logits_d, generator)
                    pooled.append(sample)
                    for key in (*keys, "source=suffix"):
                        groups[key]["samples"].append(sample)
                    margin = abs(margin_c - margin_d)
                    margins.append(margin)
                    for key in keys:
                        groups[key]["margins"].append(margin)
                    for step, (logit_c, logit_d) in enumerate(zip(greedy_c.step_logits, greedy_d.step_logits)):
                        sample = sample_differences(logit_c - logit_d, generator)
                        pooled.append(sample)
                        for key in (*keys, "source=greedy step"):
                            groups[key]["samples"].append(sample)
                        if greedy_c.token_ids[step] != greedy_d.token_ids[step]:
                            break  # as sparse_equivalence: past a mismatch the two decode different text
    found = {
        "pooled": {"logit_p99": p99(pooled), "margin_mean_abs": float(np.mean(margins))},
        "settings": {"published_tokenizer": args.published_tokenizer, "kinds": list(kinds)},
        "floor": floor,
        "reported": FIRST_RUN if args.published_tokenizer else REPORTED,
        "groups": {key: {"logit_p99": p99(g["samples"]),
                         "margin_mean_abs": float(np.mean(g["margins"])) if g["margins"] else None,
                         "comparisons": len(g["margins"])}
                   for key, g in sorted(groups.items())},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(found, indent=2))
    reference = found["reported"]
    same = all(abs(found["pooled"][k] - reference[k]) < 1e-12 for k in ("logit_p99", "margin_mean_abs"))
    if floor:
        print(f"floor p99 {floor['logit_p99']:.4f}, max {floor['logit_max']:.4f}; E3 limit {2 * floor['logit_p99']:.4f}"
              + (f" (first run: p99 {FIRST_RUN['floor_logit_p99']:.4f}, max {FIRST_RUN['floor_logit_max']:.4f})"
                 if args.published_tokenizer else ""))
    print(f"pooled p99 {found['pooled']['logit_p99']:.4f}, margin mean {found['pooled']['margin_mean_abs']:.4f} "
          f"({'same as' if same else 'DIFFERENT from'} the {'first' if args.published_tokenizer else 'NO-GO'} validate report)")
    for key, g in found["groups"].items():
        if not key.startswith("tokens="):
            print(f"{key:40s} p99 {g['logit_p99']:.4f}  margin {g['margin_mean_abs'] or 0:.4f}")
    print(f"full output: {args.out}")


if __name__ == "__main__":
    main()
