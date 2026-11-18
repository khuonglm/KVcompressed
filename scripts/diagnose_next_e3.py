"""Diagnostic only (19 November 2026): where NeXT's E3 logit p99 comes from. Changes no rule or verdict.

`kvsel validate --model llava_next` failed E3's logit criterion (p99 0.0380, limit 0.0321). This reruns E3 exactly as
`validation.sparse_equivalence` does (same items, order, kept sets and random draws), checks that the pooled p99 and
margin mean equal the validate report's, and splits them by kept-set kind, share, context and source (the suffix
pass or a greedy step).

    uv run python scripts/diagnose_next_e3.py --items manifests/gqa_dev.jsonl --images data/gqa/images \
        --out results/next/diagnose_e3.json
"""

from __future__ import annotations

import argparse
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
from kvsel.validation import MIS, _label, _prefilled, _replayed, sample_differences, validation_set

REPORTED = {"logit_p99": 0.03800487518310547, "margin_mean_abs": 0.004719372947364617}  # the NO-GO run


def p99(chunks: list[np.ndarray]) -> float:
    return float(np.percentile(np.concatenate(chunks), 99))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--sparse-items", type=int, default=20)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    bundle = load_bundle("cuda", None, args.model_dir, NEXT)
    items = load_items(args.items)[: args.sparse_items]
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
                for kind in VALIDATION.kinds:
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
                    for logit_c, logit_d in zip(greedy_c.step_logits, greedy_d.step_logits):
                        sample = sample_differences(logit_c - logit_d, generator)
                        pooled.append(sample)
                        for key in (*keys, "source=greedy step"):
                            groups[key]["samples"].append(sample)
    found = {
        "pooled": {"logit_p99": p99(pooled), "margin_mean_abs": float(np.mean(margins))},
        "reported": REPORTED,
        "groups": {key: {"logit_p99": p99(g["samples"]),
                         "margin_mean_abs": float(np.mean(g["margins"])) if g["margins"] else None,
                         "comparisons": len(g["margins"])}
                   for key, g in sorted(groups.items())},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(found, indent=2))
    same = all(abs(found["pooled"][k] - v) < 1e-12 for k, v in REPORTED.items())
    print(f"pooled p99 {found['pooled']['logit_p99']:.4f}, margin mean {found['pooled']['margin_mean_abs']:.4f} "
          f"({'same as' if same else 'DIFFERENT from'} the validate report)")
    for key, g in found["groups"].items():
        if not key.startswith("tokens="):
            print(f"{key:40s} p99 {g['logit_p99']:.4f}  margin {g['margin_mean_abs'] or 0:.4f}")
    print(f"full output: {args.out}")


if __name__ == "__main__":
    main()
