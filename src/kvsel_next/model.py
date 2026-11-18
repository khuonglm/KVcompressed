"""NeXT's per-unit runner: kvsel's row, scoring before greedy decoding so the GPU holds one cache copy fewer."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import torch

from kvsel.engine import Decoder, Prefix, State, greedy, replay
from kvsel.model import Bundle
from kvsel.normalize import outcome
from kvsel.runner import Unit, cell_name, score_candidates


def greedy_releasing(decoder: Decoder, box: list[State], **options):
    """`kvsel.engine.greedy` on the one state in ``box``, emptied first: greedy's frame then holds the only
    reference, so each decoding step frees the state before it. The caller must keep no other reference."""
    return greedy(decoder, box.pop(), **options)


def run_unit(bundle: Bundle, item: dict, unit: Unit, prefix: Prefix, kept: np.ndarray, tokens: dict) -> dict:
    """`kvsel.runner.run_unit`'s row. Candidates are scored before greedy decoding (neither changes the state) and
    greedy frees the replay state as it steps, where kvsel's keeps it alive: with NeXT's CPU-held prefix
    (`kvsel.engine.replay`), the GPU holds at most two cache copies."""
    answering = tokens[unit.answering]
    suffix = torch.tensor([answering.suffix_ids])
    box = [replay(bundle.decoder, prefix, torch.from_numpy(kept), suffix, "compact")[0]]
    scores = score_candidates(bundle, box[0], answering.prompt, item["a_true"], item["a_alt"])
    decoded = greedy_releasing(bundle.decoder, box)
    text = bundle.tokenizer.decode(decoded.token_ids, skip_special_tokens=True).strip()
    return {
        **asdict(unit),
        "cell": cell_name(unit),
        "kept": kept.tolist() if unit.selector != "full" else None,
        "scores": scores,
        "greedy_ids": decoded.token_ids,
        "greedy_text": text,
        "outcome": "other" if decoded.truncated else outcome(text, item["a_true"], item["a_alt"]),
        "truncated": decoded.truncated,
        "top_two_gaps": decoded.top_two_gaps,
    }
