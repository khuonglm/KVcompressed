"""Probe, replay, candidate scoring and greedy decoding against shared prefix entries.

Compact and dense-masked execution share every line below ``replay``; they differ only in which
entries sit in the cache and which columns the mask drops, so the dense reference tests this code.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .cache import Entries, additive_mask, cache_from_entries, compact, dropped_columns, entries_from_cache

EOS_ID = 2


@dataclass(frozen=True)
class State:
    """Decoder state after processing some tokens. Holds only references to tensors."""

    entries: Entries
    logical_next: int
    physical_next: int
    dropped: torch.Tensor | None
    last_logits: torch.Tensor  # fp32, shape (vocab,)


class Decoder:
    """Language model plus output head, with explicit positions and masks on every call.

    ``head_precision``: "fp32" projects the fp32-cast hidden state through an fp32 copy of the
    head. "native" uses the model's head and casts its logits afterwards. Log-probabilities are
    computed in fp32 either way. Neither repairs rounding already present in fp16 hidden states.
    """

    def __init__(self, language_model: nn.Module, lm_head: nn.Module, head_precision: str = "fp32"):
        if head_precision not in {"fp32", "native"}:
            raise ValueError(f"unknown head_precision {head_precision!r}")
        self.language_model = language_model
        self.lm_head = lm_head
        self.head_precision = head_precision
        self.fp32_head = copy.deepcopy(lm_head).float() if head_precision == "fp32" else None
        self.dtype = next(language_model.parameters()).dtype
        self.device = next(language_model.parameters()).device

    @torch.inference_mode()
    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.fp32_head is not None:
            return self.fp32_head(hidden.float())
        return self.lm_head(hidden).float()

    @torch.inference_mode()
    def step(
        self,
        ids: torch.Tensor,
        entries: Entries,
        logical_start: int,
        physical_start: int,
        dropped: torch.Tensor | None = None,
        output_attentions: bool = False,
    ):
        """One forward pass. ``ids`` has shape (1, T). Returns (hidden, new entries, attentions)."""
        ids = ids.to(self.device)
        new_tokens = ids.shape[1]
        output = self.language_model(
            input_ids=ids,
            past_key_values=cache_from_entries(entries),
            use_cache=True,
            position_ids=torch.arange(logical_start, logical_start + new_tokens, device=self.device)[None],
            cache_position=torch.arange(physical_start, physical_start + new_tokens, device=self.device),
            attention_mask=additive_mask(new_tokens, physical_start, dropped, self.dtype, self.device),
            output_attentions=output_attentions,
        )
        return output.last_hidden_state, entries_from_cache(output.past_key_values), output.attentions


@dataclass(frozen=True)
class Prefix:
    entries: Entries
    num_nonvisual: int
    num_visual: int

    @property
    def length(self) -> int:
        return self.num_nonvisual + self.num_visual


def probe(decoder: Decoder, prefix: Prefix, suffix_ids: torch.Tensor) -> np.ndarray:
    """Attention from each suffix token to each visual entry, averaged over heads and layers 2 up.

    Returns float32 of shape (suffix_tokens, num_visual). Probabilities are over all available
    keys and are not renormalised. The probe's cache is discarded on return.
    """
    _, _, attentions = decoder.step(suffix_ids, prefix.entries, prefix.length, prefix.length, output_attentions=True)
    start, stop = prefix.num_nonvisual, prefix.length
    total = None
    for attention in attentions[2:]:
        layer = attention[0, :, :, start:stop].float().mean(dim=0)
        total = layer if total is None else total + layer
    return (total / (len(attentions) - 2)).float().cpu().numpy()


def replay(
    decoder: Decoder,
    prefix: Prefix,
    kept_visual: torch.Tensor,
    suffix_ids: torch.Tensor,
    mode: str = "compact",
) -> tuple[State, torch.Tensor]:
    """Process the answering suffix against the kept entries. Returns (state, suffix logits)."""
    kept_visual = kept_visual.to(decoder.device)
    if mode == "compact":
        entries = compact(prefix.entries, kept_visual, prefix.num_nonvisual)
        dropped = None
    elif mode == "dense":
        entries = prefix.entries
        dropped = dropped_columns(kept_visual, prefix.num_nonvisual, prefix.num_visual)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    physical = entries[0][0].shape[2]
    hidden, new_entries, _ = decoder.step(suffix_ids, entries, prefix.length, physical, dropped)
    logits = decoder.logits(hidden)
    tokens = suffix_ids.shape[1]
    state = State(new_entries, prefix.length + tokens, physical + tokens, dropped, logits[0, -1])
    return state, logits


def advance(decoder: Decoder, state: State, token_ids: list[int]) -> tuple[State, torch.Tensor]:
    """Process more tokens from a state, leaving the state itself reusable."""
    ids = torch.tensor([token_ids], device=decoder.device)
    hidden, entries, _ = decoder.step(ids, state.entries, state.logical_next, state.physical_next, state.dropped)
    logits = decoder.logits(hidden)
    n = len(token_ids)
    return State(entries, state.logical_next + n, state.physical_next + n, state.dropped, logits[0, -1]), logits


def score_sequence(decoder: Decoder, state: State, token_ids: list[int]) -> list[float]:
    """Log-probability of each token in sequence, computed in fp32."""
    log_probs = [torch.log_softmax(state.last_logits, dim=-1)[token_ids[0]]]
    if len(token_ids) > 1:
        _, logits = advance(decoder, state, token_ids[:-1])
        step_log_probs = torch.log_softmax(logits[0], dim=-1)
        log_probs.extend(step_log_probs[i, token] for i, token in enumerate(token_ids[1:]))
    return [float(value) for value in log_probs]


@dataclass(frozen=True)
class Greedy:
    token_ids: list[int]
    top_two_gaps: list[float]
    truncated: bool
    step_logits: list[torch.Tensor]


def greedy(decoder: Decoder, state: State, max_new: int = 16, stop_at_eos: bool = True) -> Greedy:
    """Greedy decode. The returned ids exclude EOS; ``truncated`` means no EOS within ``max_new``."""
    tokens: list[int] = []
    gaps: list[float] = []
    step_logits: list[torch.Tensor] = []
    finished = False
    for _ in range(max_new):
        step_logits.append(state.last_logits)
        top = torch.topk(state.last_logits, 2)
        gaps.append(float(top.values[0] - top.values[1]))
        token = int(top.indices[0])
        if token == EOS_ID and stop_at_eos:
            finished = True
            break
        tokens.append(token)
        state, _ = advance(decoder, state, [token])
    return Greedy(tokens, gaps, not finished, step_logits)
