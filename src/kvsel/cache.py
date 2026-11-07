"""Prefix KV entries, compaction and masking.

The only module that touches the transformers cache API. Entries are plain tuples of
``(key, value)`` tensors shaped (batch, heads, seq, head_dim), one tuple per layer. Rotary
embeddings are applied before keys are cached, so gathered entries keep their original positions.
"""

from __future__ import annotations

import hashlib

import torch
from transformers import DynamicCache

Entries = tuple[tuple[torch.Tensor, torch.Tensor], ...]


def entries_from_cache(cache: DynamicCache) -> Entries:
    return tuple((layer.keys, layer.values) for layer in cache.layers)


def cache_from_entries(entries: Entries) -> DynamicCache:
    """Wrap entries without copying. Forward passes concatenate into new tensors."""
    cache = DynamicCache()
    for layer_idx, (key, value) in enumerate(entries):
        cache.update(key, value, layer_idx)
    return cache


def compact(prefix: Entries, kept_visual: torch.Tensor, num_nonvisual: int) -> Entries:
    """Keep every non-visual entry and the kept visual entries, in sequence order, in every layer."""
    kept_visual = kept_visual.to(prefix[0][0].device).sort().values
    idx = torch.cat([torch.arange(num_nonvisual, device=kept_visual.device), num_nonvisual + kept_visual])
    return tuple((k.index_select(2, idx), v.index_select(2, idx)) for k, v in prefix)


def dropped_columns(kept_visual: torch.Tensor, num_nonvisual: int, num_visual: int) -> torch.Tensor:
    """Physical prefix columns of the visual entries that are not kept."""
    mask = torch.ones(num_visual, dtype=torch.bool, device=kept_visual.device)
    mask[kept_visual] = False
    return num_nonvisual + mask.nonzero().squeeze(1)


def additive_mask(
    new_tokens: int,
    past_len: int,
    dropped: torch.Tensor | None,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Shape (1, 1, T, past + T): causal among new tokens, -inf on dropped prefix columns."""
    inf = float("-inf")
    mask = torch.zeros(1, 1, new_tokens, past_len + new_tokens, dtype=dtype, device=device)
    mask[0, 0, :, past_len:] = torch.triu(
        torch.full((new_tokens, new_tokens), inf, dtype=dtype, device=device), diagonal=1
    )
    if dropped is not None and dropped.numel():
        mask[0, 0, :, dropped] = inf
    return mask


def entries_hash(entries: Entries) -> str:
    """SHA-256 over all tensors. For test E4 only; never enabled during timing."""
    digest = hashlib.sha256()
    for key, value in entries:
        for tensor in (key, value):
            digest.update(tensor.detach().cpu().float().numpy().tobytes())
    return digest.hexdigest()


def entries_bytes(entries: Entries) -> int:
    return sum(k.numel() * k.element_size() + v.numel() * v.element_size() for k, v in entries)
