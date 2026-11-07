"""Eager attention with scores, softmax and the weighted sum of values in fp32.

The library's eager attention rounds the QK^T scores to the model dtype before the softmax and the
attention weights to it before multiplying by V. In bf16 that is 8 bits of mantissa: one step is 1.0
at scores above 128. Compact and dense replay differ only inside attention, so this rounding is what
separates them. Flash attention, which Qwen2.5-VL was trained with, keeps scores in fp32 too.

Queries, keys and values arrive in the model dtype and the output returns in it; the cache is not
changed. The attention weights returned for probes are fp32.
"""

from __future__ import annotations

import torch
from torch import nn
from transformers import AttentionInterface
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, eager_mask

FP32_ATTENTION = "kvsel_eager_fp32"


def eager_attention_fp32(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    groups = module.num_key_value_groups
    key = key.repeat_interleave(groups, dim=1).float() if groups > 1 else key.float()
    value = value.repeat_interleave(groups, dim=1).float() if groups > 1 else value.float()
    scores = torch.matmul(query.float(), key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask[:, :, :, : key.shape[-2]].float()
    weights = nn.functional.softmax(scores, dim=-1, dtype=torch.float32)
    output = torch.matmul(weights, value).to(query.dtype).transpose(1, 2).contiguous()
    return output, weights


AttentionInterface.register(FP32_ATTENTION, eager_attention_fp32)
ALL_MASK_ATTENTION_FUNCTIONS.register(FP32_ATTENTION, eager_mask)  # masks are built as for eager


def set_text_attention(model: nn.Module, precision: str) -> None:
    """"fp32" switches the language model to the attention above; "native" leaves the library's eager."""
    if precision not in {"fp32", "native"}:
        raise ValueError(f"unknown attention precision {precision!r}")
    text_config = model.config.text_config
    if text_config._attn_implementation not in {"eager", FP32_ATTENTION}:
        raise RuntimeError("the language model must use eager attention")
    text_config._attn_implementation = FP32_ATTENTION if precision == "fp32" else "eager"
