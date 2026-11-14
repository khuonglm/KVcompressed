---
name: kv-cache-change
description: Use before editing src/kvsel/cache.py, engine.py, selectors.py, model.py or prompts.py, or anything that touches KV entries, attention masks, positions, probes, replay, scoring or tokenization. Lists the invariants these modules must keep and the tests that prove them.
---

# Changing the replay engine

These modules decide whether the experiment measures what it claims. A silent bug here invalidates
every result, so every change must keep the invariants below and prove it with the tests.

## Layout facts (LLaVA-1.5-7B, template v3.0)

- Prefix: 581 tokens = 5 non-visual (`<s>`, `▁US`, `ER`, `:`, `▁`) + 576 image tokens. Identical for
  Q and every QH_c.
- A compact cache at K keeps all 5 non-visual entries plus K visual entries (K = 144 gives 149).
- The last 16 suffix tokens are the fixed instruction; this is PW-attn's window. Question-span
  attention uses the question tokens instead.
- Candidates are scored as `prompt + " " + candidate`, and the prompt tokens must be an exact prefix.
- `TEMPLATE_VERSION` in `prompts.py` must be bumped for any change to the prompt text.

## Invariants

1. **Only `cache.py` touches the transformers cache API.** Entries are tuples of `(key, value)` per
   layer, shaped (batch, heads, seq, head_dim).
2. **Logical positions.** Rotary embeddings are applied before caching, so gathered entries keep
   their original positions; replay must use logical positions (past length = 581), never the
   compacted physical length.
3. **Compact equals dense.** Compact replay on the kept set must match dense replay with a -inf mask
   on dropped columns (below 1e-5 in fp32 on CPU; exactly equal at K = 576 on GPU).
4. **The shared prefix is never mutated.** Probes and replays build new tensors; `entries_hash` before
   and after must match (E4). No in-place ops on prefix tensors.
5. **Kept indices are sorted** and preserve sequence order in every layer.
6. **Numerics**: backbone, cache and attention fp16 on the real model; log-probabilities and their
   sums always fp32; output head fp32 by default (`--head-precision`).
7. `transformers==4.57.6` and `torch==2.6.0` are pinned exactly because cache internals change
   between versions. Do not bump them without rerunning Level 1 and Level 2.

## Tests to run

- `uv run pytest tests/test_level1.py` — compact vs dense on random, concentrated and scored kept
  sets at K = 288, 144, 72, plus the negative controls: physical positions must **fail**, perturbing
  dropped entries must give exactly 0, perturbing kept entries must differ, hashes must not change.
  If a negative control stops failing, the test has lost its power: treat that as a bug.
- `uv run pytest tests/test_selectors.py tests/test_prompts.py tests/test_pipeline.py`.
- Then the full suite, and hand `kvsel validate` to the user (`gpu-handoff`) for any change that
  could affect the real model's numbers.

When adding a new mechanism, add a negative control that shows the test would catch the bug it targets.
