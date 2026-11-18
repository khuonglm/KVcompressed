# Spec: LLaVA-NeXT-7B as a second model (GQA, pre-registered)

Status: draft for review. Date: 15 November 2026.
- **Validation gate:** 24 November 2026. If the model doesn't pass validation by then, it is dropped and reported as a limitation.
- **Study deadline:** 3 December 2026, 23:59 KST.

## Problem Statement

Every result so far comes from one model, LLaVA-1.5-7B. A reviewer's first question is whether text-steered KV selection is a quirk of that model's fixed 576-token image layout. That layout is a single 336-pixel tile, with no high-resolution tiles. Modern VLMs encode images at higher resolution with a variable number of visual tokens, and the effect may shrink, grow or vanish there.

The researcher needs a second, architecturally different model, tested under the same pre-registered discipline. The run has to fit on the 20 GB A100 and in the remaining time.

It must also not disturb two things:
- the frozen GQA study;
- the GQA replication, which is running now and whose freeze record hashes all code under `src/kvsel/`.

## Solution

Run LLaVA-NeXT-7B (Vicuna-7B base) on GQA under a new PREREG section 4, with its own three-stage freeze.

**Why this model.** It shares LLaVA-1.5's language model family, tokenizer and chat template, so prompts and scoring carry over unchanged. What differs is the vision side. It uses "AnyRes": a 24×24 base tile plus a high-resolution grid of tiles, with a newline token at the end of each high-resolution row. A GQA image becomes about 1,700–2,900 visual tokens instead of 576.

**Budgets.** Budgets are shares of each image's visual tokens: 12.5%, 25% and 50%. These correspond to K = 72, 144 and 288 of 576 in LLaVA-1.5.

**Workflow.** The same steps as before:
1. Validate.
2. Freeze the NeXT screening rules (stage 1).
3. Screen items that already passed hand verification (flags only).
4. Freeze the comparative protocol (stage 3).
5. Run once.
6. Report.

**Code location.** All NeXT code lives in a new top-level package next to `kvsel`, importing the frozen engine. No byte under `src/kvsel/` changes, so both GQA freeze records and the replication record stay valid.

## User Stories

1. As the researcher, I want a second model that differs from LLaVA-1.5 in visual encoding but not in language model or template, so that a difference in results points at the visual side, not at prompts.
2. As the researcher, I want NeXT's code outside `src/kvsel/`, so that the running replication's code hash and both GQA records stay valid with no code amendment.
3. As the researcher, I want a `fetch-model` step for NeXT with a pinned revision, so that the server loads exactly the weights the freeze names.
4. As the researcher, I want a CPU test suite on a tiny NeXT-shaped model (tiny Llama, tiny vision tower, AnyRes on), so that every code path runs before the GPU does.
5. As the researcher, I want the Level-1 invariants proven for NeXT's layout, so that compaction is known to be correct on a variable-length, non-square visual span. The invariants:
   - compact replay equals dense replay with masked dropped columns;
   - logical positions are kept, and physical positions fail;
   - perturbing dropped entries changes nothing;
   - the shared prefix is never mutated.
6. As the researcher, I want a GPU validation command for NeXT, the same E-checks as `kvsel validate` adapted to it, so that I know by 24 November whether NeXT goes ahead. It covers:
   - serialization and tokenization;
   - compact at 100% equals dense, exactly;
   - the noise floor;
   - the memory and time per item on the A100.
7. As the researcher, I want the prompt to be byte-identical to LLaVA-1.5's apart from the image token expansion, so that the description label, question and instruction are the same.
8. As the researcher, I want the visual span located from the processor's own expansion of the image token, and checked, so that the code never assumes 576 tokens.
9. As the researcher, I want each image's layout recorded per item:
   - base tile size;
   - high-resolution grid shape;
   - number of newline tokens;
   - total visual tokens;

   so that budgets and pooling are reproducible.
10. As the researcher, I want budgets computed as round(share × selectable tokens) per image, so that 12.5/25/50% mean the same thing on every image.
11. As the researcher, I want newline tokens always kept and not counted in the budget, like the non-visual prefix entries, so that selection is over image content only and the layout separators stay intact.
12. As the researcher, I want PW-attn's 3×3 pooling applied within each spatial map (the base tile and the high-resolution map), never across a map boundary or a newline, so that it means the same as on LLaVA-1.5's single 24×24 grid.
13. As the researcher, I want NeXT's own eligibility screen over items that already passed hand verification, so that no item is verified again.
14. As the researcher, I want the screen to apply the same rules as section 1.4 under NeXT:
    - Full_Q correct with a positive margin under the capitalized rule;
    - the blind filter.

    I want it to expose only flags and counts, so that stage 1 stays blind to comparative outcomes.
15. As the researcher, I want NeXT's main set to be the first 1,000 NeXT-eligible passed items in manifest order, from both the original main set and the replication pool, so that the set is large and needs no new verification.
16. As the researcher, I want the first 300 of that set as the control subset, so that the matching condition runs as in the original study.
17. As the researcher, I want the NeXT units on every item:
    - at 12.5%, 25% and 50%: cell A, and cells B, C and D for mis and irr;
    - Full_Q, Full_mis and Full_irr;
    - Blind_Q.

    On control items, I also want B, C and D for match at each share, and Full_match. This mirrors the replication, so the two models' tables line up.
18. As the researcher, I want one probe per context per item reused for all three shares, so that the run fits overnight.
19. As the researcher, I want the run to be resumable, atomic per item and progress-only, as before.
20. As the researcher, I want a NeXT stage-1 record and a NeXT stage-3 record, each refusing to run after any change, so that NeXT has the same three-stage discipline as LLaVA-1.5.
21. As the researcher, I want pre-registered NeXT hypotheses at 25% as their own Holm family, so that NeXT's result stands on its own and doesn't change the LLaVA-1.5 families:
    - N1: S_H^(mis) ≠ 0
    - N2: S_H^(irr) ≠ 0
    - N3: E_harm^(mis) ≠ 0
    - N4: T_irr ≠ 0
22. As the researcher, I want 12.5% and 50%, S_Q, I, behaviour, overlap and the T_match/match terms reported as intervals only.
23. As the researcher, I want the NeXT report to print LLaVA-1.5's matching estimates beside NeXT's, at 25% versus K = 144 and so on, labelled and never pooled, so that the paper can show both models side by side.
24. As the researcher, I want a README runbook and an overnight script for NeXT like the replication's, so that I can run it unattended.
25. As the researcher, I want a clear go/no-go output from validation, so that the 24 November decision is mechanical.
26. As a reader, I want NeXT's PREREG section dated and frozen before its comparative run, with the reuse of items that LLaVA-1.5 has already seen disclosed, so that I can judge the evidence.

## Implementation Decisions

- **Model.** LLaVA-NeXT (v1.6) Vicuna-7B in its transformers port, at a pinned revision. The transformers 4.57.6 and torch 2.6.0 pins stay. fp16 backbone and cache, fp32 head and log-probabilities, as for LLaVA-1.5. Eager attention, checked at load.
- **Package.** A new top-level package next to `kvsel` (for example `kvsel_next`), added to the wheel build. It imports `kvsel`'s engine (decoder, probe, replay, scoring), selectors' `top_k`, the analysis helpers, the verification loader and the replication's selection and freeze patterns. It never edits them.
  - Its freeze records hash its own code plus all of `src/kvsel/`.
  - `kvsel`'s own hashes don't cover it, so adding it changes no existing record.
- **Prefix layout.** Prefix = the non-visual tokens before the image + the expanded visual span (base tile, then high-resolution rows with newline tokens), contiguous.
  - The engine's compaction already takes an index set within the visual span, so it is reused.
  - The kept set is always the selected feature tokens plus every newline token.
  - Logical positions are kept as in LLaVA-1.5.
- **Layout record.** From the processor's image sizes and the model's AnyRes grid, the code derives a per-item layout:
  - base map 24×24;
  - high-resolution map h×w (after the model's unpadding);
  - newline positions;
  - a mapping from visual-span index to (map, row, column).

  The derived layout is checked against the actual expanded span length.
- **Selection.** PW-attn scores = mean attention from the last 16 suffix tokens, over heads and layers 2+. That is the same probe as LLaVA-1.5's, reused unchanged. The scores are then 3×3-pooled within each map, and the top-k is taken over feature tokens, with k = round(share × feature tokens). Ties go to the lower index, as `top_k` does.
- **Shares.** 0.125, 0.25 and 0.50. Rows carry the share and the per-item k.
- **Not ported, because each depends on LLaVA-1.5's single grid:** the uniform grid, the coverage reserve, question-span attention, H5/H6 evidence retention, LT-early, random sets and restoration.
- **Item pool.** `manifests/gqa_confirm.jsonl` items whose latest current-digest decision is "pass", in manifest order, flagged NeXT-eligible by NeXT's screen.
  - The main set is the first 1,000 such items; the first 300 form the control subset.
  - Items already used by LLaVA-1.5 runs are allowed: the hypotheses are about NeXT, and the reuse is disclosed.
- **Screen.** NeXT Full_Q and Blind_Q on the passed items, in manifest order, until 1,000 are eligible or the passed items run out. It writes flags only.
- **PREREG section 4.** It holds:
  - 4.1 NeXT screening rules, frozen at NeXT stage 1;
  - 4.2 configuration;
  - 4.3 the family N1–N4 at 25% (Holm over 4, α = 0.05), plus exploratory rows;
  - 4.4 records.

  Equivalence isn't tested for NeXT unless a projection from validation-time development items supports it; that is decided in the draft, before freezing.
- **Commands.** `python -m kvsel_next` with `fetch-model`, `validate`, `freeze-screening`, `screen`, `count`, `freeze`, `run` and `report`. They mirror the replication's command shapes and refusals.
- **Memory and time** (to be confirmed by `validate`): weights about 14 GB, and the prefix cache at about 2,900 tokens is under 1 GB. Prefill with eager attention is the peak. The estimate is about 1.5–2× LLaVA-1.5's per-item time, about 6–8 h for the comparative run and about 1.5 h for the screen.

## Testing Decisions

- **What a good test is.** It drives the public commands and functions on a tiny NeXT-shaped model or on synthetic files, and checks observable results, never internal helpers.
- **Seams.** Four proposed seams; please confirm.
  1. **Layout and selection.** Synthetic layouts of several aspect ratios. Checks:
     - pooled scores never mix maps or newline tokens;
     - k = round(share × features);
     - newline tokens are always kept;
     - kept sets are nested across shares.
  2. **Engine invariants on NeXT (Level 1).** On the tiny NeXT model:
     - compact equals dense with a -inf mask;
     - physical positions fail;
     - perturbing dropped entries has no effect;
     - perturbing kept entries changes the output;
     - the prefix hash doesn't change.
  3. **Screen, freeze and run.** End to end on the tiny model: flags only from the screen; freeze refusals; resume; the exact unit list for main and control items.
  4. **Report.** Planted effects. Checks: Holm over N1–N4; side-by-side with LLaVA-1.5 values; refusal of missing or extra files and of changed code.
- **Prior art.** `tests/test_level1.py`, which has the negative controls. Also `tests/test_replication.py` and `tests/test_confirm.py`.
- **Unchanged guards.** Tests assert that `kvsel`'s stage-3 hash, the replication's tree hash and the stage-1 screening hash are unaffected by the new package.
- **GPU.** `validate` is the go/no-go gate and must be handed to the researcher by about 20 November.

## Out of Scope

- TextVQA on NeXT; NeXT's confirmatory result is GQA only.
- Qwen and any third model.
- Selectors and analyses that depend on LLaVA-1.5's single grid (listed above).
- Published-method baselines beyond framing (VisionZip if time allows, as its own spec).
- Any change to `src/kvsel/`, the frozen GQA study, or the replication.

## Further Notes

- **Order.** Don't merge NeXT code that touches `src/kvsel/` while the replication is unreported. With the separate package, nothing needs to wait.
- **Risk: memory.** The largest risk is memory: eager attention over about 3,000 prefix tokens on 32 layers during prefill. `validate` measures the peak. If it exceeds 20 GB, the fallback is to cap AnyRes at the smallest grid that fits. That cap is decided and written into section 4.1 before stage 1, never after seeing outcomes.
- **Risk: eligible count.** If NeXT's eligibility is low (it may answer a_true less often under the capitalized rule), 1,000 eligible items may need more passed items than exist. `count` reports this before the freeze. The fallback is a smaller main set, decided before the freeze and disclosed.
- **Open choices for the researcher:**
  - Vicuna versus Mistral base (Vicuna proposed, for template identity);
  - newline tokens always kept (proposed) versus selectable;
  - the N1–N4 family at 25%;
  - whether to attempt equivalence.
