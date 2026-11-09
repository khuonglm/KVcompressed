# Pre-registration: does unreliable text steer visual-KV selection in LLaVA-1.5-7B?

The study follows `documents/Idea 1 proposal v3.md`. Its changes are recorded in `documents/Idea 1 implementation plan.md`, section 3. This file freezes the protocol in three stages (plan, decision 4):

1. **Stage 1 (screening rules):** section 1 below. Frozen with `kvsel freeze-screening`, which stores the sha256 of section 1, the confirmatory manifests and the screening code in `freeze/screening.json`. After stage 1, `kvsel screen-confirm` runs the eligibility screen. It exposes only item ids, eligible flags and funnel counts.
2. **Stage 2 (verification):** hand verification of eligible items under the rules in section 1.6.
3. **Stage 3 (comparative experiment):** section 2 below. It may still change until stage 3, because no comparative outcome on confirmatory items exists before then. Stage 3 freezes this whole file, the verification decisions, the screen output and the code. The comparative run then runs once.

Comparative outcomes (any cell other than the screen's Full_Q eligibility flag) are not inspected or used to modify the confirmatory protocol. Bugs found later are fixed outcome-blind: the fix is dated and documented, and every affected condition is rerun in full. Any data added or setting changed after stage 3 is labelled a follow-up.

## 1. Screening rules

### 1.1 Model and software

- Model: `llava-hf/llava-1.5-7b-hf` (LLaVA-1.5-7B). The second model (Qwen2.5-VL-7B) is parked and outside this study.
- Software: `torch==2.6.0` and `transformers==4.57.6`, pinned in `pyproject.toml` and `uv.lock`.
- Numerics: backbone, KV cache and attention in fp16. The output head is an fp32 copy applied to the fp32-cast hidden state. Log-probabilities and their sums are in fp32.
- Batch size 1 and deterministic algorithms (`CUBLAS_WORKSPACE_CONFIG=:4096:8`).

### 1.2 Image and prompt

- Image: padded to a square with the processor's mean colour, then resized to 336 × 336. This gives 576 visual tokens on a 24 × 24 grid.
- Template **v3.1**. The Q prompt, used by the screen (`<image>×576` stands for 576 image tokens):

  ```
  USER: <image>×576\nAnswer from the image. Any provided description may be inaccurate.\n{question}\nAnswer the question using a single word or phrase. ASSISTANT:
  ```

- The QH_c prompt adds the line `Unverified description, possibly wrong: {description}\n` before the authority line. It isn't used in the screen.

### 1.3 Scoring

- A candidate is scored as `prompt + " " + candidate`, followed by `</s>`. The prompt's tokens must be an exact prefix of that sequence. The candidate's score is the sum of fp32 log-probabilities over its content tokens and `</s>`.
- Margin M = score(a_true) − score(a_alt).
- Surface rule, the same on both datasets and applied to a_true and a_alt alike: **capitalized** (first character upper case, the rest lower case). On the v3.1 development pilots (200 items per dataset, every cell), greedy answers were capitalized 100% of the time on GQA and 92.5% on TextVQA. The margin's sign agreed between the lower and capitalized forms on 95.9% (GQA) and 98.0% (TextVQA) of rows. The log-sum over both forms is reported as a sensitivity margin.
- Greedy decoding: at most 16 new tokens, stopping at `</s>`. A truncated answer counts as "other".
- Behavioural normalisation: lower case; characters other than letters, digits, underscore, whitespace and hyphen become spaces; the articles a, an and the are removed; "grey" is mapped to "gray". The outcome is "correct" if the normalised greedy answer equals the normalised a_true, "alt" if it equals the normalised a_alt, and "other" otherwise.

### 1.4 Eligibility

An item is eligible if, under Full_Q (full cache, Q prompt, no description), its greedy outcome is "correct" **and** M > 0 under the capitalized surface rule.

### 1.5 Confirmatory pools

| Dataset | Manifest | Builder | Items | Images | sha256 |
| --- | --- | --- | ---: | ---: | --- |
| GQA | `manifests/gqa_confirm.jsonl` | `gqa-3` | 9,209 | 4,429 | `ded6b8295a653afa55b35f75b8f8b012671d66fcf53f2c940b270250ddb7539c` |
| TextVQA | `manifests/textvqa_confirm.jsonl` | `textvqa-3` | 5,514 | 4,905 | `16fcf6fe0eae0ea2ab15cfe9b37e44d2d3161b8bfc30fc5d54d61da9bcf6d0d7` |

- No confirmatory image appears in development.
- Manifest order is the builder's seeded image order (seed 0). Every rule below follows manifest order.
- The screen may cover a prefix of a manifest (`--limit`), extended later, if images arrive in stages. The prefix never changes which items qualify, only how far down the list the screen has reached.

### 1.6 Hand verification and the main set

- **What is verified:** eligible items, in manifest order, with `kvsel verify`. The tool shows the image, question, a_true, a_alt, descriptions and evidence box, and no model output.
- **Decisions:** "pass" or one failure reason:
  - the misleading statement isn't false
  - a_true is wrong
  - the referent isn't unique
  - a_alt is an acceptable answer
  - the wording is bad
  - the evidence box is wrong

  Decisions are append-only, and the latest decision for an item counts. Each decision is bound to a hash of the item.
- **Failures:** failed items are removed and logged, never repaired.
- **Decisions made before the screen still count.** Verifying is blind to outcomes, so it doesn't matter whether it happened before or after the screen. This covers the GQA items verified on 9 November.
- **Main set (per dataset):** the first 1,000 items in manifest order that are eligible and whose latest decision is "pass".
- **Control subset:** the first 300 items of the main set.
- **Grounding subset:** control-subset items with a non-empty evidence box.
- **Stopping:** verification stops once 1,000 eligible items have passed. If the pool runs out first, the main set is all eligible items that passed, and the shortfall is reported.
- **No audit:** every item in the main set is hand-verified, so the proposal's audit of 100 items doesn't apply.

### 1.7 Amendments

A screening rule can't change after stage 1 because of anything seen on confirmatory items. If one must change, the amendment is dated and documented here, and a fresh held-out partition is used.

## 2. Comparative experiment (frozen at stage 3)

Section 2 covers both datasets. **GQA runs first.** TextVQA is pre-registered here and runs later under this same protocol, with nothing changed because of the GQA results. Each dataset gets its own stage-3 record (`freeze/comparative_gqa.json`, `freeze/comparative_textvqa.json`), made with `kvsel freeze-comparative` before that dataset's run. GQA's record freezes this whole file, so TextVQA's protocol is fixed before any GQA comparative outcome exists.

### 2.1 Configuration

- **Primary selector:** PW-attn. Its window is the last 16 suffix tokens (the fixed instruction), and it averages attention to the 576 visual entries over heads and layers 2 and up.
- **Budget:** K = 144 visual entries, plus the 5 non-visual prefix entries.
- **Replay:** on the compact cache with logical positions.
- **Secondary selector:** question-span attention, the same formula with the window set to the question tokens. It is descriptive only.
- **Reference:** a uniform grid at K = 144, keeping the patches whose row and column are both even.
- **Cells, for each condition c:**
  - A: S_Q selection, Q answering
  - B_c: S_c selection, QH_c answering
  - C_c: S_Q selection, QH_c answering
  - D_c: S_c selection, Q answering
  - Full anchors: Full_Q and Full_c
- **Units on every main-set item:**
  - cells A, B_mis, C_mis and D_mis under both selectors
  - Full_Q and Full_mis
  - the uniform grid under Q and under QH_mis
  - PW-attn with the coverage reserve: (S_Q, Q), (S_mis, QH_mis) and (S_Q, QH_mis)
  - **Blind_Q** (section 2.2)
- **Control subset:** additionally, cells B, C and D, Full_c and the uniform grid for c = match and irr, plus the two reserve variants for match. `kvsel.confirm.comparative_settings` lists every unit exactly.
- **Run once.** Each item's file is written only when that item is complete, so a crash can resume. The run prints only progress, never margins or answers.
- **Consistency check:** Full_Q must still be eligible on every item. The count of items where it isn't is reported.

### 2.2 Blind baseline (added 10 November 2026, before any comparative outcome)

- **Why:** some items may be answerable from language alone ("what colour is the sky?"). On those items the kept visual entries matter little, which would dilute S_H toward 0.
- **Definition:** Blind_Q is Full_Q with a blank image: a single 336 × 336 image filled with the padding colour (the CLIP mean). An item is **image-independent** if Blind_Q's margin is above 0 under the capitalized surface rule, and **image-dependent** otherwise.
- **Population unchanged:** this rule doesn't change the main set or the primary analysis. It defines only the secondary analyses in section 2.3.
- **Why this rule:** it was fixed on scientific grounds. Only the share of image-independent items on development data may be inspected (`kvsel blind-rate`), never S_H split by blind status.

### 2.3 Estimands and tests

- **Estimands:**
  - S_H^(c) = M(C_c) − M(B_c)
  - T_match = S_H^(mis) − S_H^(match) and T_irr = S_H^(mis) − S_H^(irr)
  - S_Q^(c) = M(A) − M(D_c)
  - I^(c) = S_H^(c) − S_Q^(c)
  - E_harm^(c) = (M(Full_c) − M(B_c)) − (M(Full_Q) − M(A))
- **Margin:** M uses the capitalized surface rule.
- **Inference:** image-clustered paired bootstrap, 10,000 resamples, seed 0, with a 95% percentile interval.
- **Two-sided p-value:** p = min(1, 2 · (1 + min(#{θ\* ≤ 0}, #{θ\* ≥ 0})) / 10,001). Every confirmatory test is two-sided.

| Family | Tests | Set | Multiplicity |
| --- | --- | --- | --- |
| H1 (primary) | S_H^(mis) for PW-attn, GQA and TextVQA | Main | Holm over 2, α = 0.05 |
| H2 | T_match and T_irr per dataset | Control | Holm over 4 |
| H3, H4 | E_harm^(mis) and S_Q^(mis) per dataset | Main | Holm over 4 |
| Image dependence | S_H^(mis) on image-dependent items; S_H^(mis) of dependent minus independent items; share of image-independent items | Main | Intervals and two-sided p, no Holm claim |
| Exploratory | I^(mis), S_H^(match), S_H^(irr), question-span selector | Main / control | Intervals only |
| H5 | Per-item S_H^(mis) regressed on (1 − Jaccard(S_Q, S_mis)) | Main | OLS, image-clustered SE |
| H6 | ERR(S_Q) − ERR(S_mis) under all four mapping rules | Grounding | Intervals |
| Mitigation (EQ1) | Reserve and isolation variants against B_mis, on misleading and neutral answering | Main / control | Intervals |
| Behaviour | Paired differences in correct and a_alt rates, B_mis vs C_mis | Main | Intervals |

- **Reading the results:**
  - Each dataset gets its own conclusion. An effect is called general only if both datasets pass.
  - GQA's estimates, intervals and unadjusted p-values are reported when its run ends. The Holm decisions are final once TextVQA's results exist.
  - If TextVQA is never run, the H1 family becomes GQA alone, and this is disclosed as a deviation.
- **Sensitivity:** the margin is also reported as the log-sum over the lower and capitalized surface forms.
- **Equivalence:** δ = 0.10 nats, tested only if the development-projected 90% half-width is below δ/2.

### 2.4 Left out of the confirmatory run

These are not part of the confirmatory study. Any of them run later is labelled a follow-up:
- K = 288 and 72
- LT-early
- random-selection seeds
- oracle restoration
- the less-filtered population (decision 6)
