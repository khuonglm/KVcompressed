# Pre-registration: does unreliable text steer visual-KV selection in LLaVA-1.5-7B?

The study follows `documents/Idea 1 proposal v3.md`. Its changes are recorded in `documents/Idea 1 implementation plan.md`, section 3. This file freezes the protocol in three stages (plan, decision 4):

1. **Stage 1 (screening rules):** section 1 below. Frozen with `kvsel freeze-screening`, which stores the sha256 of section 1, the confirmatory manifests and the screening code in `freeze/screening.json`, superseded by amendment 1's `freeze/screening_amend1.json`. After stage 1, `kvsel screen-confirm` runs the eligibility screen. It exposes only item ids, eligible flags and funnel counts.
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

An item is eligible if both hold:

1. Under Full_Q (full cache, Q prompt, no description), its greedy outcome is "correct" **and** M > 0 under the capitalized surface rule.
2. **Amendment 1:** the same rule does **not** pass on a blank image. Blind_Q is Full_Q with the real image replaced by one 336 × 336 image filled with the padding colour (the CLIP mean). An item whose Blind_Q answer is correct with M > 0 can be answered without the image, so it is not eligible.

The screen runs Blind_Q only on items that pass condition 1. It writes four flags per item (`full_q_eligible`, `blind_eligible`, `eligible`, and the item id) and no answers or margins.

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

### 1.8 Amendment 1 (10 November 2026): items answerable without the image

- **Change:** condition 2 of section 1.4 (the blind filter) was added.
- **Reason:** while hand-verifying confirmatory GQA items, the verifier noticed questions whose answer follows from language alone ("what colour is the sky?"). Such items test the language prior, not the use of visual entries.
- **What had been seen:** the stage-1 screen's eligible flags for GQA, and the item contents shown by the verification tool. No answers, margins or comparative outcomes had been produced or seen. The trigger was item content, not model output.
- **Deviation from 1.7:** no fresh held-out partition is used, because GQA has no unused validation images left. The amended screen reruns on the same pool. The flags already seen carry no information about any estimand in section 2. This deviation is disclosed in the report.
- **Records:**
  - The original stage-1 record `freeze/screening.json` is kept.
  - `freeze/screening_amend1.json` supersedes it and names it with its sha256.
  - The amended screen writes to a new folder; the first screen's output is not used.
- **Verification decisions carry over.** The main-set rule (1.6) is unchanged and is applied to the amended eligibility flags.
- **Before this screen ran,** the share of development items the rule removes was measured with `kvsel blind-rate` (rates only). On 150 verified GQA development items, all eligible, the rule removes 20%: Blind_Q's greedy answer is correct with M > 0. Blind_Q's margin alone is above 0 on 76%, so the blank-image prior often prefers a_true over a_alt without producing a_true. The rule was fixed before these rates were seen and was not changed because of them.

### 1.9 Amendment 2 (15 November 2026): code clean-up, no change to any rule

- **Change:** code only. Support for the parked Qwen2.5-VL model, its fp32 attention module (`attention.py`), the prompt-variant machinery used once for the 9 November prompt test, and one-off development commands were removed. The screening-code hash now also covers `geometry.py` (image padding and the blank image's colour), closing a gap recorded on 10 November. No screening rule, prompt text, scoring rule or setting changed.
- **Reason:** to simplify the code before the replication and the second model. No confirmatory outcome motivated it.
- **Evidence that nothing a model sees or a rule decides changed:**
  - For every item of the four manifests (GQA and TextVQA, development and confirmatory), the prompt strings, prefix and suffix tokens, question and description spans and the scored candidate tokens are identical before and after, and so are the item digests that bind verification decisions.
  - On the tiny test model, the confirmatory run's rows for every unit and the screen's flags are byte-identical before and after.
  - The GQA confirmatory analysis and the follow-up report are rerun under the new code and must reproduce the frozen results exactly. That check and its outcome are recorded in `freeze/code_amendments.json`, which links the stage-3 record's code hash to the new one.
- **Records:** `freeze/screening_amend2.json` supersedes `freeze/screening_amend1.json` and names it with its sha256. Screens already run, eligibility flags and verification decisions carry over unchanged.

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

### 2.2 Blind baseline

Items answerable without the image are excluded at screening (section 1.4, amendment 1). The run computes Blind_Q again on every main item as a consistency check: the count of items where it now passes the eligibility rule is reported, and it should be 0. The mean Blind_Q margin is reported descriptively.

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
| Exploratory | I^(mis), S_H^(match), S_H^(irr), question-span selector | Main / control | Intervals only |
| H5 | Per-item S_H^(mis) regressed on (1 − Jaccard(S_Q, S_mis)) | Main | OLS, image-clustered SE |
| H6 | ERR(S_Q) − ERR(S_mis) under all four mapping rules | Grounding | Intervals |
| Mitigation (EQ1) | Reserve and isolation variants against B_mis, on misleading and neutral answering | Main / control | Intervals |
| Behaviour | Paired differences in correct and a_alt rates, B_mis vs C_mis | Main | Intervals |

- **Reading the results:**
  - Each dataset gets its own conclusion. An effect is called general only if both datasets pass.
  - GQA's estimates, intervals and unadjusted p-values are reported when its run ends. The Holm decisions are final once TextVQA's results exist.
  - If TextVQA is never run, the H1 family becomes GQA alone, and this is disclosed as a deviation.
- **Sensitivity:** S_H^(mis) is also reported with three other margin forms:
  - the log-sum over the distinct lower, capitalized and upper forms;
  - content tokens only, without `</s>`;
  - length-normalised: the mean log-probability per token, with `</s>` counted.
- **Analysis code:** `kvsel confirm-report` (`src/kvsel/confirm_analysis.py`). It is frozen with the rest of the code by the stage-3 record. It was checked on simulated runs with planted effects and rehearsed on the development pilots before any confirmatory run.
- **Definitions:**
  - **T and the exploratory S_H^(match), S_H^(irr):** computed on control-subset items only.
  - **Behaviour:** per-item differences in the indicator of a correct (or a_alt) greedy answer, C_mis − B_mis, with the same clustered bootstrap.
  - **H5:**
    - Model: OLS of per-item S_H^(mis) on 1 − Jaccard(S_Q, S_mis), with an intercept. S_Q and S_mis are the PW-attn kept sets at K = 144, without the reserve.
    - Standard errors: CR1 image-clustered.
    - Inference: normal-approximation 95% interval and two-sided p.
  - **H6:**
    - Population: grounding-subset items, which are control items with a non-empty evidence box.
    - Measure: ERR(S) = |S ∩ E| / |E|, where E is the evidence patches under each of the four mapping rules. Items whose E is empty under a rule are left out for that rule.
    - Inference: the same bootstrap.
  - **Mitigation (EQ1):** margin differences against PW-attn B_mis:
    - isolation C_mis
    - reserve B_mis + r
    - both C_mis + r
    - the uniform grid with QH_mis

    The costs are A + r − A on neutral answering, and B_match + r − B_match and C_match − B_match on matching answering (control).
  - **Checks reported:** Full_Q no longer eligible; Blind_Q eligible (should be 0); the mean Blind_Q margin; the share of truncated answers.
  - **Reading:** the interval readings (positive, inconclusive, practically small, opposite direction) apply to margins in nats only.
- **Equivalence:** δ = 0.10 nats, tested only if the development-projected 90% half-width is below δ/2 = 0.05. The test is a 90% clustered interval inside ±δ.
  - *Projection* (10 November, v3.1 development pilots, PW-attn, verified items, capitalized). The clustered SE of S_H^(mis) was scaled by √(development images / expected confirmatory images). Expected images = 1,000 ÷ the pool's items per image.
    - GQA: SE 0.042 on 137 images; 2.08 items per image, so about 480 images; projected 90% half-width about 0.037.
    - TextVQA: SE 0.017 on 176 images; 1.12 items per image, so about 890 images; projected half-width about 0.012.
  - Both are below 0.05, so equivalence is tested on both datasets.
  - The development items predate amendment 1, so these are planning numbers only.

### 2.4 Left out of the confirmatory run

These are not part of the confirmatory study. Any of them run later is labelled a follow-up:
- K = 288 and 72
- LT-early
- random-selection seeds
- oracle restoration
- the less-filtered population (decision 6)

## 3. GQA replication on fresh items

Section 2's GQA run and its follow-ups (2.4) suggested that steering grows as the budget shrinks, that E_harm^(mis) changes sign at K = 72, and that irrelevant text steers selection too. Those observations came from the same 1,000 items after their outcomes were seen, so they are exploratory. This section tests them once on fresh items, under rules fixed before any outcome on those items exists. The hypotheses below were chosen *because* of the earlier results; that is the point of a replication, and the report says so.

Everything not stated here is as in sections 1 and 2: model, prompt, surface rule (`capitalized`), margin, PW-attn's definition, replay, bootstrap (image-clustered, 10,000 resamples, seed 0, 95% percentile interval), the two-sided p-value formula, and the interval readings.

### 3.1 Items

- **Pool:** `manifests/gqa_confirm.jsonl`, with the amendment-1 screen flags and the hand-verification decisions of section 1.6 (latest decision per item, bound by item digest; a stale decision counts as unverified).
- **Rule:** in manifest order, starting right after the last item of the frozen main set (2.1), take eligible items, then:
  1. leave out items whose image is the image of any main-set item;
  2. leave out items with a result file in any earlier run folder (the confirmatory run, the follow-ups, and the folder holding the first run's extra items);
  3. stop at the first eligible item that is not verified; pass over failed items;
  4. the first 1,000 that passed form the **replication set**; its first 300 form the **control subset**.
- No item is verified again, and no item is added, dropped or repaired after the freeze.
- The excluded ids and the reason for each are stored in the freeze record (3.5). `python -m kvsel.replication count` applies this rule and prints counts only.

### 3.2 Configuration

- **Selector:** PW-attn only. The question-span selector, coverage reserve, uniform grid, LT-early, random sets and restoration are not run.
- **Budgets:** K = 72, 144 and 288 visual entries, plus the 5 non-visual prefix entries. One probe per context and item gives the kept sets for all three budgets (top-K of the same scores).
- **Units on every replication item:**
  - for each K: cell A, and cells B_c, C_c, D_c for c = mis and irr;
  - Full_Q, Full_mis and Full_irr;
  - Blind_Q, as in 2.2.
- **Control subset, additionally:** for each K, cells B_match, C_match, D_match; and Full_match.
- **Run once,** resumable, printing progress only, with the same consistency checks as 2.1 and 2.2 (Full_Q still eligible; Blind_Q eligible should be 0).

### 3.3 Confirmatory tests

Estimands are those of 2.3, computed separately at each K. One change: **T_irr = S_H^(mis) − S_H^(irr) is computed on all replication items**, not on the control subset, because irr now runs on every item.

| Hypothesis | Test | K | Set |
| --- | --- | --- | --- |
| R1 | S_H^(mis) ≠ 0 | 72; 144 | all items |
| R2 | S_H^(irr) ≠ 0 | 72; 144 | all items |
| R3 | E_harm^(mis) ≠ 0 | 72 | all items |
| R5 | T_irr ≠ 0 | 72 | all items |

- **Multiplicity:** R1, R2, R3 and R5 are one Holm family of **6 tests** at α = 0.05. The replication is GQA-only by design, so the Holm decisions are final when the report runs.
- **R4, equivalence:** S_H^(mis) at K = 288 is called practically null if its 90% image-clustered interval lies inside ±δ, δ = 0.10 nats (as in 2.3). R4 is reported outside the Holm family.
- **Projection for R4** (planning number only): the original follow-up gave S_H^(mis) = +0.050 [+0.032, +0.070] at K = 288 on 708 images, a clustered SE of about 0.010. With a similar number of images, the 90% half-width is about 0.016, below δ/2 = 0.05, so R4 is tested.
- **Reading:** a test supports its hypothesis if Holm rejects; the sign is reported, and a rejection in the direction opposite to the original estimate is reported as a failed replication of that direction.

### 3.4 Exploratory (intervals only)

- S_Q^(c), I^(c) and E_harm^(irr) at every K; S_H^(mis), S_H^(irr) and E_harm^(mis) at the budgets not tested above.
- T_match and S_H^(match) on the control subset, at every K.
- Behaviour rates and the paired C − B differences in correct and a_alt rates, per K and condition.
- Jaccard overlap of each condition's kept set with S_Q, per K.
- The original estimates (section 2 and its follow-ups) printed beside the replication's, labelled as the original exploratory values. The two sets are never pooled.

### 3.5 Records and code

- **Freeze:** `python -m kvsel.replication freeze` writes `freeze/replication_gqa.json`. It stores the sha256 of this file, of the latest stage-1 record (amendment 2), of the GQA stage-3 record (section 2), of the manifest and of the screen flags, the hash of the replication set with its verification digests, the earlier run folders given to it, the excluded ids with reasons, the settings (units, budgets, Blind_Q, surface rule, the hypotheses, α and δ), and a hash of all code under `src/kvsel/`, subfolders included. It refuses while this section is marked DRAFT and refuses to overwrite a different record.
- **Run and report** recompute this state and refuse if anything in the record changed. Code changes after the freeze follow 1.9: dated, outcome-blind, shown to reproduce, and listed in `freeze/code_amendments.json`.
- **Order:** no comparative unit runs on any replication item before the freeze record is committed.
- The replication code is a subpackage outside `src/kvsel/*.py`, so the GQA stage-3 record's code hash (section 2) is unaffected.
