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
- **Run** recomputes this state and refuses if anything in the record changed. **Report** does the same, except that this file may have grown by dated amendments written after the run (git keeps its history). Code changes after the freeze follow 1.9: dated, shown to reproduce, and listed in `freeze/code_amendments.json`.
- **Order:** no comparative unit runs on any replication item before the freeze record is committed.
- The replication code is a subpackage outside `src/kvsel/*.py`, so the GQA stage-3 record's code hash (section 2) is unaffected.

### 3.6 Code amendment 3 (15 November 2026): review fixes and clean-up, no rule changed

- **Change:** code only, in the stage-3 and replication hashes; no stage-1 (screening) module changed. Replication: the DRAFT check matches only a line starting with "DRAFT"; R4 uses the same "inside ±δ" rule as the interval readings (`<=`, as 2.3's equivalence now does too); the 3.3 sign rule is applied against the original estimates; the report rechecks the whole freeze, not only the code; the record stores the earlier run folders and the full settings. Clean-up: the consistency checks of 2.1 and 3.2 are one function; `validate` takes its surface rule as an argument; the model-file lookup has no LLaVA defaults; the 576 constant is written once. The analysis is otherwise the frozen code.
- **Timing, stated plainly:** these findings came from a code review finished before the replication run ended, but they were applied after its report (the original code's) had been seen. None changes an estimate, interval, p-value or Holm decision.
- **Evidence:** `scripts/verify_amendment3.py` reruns the confirmatory, follow-up and replication reports with the amended code: the first two `results.json` files are byte-identical to the stored ones, and the replication's is identical on every key the frozen code produced. The entries in `freeze/code_amendments.json` carry that evidence once the script has run.

## 4. Second model: LLaVA-NeXT-7B on GQA

This section tests whether the steering found on LLaVA-1.5-7B also appears in a model that encodes images differently: LLaVA-NeXT (v1.6) with AnyRes, a 24 × 24 base tile plus a high-resolution grid of tiles, about 1,700–2,900 visual entries per GQA image instead of 576. It has its own three-stage freeze (4.4). Sections 1 to 3 are unchanged by it, and its results never enter their families.

Everything not stated here is as in sections 1 to 3: the margin, the capitalized surface rule, scoring, greedy decoding and normalisation (1.3), PW-attn's probe, replay on the compact cache with logical positions, the bootstrap (image-clustered, 10,000 resamples, seed 0, 95% percentile interval), the two-sided p-value formula and the interval readings (2.3).

**Items already seen.** NeXT's items come from the passed GQA confirmatory items, which include the original main set (section 2) and the replication set (section 3). LLaVA-1.5's comparative outcomes on these items have been seen. The hypotheses below are about NeXT, on which no comparative outcome exists, but they were chosen because of LLaVA-1.5's results on many of the same items. The report says so.

### 4.1 Screening rules (frozen at NeXT stage 1)

- **Model:** `llava-hf/llava-v1.6-vicuna-7b-hf` (LLaVA-NeXT, Vicuna-7B base, chosen on 15 November for template identity), at revision `c916e6cdcd760b4cecd1dd4907f84ac649f93b23` (the one `validate` loaded, 16 November 2026), loaded with eager attention.
- **Software and numerics:** as 1.1: `torch==2.6.0`, `transformers==4.57.6`; fp16 backbone, KV cache and attention; fp32 output head on the fp32-cast hidden state; fp32 log-probabilities and sums; batch size 1; deterministic algorithms.
- **Image:** the NeXT processor's own AnyRes handling: a 24 × 24 base tile, then the high-resolution feature map after the model's unpadding, with a newline entry at the end of each high-resolution row.
- **AnyRes cap:** none, set from `validate`'s peak-memory numbers on development items (16 November 2026): the largest grid AnyRes produces, 48 × 48 (2,928 visual entries), peaked at 16.8 GiB on the 20 GiB GPU slice. Set before stage 1 and never changed after any NeXT outcome on confirmatory items.
- **Prompt:** template v3.1 (1.2), byte-identical apart from the image token's expansion into the visual span. The visual span is located from the processor's own expansion, never assumed.
- **Eligibility:** as 1.4, under NeXT. An item is eligible if (1) under Full_Q its greedy outcome is "correct" and M > 0 under the capitalized rule, and (2) the same rule does not pass under Blind_Q. Blind_Q replaces the real image with one of the same width and height filled with the padding colour of 1.4 (the CLIP mean), so the blank gets the same AnyRes layout as the real image. Blind_Q runs only on items that pass condition 1.
- **Pool:** items of `manifests/gqa_confirm.jsonl` whose latest decision, bound to the item's current digest, is "pass" (1.6), in manifest order. No item is verified again; a stale decision counts as unverified.
- **Screen:** in pool order, until 1,000 items are eligible or the pool runs out. It writes the item id and the flags (`full_q_eligible`, `blind_eligible`, `eligible`) only, with no answers or margins, and prints progress and funnel counts only.
- **Main set:** the first 1,000 eligible items in manifest order. **Control subset:** its first 300. If fewer than 1,000 are eligible once the whole pool is screened (4.5(d), settled 16 November 2026): as 1.6, the main set is every eligible item and the shortfall is reported; the control subset stays the first 300; `count` reports the size before the stage-3 freeze, which then freezes that size.

### 4.2 Configuration

- **Shares:** 12.5%, 25% and 50% of each image's feature entries, the counterparts of K = 72, 144 and 288 of 576. Feature entries are the visual span minus the newline entries. Per item and share, k = share × feature entries, which is always a whole number: unpadding trims only one axis of the high-resolution map, so the feature count is a multiple of 24 and no rounding is needed. The code refuses a share that would not give a whole k. Rows carry the share, k and the item's layout.
- **Always kept, outside the budget:** the non-visual prefix entries (as before) and every newline entry.
- **Selector:** PW-attn only, with the probe of 2.1 unchanged (mean attention from the last 16 suffix tokens, over heads and layers 2 and up). Scores are 3 × 3-pooled within each spatial map (the base tile and the high-resolution map); at a map's edge the pool averages that map's in-bounds neighbours only, and it never crosses a map boundary or a newline entry. The top k feature entries are kept, ties to the lower index. One probe per context and item gives the kept sets for all three shares, so they are nested.
- **Units on every main-set item:**
  - for each share: cell A, and cells B_c, C_c, D_c for c = mis and irr;
  - Full_Q, Full_mis and Full_irr;
  - Blind_Q, with the blank of 4.1, as a consistency check (2.2).
- **Control subset, additionally:** for each share, cells B_match, C_match, D_match; and Full_match.
- **Run once,** resumable, each item's file written atomically when complete, printing progress and the two consistency counts only (Full_Q no longer eligible; Blind_Q now eligible, which should be 0).
- **Not ported:**
  - the uniform grid, the coverage reserve and H6's evidence retention, because each is defined on LLaVA-1.5's single 24 × 24 grid (even rows and columns, coverage cells, evidence boxes mapped to that grid) and has no agreed counterpart on a variable AnyRes layout;
  - H5 (steering against kept-set overlap), because the replication did not carry it either; overlap is reported as an interval only;
  - the question-span selector, LT-early, random sets and restoration, which the replication dropped too, so the two models' tables line up and the run fits overnight.

### 4.3 Hypotheses and tests

Estimands are those of 2.3, computed separately at each share. As in 3.3, T_irr = S_H^(mis) − S_H^(irr) is computed on all main-set items, because irr runs on every item.

| Hypothesis | Test | Share | Set |
| --- | --- | --- | --- |
| N1 | S_H^(mis) ≠ 0 | 25% | main |
| N2 | S_H^(irr) ≠ 0 | 25% | main |
| N3 | E_harm^(mis) ≠ 0 | 25% | main |
| N4 | T_irr ≠ 0 | 25% | main |

- **Multiplicity:** N1–N4 are one Holm family of 4 tests at α = 0.05, separate from H1–H6 and R1–R5. NeXT is GQA-only by design, so the Holm decisions are final when the report runs.
- **Reading:** a test supports its hypothesis if Holm rejects; the sign is reported. Whether NeXT's sign agrees with LLaVA-1.5's is reported descriptively, not tested.
- **Equivalence:** not tested (see 4.5(c)).
- **Intervals only:** N1–N4's estimands at 12.5% and 50%; S_Q^(c), I^(c) and E_harm^(irr) at every share; T_match and S_H^(match) on the control subset at every share; behaviour rates and the paired C − B differences in correct and a_alt rates; Jaccard overlap of each condition's kept set with S_Q.
- **Beside LLaVA-1.5:** the report prints the replication's estimates (section 3) beside NeXT's, matched as 12.5% to K = 72, 25% to K = 144 and 50% to K = 288, labelled and never pooled. The item sets differ and the absolute budgets differ (25% of NeXT's feature entries is several hundred entries, not 144), so the comparison is descriptive.
- **Checks reported:** Full_Q no longer eligible; Blind_Q eligible (should be 0); the share of truncated answers; the range of visual-entry counts and layouts.

### 4.4 Records

- **Stage 1:** `python -m kvsel_next freeze-screening` writes `freeze/next_screening.json`. It stores the sha256 of section 4.1 (from its `### 4.1` heading line up to the `### 4.2` heading, which is not included), of the manifest, of all code under `src/kvsel_next/` and `src/kvsel/`, and the screening settings (model id and revision, AnyRes cap, template, surface rule, eligibility rule, Blind_Q image rule). It refuses while this section still has its draft line and refuses to overwrite a different record. `screen` recomputes this state and refuses, before loading the model, if anything changed.
- **Stage 2:** none. The pool is items already verified under 1.6.
- **Stage 3:** `python -m kvsel_next freeze` writes `freeze/next_gqa.json`. It stores the sha256 of this file, of the stage-1 record, of the manifest and of the screen flags, the hash of the main set with its verification digests, the settings (shares, the exact unit lists, the hypotheses and α), and the code hash of both packages. It refuses while this section still has its draft line, if fewer items qualify than the frozen size, or if a different record exists.
- **Run** recomputes the stage-3 state and refuses if anything changed. **Report** does the same, except that this file may have grown by dated amendments written after the run.
- **Order:** section 4.1 is not edited after stage 1. No comparative unit runs on NeXT before the stage-3 record and this file are committed. Code changes after a NeXT freeze follow 1.9: dated, shown to reproduce, and listed in `freeze/code_amendments.json`.
- `kvsel`'s own records do not hash `src/kvsel_next/`, so adding it changes no earlier record.

### 4.5 Open choices (for the researcher; settled before stage 1)

Each is written with the proposal. Once settled, the decision moves into 4.1–4.3 and this subsection records what was chosen.

- **(a) Newline entries.** Settled 16 November 2026: the proposal, now in 4.2. Proposed: always kept and outside the budget, like the non-visual prefix entries, so selection is over image content only and the row separators stay intact. Alternative: selectable like feature entries, which makes k count them and lets PW-attn drop them.
- **(b) The confirmatory family.** Settled 16 November 2026: the proposal, now in 4.3: the budget is a share of each image's feature entries, so k differs between images and is not 144. Proposed: N1–N4 at 25%, the counterpart of LLaVA-1.5's primary K = 144. Alternatives: at 12.5%, the counterpart of K = 72, where LLaVA-1.5's replication tested R1–R3 and R5; or the replication's shape (N1 and N2 at two shares, six tests).
- **(c) Equivalence.** Settled 16 November 2026: the proposal, now in 4.3. Any equivalence analysis added after NeXT's results are seen is exploratory and labelled so in the report. Proposed: not tested. No NeXT development estimate of S_H^(mis) exists to project the interval's width, as 2.3 and 3.3 did, and getting one would need comparative runs on development items before the freeze. Alternative: run the 25% cells on the 150 verified GQA development items during validation, and test equivalence at 50% (δ = 0.10, 90% interval) only if the projected half-width is below 0.05.
- **(d) Fewer than 1,000 eligible.** Settled 16 November 2026: the proposal, now in 4.1. The pool holds 2,335 passed items (16 November 2026). Proposed: as 1.6, the main set is every eligible item and the shortfall is reported; the control subset stays the first 300; `count` reports the size before the stage-3 freeze, which then freezes that size. Alternative: a floor (for example 500) below which NeXT is dropped and reported as a limitation.
- **(e) The Blind_Q image.** Settled 16 November 2026: the proposal, in 4.1. The existing blank is 336 × 336, which under AnyRes gets a different grid, and so a different number of visual entries, from the real image. Proposed (and written into 4.1): a blank of the padding colour at the real image's width and height, so Blind_Q differs from Full_Q only in pixel content. Alternative: keep the 336 × 336 blank, as in section 1.4.

### 4.6 Amendment (19 November 2026): corrections before NeXT is frozen again

Two code reviews (of the NeXT code and of the reorganization) found the problems below. They change NeXT's rules and outputs only. No LLaVA-1.5 output changes (the characterization suite's LLaVA-1.5 golden outputs are unchanged), and no NeXT comparative unit had run on confirmatory items.

- **Prompt tokens (4.1).** NeXT's tokenizer, as published, drops the space token (id 29871) that LLaVA-1.5's tokenizer puts between the image token and the newline, so NeXT's prompt was one token shorter than template v3.1 tokenized by LLaVA-1.5. 4.1's "byte-identical apart from the image token's expansion" held for the text, not the tokens. Now the tokenizer loads with `legacy=True`. With that, the ids outside the image span equal LLaVA-1.5's for Q and every condition. The option is recorded in the stage-1 settings (`tokenizer_options`). Validation's E1 compares against LLaVA-1.5's own tokenizer, not NeXT's.
- **Image processor (4.1).** The slow `LlavaNextImageProcessor`, the one the model was saved with, is loaded by name and recorded in the stage-1 settings (`image_processor`). Before, it was whatever transformers' default resolved to, and that default is moving to the fast processor, whose outputs differ slightly.
- **Full rows (4.2).** A Full row's k is the item's feature count, not its visual-span length with newlines included. Every row's layout record also carries the newline count.
- **Overlap (4.3).** Jaccard overlap with S_Q is computed on feature positions, with newline entries removed. Every kept set holds every newline, which pushed the overlap towards 1.
- **Validation (tests/README.md, "LLaVA-NeXT-7B, Level 2").**
  - E3 adds a "tail" kept set: the last k feature tokens plus every newline. At 12.5% and 25% the concentrated set lies wholly in the base map, so the tail set makes every share cover the high-resolution map.
  - E4 hashes both the prefix as the family holds it (CPU memory) and a copy on the GPU, which probes and replays then read in place.
  - Both noise floors, under the fp32 and the native output head, are reported whichever head the run uses (in place since 18 November, ticket 15; listed for the same reason as E7).
  - E7 reruns every unit of the control list, the probe-scored ones included (in place since 18 November, ticket 09; listed here because the first stage-1 freeze predates it).
- **Report records (4.3).** `results.json` records the LLaVA-1.5 comparison file's path and sha256, and the number of bootstrap resamples. The report refuses a comparison file that lacks K = 72, 144 or 288. Before, the report showed "—" for each missing budget.
- **Records (4.4).** The first NeXT records, `freeze/next_screening.json` (16 November 2026, 07:19 UTC) and `freeze/next_gqa.json` (08:34 UTC), are moved, not deleted, to `freeze/superseded/next-first-freeze/`. Their screen folder `results/next/screen`, any run folder `results/next/gqa_run` and the validation output `results/validate_llava_next.json` move to `results/next/superseded/first-freeze/`. No command reads them, and no number produced under them is reported. NeXT then repeats validation, stage 1, the screen, the count, stage 3, the run and the report under fresh records. 4.1's text is unchanged. The new stage-1 record pins the two options above in its settings.
- **Timing, stated plainly:** written after the superseded records were frozen. The researcher states in the report how far the first screen got and whether any first run started. A first screen writes eligibility flags only (4.1). The changes were motivated by the code reviews, not by any NeXT outcome.

## 5. Code amendment 4 (18 November 2026): reorganization, no rule changed

- **Change:** code only. `src/kvsel` and `src/kvsel_next` become one package, organized by what the code does rather than by study: items, verification, model families with an explicit visual layout, the engine, selection, units and runs, one freeze mechanism (`kvsel.protocol`), shared analysis blocks, validation for both models, one package per study under `kvsel.studies`, and one command line. No screening rule, prompt, scoring rule, selection rule, unit list, setting, statistic or report formula changed: every live record's settings equal what the code declares, and every stored report is reproduced (evidence below).
- **Freeze mechanics, now the same for every stage** (disclosed because they differ from what sections 1–4 describe):
  - *Code identity* is one hash of every source file of the package (`.py`, and `.html` resources) by relative path. Every stage, stage 1 included, accepts its frozen code or code reached from it through `freeze/code_amendments.json` (1.9). Before, stage 1 hashed a listed set of modules and compared exactly; stage 3 hashed the top-level modules; the replication hashed the tree; NeXT hashed both packages.
  - *One draft rule:* a freeze refuses while its section has a line starting with "DRAFT" or a "[to fill" blank (section 2 for the GQA stage 3, section 3 for the replication, section 4 for NeXT). Before, stage 3 refused the word DRAFT anywhere before section 1.
  - Stage 1's check reads every manifest the record froze (GQA and TextVQA) from the screened manifest's folder, so a changed manifest blocks either screen.
  - Report-time checks compare nothing in this file, stage 1 included (as 3.5 and 4.4 already said for the later stages); run-time checks still refuse any change.
  - Item files written by screens and runs carry the hash of the record they were produced under, and a resumed run refuses a file from another record; a screen folder records its kind. A refusal exits with code 2.
- **Names in sections 1–4 that moved:** `kvsel.confirm.comparative_settings` (2.1) is `kvsel.studies.confirmatory.comparative.comparative_settings`; `src/kvsel/confirm_analysis.py` (2.3) is `src/kvsel/studies/confirmatory/report.py`; "all code under `src/kvsel/`" (3.5) and "both packages" (4.4) are the code identity above.

  | Before | Now |
  | --- | --- |
  | `kvsel build-gqa`, `build-textvqa`, `fetch-images` | `kvsel data build-gqa`, `build-textvqa`, `fetch-images` |
  | `kvsel fetch-model`, `python -m kvsel_next fetch-model` | `kvsel model fetch [--model llava_next]` |
  | `kvsel validate`, `python -m kvsel_next validate` | `kvsel validate [--model llava_next]` |
  | `kvsel screen`, `pilot`, `confirm-report` on a pilot | `kvsel pilot screen`, `run`, `report` |
  | `kvsel freeze-screening`, `screen-confirm`, `freeze-comparative`, `confirm`, `confirm-report` | `kvsel confirmatory freeze-screening`, `screen`, `freeze`, `run`, `report` |
  | `python -m kvsel.followup run`, `report` | `kvsel followups run`, `report` |
  | `python -m kvsel.replication count`, `freeze`, `run`, `report` | `kvsel replication count`, `freeze`, `run`, `report` |
  | `python -m kvsel_next freeze-screening`, `screen`, `count`, `freeze`, `run`, `report` | `kvsel next` with the same commands |
  | `scripts/verify_amendment3.py` | `kvsel reproduce` |

- **Reason:** each study had copied the freeze, code-hash and report code of the one before it, and LLaVA-1.5's 576-token layout was written into shared code, so a reviewer had to check the same rule in several places. No confirmatory outcome motivated the change.
- **Timing, stated plainly:** written after the GQA confirmatory, follow-up and replication reports had been seen, and before any LLaVA-NeXT comparative unit runs on the reorganized code. None of the changes alters an estimate, interval, p-value or Holm decision; the evidence shows it.
- **Evidence:** `kvsel reproduce --reason "amendment 4: reorganization, no rule changed (PREREG 5)" --images data/gqa/images` writes one pending entry per live record into `freeze/code_amendments.json`, from that record's latest accepted code to the reorganized code. It reruns the stored reports listed in `stored_reports.json` (the confirmatory and follow-up `results.json` byte for byte; the replication's on every stored key, since amendment 3 added keys) and three main and three control items of each stored LLaVA-1.5 run row for row on the GPU. Only if everything reproduces does it write the evidence into those entries; until then no check accepts the new code through them. Model inputs and item digests are identical on all four manifests (`uv run pytest -m slow`). The records themselves are not rewritten.
