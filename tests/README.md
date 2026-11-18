# Acceptance policy

Written before these tests first ran. A throwaway exploratory spike on a tiny Llama preceded this
file; it applied no tolerances and is not part of this suite.

## Rules

1. **Tolerances are never loosened after a failure.** A failure closes only when its cause is identified.
2. **Numerical-policy changes** (for example output-head precision) are allowed during development
   only. Each needs a documented cause and a rerun of every affected test. None may respond to a
   confirmatory scientific outcome.
3. **Timing runs** have test instrumentation (hashing, extra logging) switched off.

## Numerical policy

- Backbone, cache and attention stay in fp16 on the real model.
- Log-probabilities and their sums are always computed in fp32.
- Output-head default: fp32, a copy of `lm_head` applied to the fp32-cast hidden state. This reduces
  rounding in the projection only and cannot repair rounding already in fp16 hidden states, so it
  is no guarantee that E3 passes. `kvsel validate` reports the noise floor under both head
  precisions, so the choice rests on a measurement. If the fp16 head is no worse, drop the fp32 copy.

## Level 1: operator correctness (`tests/test_level1.py`, CPU)

A tiny random Llama in fp32 runs the same gather, mask and replay code. Compact vs dense on random,
concentrated and scored kept sets, K = 288, 144, 72, with generated tokens.
**Pass: max absolute logit difference below 1e-5.**

Added to the proposal's Level 1, to show the test can detect the bugs it targets:

| Control | Pass |
| --- | --- |
| Physical instead of logical positions in compact replay | Difference above 1e-5 |
| Perturbing dropped entries under dense execution | Difference exactly 0 |
| Perturbing kept entries under dense execution | Difference above 1e-5 |
| A probe or replay leaves prefix tensors unchanged | Hash unchanged |

## Level 2: LLaVA pipeline (`kvsel validate`, GPU, fp16)

The checks are `kvsel.validation`, shared with LLaVA-NeXT's Level 2 below; `tests/test_validation.py` runs each
one on both tiny models and the whole command to GO and NO-GO.

**Noise floor.** One full forward pass vs prefix-then-suffix with the full cache, on 50 development
items, on each of Q and QH_mis. The distribution of logit and margin differences is the fp16 noise
floor. E3 and E5 keep the Q floor, which was the tighter limit on the first GPU run.

| ID | Test | Pass |
| --- | --- | --- |
| E1 | Identical prefix tokens for Q and every QH_c; prompt tokens an exact prefix of prompt + candidate; window span logged | Exact, on all development items |
| E2 | Compact replay at K = 576 on Q and QH_mis, vs one full forward pass and vs dense with nothing dropped | Max logit difference at most the maximum of that suffix's own noise floor, and compact equal to dense exactly |
| E3 | Compact vs dense for irregular random, spatially concentrated and scored kept sets; K = 288, 144, 72; Q and QH_mis; 10 generated tokens | See below |
| E4 | Hash the shared prefix before and after each probe and replay | Unchanged |
| E5 | Placebo: B_c and C_c with kept indices forced equal | Within the noise floor's maximum; identical answers |
| E6 | Selector unit tests (`tests/test_selectors.py`) | All pass |
| E7 | Rerunning an item reproduces its rows | Identical |

**E3 criteria.**

- The 99th percentile of compact-vs-dense logit differences is at most twice that of the noise floor.
- Mean absolute margin difference is below 0.01 nats.
- Every greedy mismatch is investigated. It is accepted only as a near tie: the top-two logit gap at
  the diverging step is below twice the noise floor's maximum. Any other mismatch is a failure.

## Closed failure: E2, first GPU run (7 November 2026)

**Failure.** Max logit difference 0.208 against a limit of 0.110. All other tests passed.

**Cause.** E2 ran on the QH_mis suffix, but its limit came from a noise floor measured on the Q
suffix. The reading of the diagnostic was fixed before it first ran, and `kvsel diagnose-e2` matched
its first row:

| Suffix | Full vs compact, max | Full vs dense, max | Compact vs dense, max |
| --- | ---: | ---: | ---: |
| Q | 0.1104 | 0.1104 | 0 |
| QH_mis | 0.2084 | 0.2084 | 0 |

Gathering adds no error at K = 576. The 0.2084 is the same fp16 difference between one full pass and
prefix-then-suffix that the noise floor measures, and it occurs on one item.

**Change.** E2 now compares each suffix with its own noise floor, under the same criterion, and also
requires compact to equal dense exactly. That second condition is new and stricter. Without it, E2
at K = 576 would repeat the noise-floor computation and could not fail. The failing limit was not
raised on the Q suffix, and E3 and E5 keep the Q floor. Every test reruns under `kvsel validate`.

## Level 2 outcomes, LLaVA (development items)

| Run | Result |
| --- | --- |
| GQA validate (7 Nov 2026, after the E2 fix) | E1-E7 pass |
| TextVQA validate (7 Nov 2026) | E1-E7 pass. Noise floor: Q margin mean 0.0055, QH_mis 0.0056, max logit 0.121. E3 margin mean 0.0048; 2 greedy mismatches, both near ties |
| TextVQA screen | 888 of 1,500 eligible under `capitalized`; greedy Full_Q answers 87.9% capitalized, 11.0% lower, 0.1% upper |
| check-tokenization | GQA pilot 200 of 200 agree. TextVQA screen 1,491 of 1,500 (99.4%); the 9 disagreements are rare names (Olivetti, Expedia, Powerade) the model spells with a non-canonical token split |

## Qwen2.5-VL-7B (second model, added 7 November 2026)

Written before any Qwen test ran on the real model. The same rules and the same E1-E7 criteria apply;
every limit that comes from a noise floor uses **Qwen's own** noise floor, measured the same way.

**Numerical policy for Qwen.**

- Backbone, cache and attention in **bf16**, the model's training precision. Qwen2.5-VL is known to be
  unstable in fp16, so the LLaVA rule of fp16 does not carry over.
- Output head: **native** (bf16 projection, logits cast to fp32). An fp32 copy of the 152k-row head is
  2.2 GB and does not fit beside 16.6 GB of weights on the 20 GB GPU. *Superseded by the amendment below.*
- Log-probabilities and their sums in fp32, as for LLaVA.

**Known risk, stated in advance.** bf16 keeps 8 bits of mantissa against fp16's 11, so Qwen's noise
floor will be larger than LLaVA's. E3's margin limit (mean below 0.01 nats) is absolute, not relative
to the noise floor, so it is the criterion most likely to fail. If it does, the rules above apply: the
tolerance is not loosened, and the failure closes only once its cause is identified.

**Amendment (7 November 2026, after the failure below).** Output head changed to **fp32**, computed
by upcasting the bf16 head 16,384 rows at a time, so no 2.2 GB copy is stored. The arithmetic is the
same as an fp32 copy, since bf16 weights are exact in fp32 (`test_sliced_fp32_head_matches_a_full_fp32_copy`).
This is a numerical-policy change under rule 2; no tolerance changed.

### Open failure: Qwen E3, first GPU run (7 November 2026)

**Failure.** E3 margin mean 0.204 against the 0.01 limit. E1, E2, E4, E5 and E7 passed, and E3's
logit p99 (1.69) was within its limit (4.25). Noise floor: margin mean 0.224, logit p99 2.13, logit
max 17.3, against LLaVA's 0.003, 0.017 and 0.12.

**Weakness found in the test.** All 73 greedy mismatches counted as near ties, but the near-tie limit
is twice the noise floor's maximum, 34.6 nats, so that check could not fail. It regains its power only
once the noise floor is small. The criterion is unchanged.

**Two candidate causes, stated before the rerun.**

1. *The bf16 output head.* Every greedy gap in the report is 0, 0.125, 0.25 or 0.375. That is the
   bf16 spacing for logits between 16 and 32, and gaps of exactly 0 are ties that exist only because
   the logits were rounded. The amendment above removes this source.
2. *bf16 attention.* The head alone cannot explain a logit p99 of 2.13, because rounding in the head
   is at most 0.0625 for logits below 32. Qwen's eager attention rounds the QK^T scores to bf16
   before the softmax, and the attention weights to bf16 before multiplying by V. Compact and dense
   differ only inside attention (different key lengths, so a different summation order), so E3's
   difference must start there. At scores above 128, one bf16 step is 1.0 before the softmax. The
   model was trained with flash attention, which keeps scores in fp32.

**Predictions for the rerun (fp32 head, otherwise unchanged).**

- The native-head noise floor, now measured separately, reproduces the first run exactly (margin mean
  0.2240, logit max 17.3047). If it does not, something else changed, and that is investigated first.
- Greedy gaps are no longer multiples of 0.125.
- If the fp32-head E3 margin mean falls below 0.01, cause 1 was enough and the failure closes.
- If it stays above 0.01, which is the expected outcome given the logit p99, cause 2 is next: an
  attention that keeps scores and weights in fp32, tested the same way. The 0.01 limit does not change.

**Second run, fp32 head (7 November 2026).** E3 failed again. The native-head noise floor reproduced
the first run exactly (margin mean 0.22395785964472453, logit max 17.3046875). The fp32-head floor
was almost the same (margin mean 0.2235, logit p99 2.127, max 17.30). So the head contributes almost
nothing, and cause 1 is ruled out as the main source. The fp32 head stays, since it removes the
rounding ties and costs no memory.

**Amendment 2 (7 November 2026).** The language model's attention now uses `kvsel.attention`: scores,
softmax and the weighted sum of values in fp32, output returned in bf16, cache unchanged. Qwen's
vision tower and all of LLaVA keep the library's eager attention. In an fp32 model it equals eager
attention below 1e-6, in both logits and probe weights (`test_fp32_attention_equals_eager_in_an_fp32_model`).
On CPU, compact and dense already agree exactly in bf16 under both attentions, so the effect can
only be measured on the GPU.

**Predictions for the third run (fp32 head, fp32 attention).**

- E3's margin mean falls well below 0.204, because compact and dense differ only inside attention.
  Below 0.01 closes the failure.
- The noise floor may stay large. It also compares bf16 linear layers run over different numbers of
  rows (one full pass against prefix then suffix), which this change does not touch.
- If E3 falls a lot but stays above 0.01, the remaining source is the bf16 rounding of the attention
  output. Further precision cannot remove that without running the model in fp32, which does not fit.
  The choice then is the user's: report Qwen as exploratory, with its measured compact-vs-dense error,
  or drop it. The limit does not change.
- If E3 barely moves, cause 2 is wrong too, and the next step is diagnosis, not another precision change.

**Third run, fp32 head and fp32 attention (7 November 2026).** E3 still fails, on the margin criterion
alone. Every other test passes.

| | Run 1 (all bf16) | Run 2 (fp32 head) | Run 3 (fp32 head and attention) |
| --- | ---: | ---: | ---: |
| Noise floor, Q: margin mean | 0.224 | 0.224 | 0.034 |
| Noise floor, Q: logit max | 17.3 | 17.3 | 2.26 |
| Noise floor, Q: logit p99 | 2.13 | 2.13 | 0.30 |
| E3 margin mean (limit 0.01) | 0.204 | not recorded | **0.038** |
| E3 logit p99 (limit twice the floor's p99) | 1.69 (4.25) | not recorded | 0.33 (0.61), pass |
| E3 greedy mismatches, of 360 | 73, check powerless | not recorded | 17; largest gap 0.26, near-tie limit 4.5 |

- Attention was the main source: fp32 attention cut the E3 margin error about 5x and the noise floor about 7x.
- In run 3 the native-head floor had a margin mean of 0.050, against 0.034 for the fp32 head, so the head
  matters once attention noise is gone.
- E3's remaining error (0.038) is about the size of Qwen's own noise floor (0.034, from one full pass
  against prefix-then-suffix). Compaction adds about as much difference as computing the same thing in
  a different order.
- The remaining source is most likely bf16 rounding outside the attention scores: the attention output
  and the linear layers. That is inferred, not shown. Removing it would need the model in fp32, which
  does not fit on the GPU.

**Status.** This was the outcome predicted under amendment 2: E3 falls a lot but stays above 0.01. The
limit stays. Qwen does not pass Level 2, and whether it continues is the user's decision: as an
exploratory model reported with its measured compact-vs-dense error, or not at all.

**CPU evidence so far** (`tests/test_qwen.py`, tiny random Qwen2.5-VL in fp32, real tokenizer and image
processor): prefix-then-replay equals one full forward pass, which computes M-RoPE itself; replaying
from the physical entry count instead of the M-RoPE position fails, as it should; compact equals dense
at K = 288, 144 and 72 with generated tokens; the prefix is not mutated; E1 passes on all 600 GQA and
1,500 TextVQA development items.

## LLaVA-NeXT-7B, Level 1 (`tests/test_next.py`, CPU, added 16 November 2026)

A tiny random LLaVA-NeXT (real processor, AnyRes on, real grid pinpoints) in fp32 runs the same
engine on the expanded visual span of a wide and a tall image (1656 and 1676 visual tokens). Kept sets
are feature tokens (random, concentrated, tail, probe-scored; 50, 25 and 12.5%) plus every newline token.
Limits and controls are Level 1's, unchanged: compact vs dense below 1e-5; physical positions above
1e-5; perturbed dropped entries exactly 0; perturbed kept entries above 1e-5; prefix hash unchanged.
Numerics on the real model are LLaVA-1.5's: fp16 backbone and cache, fp32 head and log-probabilities.
Selection by share (`tests/test_next_selection.py`, NeXT's E6): 3 x 3 pooling within each map, never across
a map boundary or a newline; k = share x feature tokens, refused unless whole (real feature counts are multiples of 24);
newlines always kept and outside k; kept sets nested across 12.5, 25 and 50%.

## LLaVA-NeXT-7B, Level 2 (`kvsel validate --model llava_next`, GPU, fp16; written 16 November 2026, before the run)

Level 2's criteria, unchanged, on NeXT's expanded visual span; on the CPU it runs end to end on the tiny
model in `tests/test_validation.py`, with the same code as LLaVA-1.5's Level 2 (`kvsel.validation`). Every context
shares Q's prefix. A kept set is k = round(share × feature tokens) feature tokens plus every newline token.

**Noise floor.** As Level 2, on 50 development items, on each of Q and QH_mis. E3 uses the Q floor.

| ID | Test | Pass |
| --- | --- | --- |
| E1 | On every given item: the processor's expansion leaves the tokens outside the span equal to LLaVA-1.5's; identical prefix for Q and every QH_c; identical question tokens; prompt tokens an exact prefix of prompt + candidate; derived layout equal to the span length | Exact, on all given items |
| E2 | Compact replay with every visual token kept, on Q and QH_mis, vs one full forward pass and vs dense with nothing dropped | Max logit difference at most the maximum of that suffix's own noise floor, and compact equal to dense exactly |
| E3 | Compact vs dense for random and concentrated (first k) feature sets; shares 12.5%, 25%, 50%; Q and QH_mis; 10 generated tokens; first 20 items | Level 2's E3 criteria |
| E4 | Hash the shared prefix before and after each probe and replay | Unchanged |
| E7 | Prefill, the mis probe and two units (Full_Q; QH_mis at a 25% random set), run twice on 5 items | Identical |

(Note, 18 November 2026, ticket 09: E7 now reruns every unit of the control list and compares the rows and the
probe attention, as LLaVA-1.5's E7 reruns every pilot unit; the pass rule, identical, is unchanged.)

Not run for NeXT: E5 (placebo, which needs selection) and scored kept sets (selection is ticket 02).
(Note, 16 November 2026: selection now exists; validate still uses random and concentrated sets only,
since E3 compares compact and dense replay on any sparse set. The criteria above are unchanged.)
(Note, 16 November 2026, after the first GPU attempts ran out of memory on the 20 GiB slice, before any check
had a result: the prefix cache now waits in CPU memory. Compact replays gather the kept entries on the CPU and
move them to the GPU; dense replays and probes use a GPU copy of the prefix, freed before scoring. Both copy
values exactly, and `tests/test_next.py` checks the rows, logits and states equal kvsel's own path. The
checks and limits are unchanged. E4 now hashes the copy each probe and replay receives, before and after, since
the CPU-held prefix itself is never given to the engine. The fp32 head is upcast in 4,096-row slices on each call
instead of kept as an fp32 copy: each logit is the same fp32 dot product, though on CUDA a different matrix shape
may move its last bits. Prefill, probe and unit times now include moving entries between CPU and GPU, as the run
does, so they are not comparable with earlier attempts.)
(Note, 18 November 2026, codebase reorganization: the shared engine now does this gathering and moving itself, so
compact replays receive the CPU-held prefix itself. E4 hashes the shared prefix before and after each probe and
replay, as the table says, and also the GPU copy that probes and dense replays still receive; a compact replay's
gathered copy is made inside the engine and is not hashed (the same step code's input is, in dense replays). `tests/test_next.py` checks that a device-held and a CPU-held prefix give identical rows, and the
Level-1 invariants run for both residencies when an accelerator is present. The checks and limits are unchanged.)
(Note, 18 November 2026, ticket 09: one validation module now runs both families, one check at a time, so E4 hashes the
shared prefix before and after each probe and each compact and dense replay of E3's kept sets; the engine makes
every device copy itself. The noise floor under the native head is now reported for NeXT too. The checks and limits
are unchanged; on the tiny models every verdict and measured value equals the earlier module's.)
**Stricter checks for the re-freeze (PREREG 4.6, written 19 November 2026, before any GPU run under it).** These
replace the rows above where they differ. The pass rules and limits are unchanged.

| ID | Test | Pass |
| --- | --- | --- |
| E1 | As above, but the ids outside the span are compared with LLaVA-1.5's own tokenizer (`llava-hf/llava-1.5-7b-hf`) for Q and every QH_c, not with NeXT's. NeXT's tokenizer loads with `legacy=True`, which keeps the space token before the newline, and the slow `LlavaNextImageProcessor` is loaded by name | Exact, on all given items |
| E3 | Random, concentrated (first k) and tail (last k) feature sets plus every newline; shares 12.5%, 25%, 50%; Q and QH_mis; 10 generated tokens; first 20 items. The tail set covers the high-resolution map at shares where the concentrated set lies in the base map | Level 2's E3 criteria |
| E4 | Hash the prefix as held (CPU memory) and a GPU copy that probes and replays read in place, each before and after every probe and every compact and dense replay of E3's kept sets | Both unchanged |
| E7 | Prefill, every probe and every unit of the control list (the probe-scored PW-attn cells included), run twice on 5 items; rows and probe attention compared | Identical |

Both noise floors (fp32 and native head) are reported whichever head runs, as since ticket 15.

On the CPU: `tests/test_next.py` checks that NeXT's ids outside the span equal LLaVA-1.5's for Q and every condition,
and that on a 4:3 image the tail set at 12.5% lies wholly in the high-resolution map. `tests/test_validation.py`
checks that E1 fails with the tokenizer as published. `tests/test_next_comparative.py` checks the overlap on
feature positions against hand-computed Jaccards, the report's comparison-file record and resample count, and its
refusal of a file missing K = 72, 144 or 288. E4's GPU copy only differs from the held prefix on an accelerator.

GO only if every check passes. Peak GPU memory, the visual-token range and time per item are reported,
not judged. The memory that counts is the run peak (prefill, the probe, then a Full unit on the cache), not validate's
own full forward passes; a run peak above 20 GB means an AnyRes cap in PREREG 4.1 before stage 1.
Projected times are upper bounds: every unit is costed as a Full unit on the longest suffix.

## LLaVA-NeXT-7B, stage 1 (`tests/test_next_screen.py`, CPU, added 16 November 2026)

`freeze-screening` refuses while section 4 has a line starting with DRAFT (a mention in prose is not the
marker), refuses a different existing record and returns the same record on an unchanged rerun. The
section hash runs from the `### 4.1` heading to the next `### ` or `## ` heading. `screen` refuses before
loading the model if section 4.1, the manifest, the settings or the code changed; screens only items whose
latest current-digest decision is pass, in manifest order; writes the item id and three flags only (every
file's keys are checked); stops once the main set's size is eligible; and resumes after a stop without
redoing an item. `count` works mid-screen and never loads the model. On the tiny model, Full_Q and Blind_Q
run end to end with a resume; Blind_Q's blank gets the real image's layout (1656 visual tokens).

## LLaVA-NeXT-7B, stage 3 (`tests/test_next_comparative.py`, CPU, added 16 November 2026)

`freeze` records the main set (the first eligible passed items in manifest order, stopping at an unscreened
one) with its digests, the control subset, and hashes of PREREG.md, the stage-1 record, the manifest, the
flags, the settings and both packages' code. It refuses a DRAFT line, a short set and a different record.
`run` refuses any change since the freeze, and missing images, before loading the model; code passes only
through `freeze/code_amendments.json`, at stage 1 and stage 3 alike. On the tiny model: the run stops and
resumes without redoing an item; rows follow section 4.2's unit lists (written out by hand in the test) and
carry share, k and the layout record; each context is probed once per item and its kept sets are nested
across shares; planting +0.7 nats on every C_mis margin moves N1 by exactly 0.7. On synthetic runs with
planted margins, every estimand (S_H, S_Q, I, E_harm per condition, T_irr, T_match) is recovered at every
share, match terms come from the 20 control items only, Holm over N1-N4 at 25% rejects the three planted
tests and not N2 (Holm's step-down thresholds written out), and LLaVA-1.5's K = 72, 144 and 288 estimates
are printed beside 12.5%, 25% and 50% with sign agreement; an estimand absent from the replication shows "—".
Settings changed since a freeze (the thin path's unit lists) are refused. `report` refuses missing or extra
item files and allows PREREG.md to grow, which `run` refuses.

## Not yet covered in the minimal version

E6's coverage plots (the radius is checked), hand verification (D1, D2) and the timing test R1.
