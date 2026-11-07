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

**CPU evidence so far** (`tests/test_qwen.py`, tiny random Qwen2.5-VL in fp32, real tokenizer and image
processor): prefix-then-replay equals one full forward pass, which computes M-RoPE itself; replaying
from the physical entry count instead of the M-RoPE position fails, as it should; compact equals dense
at K = 288, 144 and 72 with generated tokens; the prefix is not mutated; E1 passes on all 600 GQA and
1,500 TextVQA development items.

## Not yet covered in the minimal version

E6's coverage plots (the radius is checked), hand verification (D1, D2) and the timing test R1.
