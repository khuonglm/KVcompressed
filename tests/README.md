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
items. The distribution of logit and margin differences is the fp16 noise floor.

| ID | Test | Pass |
| --- | --- | --- |
| E1 | Identical prefix tokens for Q and every QH_c; prompt tokens an exact prefix of prompt + candidate; window span logged | Exact, on all development items |
| E2 | Compact replay at K = 576 vs one full forward pass | Max logit difference at most the noise floor's maximum |
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

## Open failure: E2, first GPU run (7 November 2026)

Max logit difference 0.208 against a limit of 0.110. All other tests passed. Possible cause: E2 ran on
the QH_mis suffix, but its limit came from a noise floor measured on the Q suffix.

`kvsel diagnose-e2` measures compact K = 576, dense with nothing dropped, and one full forward pass,
on both suffixes and the same items. This reading was fixed before the diagnostic first ran:

| Observed | Conclusion | Action |
| --- | --- | --- |
| Compact equals dense exactly on every item and suffix, and the Q and QH_mis noise-floor maxima fall in line with 0.11 and 0.21 | Cause identified: the limit came from a different suffix. Gathering adds no error | Compare E2 with a noise floor from the same suffix, keeping the same criterion, and rerun `kvsel validate` |
| Compact differs from dense on any item | Compaction itself adds error at K = 576 | Stays failed; debug the gather |
| Compact equals dense, but the suffix difference doesn't explain the size | Cause not identified | Stays failed; investigate further |

## Not yet covered in the minimal version

E6's coverage plots (the radius is checked), hand verification (D1, D2) and the timing test R1.
