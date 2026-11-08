# Status report, 8 November 2026

## The study
Does unreliable text steer which visual entries LLaVA-1.5-7B keeps when its cache is compressed?

The prompt includes a misleading description. We compare two kept sets under that same prompt:
- **B:** visual entries chosen with the description in view.
- **C:** visual entries chosen from the question only.

**Main measure:** S_H^(mis) = margin(C) − margin(B), where the margin is log p(true answer) − log p(wrong answer), in nats. T_match and T_irr check that the effect comes from *false* text, not from any added text.

## Where we are
| Stage | Status |
| --- | --- |
| 1. Pipeline and tests | Done |
| 2. Datasets (GQA, TextVQA) | Done |
| 3. GPU validation (E1–E7) | Done. LLaVA passes everything on both datasets |
| 4. Screen, pilot, hand verification (development items) | Done for both datasets |
| 5. Your decisions, pre-registration, timing test, coverage plots | **Next** |
| 6. Freeze, then the confirmatory run once, then the analysis | Not started |
| Second model, Qwen2.5-VL | Parked (see below) |

## Data
| | GQA | TextVQA |
| --- | ---: | ---: |
| Development items | 600 | 1,500 |
| Confirmatory pool | 1,634 | 5,720 |
| Eligible (model answers correctly with no description) | 351 of 600 (58%) | 888 of 1,500 (59%) |
| Passed hand verification | 150 of 200 (75%) | 188 of 200 (94%) |

Expected eligible confirmatory items for GQA: about 950. The plan asks for 1,000, so the pool will be widened. Only the control subset (300) and an audit (100) are hand-verified, and the audit fails above 5% errors. GQA development verification failed 25% of items, so the builder needs fixing first.

## GQA pilot results
*Development items only, so these are not claims.* 150 verified items.

- **Main effect:** S_H^(mis) = **+0.23 nats [+0.14, +0.32]**. Choosing visual entries from the question only is better than choosing them with the misleading description in view.
- **Specific to false text:** with a matching or an irrelevant description, the effect is about +0.02. T ≈ +0.20.
- **But answers don't change:** the model follows the misleading description about 99% of the time, even with the full cache. Its margin there is about −3.5 nats, and a 0.23 shift can't flip that.
- **The misleading kept set hurts on its own:** with no description in the prompt, correct answers drop from 97% to 93% (cell D).
- **H3 isn't supported:** extra compression damage under the misleading description is −0.12, inconclusive. H3 predicted a positive value.
- **Sanity check:** the attention selector works. Uniform selection gets 81% correct, against 97% for the selector.

## TextVQA pilot results
188 verified items. This is the opposite picture from GQA.

- **The model mostly resists the misleading text:** 90% correct with the full cache; only 4% give the wrong alternative.
- **Main effect reversed:** S_H^(mis) = **−0.10 [−0.15, −0.05]**. Here, choosing entries with the misleading description in view does slightly *better*. T_match and T_irr are negative too.
- **H3 holds here:** E_harm^(mis) = +0.53. Compression hurts more under the misleading description.
- **Question-span selection** shows no effect (S_H ≈ 0).

## Qwen2.5-VL-7B
Added as a model that follows text less readily. After two precision fixes it still fails one validation test:
- **The failure:** compressed replay differs from the reference by 0.038 nats, against a 0.01 limit.
- **The cause:** its bf16 numbers. The full model in fp32 doesn't fit on the GPU.

It's parked, and the code is kept.

## Decisions for you
| # | Question | Options (★ my recommendation) |
| --- | --- | --- |
| 1 | The model almost always follows the misleading text | ★ **A.** Keep the prompt; report margin effects, with cell D as the evidence about answers. **B.** Change the prompt; this needs a new pilot and several days |
| 2 | Hypothesis H3 points the wrong way in the pilot | ★ Test it two-sided. Or keep it as written. Flipping it would look like fitting to the data |
| 3 | I^(mis): exploratory, but also listed as a test | ★ Report it with intervals only, outside the Holm correction |
| 4 | GQA size | ★ Accept about 700 items. Or widen the item pool to reach 1,000 |
| 5 | TextVQA | ★ Keep it, pending its pilot |
| 6 | Primary selector | ★ PW-attn, as specified in advance. Question-span stays secondary |

## After you decide
I write the pre-registration and build the timing test and coverage plots. Then we freeze the protocol and run the confirmatory items once.
