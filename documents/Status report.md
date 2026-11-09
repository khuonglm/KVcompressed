# Status report, 9 November 2026

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
| 5. Your decisions, then v3.1 pilots on both datasets | Done |
| 6. Builder fixes, GQA widening, timing test, coverage plots | Done |
| 6b. Pre-registration and freeze tooling | **Next** |
| 7. Freeze, then the confirmatory run once, then the analysis | Not started |
| Second model, Qwen2.5-VL | Parked (see below) |

## Data
| | GQA | TextVQA |
| --- | ---: | ---: |
| Development items | 600 | 1,500 |
| Confirmatory pool | 9,209 | 5,514 |
| Eligible (model answers correctly with no description) | 351 of 600 (58%) | 888 of 1,500 (59%) |
| Passed hand verification | 150 of 200 (75%) | 188 of 200 (94%) |

GQA's confirmatory pool is now 9,209 candidates (builder `gqa-3`, from `val_all`). GQA labels fail hand verification 25% of the time and TextVQA's 6%, so every confirmatory item on both datasets will be hand-verified (no audit).

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

## v3.1 pilots (9 November)
*Development only.* Template v3.1 labels the description "Unverified description, possibly wrong:".

| | GQA (150) | TextVQA (188) |
| --- | --- | --- |
| Full cache, misleading answer given | 46% (was 99%) | 2% (was 4%) |
| S_H^(mis) | **+0.09 [+0.02, +0.18]** | −0.01 [−0.05, +0.02] |
| T_match / T_irr | +0.07 / +0.06, inconclusive | −0.07 / −0.07 |
| I^(mis), exploratory | −0.00 [−0.05, +0.06] | −0.04 |
| E_harm^(mis), H3 | −0.21, opposite to the prediction | +0.45, but positive for every description |

- **GQA:** a smaller effect than under v3.0, but answers can now flip. I ≈ 0 means no evidence that the steered kept set hurts *more* when the lie is present.
- **TextVQA:** no steering effect.
- **Tokenization:** 200/200 and 199/200.

## Qwen2.5-VL-7B
Added as a model that follows text less readily. After two precision fixes it still fails one validation test:
- **The failure:** compressed replay differs from the reference by 0.038 nats, against a 0.01 limit.
- **The cause:** its bf16 numbers. The full model in fp32 doesn't fit on the GPU.

It's parked, and the code is kept.

## Decisions
| # | Question | Status |
| --- | --- | --- |
| 1 | The model almost always follows the misleading text | **Done:** prompt test → template v3.1 ("Unverified description, possibly wrong:"). GQA adoption 99% → 46% |
| 2 | H3's direction | **Done:** two-sided |
| 3 | I^(mis) | **Done:** reported outside Holm, exploratory |
| 4 | GQA size | **Done:** widen to 1,000, together with a builder fix for the 25% verification failures |
| 5 | TextVQA | **Done:** kept |
| 6 | Primary selector | **Done:** PW-attn |
| 7 | H1, H2 one-sided, but TextVQA points the other way | **Done:** all confirmatory tests two-sided |

## Next
1. GQA: builder `gqa-3` is done (9,209 confirmatory candidates from `val_all`; development items unchanged). Automatic filters couldn't bring failures from 25% to 5%, so every confirmatory item will be hand-verified (about 1,350 to reach 1,000 passes).
2. TextVQA: builder `textvqa-3` is done (at least 5 of 10 annotators must agree on the answer; 5,514 confirmatory candidates; development items unchanged). Every confirmatory item will be hand-verified too, about 1,100 items.
3. **Done:** S1 (GQA 73%, TextVQA 60%, both in the expected ranges) and R1 (screens 1.6 h and 1.1 h, confirmatory run about 10 h, 15.3 GiB peak). E6 coverage plots are done.
4. **Built:** `PREREG.md` draft (for your review), `freeze-screening`, `screen-confirm`. Next: review and commit PREREG, freeze, screen, verify eligible items. Then the stage-3 tooling and the confirmatory run.
