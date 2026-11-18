# LLaVA-NeXT-7B on GQA: report

Surface rule `capitalized`; PW-attn; image-clustered bootstrap, two-sided tests. 2 items on 2 images; 1 control items. LLaVA-1.5's comparative outcomes on many of these items were seen before NeXT's hypotheses were chosen (PREREG 4).

## Confirmatory: Holm family of 4 (α = 0.05), final

| Test | Mean (nats) | 95% interval | Two-sided p | Reading | Items / images | Holm |
| --- | ---: | --- | ---: | --- | ---: | --- |
| N1: S_H^(mis), 25.0% | -0.001 | [-0.005, +0.004] | 0.4314 | practically small | 2 / 2 | not rejected |
| N2: S_H^(irr), 25.0% | +0.000 | [-0.005, +0.005] | 0.5098 | practically small | 2 / 2 | not rejected |
| N3: E_harm^(mis), 25.0% | +0.001 | [-0.005, +0.006] | 0.5098 | practically small | 2 / 2 | not rejected |
| N4: T_irr, 25.0% | -0.001 | [-0.001, -0.000] | 0.0392 | opposite direction, practically small | 2 / 2 | not rejected |

Equivalence: not tested (PREREG 4.3).

Checks: Full_Q no longer eligible on 0 items; Blind_Q eligible on 0 of 2 (should be 0); truncated answers 25.00%; visual tokens 984-1320; high-resolution grids 24x30 (1), 24x16 (1).

## Every estimand by share (intervals only unless listed above)

### 12.5%

| Estimand | Mean (nats) | 95% interval | Two-sided p | Reading | Items / images |
| --- | ---: | --- | ---: | --- | ---: |
| S_H^(mis) | +0.019 | [+0.013, +0.025] | 0.0392 | positive, practically small | 2 / 2 |
| S_Q^(mis) | +0.020 | [+0.014, +0.026] | 0.0392 | positive, practically small | 2 / 2 |
| I^(mis) | -0.001 | [-0.001, -0.001] | 0.0392 | opposite direction, practically small | 2 / 2 |
| E_harm^(mis) | +0.019 | [+0.015, +0.024] | 0.0392 | positive, practically small | 2 / 2 |
| S_H^(irr) | +0.018 | [+0.011, +0.025] | 0.0392 | positive, practically small | 2 / 2 |
| S_Q^(irr) | +0.019 | [+0.013, +0.026] | 0.0392 | positive, practically small | 2 / 2 |
| I^(irr) | -0.001 | [-0.001, -0.001] | 0.0392 | opposite direction, practically small | 2 / 2 |
| E_harm^(irr) | +0.018 | [+0.013, +0.023] | 0.0392 | positive, practically small | 2 / 2 |
| S_H^(match) | +0.025 | [+0.025, +0.025] | 0.0392 | positive, practically small | 1 / 1 |
| S_Q^(match) | +0.026 | [+0.026, +0.026] | 0.0392 | positive, practically small | 1 / 1 |
| I^(match) | -0.001 | [-0.001, -0.001] | 0.0392 | opposite direction, practically small | 1 / 1 |
| E_harm^(match) | +0.023 | [+0.023, +0.023] | 0.0392 | positive, practically small | 1 / 1 |
| T_irr | +0.001 | [+0.000, +0.002] | 0.0392 | positive, practically small | 2 / 2 |
| T_match | -0.000 | [-0.000, -0.000] | 0.0392 | opposite direction, practically small | 1 / 1 |

### 12.5%: behaviour, paired C − B

| Estimand | Mean (share) | 95% interval | Two-sided p | — | Items / images |
| --- | ---: | --- | ---: | --- | ---: |
| correct: C_mis − B_mis | +0.000 | [+0.000, +0.000] | 1.0000 |  | 2 / 2 |
| alt: C_mis − B_mis | +0.000 | [+0.000, +0.000] | 1.0000 |  | 2 / 2 |
| correct: C_irr − B_irr | +0.000 | [+0.000, +0.000] | 1.0000 |  | 2 / 2 |
| alt: C_irr − B_irr | +0.000 | [+0.000, +0.000] | 1.0000 |  | 2 / 2 |
| correct: C_match − B_match | +0.000 | [+0.000, +0.000] | 1.0000 |  | 1 / 1 |
| alt: C_match − B_match | +0.000 | [+0.000, +0.000] | 1.0000 |  | 1 / 1 |

Mean Jaccard with S_Q: mis 0.559, irr 0.554, match 0.500

### 25.0%

| Estimand | Mean (nats) | 95% interval | Two-sided p | Reading | Items / images |
| --- | ---: | --- | ---: | --- | ---: |
| S_H^(mis) | -0.001 | [-0.005, +0.004] | 0.4314 | practically small | 2 / 2 |
| S_Q^(mis) | -0.001 | [-0.006, +0.004] | 0.4314 | practically small | 2 / 2 |
| I^(mis) | -0.000 | [-0.001, +0.000] | 0.5098 | practically small | 2 / 2 |
| E_harm^(mis) | +0.001 | [-0.005, +0.006] | 0.5098 | practically small | 2 / 2 |
| S_H^(irr) | +0.000 | [-0.005, +0.005] | 0.5098 | practically small | 2 / 2 |
| S_Q^(irr) | +0.000 | [-0.005, +0.006] | 0.5098 | practically small | 2 / 2 |
| I^(irr) | -0.000 | [-0.000, +0.000] | 0.5098 | practically small | 2 / 2 |
| E_harm^(irr) | +0.001 | [-0.005, +0.007] | 0.5098 | practically small | 2 / 2 |
| S_H^(match) | -0.005 | [-0.005, -0.005] | 0.0392 | opposite direction, practically small | 1 / 1 |
| S_Q^(match) | -0.006 | [-0.006, -0.006] | 0.0392 | opposite direction, practically small | 1 / 1 |
| I^(match) | +0.000 | [+0.000, +0.000] | 0.0392 | positive, practically small | 1 / 1 |
| E_harm^(match) | -0.006 | [-0.006, -0.006] | 0.0392 | opposite direction, practically small | 1 / 1 |
| T_irr | -0.001 | [-0.001, -0.000] | 0.0392 | opposite direction, practically small | 2 / 2 |
| T_match | -0.000 | [-0.000, -0.000] | 0.0392 | opposite direction, practically small | 1 / 1 |

### 25.0%: behaviour, paired C − B

| Estimand | Mean (share) | 95% interval | Two-sided p | — | Items / images |
| --- | ---: | --- | ---: | --- | ---: |
| correct: C_mis − B_mis | +0.000 | [+0.000, +0.000] | 1.0000 |  | 2 / 2 |
| alt: C_mis − B_mis | +0.000 | [+0.000, +0.000] | 1.0000 |  | 2 / 2 |
| correct: C_irr − B_irr | +0.000 | [+0.000, +0.000] | 1.0000 |  | 2 / 2 |
| alt: C_irr − B_irr | +0.000 | [+0.000, +0.000] | 1.0000 |  | 2 / 2 |
| correct: C_match − B_match | +0.000 | [+0.000, +0.000] | 1.0000 |  | 1 / 1 |
| alt: C_match − B_match | +0.000 | [+0.000, +0.000] | 1.0000 |  | 1 / 1 |

Mean Jaccard with S_Q: mis 0.562, irr 0.558, match 0.593

### 50.0%

| Estimand | Mean (nats) | 95% interval | Two-sided p | Reading | Items / images |
| --- | ---: | --- | ---: | --- | ---: |
| S_H^(mis) | +0.001 | [-0.001, +0.002] | 0.5098 | practically small | 2 / 2 |
| S_Q^(mis) | +0.001 | [-0.001, +0.002] | 0.5098 | practically small | 2 / 2 |
| I^(mis) | +0.000 | [-0.000, +0.000] | 0.4314 | practically small | 2 / 2 |
| E_harm^(mis) | +0.001 | [-0.001, +0.003] | 0.5098 | practically small | 2 / 2 |
| S_H^(irr) | +0.001 | [-0.001, +0.003] | 0.5098 | practically small | 2 / 2 |
| S_Q^(irr) | +0.001 | [-0.002, +0.003] | 0.5098 | practically small | 2 / 2 |
| I^(irr) | +0.000 | [-0.000, +0.000] | 0.4314 | practically small | 2 / 2 |
| E_harm^(irr) | +0.001 | [-0.001, +0.003] | 0.5098 | practically small | 2 / 2 |
| S_H^(match) | -0.001 | [-0.001, -0.001] | 0.0392 | opposite direction, practically small | 1 / 1 |
| S_Q^(match) | -0.001 | [-0.001, -0.001] | 0.0392 | opposite direction, practically small | 1 / 1 |
| I^(match) | +0.000 | [+0.000, +0.000] | 0.0392 | positive, practically small | 1 / 1 |
| E_harm^(match) | -0.001 | [-0.001, -0.001] | 0.0392 | opposite direction, practically small | 1 / 1 |
| T_irr | +0.000 | [-0.000, +0.000] | 0.4314 | practically small | 2 / 2 |
| T_match | -0.000 | [-0.000, -0.000] | 0.0392 | opposite direction, practically small | 1 / 1 |

### 50.0%: behaviour, paired C − B

| Estimand | Mean (share) | 95% interval | Two-sided p | — | Items / images |
| --- | ---: | --- | ---: | --- | ---: |
| correct: C_mis − B_mis | +0.000 | [+0.000, +0.000] | 1.0000 |  | 2 / 2 |
| alt: C_mis − B_mis | +0.000 | [+0.000, +0.000] | 1.0000 |  | 2 / 2 |
| correct: C_irr − B_irr | +0.000 | [+0.000, +0.000] | 1.0000 |  | 2 / 2 |
| alt: C_irr − B_irr | +0.000 | [+0.000, +0.000] | 1.0000 |  | 2 / 2 |
| correct: C_match − B_match | +0.000 | [+0.000, +0.000] | 1.0000 |  | 1 / 1 |
| alt: C_match − B_match | +0.000 | [+0.000, +0.000] | 1.0000 |  | 1 / 1 |

Mean Jaccard with S_Q: mis 0.737, irr 0.737, match 0.766

## LLaVA-1.5 (replication, section 3) beside NeXT: descriptive, never pooled

The item sets and absolute budgets differ; sign agreement is reported, not tested.

| Estimand | NeXT share | LLaVA-1.5 K | LLaVA-1.5 | NeXT | Same sign |
| --- | ---: | ---: | --- | --- | --- |
| S_H^(mis) | 12.5% | 72 | +0.001 [-0.004, +0.006] | +0.019 [+0.013, +0.025] | yes |
| S_Q^(mis) | 12.5% | 72 | +0.001 [-0.005, +0.007] | +0.020 [+0.014, +0.026] | yes |
| I^(mis) | 12.5% | 72 | -0.000 [-0.001, +0.000] | -0.001 [-0.001, -0.001] | yes |
| E_harm^(mis) | 12.5% | 72 | -0.000 [-0.006, +0.006] | +0.019 [+0.015, +0.024] | no |
| S_H^(irr) | 12.5% | 72 | +0.001 [-0.005, +0.006] | +0.018 [+0.011, +0.025] | yes |
| S_Q^(irr) | 12.5% | 72 | +0.001 [-0.005, +0.007] | +0.019 [+0.013, +0.026] | yes |
| I^(irr) | 12.5% | 72 | -0.001 [-0.001, -0.000] | -0.001 [-0.001, -0.001] | yes |
| E_harm^(irr) | 12.5% | 72 | +0.002 [-0.003, +0.008] | +0.018 [+0.013, +0.023] | yes |
| S_H^(match) | 12.5% | 72 | +0.006 [+0.006, +0.006] | +0.025 [+0.025, +0.025] | yes |
| S_Q^(match) | 12.5% | 72 | +0.007 [+0.007, +0.007] | +0.026 [+0.026, +0.026] | yes |
| I^(match) | 12.5% | 72 | -0.001 [-0.001, -0.001] | -0.001 [-0.001, -0.001] | yes |
| E_harm^(match) | 12.5% | 72 | +0.006 [+0.006, +0.006] | +0.023 [+0.023, +0.023] | yes |
| T_irr | 12.5% | 72 | +0.000 [+0.000, +0.000] | +0.001 [+0.000, +0.002] | yes |
| T_match | 12.5% | 72 | +0.000 [+0.000, +0.000] | -0.000 [-0.000, -0.000] | no |
| S_H^(mis) | 25.0% | 144 | -0.005 [-0.007, -0.003] | -0.001 [-0.005, +0.004] | yes |
| S_Q^(mis) | 25.0% | 144 | -0.005 [-0.007, -0.003] | -0.001 [-0.006, +0.004] | yes |
| I^(mis) | 25.0% | 144 | +0.000 [+0.000, +0.000] | -0.000 [-0.001, +0.000] | no |
| E_harm^(mis) | 25.0% | 144 | -0.007 [-0.010, -0.005] | +0.001 [-0.005, +0.006] | no |
| S_H^(irr) | 25.0% | 144 | -0.005 [-0.007, -0.003] | +0.000 [-0.005, +0.005] | no |
| S_Q^(irr) | 25.0% | 144 | -0.005 [-0.007, -0.003] | +0.000 [-0.005, +0.006] | no |
| I^(irr) | 25.0% | 144 | +0.000 [+0.000, +0.000] | -0.000 [-0.000, +0.000] | no |
| E_harm^(irr) | 25.0% | 144 | -0.005 [-0.008, -0.003] | +0.001 [-0.005, +0.007] | no |
| S_H^(match) | 25.0% | 144 | -0.003 [-0.003, -0.003] | -0.005 [-0.005, -0.005] | yes |
| S_Q^(match) | 25.0% | 144 | -0.003 [-0.003, -0.003] | -0.006 [-0.006, -0.006] | yes |
| I^(match) | 25.0% | 144 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | yes |
| E_harm^(match) | 25.0% | 144 | -0.004 [-0.004, -0.004] | -0.006 [-0.006, -0.006] | yes |
| T_irr | 25.0% | 144 | +0.000 [+0.000, +0.000] | -0.001 [-0.001, -0.000] | no |
| T_match | 25.0% | 144 | +0.000 [+0.000, +0.000] | -0.000 [-0.000, -0.000] | no |
| S_H^(mis) | 50.0% | 288 | -0.004 [-0.005, -0.003] | +0.001 [-0.001, +0.002] | no |
| S_Q^(mis) | 50.0% | 288 | -0.004 [-0.005, -0.004] | +0.001 [-0.001, +0.002] | no |
| I^(mis) | 50.0% | 288 | +0.000 [+0.000, +0.000] | +0.000 [-0.000, +0.000] | yes |
| E_harm^(mis) | 50.0% | 288 | -0.005 [-0.006, -0.004] | +0.001 [-0.001, +0.003] | no |
| S_H^(irr) | 50.0% | 288 | -0.004 [-0.006, -0.003] | +0.001 [-0.001, +0.003] | no |
| S_Q^(irr) | 50.0% | 288 | -0.005 [-0.006, -0.004] | +0.001 [-0.002, +0.003] | no |
| I^(irr) | 50.0% | 288 | +0.000 [+0.000, +0.000] | +0.000 [-0.000, +0.000] | yes |
| E_harm^(irr) | 50.0% | 288 | -0.005 [-0.006, -0.003] | +0.001 [-0.001, +0.003] | no |
| S_H^(match) | 50.0% | 288 | -0.005 [-0.005, -0.005] | -0.001 [-0.001, -0.001] | yes |
| S_Q^(match) | 50.0% | 288 | -0.005 [-0.005, -0.005] | -0.001 [-0.001, -0.001] | yes |
| I^(match) | 50.0% | 288 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | yes |
| E_harm^(match) | 50.0% | 288 | -0.006 [-0.006, -0.006] | -0.001 [-0.001, -0.001] | yes |
| T_irr | 50.0% | 288 | +0.000 [+0.000, +0.000] | +0.000 [-0.000, +0.000] | yes |
| T_match | 50.0% | 288 | +0.000 [+0.000, +0.000] | -0.000 [-0.000, -0.000] | no |

## Behaviour rates

### 12.5%

| Run | Correct | Alternative | Other |
| --- | ---: | ---: | ---: |
| blind|Q | 0.000 | 0.000 | 1.000 |
| full|Q | 1.000 | 0.000 | 0.000 |
| full|irr | 1.000 | 0.000 | 0.000 |
| full|match | 1.000 | 0.000 | 0.000 |
| full|mis | 1.000 | 0.000 | 0.000 |
| pw|A | 0.000 | 0.000 | 1.000 |
| pw|B_irr | 0.000 | 0.000 | 1.000 |
| pw|B_match | 0.000 | 0.000 | 1.000 |
| pw|B_mis | 0.000 | 0.000 | 1.000 |
| pw|C_irr | 0.000 | 0.000 | 1.000 |
| pw|C_match | 0.000 | 0.000 | 1.000 |
| pw|C_mis | 0.000 | 0.000 | 1.000 |
| pw|D_irr | 0.000 | 0.000 | 1.000 |
| pw|D_match | 0.000 | 0.000 | 1.000 |
| pw|D_mis | 0.000 | 0.000 | 1.000 |

### 25.0%

| Run | Correct | Alternative | Other |
| --- | ---: | ---: | ---: |
| blind|Q | 0.000 | 0.000 | 1.000 |
| full|Q | 1.000 | 0.000 | 0.000 |
| full|irr | 1.000 | 0.000 | 0.000 |
| full|match | 1.000 | 0.000 | 0.000 |
| full|mis | 1.000 | 0.000 | 0.000 |
| pw|A | 0.000 | 0.000 | 1.000 |
| pw|B_irr | 0.000 | 0.000 | 1.000 |
| pw|B_match | 0.000 | 0.000 | 1.000 |
| pw|B_mis | 0.000 | 0.000 | 1.000 |
| pw|C_irr | 0.000 | 0.000 | 1.000 |
| pw|C_match | 0.000 | 0.000 | 1.000 |
| pw|C_mis | 0.000 | 0.000 | 1.000 |
| pw|D_irr | 0.000 | 0.000 | 1.000 |
| pw|D_match | 0.000 | 0.000 | 1.000 |
| pw|D_mis | 0.000 | 0.000 | 1.000 |

### 50.0%

| Run | Correct | Alternative | Other |
| --- | ---: | ---: | ---: |
| blind|Q | 0.000 | 0.000 | 1.000 |
| full|Q | 1.000 | 0.000 | 0.000 |
| full|irr | 1.000 | 0.000 | 0.000 |
| full|match | 1.000 | 0.000 | 0.000 |
| full|mis | 1.000 | 0.000 | 0.000 |
| pw|A | 0.000 | 0.000 | 1.000 |
| pw|B_irr | 0.000 | 0.000 | 1.000 |
| pw|B_match | 0.000 | 0.000 | 1.000 |
| pw|B_mis | 0.000 | 0.000 | 1.000 |
| pw|C_irr | 0.000 | 0.000 | 1.000 |
| pw|C_match | 0.000 | 0.000 | 1.000 |
| pw|C_mis | 0.000 | 0.000 | 1.000 |
| pw|D_irr | 0.000 | 0.000 | 1.000 |
| pw|D_match | 0.000 | 0.000 | 1.000 |
| pw|D_mis | 0.000 | 0.000 | 1.000 |
