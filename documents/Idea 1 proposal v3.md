# Does unreliable text steer visual-KV selection?

**Idea 1 — research proposal, v3 (7 November 2026)**

**Abstract.** Text-conditioned compression of a VLM's visual KV cache scores image entries with attention from the prompt. If the prompt carries an unreliable description, that description can change which image entries are kept. We intervene separately on the context used to *select* visual entries and the context used to *answer*, holding the image prefix, budget and answering text fixed. This isolates whether the description harms image-grounded answering through selection, whether the harm is specific to false content, and whether selecting with the trusted question alone repairs it.

**Scope.** The pipeline is a replay-based compression policy: probe, compress, then process the answering text against the compressed cache. Conclusions are about this policy, not about standard one-pass SnapKV or FastV execution. One model (LLaVA-1.5-7B), two dataset settings, and a population of items the uncompressed model answers correctly.

## 1. Background

### How LLaVA-1.5 reads an image

LLaVA-1.5-7B encodes a 336 × 336 image with a CLIP vision encoder into a 24 × 24 grid of patch features. A projector turns each patch into one *visual token*, so every image becomes 576 tokens in the language model's input, placed before the text. The language model (Vicuna-7B, 32 layers, 32 attention heads per layer) then reads the text tokens, which attend to the visual tokens.

### The KV cache and its compression

While processing a prompt, every layer stores a key vector and a value vector for each token. These stored vectors are the *KV cache*. Later tokens, including every generated answer token, can read the image only through these stored entries.

Visual entries dominate the cache: 576 of roughly 620 prompt tokens. Compression keeps only K of the 576 visual entries and drops the rest, so later tokens can attend only to the kept ones.

### How compression chooses what to keep

Most training-free methods score each visual entry by the attention it receives from text tokens, and keep the top K. FastV uses the last prompt token's attention at an early layer. SnapKV uses attention from a window of the final prompt tokens. Text-agnostic alternatives ignore the prompt: a uniform grid, a random set, or scores from the vision encoder alone.

### The problem

Prompts often carry auxiliary text besides the question: a caption, OCR output, retrieved notes or a user's claim. That text may be wrong. With text-conditioned selection, it becomes part of what scores the visual entries.

**Example.** An image shows a red bus. The question is "What colour is the bus?" The prompt also says "The bus in this image is blue." With the full cache the model answers "red"; after compression it answers "blue". Two explanations are possible:

- The decoder trusted the text more once the visual evidence was thinner.
- The description changed which visual entries were kept, removing evidence that would have refuted it.

Comparing compressed runs with and without the description cannot tell these apart, because the description changes both the selection and the text the model answers from.

### The design idea

The project separates the two roles of the prompt. The **selection context** is the text used to score and pick visual entries. The **answering context** is the text the model reads when it answers. Crossing the two gives four runs per item. The key comparison keeps the answering text identical and changes only which visual entries were kept.

To make "dropped" really mean dropped, selection runs as a separate **probe** whose text states are thrown away. The answering text is then processed afresh against the compressed cache, a step called **replay**. Without replay, text tokens processed before compression would already carry information from entries that were later dropped.

&#91;embedded content: Experimental pipeline · 1 shared prefix, 2 probes, 4 cells per description type\]

B\_c is what a normal text-conditioned pipeline does; C\_c keeps the same answering text but selects with the question alone. Section 3 defines every symbol in the figure.

## 2. Motivation and contribution

VLMs often follow conflicting text over the image ([Words or Vision](https://openaccess.thecvf.com/content/CVPR2025/html/Deng_Words_or_Vision_Do_Vision-Language_Models_Have_Blind_Faith_in_CVPR_2025_paper.html), CVPR 2025; [ConText-VQA](https://aclanthology.org/2026.eacl-long.323), EACL 2026). Separately, visual-token compression increases reliance on textual priors ([PAA-Bench](https://arxiv.org/abs/2610.24740); [COAST](https://arxiv.org/html/2605.09429)), and text-guided scoring is known to be misaligned with visual evidence ([VisionDrop](https://arxiv.org/html/2506.22283v3); [PriorTR](https://arxiv.org/pdf/2606.24156); [DeFT](https://arxiv.org/html/2609.34861v1)). In text-only LLMs, KV eviction degrades some instructions far more than others and is sensitive to instruction order ([The Pitfalls of KV Cache Compression](https://aclanthology.org/2026.acl-long.1926.pdf), ACL 2026).

What no prior work isolates is the pathway between these findings. When auxiliary text is part of the selector's input, does it change *which visual evidence the model can still see*, and does that, by itself, harm the answer?

**Contribution.** We intervene separately on selection context and answering context to test whether unreliable auxiliary text harms image-grounded answering through visual-KV selection. The design has three parts:

1. A crossed selection × answering experiment with fixed answer candidates, run for misleading, matching and irrelevant descriptions, so misinformation-specific effects can be separated from the effect of adding any description.
2. A validated replay pipeline whose compact cache is tested against a dense masked reference.
3. A targeted, assumption-explicit mitigation, question-only selection with an optional coverage reserve, evaluated for both robustness and clean-quality cost.

## 3. Notation and terms

Every symbol used later is defined here; sections 5–8 give full detail.

### Prompt and pipeline

| Term | Meaning |
| --- | --- |
| Prefix | The tokens up to and including the last image token: BOS (the beginning-of-sequence token), ` USER:  ` and the 576 visual tokens. Computed once per item and shared by all runs |
| Suffix | All tokens after the prefix: optional description, authority line, question, answer instruction, `ASSISTANT:` |
| Q | The suffix with **no** description (question only) |
| c | Description type: **mis** (misleading, asserts a wrong answer), **match** (matching, asserts the right answer) or **irr** (irrelevant, a true statement about something else) |
| H\_c | The description sentence of type c, e.g. H\_mis = "The bus in this image is blue." |
| QH\_c | The suffix Q with the line `Description: H_c` inserted. QH\_mis, QH\_match and QH\_irr differ from Q only in that line |
| Visual entry | The stored key and value vectors of one visual token, in every layer |
| K | Budget: the number of visual entries kept, out of 576 |
| Selector | A rule that scores the 576 visual entries and keeps K. *Text-conditioned* selectors use attention from the suffix; *text-agnostic* ones (uniform grid, random) do not |
| PW-attn, LT-early | The two text-conditioned selectors (section 6): attention from the final prompt window, and last-token early-layer attention |
| Probe | A forward pass of a suffix over a copy of the prefix cache, used only to score visual entries; its own states are then discarded |
| Selection context | The suffix used in the probe (Q or QH\_c). Selecting with Q is called Q-only selection; selecting with the actual prompt QH\_c is called complete-prompt selection |
| Answering context | The suffix used in replay (Q or QH\_c) |
| S\_Q | The K entries kept when the probe uses Q |
| S\_c | The K entries kept when the probe uses QH\_c (S\_mis, S\_match, S\_irr) |
| Compact cache | The prefix cache with only the non-visual entries and the kept visual entries |
| Replay | Processing the answering suffix against the compact cache, then scoring answers |
| Dense masked reference | The same computation with all entries kept but dropped visual entries masked out; the specification the compact cache must reproduce |

### Runs and quantities

| Term | Meaning |
| --- | --- |
| Cell A | Select with S\_Q, answer with Q |
| Cell B\_c | Select with S\_c, answer with QH\_c: what a normal text-conditioned pipeline does |
| Cell C\_c | Select with S\_Q, answer with QH\_c: same answering text as B\_c, different kept set |
| Cell D\_c | Select with S\_c, answer with Q |
| Full\_Q, Full\_c | No compression, answering with Q or QH\_c |
| a\_true, a\_alt | Two answer candidates fixed per item: the image-supported answer, and a predetermined wrong alternative (the one H\_mis asserts) |
| M | Margin: log p(a\_true) − log p(a\_alt) given the run's context, in **nats** (natural-log units). Higher is better |
| S\_H^(c) | M(C\_c) − M(B\_c): the benefit of selecting with Q instead of QH\_c when answering with QH\_c. The primary quantity is S\_H^(mis) |
| T\_match, T\_irr | S\_H^(mis) − S\_H^(match) and S\_H^(mis) − S\_H^(irr): how much of the benefit is specific to false content |
| S\_Q^(c), I^(c), E\_harm^(c) | Carry-over harm, answering-context dependence and extra compression damage (section 5) |
| E | Evidence patches: the grid cells covering the annotated answer region |
| ERR | Annotated-region retention: the share of E that a kept set contains |
| Jaccard | Overlap of two kept sets: size of intersection ÷ size of union (1 = identical) |

### Data and statistics

| Term | Meaning |
| --- | --- |
| Eligible item | One the uncompressed model answers correctly with Q, with a positive margin |
| Development set | Items used for building and debugging; never used for claims |
| Confirmatory main set | Held-out items run once after the freeze, with misleading descriptions |
| Control subset | 300 confirmatory items per dataset run with all three description types |
| Grounding subset | Control-subset items with a hand-verified evidence box |
| Freeze | The dated commit of `PREREG.md` fixing every choice before confirmatory runs |
| Image-clustered paired bootstrap | Confidence intervals from resampling *images* (with all their questions), since questions on the same image are not independent |
| Holm correction | A standard procedure that adjusts several tests at once so the chance of any false positive stays at 5% |
| δ | 0.10 nats: the smallest effect treated as practically meaningful |
| fp16, eager attention | Half-precision weights; the attention implementation that returns attention weights, needed by the probes |

## 4. Research questions

**RQ.** Does misleading auxiliary text change visual-KV selection in ways that harm subsequent image-grounded answering?

**RQ-a.** Is that harm specific to false content, or does any added description produce it?

**RQ-b.** Does compression increase susceptibility to misleading text beyond its ordinary damage on the same items?

**RQ-c.** When the selection context changes the answer, is that associated with lower retention of annotated evidence regions?

### Ladder of claims

Each rung requires its own evidence. Rungs 1–4 are the core contribution; rung 5 is mechanism evidence; rung 6 is an optional strengthening experiment.

| Rung | Claim | Evidence required |
| --- | --- | --- |
| 1 | The description changes the selected entries | Overlap between S\_Q and S\_c below 1 on a meaningful share of items |
| 2 | That selection change affects answering | S\_H ≠ 0 under otherwise identical replay |
| 3 | The change is harmful | S\_H > 0 for misleading descriptions |
| 4 | The harm is specific to false content | S\_H(misleading) > S\_H(matching) and > S\_H(irrelevant) |
| 5 | The harm is associated with annotated-evidence loss | Retention differences and their association with S\_H on the grounding subset |
| 6 | Losing that evidence contributes causally | Restoring annotated entries at fixed budget recovers more than restoring equal-count non-evidence entries |

## 5. Setting and estimands

### Prompt template `v3.0`

```
USER: <image>
[Description: {description}]                       <- QH_c only
Answer from the image. Any provided description may be inaccurate.
{question}
Answer the question using a single word or phrase. ASSISTANT:
```

Q and QH\_c differ only in the description line. The authority line ("Answer from the image. Any provided description may be inaccurate.") is part of the task definition and is identical in every condition. The prefix is every token up to and including the last expanded image token, taken from the fully tokenized prompt; the suffix is the rest.

### Description conditions

c ∈ {mis, match, irr}: a misleading, matching or irrelevant description H\_c. Each gives its own selection set S\_c, scored from a probe with suffix QH\_c. S\_Q is scored from a probe with suffix Q.

### Cells, for each condition c

| Selection set | Answer with Q | Answer with QH\_c |
| --- | --- | --- |
| **S\_Q** | A | C\_c |
| **S\_c** | D\_c | B\_c |

Full-cache anchors: Full\_Q and Full\_c.

### Fixed answer candidates

Each item has two candidates fixed at construction time: a\_true, the image-supported answer, and a\_alt, a predetermined incorrect alternative. The misleading description asserts a\_alt; the matching description asserts a\_true; the irrelevant description asserts neither. The same pair is scored in every cell and condition:

```latex
M(x) = \log p(a_{\text{true}} \mid x) - \log p(a_{\text{alt}} \mid x)
```

Higher M is better everywhere. Behavioural outcomes use the same pair: correct, alternative, or other. The alternative outcome is called *misleading-answer adoption* only under c = mis.

### Estimands

```latex
\begin{aligned}
S_H^{(c)} &= M(C_c) - M(B_c) && \text{benefit of Q-only selection, answering with } H_c \\
T_{\text{match}} &= S_H^{(\text{mis})} - S_H^{(\text{match})}, \quad T_{\text{irr}} = S_H^{(\text{mis})} - S_H^{(\text{irr})} && \text{misinformation-specific part} \\
S_Q^{(c)} &= M(A) - M(D_c) && \text{benefit of Q-only selection, answering without } H_c \\
I^{(c)} &= S_H^{(c)} - S_Q^{(c)} && \text{dependence on the answering context} \\
E_{\text{harm}}^{(c)} &= \big(M(\text{Full}_c) - M(B_c)\big) - \big(M(\text{Full}_Q) - M(A)\big) && \text{extra compression damage under } H_c
\end{aligned}
```

The primary estimand is S\_H^(mis). E\_harm decomposes exactly into the selection-context contribution S\_H^(c) plus a remaining differential degradation, (M(Full\_c) − M(C\_c)) − (M(Full\_Q) − M(A)), which still involves compression and its interaction with the answering text.

### What the comparisons establish

B\_c vs C\_c substitutes one kept set for another in this pipeline, with identical answering text and budget. Adding a description also changes token count, positions and the contents of the selector's prompt window (section 6), so S\_H^(mis) alone cannot separate false content from added context. T\_match and T\_irr provide that separation, within the limits of the description templates. None of these comparisons shows that attention is a faithful explanation or that other pipelines share the effect.

## 6. Method

### Pipeline

1. **Prefix.** Prefill BOS, ` USER:  ` and the 576 expanded image tokens once; keep the prefix KV cache.
2. **Probe.** Run a probe suffix on a copy of the prefix cache and record a score for each of the 576 visual positions. Discard the probe's cache entirely.
3. **Select.** Derive kept sets for every budget and reserve from the stored scores.
4. **Compress.** Gather the non-visual prefix entries and the kept visual entries in every layer.
5. **Replay.** Process the answering suffix against the compact cache; score the two candidates and greedy-decode.

| Purpose | Probes run |
| --- | --- |
| Research comparison | Q and every QH\_c, to construct the counterfactual kept sets |
| Deployed complete-prompt policy | QH\_c only (the actual prompt) |
| Deployed Q-only policy | Q only |

A deployed policy's cost is one probe, compression and replay. That is what the cost comparison reports, not the cost of the experimental grid.

### Dense masked reference

The compact cache must reproduce a dense reference, defined operationally:

- The prefix is computed normally, with no mask. Its states are never recomputed.
- All prefix entries stay physically present. When suffix or generated queries attend to the prefix, discarded visual keys receive an additive −∞ mask.
- The suffix uses its normal positions, continuing from the original prefix length.

The compact implementation must satisfy these invariants:

- K and V are gathered with exactly the same indices.
- Kept entries stay in original sequence order.
- Every non-visual prefix entry is kept.
- Suffix and generated tokens keep their original logical positions.
- The same kept set is applied in every layer.

### Selectors

**PW-attn (primary): attention from the final prompt window.** Inspired by SnapKV; not a reproduction of it.

```latex
s_i = \mathrm{Pool}_{3\times3}\!\left(\frac{1}{|\mathcal L||\mathcal H||\mathcal W|}\sum_{\ell\in\mathcal L}\sum_{h\in\mathcal H}\sum_{t\in\mathcal W} \alpha^{(\ell,h)}_{t,i}\right)_{i}
```

**LT-early (secondary): last-token early-layer attention.** Inspired by FastV.

```latex
s_i = \frac{1}{|\mathcal H|}\sum_{h\in\mathcal H} \alpha^{(2,h)}_{t_{\text{last}},\,i}
```

Symbols: s\_i is the score of visual entry i (i = 1, …, 576). α^(ℓ,h)\_{t,i} is the attention probability from suffix token t to visual entry i in layer ℓ and head h. 𝓛, 𝓗 and 𝓦 are the sets of layers, heads and window tokens used; t\_last is the final suffix token. The K highest scores form the kept set.

| Choice | Value |
| --- | --- |
| α | Softmax attention probabilities over all available keys, not renormalised over visual keys |
| Layers 𝓛 | Zero-based. PW-attn: 2–31. LT-early: layer 2 only |
| Heads 𝓗 | All 32, equal weight |
| Window 𝓦 | Last 16 tokens of the serialized probe suffix, ending at `ASSISTANT:` |
| Pooling | 3 × 3 mean on the 24 × 24 grid, stride 1; edge cells average over in-image neighbours only |
| Ties | Lower position index first |
| Output | One kept set shared by all layers and heads |

The window is contextualized by the question and description but consists largely of fixed instruction tokens, so PW-attn is attention from the final prompt window, not from the question. The exact window span is logged per item. During development, check whether question tokens enter the window, whether the description does for short questions, and whether this differs across conditions. A question-span selector is an optional ablation once PW-attn works.

**Reference selectors (text-agnostic).** Uniform grid, and random with seeds 0–4. For these, S\_Q = S\_c by construction.

### Budgets and coverage reserve

- K ∈ {288, 144, 72} (50%, 25%, 12.5%); primary K = 144.
- Reserve fraction r = 10%: k\_r = round(rK), giving 29, 14 and 7. Take the top K − k\_r entries by score, then add entries from the coverage ordering, skipping any already kept, until exactly K are kept.
- **Coverage ordering:** farthest-point order on the 24 × 24 grid. Start at cell (11, 11); repeatedly add the cell with the largest Euclidean distance to its nearest chosen cell; break ties by row-major index. Every prefix of this order is spatially spread, not just the full permutation. The order is computed once and committed.
- Before freezing, plot the reserve alone and the complete kept set for each k\_r, and report the coverage radius (largest distance from any grid cell to the nearest kept cell) against a random set of the same size.

### Mitigation

All four variants answer with QH\_c and keep exactly K entries.

|  | No reserve | Reserve r = 10% |
| --- | --- | --- |
| **Complete-prompt selection (S\_c)** | Existing behaviour (B\_c) | Coverage only |
| **Q-only selection (S\_Q)** | Isolation only (C\_c) | Isolation + coverage |

Uniform selection is the reference: if it matches Q-only selection, attention scoring is not justified under conflict. The method assumes the question is trusted and the description may not be; false-premise questions violate this assumption and are out of scope.

### Oracle restoration (rung 6, optional)

On grounding-subset items where S\_Q keeps annotated evidence entries that S\_mis omits:

1. **Shared removal set.** Remove the m lowest-scoring non-evidence entries of S\_mis (scored by the QH\_mis probe).
2. **Treatment.** Add the m omitted evidence entries with the highest S\_Q scores.
3. **Control.** Add m omitted non-evidence entries chosen at random (seeds 0–4).
4. Both arms keep exactly K entries; m ∈ {4, 8} is fixed.

Items with fewer than m omitted evidence entries are excluded for that m, and qualifying counts are reported. The result is an intervention on entries associated with annotated regions, not proof that each entry carries only local evidence.

## 7. Data

**Model.** `llava-hf/llava-1.5-7b-hf`, fp16, eager attention. Images are padded to square as in the original LLaVA-1.5 evaluation; the processor is configured not to centre-crop.

### Setting 1: GQA direct attributes

GQA pairs real photographs with questions generated from *scene graphs*: annotated objects with names, attributes, relations and bounding boxes. Each question comes with a semantic program, the sequence of reasoning steps that produces its answer.

- **Questions:** semantic program `select(object) → query(color|material)` with no relate step, so the queried object is the answer object and its scene-graph box is the evidence.
- **Referent uniqueness:** exclude the item if the scene graph has another object with the same name.
- **a\_alt:** from the same vocabulary (colours or materials); not the true value, not any attribute of the same object, not a word in the question.
- **Conflict type:** `same_object_attribute`. The description names the right object with a wrong value.

### Setting 2: TextVQA scene text

TextVQA asks questions about text in natural images. Each image comes with OCR (optical character recognition) tokens and their bounding boxes, and each question has 10 human answers.

- **Questions:** the most common answer (≥ 3 of 10 annotators) matches one complete OCR token or a contiguous span of complete tokens after normalisation; substring-only matches are excluded.
- **Repeated answers:** the evidence set is the union of all matching locations.
- **Answers located through context** (a logo, object or neighbouring label): kept for answer outcomes, flagged, and excluded from the grounding subset.
- **a\_alt:** a complete OCR token that appears exactly once in the same image, has the same character class, length within ±2, and is not equal to any accepted answer after normalisation.
- **Conflict type:** `other_region_text`. The description points to a string elsewhere in the image.

Dataset and conflict type are confounded by construction. Findings are reported per dataset–conflict setting; any comparison between conflict types is exploratory.

### Descriptions

| Condition | GQA | TextVQA |
| --- | --- | --- |
| Misleading | "The {object} in this image is {a\_alt}." | "The relevant text in this image reads '{a\_alt}'." |
| Matching | "The {object} in this image is {a\_true}." | "The relevant text in this image reads '{a\_true}'." |
| Irrelevant | "The {other object} in this image is {its true value}." (non-overlapping object) | "Other text in this image reads '{other OCR string}'." (not a\_alt, not an answer) |

Matching and misleading differ only in the value. Token lengths across the three conditions are recorded; differences within ±3 tokens are accepted, and no description is padded to force equality.

### Sets

| Set | Content | Conditions |
| --- | --- | --- |
| Development | 200 items per dataset, disjoint images | All three |
| Confirmatory main | All remaining eligible items, up to 1,000 per dataset | Misleading |
| Control subset | 300 confirmatory items per dataset, drawn with seed 0 before any outcome is seen | All three |
| Grounding subset | Control-subset items with hand-verified, non-empty evidence | All three |

Splits are by image id. **Eligibility** is conditional: Full\_Q gives a correct greedy answer and a positive margin. Results describe this population; a secondary analysis drops the margin condition. Funnel counts and answer-type distributions are reported.

### Verification

Label quality matters because some effects may be a few points. Human verification is prioritised where errors would bias the estimands:

- **Fully hand-verified:** every development item and every control-subset item, for falsity of the misleading statement, referent uniqueness in the image, a\_alt not being an acceptable answer, minimal-pair wording, and the evidence box.
- **Audited:** a random 100 of the remaining confirmatory items per dataset. Report the error rate with a 95% interval. If it exceeds 5%, fix the builder and rebuild before the freeze.
- Failed items are removed, not repaired by hand, so the construction stays reproducible.

## 8. Measurements

| Measure | Definition |
| --- | --- |
| Margin M (primary) | log p(a\_true ∣ x) − log p(a\_alt ∣ x); candidate content plus `</s>`, summed over tokens |
| Margin sensitivity | Content tokens only; length-normalised. Reported, never substituted for the primary score |
| Behaviour | Greedy answer after dataset normalisation: correct, a\_alt, or other |
| Misleading-answer adoption | Share of a\_alt answers under c = mis |
| Annotated-region retention | ERR(S) = ∣S ∩ E∣ / ∣E∣ on the grounding subset |
| Set overlap | Jaccard for S\_Q vs S\_mis, S\_Q vs S\_match, S\_Q vs S\_irr, S\_mis vs S\_match |
| Window composition | Share of window tokens from question, description and fixed instruction, per condition |
| Cost per deployed policy | Probe, compression and replay time; retained visual-KV bytes; total prompt-KV bytes; peak allocated and reserved memory |

**Candidate tokenization.** Tokenize prompt and prompt + candidate; assert that the first is an exact prefix of the second before slicing out the candidate's tokens. Items that fail are inspected; if a stable delimiter fixes them, it is added to the template for all items.

**TextVQA scoring.** Exact match on a\_true is the project metric; the standard multi-reference VQA score is reported alongside it.

**Evidence patches.** Boxes are mapped through the same resize and pad onto the 24 × 24 grid of 14-pixel patches.

| Rule | Use |
| --- | --- |
| Patch overlaps the box by ≥ 25% of its area | Primary |
| If that is empty: patches containing the box centre and corners | Tiny-box fallback |
| Patch centre inside the box | Sensitivity |
| Box expanded by one patch on each side | Sensitivity |

ERR measures retention of *annotated regions*. Annotations may omit context, and kept KV entries are contextual representations rather than raw patches.

**Timing.** Cost measurements run with test instrumentation (cache hashing, extra logging) switched off.

## 9. Hypotheses and evaluation questions

Text-conditioned selectors only. Each hypothesis is directional; section 10 defines how its interval is read.

| ID | Hypothesis | Set |
| --- | --- | --- |
| H1 (primary) | S\_H^(mis) > 0: Q-only selection improves answering under a misleading description | Confirmatory main |
| H2 | T\_match > 0 and T\_irr > 0: the benefit is larger for misleading than for matching or irrelevant descriptions | Control subset |
| H3 | E\_harm^(mis) > 0: compression damages the misleading condition more than the neutral one | Confirmatory main |
| H4 | S\_Q^(mis) > 0: the misleading-selected set harms even when answering without the description | Confirmatory main |
| H5 | Items with lower Jaccard(S\_Q, S\_mis) show larger S\_H^(mis) | Confirmatory main |
| H6 | ERR(S\_Q) > ERR(S\_mis) | Grounding subset |

**Exploratory, no directional prediction:** the signs of S\_H^(match) and S\_H^(irr); I^(c); whether H6 differs between the two dataset settings.

**Evaluation questions (measured, not hypothesised):**

- **EQ1.** What robustness gain does each mitigation variant give on misleading items, and at what cost on neutral (Q-only answering) and matching items? Isolation alone cannot change neutral quality, because A already uses S\_Q; the reserve can, so its cost is measured.
- **EQ2.** What does each deployed policy cost in time and memory?
- **EQ3 (optional).** Does oracle restoration of annotated entries recover more margin than equal-count non-evidence restoration?

## 10. Analysis plan

### Primary analysis

| Element | Specification |
| --- | --- |
| Estimand | S\_H^(mis), mean per-item M(C\_mis) − M(B\_mis), in nats |
| Configuration | PW-attn, K = 144, confirmatory main set, each dataset separately |
| Inference | Image-clustered paired bootstrap: resample images with replacement, keep all their items, recompute the mean; 10,000 resamples, seed 0 |
| Test | One-sided bootstrap p-value, p = (1 + #{S\_H\* ≤ 0}) / (10,001) per dataset, where S\_H\* is the estimate in one bootstrap resample |
| Multiplicity | Holm across the two dataset tests, family-wise α = 0.05 |
| Reported | Point estimate and 95% percentile interval per dataset; Holm decision per dataset |
| Conclusion | Each dataset receives its own conclusion. An effect is claimed as general only if both datasets pass |

### Reading an interval

| Interval pattern | Interpretation |
| --- | --- |
| Entirely above 0 | Evidence for a positive effect |
| Includes 0 and practically meaningful positive values | Inconclusive: the design did not estimate the effect precisely enough to support superiority or equivalence |
| Entirely within ±δ | Evidence that the effect is practically small under that criterion |
| Entirely below 0 | Evidence for an effect in the opposite direction |

### Equivalence (secondary)

δ = 0.10 nats, a change of about 10% in the relative odds of a\_true over a\_alt. δ is also reported as a fraction of the mean Full\_Q margin on development items, so its practical size is visible. Equivalence is tested (90% clustered interval inside ±δ) only if the development-projected 90% half-width is below δ / 2; otherwise it is not attempted. A finding of equivalence means no practically important selection-context effect was detected in this configuration. It does not mean compression is safe at other budgets, on other data, or in other pipelines.

### Secondary analyses

| Family | Contents | Multiplicity |
| --- | --- | --- |
| Specificity | T\_match, T\_irr per dataset, control subset | Holm over 4 tests |
| Compression and carry-over | E\_harm^(mis), S\_Q^(mis), I^(mis) per dataset | Holm over 6 tests |
| Robustness of the primary | K = 288 and 72; LT-early; content-only and length-normalised margins; less-filtered population | Reported with intervals, no claims |
| Behaviour | Paired differences in correct and a\_alt rates between B\_mis and C\_mis, image-clustered | Reported with intervals |
| Mechanism | Per-item S\_H^(mis) regressed on (1 − Jaccard), image-clustered standard errors; ERR(S\_Q) − ERR(S\_mis) under all four mapping rules | Reported with intervals |
| Mitigation (EQ1) | Four factorial variants and uniform, on misleading, neutral and matching items of the control subset | Reported with intervals |
| Restoration (EQ3) | Treatment − control margin gain per m | Reported with intervals |

A selection × answering × budget × selector mixed model is optional and descriptive.

### Planning precision

Estimate the image-clustered SD of per-item S\_H^(mis) on development items and project interval widths for the confirmatory sample. These projections are planning numbers, not achieved precision.

### Confirmatory discipline

- All choices in sections 5–10 are frozen in `PREREG.md` before the confirmatory run.
- The confirmatory sets are evaluated once. No change is made because of an outcome.
- Bugs found later are fixed outcome-blind: the fix is documented with its date, and every affected condition is rerun in full.
- Any added data or changed setting after the freeze is labelled a follow-up.

## 11. Validation

The acceptance policy is written into `tests/README.md` before the tests first run. Tolerances are never loosened after a failure; a failure is closed only when its cause is identified.

### Level 1: operator correctness (fp32, small model)

A small randomly initialised model with the same decoder architecture (a few layers, small hidden size) runs the same prefix, gather, mask and replay code. Compact and dense-masked execution are compared on random kept sets, several budgets, multiple layers and generated tokens. Pass: maximum absolute logit difference below 1e-5.

### Level 2: LLaVA pipeline (fp16, on the 20 GB GPU)

**Numerical baseline.** Compare two mathematically equivalent dense computations on 50 development items: one full-sequence forward pass, and prefix-then-suffix with the full cache. The distribution of logit and margin differences is the fp16 noise floor.

| ID | Test | Pass |
| --- | --- | --- |
| E1 | Serialization: identical prefix tokens for Q and every QH\_c; template round-trips; prompt tokens are an exact prefix of prompt + candidate; window span logged | Exact, on all development items |
| E2 | Full fidelity: compact replay at K = 576 vs one full forward pass | Within the noise floor |
| E3 | Sparse equivalence: compact vs dense-masked for irregular random, spatially concentrated and scored kept sets; K = 288, 144, 72; Q and QH\_mis suffixes; 10 generated tokens | See criteria below |
| E4 | Cache immutability: hash the shared prefix cache before and after each probe and replay (tests only) | Hashes unchanged |
| E5 | Placebo: identical kept indices and suffix ids in B\_c and C\_c | Equal within the noise floor; identical answers |
| E6 | Selectors: known top-K on hand-built attention; edge pooling; ties; exactly K unique entries with and without reserve; coverage-ordering plots and radius | All pass |
| E7 | Determinism: fixed seeds give identical random sets; rerunning an item reproduces its row | Identical |

**E3 criteria.**

- The 99th percentile of compact-vs-dense logit differences is at most twice that of the noise floor.
- Mean absolute margin difference is below 0.01 nats, one tenth of δ.
- Every greedy-answer mismatch is investigated. It is accepted only as a near tie: the top-two logit gap at the diverging step is below twice the noise floor's maximum. Any other mismatch is a failure.

**If E2 or E3 fails,** debug against the dense reference, or pin a different library version. Re-prefilling only the kept visual tokens is not a substitute, because it recomputes their states without the omitted tokens. If cache surgery cannot be made correct, the project becomes a separately labelled token-pruning study with revised claims.

### Data and sanity checks

| ID | Check |
| --- | --- |
| D1 | Hand verification and audit, section 7 |
| D2 | Evidence-patch overlays under all four mapping rules for every grounding-subset item |
| D3 | Funnel, answer types and description token lengths recorded before the freeze |
| S1 | Unfiltered random 200 development questions with LLaVA's standard prompt; accuracy in a plausible range. Detects implementation and prompt bugs only |
| S2 | Full-cache susceptibility on development items: Full\_mis adoption and margin. Reported; the template does not change in response |
| R1 | Per-stage time and peak allocated/reserved memory for each deployed policy, short and long suffixes. The confirmatory runtime estimate comes from these numbers |

## 12. Phases and schedule

Development uses only development items and may fix bugs, prompts and builders. Confirmation starts after the freeze and runs once.

| Phase | Dates | Work | Gate |
| --- | --- | --- | --- |
| P0 Engineering | 7–11 Nov | Level 1, then E2–E3 on single items; then E1, E4–E7 | All validation tests pass |
| P1 Data | 7–12 Nov (GQA), to 14 Nov (TextVQA) | Builders; development set hand-verified; control subset drawn | D1 for the development set; D3 started |
| P2 Development pilot | 12–14 Nov | B\_c vs C\_c for all three description types on development items; window composition; overlaps; clustered SD; S1, S2, R1 | Measured runtime fits the schedule |
| P2b Control verification | 12–15 Nov | Hand-verify the control subset; audit the main set | D1 and D2 complete |
| P3 Freeze | 15 Nov | `PREREG.md` committed with every frozen choice | Timestamped before any confirmatory output |
| P4 Confirmatory core | 16–20 Nov | PW-attn, K = 144: cells A–D and Full anchors on the main set (misleading) and control subset (all three); uniform reference | Primary analysis computed once on 21 Nov |
| P5 Extensions | 21–27 Nov | Other budgets; LT-early; random seeds; mitigation factorial; ERR and overlap analyses; optional restoration | Each result labelled with its pre-registered family |
| P6 Results freeze | 29 Nov | Every table regenerates from raw results with one command | No new runs |

### Minimum scope

One model; PW-attn; GQA setting; K = 144; cells A–D with Full anchors; uniform reference; matching and irrelevant controls on the control subset; Q-only vs complete-prompt selection with one 10% reserve; tests E1–E7; image-clustered analysis.

### Cut order

1. Oracle restoration.
2. K = 288 and 72.
3. LT-early and extra random seeds.
4. TextVQA setting.

The matching and irrelevant controls are part of the core and are kept ahead of every secondary analysis.
