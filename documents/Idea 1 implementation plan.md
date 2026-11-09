# Idea 1: implementation plan and design

**Implements:** `Idea 1 proposal v3.md` · **Written:** 7 November 2026

Symbols (Q, QH\_c, S\_Q, S\_c, cells A–D, M, S\_H and so on) mean what section 3 of the proposal says. Section numbers written as "§" refer to the proposal.

## 1. Summary

The build has four parts, in dependency order:

1. **Replay engine.** Prefix, probe, select, compress, replay and score. It is the only code that touches the model, and it must match the dense masked reference.
2. **Dataset builders and a verification tool.** These turn GQA and TextVQA into frozen item manifests, each with a\_true, a\_alt, three descriptions and evidence boxes.
3. **Experiment runner.** It runs the grid of cells for each item, can resume after a crash, and writes append-only results. A guard stops confirmatory runs unless the freeze is in place.
4. **Analysis package.** One command regenerates every table from raw results.

Two conclusions shaped the plan:

- **Human verification limits the schedule, not the GPU.** The proposal requires 1,200 hand-checked items. The GPU work looks like hours (section 9).
- **Eight decisions should be settled before the tests first run or before the freeze** (section 3). The first one affects what the primary experiment measures.

### Build order: minimal version first

You run the GPU jobs and do the hand verification. The first version is a vertical slice that gets a development pilot on GQA running. This is narrower than §12's Minimum scope, which it grows into.

| In the first version | Added later |
| --- | --- |
| Replay engine, compact and dense | Verification tool (next) |
| PW-attn, question-span, uniform, 10% reserve; K = 144 by default | LT-early, random seeds, restoration |
| GQA builder and funnel | TextVQA |
| Eligibility screen and pilot, development items only | Confirmatory runs: the runner refuses non-development items until the freeze guard exists |
| Level 1, E1–E7, noise floor under both heads | R1 timing, D1–D3, coverage plots |
| Estimands, clustered bootstrap, development report | H5 regression, ERR and evidence mapping, `make results` |

Sections 4–10 describe the full design. Module names in the first version are simpler than the layout in section 4 and will split as they grow.

**GQA funnel on `val_balanced` (builder `gqa-2`):** 4,503 questions with the required program → 4,100 in the closed vocabulary → 2,253 with a usable irrelevant object → 2,234 within the ±3 token rule, on 1,799 images and 92% colour. That is 600 development candidates and a 1,634-item confirmatory pool, before eligibility and verification.

**TextVQA funnel, train and val pooled (builder `textvqa-2`):** 39,602 questions → 36,178 with at least 3 votes for the modal answer → 7,220 items on 6,359 images, split into 1,500 development candidates and a 5,720-item confirmatory pool, before eligibility and verification. Eligibility is unknown until the screen runs. All images are in the archive.

Builder choices that refine §7 and should go into `PREREG.md`:

- **Closed vocabulary.** a\_true, a\_alt and irrelevant values come from 11 clear colours and 8 materials. Raw GQA answers include activities (`brushing teeth`), vague values (`dark`) and near neighbours (`tan`, `light blue`), which could make a\_alt acceptable.
- **TextVQA fragments.** a\_alt and irrelevant strings can't contain, or be contained in, an accepted answer or a\_alt. §7's rules alone allowed `a_true = toyota` with `a_alt = yota`, an OCR fragment of the right answer.
- **Verb agreement.** Descriptions use "are" when the dataset's question or full answer does, because about 15% of objects look plural and "The pants in this image is white" would fail verification. Irrelevant descriptions use singular objects only. Matching and misleading still differ only in the value.

## 2. What I checked

I checked these against the real model files and a throwaway CPU spike. The scripts are in the session scratchpad, not the repo.

| Fact | Result | Effect on the design |
| --- | --- | --- |
| Prefix layout | 581 tokens: `<s>`, `▁US`, `ER`, `:`, `▁`, then 576 image tokens. Identical for Q and all QH\_c in the samples tried | Non-visual prefix is 5 entries. A compact cache at K = 144 has 149 entries |
| Suffix length | 40–46 tokens for Q, 51–57 for QH\_c in the samples tried | Total ≈ 621, matching the proposal's "roughly 620" |
| **Window `𝓦`** | **The last 16 tokens are always `\n Answer the question using a single word or phrase. ASSISTANT:`** | **No question or description token ever enters it. See decision 1** |
| Window reach | 17 tokens touches the question's `?`; 22–28 covers the whole question; 40–46 reaches the description | Numbers for the alternative selector in decision 1 |
| Candidate delimiter | `prompt + " " + candidate` passes the exact-prefix check on all samples (`▁red`, `▁dark ▁brown`, digits split as `▁ 2 0 1 1`) | A single space is the delimiter |
| Casing changes tokens | `canon` → `▁canon`, but `Canon` → `▁Can on`, and `CANON` → `▁C AN ON` | See decision 2 |
| Checkpoint size | 14.13 GB in fp16 | About 5 GB of headroom on the 20 GB GPU. **Your 16 GB M2 can't run the real model in practice**, so only Level 1 runs locally |
| Memory | Each prefix entry is 0.52 MB across all layers. Prefix cache ≈ 305 MB; K = 144 ≈ 78 MB; probe attention ≈ 93 MB for T = 70 | Fits comfortably, without deep-copying the cache |
| Image processor | The default config has `do_center_crop: true` and `shortest_edge: 336` | It must be overridden explicitly, as §7 requires |
| **Compact vs dense** | Tiny random Llama, fp32, `transformers` 4.57.6, plain `DynamicCache` and a 4D mask: max logit difference ≤ 2.4e-7 at K = 576, 288, 144, 72 with 10 generated tokens | The core mechanism works through public API, with no patching |
| Negative controls | Using physical positions instead of logical ones gives 4.5e-3 (fails, as it should). Perturbing dropped entries in dense gives exactly 0. Perturbing kept entries gives 0.81 | The test can actually detect these bugs |

**Not yet checked:** the same path through `LlavaForConditionalGeneration` in fp16, GQA and TextVQA schemas and funnel sizes, and LLaVA's actual answer casing. All need the GPU or the datasets.

## 3. Decisions to settle

These are gaps or conflicts in the proposal. Each has a recommendation, and the plan assumes that recommendation unless you change it.

### Affects what the study measures

**1. The primary selector's window never sees the question or the description.**
§6 says the window "consists largely of fixed instruction tokens" and asks to check whether question or description tokens enter it. They never do under template `v3.0`: the 16 tokens are the same fixed text on every sample, and the window would need at least 40 tokens to reach the description.

PW-attn is still a text-conditioned selector. Identical token ids don't mean identical query vectors: from layer 2 up, these fixed tokens have already processed the question and description, and those states can change their attention to the image. How strong that effect is remains unmeasured. A null H1 would apply to this selector, model, budget and population, and would not rule out an effect for selectors that query the question directly. The same limit applies to any single selector.

*Decision:* keep PW-attn as primary, as written. Promote the optional ablation to one pre-registered secondary selector, **question-span attention**: the same formula, with 𝓦 set to the question tokens. The query span is the same in Q and QH\_c, and only its context changes. Probes store enough to derive it at no extra probe cost (section 5.3), though scoring its kept sets adds replays. Check rung 1 on development items in P2. Choose on scientific grounds, freeze the choice before held-out comparisons, and never pick by which selector gives the largest effect.

**2. Candidate casing is unspecified, and it may change the margin.**
M scores fixed strings. If LLaVA tends to answer `Red` and a\_true is stored as `red`, M would measure a form the model rarely emits. Whether that happens is an empirical question. The two casings can also differ in token count (`canon` vs `Canon`). Sequence likelihood handles that correctly, so the real question is whether the chosen form represents the answer well.

*Decision:* keep the semantic answer, the scored surface string and behavioural normalisation separate. Every run stores per-token scores for both `lower` and `Capitalized`, so nothing needs rerunning. On development items, measure which form greedy answers use, then fix one deterministic rule per dataset, applied to a\_true and a\_alt alike, before the eligibility screen. Report one sensitivity margin: the log-sum over those two forms.

**3. fp16 logits are coarse relative to the E3 tolerance.**
fp16 has a spacing of 1/64 ≈ 0.016 for values between 16 and 32, where top logits can plausibly sit. E3 requires a mean absolute margin difference below 0.01 nats. The spacing is not a lower bound on that mean, because errors can cancel, but rounding could still matter.

*Decision:* three separate rules, recorded in `tests/README.md`:

- The backbone, cache and attention stay in fp16.
- Log-probabilities and their sums are always computed in fp32.
- The output head defaults to an fp32 copy of `lm_head` applied to the fp32-cast hidden state (about 525 MB). This reduces rounding in the projection only. It can't repair rounding already in fp16 hidden states, so it doesn't guarantee E3 passes. `kvsel validate` reports the noise floor under both heads, and the fp32 copy goes if it doesn't help.

Tolerances still aren't loosened after a failure, as §11 requires. Numerical-policy changes are allowed during development with a documented cause and a rerun of every affected test, and never in response to a confirmatory outcome.

### Affects the confirmatory protocol

**4. Eligibility requires model output on held-out items before the freeze.**
The schedule draws and verifies the control subset before the freeze, but eligibility depends on Full\_Q. Computing Full\_Q is part of defining the target population, so it isn't contamination. The ambiguity is procedural: "confirmatory sets are evaluated once" can't be literally true.

*Decision:* three stages. First freeze the screening rules: model, prompt, surface forms, normalisation, eligibility and seed. Then run an automated screen using Full\_Q only, with no compression and no descriptions, exposing item ids, eligibility flags and funnel counts. Then freeze the comparative experiment. `PREREG.md` replaces "evaluated once" with "comparative outcomes are not inspected or used to modify the confirmatory protocol". If screening rules must change after held-out failures have been inspected, document an amendment and use a fresh held-out partition.

**5. Hand verification would shrink the control subset below 300.**
Failed items are removed, not repaired.

*Decision:* draw a seed-0 ordering of the eligible items and verify in that order, blind to B and C outcomes, until 300 pass per dataset, the pool runs out or a fixed verification budget is spent. Log every failure. Items that fail verification also leave the main set.

**6. The "less-filtered population" needs items outside the main set.**
§7 and §10 analyse a population that drops the margin condition, but a main set capped at eligible items contains none of those extra items.

*Decision:* keep three nested groups: structurally valid items; items Full\_Q answers correctly; and items Full\_Q answers correctly with a positive margin. The third is the primary population. Take a predefined sample from the second for cells A–D and the Full anchors at PW-attn, K = 144, in a separate manifest the primary analysis can't load.

**7. I^(mis) is both exploratory and in a Holm family.**
Separate Holm families are fine in themselves, but each must state which conclusions it supports, and separate 5% families don't give 5% control across the whole project. The specific conflict is that §9 gives I^(c) no directional prediction, while §10 puts it in a six-test family alongside the one-sided H3 and H4.

*Recommendation:* make the family four one-sided tests (E\_harm and S\_Q per dataset), and report I^(mis) with intervals only. Your call; it doesn't affect the first version. The question-span selector stays descriptive and outside the primary family.

### Smaller gaps

**8. Remaining unspecified choices.** I'll use these unless you object:

| Gap | Choice |
| --- | --- |
| The standalone "uniform grid" baseline has no rule (the reserve's coverage ordering is defined) | Nested regular lattices that hit K exactly. K = 288: `(r+c)` even. K = 144: `r` and `c` both even. K = 72: the K = 144 set with `(r/2 + c/2)` even. I chose this over a prefix of the coverage ordering because §6 calls it a uniform grid, and it stays distinct from the reserve mechanism it's compared against |
| Dataset splits | GQA: `val_balanced`, widening to `val_all` before the screen only if the funnel is short (scene graphs exist only for train and val). TextVQA: train and val pooled, since I expect a low eligibility rate. Confirm both against the funnel |
| Descriptions outside the ±3 token rule | Excluded at build time and counted in the funnel |
| Greedy length | At most 16 new tokens, stopping at `</s>`. A truncated answer scores as "other", and the truncation rate is reported |
| Mitigation on neutral items | Only S\_Q and S\_Q + reserve exist, because without a description complete-prompt selection equals S\_Q |
| Batching | Batch size 1 everywhere, so padding can never affect results |

### Decision record: second model (7 November 2026)

**Decision.** Add Qwen2.5-VL-7B-Instruct as a second model. LLaVA-1.5-7B stays primary.

**Why, fixed before any Qwen outcome was seen.** The GQA pilot showed LLaVA-1.5 adopting the
misleading description on 99.5% of items even with the full cache, which leaves selection no room to
change answers. Published measurements put Qwen at the other end: text preference under conflicting
text of 29.2% for Qwen2-VL-7B against 85.5% for LLaVA-NeXT-7B ([Words or Vision](https://arxiv.org/html/2503.02199v1)),
and the most robust open model in [ConText-VQA](https://arxiv.org/html/2601.19202) (Qwen2.5-VL-7B,
64.4% accuracy kept). Qwen2.5-VL-7B is also used in the compression literature ([COAST](https://arxiv.org/html/2605.09429)).
The pair tests whether selection harm depends on how strongly a model follows text. The choice must
never be revisited because of which model shows the larger effect.

**What differs for Qwen.**

| | LLaVA-1.5-7B | Qwen2.5-VL-7B |
| --- | --- | --- |
| Image | Padded, 336 px, 14 px patches | Padded, 672 px, 14 px patches merged 2 x 2 |
| Visual tokens | 576 on a 24 x 24 grid | 576 on a 24 x 24 grid |
| Prefix | 581 entries; 5 non-visual | 591 entries; 15 non-visual (system prompt) |
| Replay start | Position 581 | M-RoPE position 39: the grid uses only 24 positions |
| Template | v3.0 | v3.0-qwen: the same user text inside Qwen's chat template |
| Candidate | `" " + answer`, then `</s>` | `answer`, then `<|im_end|>` |
| PW-attn window | 16 fixed instruction tokens | 15 fixed tokens plus the question's last token (`'?\n'`) |
| Probe layers | 2-31 | 2-27 (Qwen has 28 layers) |
| Precision | fp16, fp32 head | bf16 weights and cache; fp32 head upcast slice by slice; fp32 attention scores, softmax and value sum in the language model (E3 failed with bf16 versions, 7 Nov 2026; see tests/README.md) |

Items are shared: both models run the same manifests, built with LLaVA's tokenizer.

**Decision (7 November 2026): Qwen is parked.** After three validation runs, E3 still failed on its
margin criterion: 0.038 nats against the 0.01 limit, with every other test passing (tests/README.md).
Qwen is outside the freeze and the confirmatory claims. The code, the fp32 head and the fp32 attention
stay. Any later Qwen work starts from that record and is reported as a labelled follow-up.

### Decision record: user decisions (9 November 2026)

- **Primary selector: PW-attn.** It was specified in advance. Question-span selection stays a descriptive secondary.
- **GQA main set: aim for 1,000 items.** Widen the confirmatory pool with more questions from GQA `val_all`, on images not used in development. Not GQA train, which LLaVA-1.5 was trained on. This is a builder change (new builder version) and must happen before the screening rules freeze. Combine it with any builder fix that the development verification failures call for.
- **H3 is two-sided** (9 November). GQA's development pilot showed E_harm^(mis) = −0.12, the opposite of H3's prediction; TextVQA's showed +0.53. A two-sided test can detect either direction. Disclosed as a revision made on development data, before the freeze.
- **Adoption: a prompt test first** (9 November). The rule below is fixed before the test runs.
  - *What runs:* `kvsel prompt-test`, full cache only, on the verified development passes (GQA 150, TextVQA 188). It computes no kept sets, so no S_H is seen for any variant.
  - *Variants,* each changing one thing about how the description is shown:
    - `base`: v3.0 itself.
    - `unverified`: the label becomes "Unverified description, possibly wrong:".
    - `tiebreak`: the authority line adds "if it conflicts with the image, trust the image". This also changes the Q prompt.
    - `after`: the description line moves after the question.
  - *A variant qualifies if both hold:*
    - On GQA, Full_mis gives the wrong alternative on at most 80% of items (base: about 99%). This leaves room for selection to change answers.
    - Full_Q stays correct on at least 95% of items, on each dataset.
  - *Choice:* the qualifying variant with the lowest GQA alternative rate. If two are within 5 points, prefer the smaller change, in the order `unverified`, `after`, `tiebreak`.
  - *If none qualifies:* keep v3.0 (option A).
  - *If one is chosen:* it becomes template v3.1 for both datasets. Rescreen and rerun the pilot on both; hand-verification decisions carry over, since the items don't change.
  - *Result (9 November, GQA, 150 items).* Rates of the wrong alternative under Full_mis: `base` 99.3%, `unverified` **46%**, `tiebreak` 100%, `after` 78%. Full_Q stayed at 99% or more under all four. Under the rule, `unverified` is chosen: it has the lowest rate, and `after` is not within 5 points of it. Its Q prompt is identical to v3.0's, so Full_Q on TextVQA is unchanged and its guard holds by construction. Mean Full_mis margin moved from −3.76 to +0.32 nats. (The first printed summary showed 0% for every alternative rate because of a label bug; the rates above are 1 − correct − other, and the bug is fixed.)
  - *Applied:* LLaVA template **v3.1** = v3.0 with the description labelled "Unverified description, possibly wrong:". E1 passes on all 2,100 development items. Screens stay valid, since the Q prompt is unchanged. The pilots on both datasets must be rerun; verification decisions carry over. Qwen (parked) keeps v3.0's text.
- **I^(mis) is reported outside Holm** (9 November), with its estimate and interval, labelled exploratory.
- **All confirmatory tests are two-sided** (9 November): H1, H2, H3 and H4. TextVQA's development pilot pointed opposite to H1 and H2 (S_H^(mis) = −0.10, T_match = −0.16 under v3.0), and a one-sided test can't report a reliable effect in that direction. One rule for every test; no direction was flipped. Holm families are unchanged apart from I^(mis), which is outside. Reports now show two-sided p-values.
- **TextVQA stays in the core study** (9 November).
- **v3.1 development pilots** (9 November, overnight run, `results/v31/`). Development only; no claims.
  - Prompt test rerun: same rates as above on GQA; TextVQA `unverified` gives the alternative on 2% (Q 100%). The rule holds on both datasets.
  - GQA, 150 items, PW-attn: S_H^(mis) +0.09 [+0.02, +0.18]; S_Q^(mis) +0.09; I^(mis) −0.00 [−0.05, +0.06]; T_match +0.07 and T_irr +0.06, both inconclusive; E_harm^(mis) −0.21 [−0.34, −0.08]. Full_mis 52% correct; PW at K = 144 under mis 64%.
  - TextVQA, 188 items: S_H^(mis) −0.01 [−0.05, +0.02]; T_match −0.07 and T_irr −0.07; E_harm^(mis) +0.45, but E_harm^(match) +0.34 and E_harm^(irr) +0.23 are positive too.
  - Tokenization check: GQA 200/200, TextVQA 199/200 (one "Powerade" split).
  - v3.1 was fixed by the prompt-test rule before these numbers existed. It isn't revisited because of them.
- **GQA verification failures: automatic filters don't reach 5%** (9 November). Using the scene graphs on the 200 verified development items (25% failed), the best combination tried (box ≥ 5% of the image, no clothing or hair when the scene has 2+ people, no scene-level nouns) still fails 12%, keeps a quarter of the items, and is fitted to these 200 items. Most failures are label noise (wrong a_true) or people the scene graph omits.
- **GQA: every confirmatory item is hand-verified** (9 November, user decision). Only passes enter the main set: the first 1,000 eligible items that pass, in manifest order. The first 300 of those form the control subset. Since every item is checked, the 5% audit is no longer needed. No new automatic filters (the ones above were fitted to 200 items and only saved time). Expected work: about 1,350 items at about 15 s each.
- **Builder `gqa-3`** (9 November). Source `val_all_questions.json`, which contains every `val_balanced` question. Changes from `gqa-2`: one item per (image, object, attribute kind), keeping the lowest question id, because `val_all` asks about the same object in several wordings; and `--dev-manifest` keeps the 600 development items byte for byte (sha256 `c5848b11…`, so verification decisions stay valid), with the confirmatory pool on every other image. Funnel: 2,011,853 questions → 25,474 with the program shape → 23,678 in the vocabulary → 13,871 with an irrelevant object → 13,785 within 3 tokens → 10,455 after de-duplication. Confirmatory pool **9,209 items on 4,429 images**, none shared with development.
- **TextVQA: majority filter, and every confirmatory item hand-verified** (9 November, user decision). Development verification failed 12 of 200 (6%): 11 wording, mostly questions with several valid answers, and 1 wrong a_true. Items with fewer than 5 of 10 votes for the modal answer failed 3 of 6; with 5 or more, 9 of 194 (4.6%). That is too close to the 5% audit limit, so TextVQA follows GQA's rule: the first 1,000 eligible items that pass, in manifest order; the first 300 form the control subset; no audit.
- **Builder `textvqa-3`** (9 November): the modal answer needs at least 5 of 10 votes (3 under `textvqa-2`), and `--dev-manifest` keeps the 1,500 development items byte for byte (sha256 `957eecf3…`). Funnel: 39,602 → 38,611 with OCR → 32,988 with 5+ votes → 6,871 items. Confirmatory pool **5,514 items on 4,905 images**, none shared with development.
- **Fixed:** the model-location message goes to stderr, so `check-tokenization` output saved with `>` parses as JSON.
- **Next, not yet done:** S1 and R1 on the GPU, coverage plots (E6), pre-registration, and the freeze tooling (screen of confirmatory items exposing only eligibility; the verification tool opened to confirmatory items).

## 4. Repository layout

```
KVcompressed/
├── pyproject.toml, uv.lock        Python 3.11; transformers pinned exactly
├── PREREG.md                      written at the freeze (P3)
├── Makefile                       test-l1, test-l2, build, screen, pilot, confirm, results
├── configs/
│   ├── template_v3.0.yaml         prompt text, candidate rule
│   ├── selectors.yaml             layers, window, pooling, budgets, reserve
│   └── runs/                      dev_pilot, confirm_core, confirm_ext
├── src/kvsel/
│   ├── prompts.py                 template, prefix/suffix split, candidate tokenization
│   ├── geometry.py                pad + resize; box → grid under all 4 mapping rules
│   ├── model.py                   load, vision features, prefix pass, fp32 logits
│   ├── cache.py                   PrefixEntries, compact(), dense_mask(), to_cache()
│   ├── engine.py                  probe(), replay(), score_candidates(), greedy()
│   ├── selectors.py               scoring, top-K, uniform, random, coverage, reserve
│   ├── grid.py                    expands a config + item into work units
│   ├── runner.py                  executes units, resumes, guards the freeze
│   ├── store.py                   append-only results, score and kept-set archives
│   ├── data/
│   │   ├── gqa.py, textvqa.py     per-dataset filters and a_alt rules
│   │   ├── descriptions.py        three description templates
│   │   ├── normalize.py           answer normalisation, VQA multi-reference score
│   │   ├── split.py               image-level splits, control-subset ordering
│   │   └── screen.py              eligibility screen
│   ├── analysis/
│   │   ├── estimands.py           S_H, T, S_Q, I, E_harm from cell rows
│   │   ├── bootstrap.py           image-clustered bootstrap, Holm, equivalence
│   │   ├── mechanism.py           Jaccard regression, ERR
│   │   └── tables.py              every table and figure
│   └── verify/app.py              local hand-verification UI
├── tests/
│   ├── README.md                  acceptance policy, written before tests first run
│   ├── level1/                    tiny fp32 model; runs on the Mac
│   ├── level2/                    E1–E7; runs on the GPU
│   └── analysis/                  simulated data with known answers
├── manifests/                     git-tracked item lists with SHA-256
├── artifacts/coverage_order.json  committed farthest-point ordering
└── data/, results/                gitignored; results are hashed in release notes
```

## 5. Replay engine

### 5.1 Principles

- **One module owns the cache API.** `transformers` has changed its cache internals across versions. Only `cache.py` touches them, and the version is pinned exactly. 4.57.6 is the starting candidate because the spike passed on it.
- **The prefix is immutable and shared.** It is stored as tuples of tensors. Each probe or replay builds a new `DynamicCache` referencing those tensors. `update` concatenates into new tensors, so no deep copy is needed. The spike confirmed the shared tensors stay unchanged, and E4 guards this on the real model.
- **Both modes share one code path.** Compact and dense differ only in which entries go in the cache and which mask is passed, so the dense reference tests the real code.
- **Positions and cache indices are always explicit.** Cached keys already have their rotary embedding applied, so gathered entries keep their original positions. New tokens get logical positions from `position_ids` and physical indices from `cache_position`. Inferring either from the cache length is the classic bug, and the spike's negative control shows it changes the output.

### 5.2 Core operations

```python
# cache.py
def compact(prefix, kept_visual):                    # kept_visual: sorted tensor, values 0..575
    idx = cat([arange(N_NONVISUAL), N_NONVISUAL + kept_visual])
    return tuple((k.index_select(2, idx), v.index_select(2, idx)) for k, v in prefix)

def additive_mask(T, past_len, dropped=None):        # shape (1, 1, T, past_len + T)
    m = zeros(1, 1, T, past_len + T)
    m[..., past_len:] = triu(full((T, T), -inf), diagonal=1)
    if dropped is not None:
        m[..., dropped] = -inf
    return m

# engine.py
def forward_step(ids, cache, logical_start, physical_start, dropped=None, attn=False):
    T = ids.shape[1]
    return decoder(ids, past_key_values=cache,
                   position_ids=arange(logical_start, logical_start + T)[None],
                   cache_position=arange(physical_start, physical_start + T),
                   attention_mask=additive_mask(T, physical_start, dropped),
                   output_attentions=attn, use_cache=True)
```

The public operations are:

| Function | Inputs | Output |
| --- | --- | --- |
| `prefill(item)` | image, prefix ids | Immutable prefix entries |
| `probe(prefix, suffix_ids)` | Full prefix, a suffix | Per-token attention arrays (5.3). The cache is discarded |
| `replay(prefix, kept, suffix_ids, mode)` | `mode` is `compact` or `dense` | Suffix logits and post-suffix cache entries |
| `score_candidates(state, a_true, a_alt)` | Post-suffix state | Log-probabilities per token for each candidate, including `</s>` |
| `greedy(state, max_new=16)` | Post-suffix state | Token ids, text, top-two gap at each step |

**Scoring.** Each candidate is `" " + text` plus the `</s>` id. Its first-token probability comes from the last suffix logit, and the remaining tokens come from a forward pass on a fresh cache wrapper, so candidates can't interfere with each other. Raw per-token log-probabilities are stored, so the content-only, length-normalised and casing sensitivity margins need no rerun.

**Real-model path.** The prefix pass goes through `LlavaForConditionalGeneration`, using `pixel_values` so the library's own image merge is used. The suffix pass calls the language model directly with the 4D mask, and logits go through the fp32 head (decision 3). E2 compares this whole path against one full forward pass.

### 5.3 Probe and selectors

Attention tensors have the shape `(1, 32, T, P+T)` for each layer. The probe reduces them immediately to two arrays, stored per item and probe:

- `A_late[t, i]`: mean over layers 2–31 and all heads, for each suffix token t and visual entry i, float32.
- `A_l2[i]`: mean over heads at layer 2, for the last suffix token only.

These take about 115 KB per probe at T = 50. PW-attn, LT-early and Span-attn all derive from them with no further GPU pass. Accumulation happens in fp32. HF returns attention weights cast back to fp16, so very small probabilities lose precision. E6 logs tie rates for this reason, especially for LT-early, which averages the fewest values.

`selectors.py` is pure NumPy and runs without a GPU:

```python
pw_attn(A_late, window)        # mean over the window, then 3×3 pool with edge handling
lt_early(A_l2)
top_k(scores, K)               # sort by (-score, index); ties go to the lower index
uniform(K), random_set(K, seed)
coverage_order()               # exact integer squared distances; start (11, 11); row-major ties
with_reserve(scores, K, r)     # top K - k_r, then coverage order, skipping kept ones, until exactly K
```

Random sets are seeded per item with `sha256(item_id, seed)`, so a set doesn't depend on run order. `k_r` values are asserted to be 29, 14 and 7. Python's `round` is safe here because none of these falls on a .5 boundary.

### 5.4 Image and evidence geometry

`geometry.py` has one function that defines the transform, and both preprocessing and box mapping use it. Otherwise a box could silently misalign.

- **Image.** Pad to a square using the CLIP mean colour, then resize to 336 × 336 with bicubic resampling, with centre crop disabled. A test checks that the processor's output matches this.
- **Box.** `x' = (x + pad_x) · 336 / side`, and likewise for y.
- **Patches.** A 14-pixel grid, under all four rules in §8. D2 overlays each one for visual review.

## 6. Data pipeline

### 6.1 Flow

```
raw dataset → filters → a_alt + descriptions → image-level split
   → development: screen, hand-verify 200 per dataset
   → confirmatory pool: eligibility screen → seed-0 ordering
        → main set (up to 1,000) + control subset (first 300 that pass verification)
        → grounding subset (control items with verified evidence)
        → audit 100 of the rest
   → manifests with SHA-256, committed
```

The builder records a count at every filter stage, which is the D3 funnel. It's a pure function of raw data, config and seed, and it derives each item's choices from `sha256(item_id, builder_version)`.

### 6.2 GQA

Filter to semantic programs of exactly `select(object) → query(color|material)`, with no relate or filter steps. Exclude items where the scene graph has another object with the same name.

The a\_alt vocabulary is built from the colours and materials in the scene graphs. a\_alt must not be the true value, any attribute of that object, or a word in the question. Spelling variants such as `gray`/`grey` count as the same answer in a small, frozen equivalence table, so one can't be chosen as a wrong alternative to the other.

The irrelevant description uses another object with a known attribute and a non-overlapping box.

**Risk:** the two-step filter plus unique referents may leave few items in `val_balanced`, so the funnel gets checked first, on 8 November.

### 6.3 TextVQA

Normalise OCR tokens and answers with one function. Keep an item if the modal answer has at least 3 of 10 votes and equals one complete OCR token or a contiguous token span. The evidence set is the union of all matching locations.

a\_alt must be a complete OCR token appearing exactly once in the image, with the same character class, length within ±2, and not an accepted answer.

Answers located through context are flagged during hand verification, since automatic detection isn't reliable. The flag only affects the grounding subset, which is fully hand-verified anyway.

**Risk:** I expect LLaVA-1.5's exact-match accuracy on text questions to be low, and that's why the plan pools train and val. If there are still fewer than about 1,200 eligible items, the main set is simply smaller, as "up to 1,000" allows. The precision projection then reports the consequence.

### 6.4 Verification tool

Verifying 1,200 items by clicking through folders isn't realistic, so this tool is worth building properly. It's a small local web app showing the image with evidence boxes and the grid overlay, the question, a\_true, a\_alt and all three descriptions.

Keyboard flow: `y` to pass, or a number key for a failure reason (statement not false, referent not unique, a\_alt acceptable, wording, bad box, located through context). Decisions go to an append-only JSONL file with item id, verifier and timestamp. It also renders the D2 overlays for all four mapping rules.

At 30–45 seconds per item, the full load takes 10–15 hours:

| Set | Items |
| --- | --- |
| Development | 400 |
| Control subset | 600, plus replacements for failures |
| Audit | 200 |

## 7. Experiment runner

### 7.1 Work units

`grid.py` expands a config and item into units. Each is identified by:

```
(item_id, selector, K, reserve, seed, selection_context, answering_context)
```

| Stage | Set | Units per item |
| --- | --- | --- |
| P4 core | Main | 2 probes; A, B\_mis, C\_mis, D\_mis; Full\_Q, Full\_mis; uniform with Q and QH\_mis (8 replays) |
| P4 core | Control, additional | 2 probes; B, C, D for match and irr; Full\_match, Full\_irr; uniform with QH\_match and QH\_irr (10 replays) |
| P5 | Main | K = 288, 72: A–D and uniform (12); LT-early (4); random seeds 0–4 with Q and QH\_mis at K = 144 (10) |
| P5 | Main | Span-attn at K = 144, if adopted (4) |
| P5 | Control | Mitigation: 2 reserve variants × {mis, match}, plus S\_Q + reserve with Q (5); random with match and irr can be added if wanted |
| P5 | Grounding | Restoration (optional): treatment and 5 control seeds × m ∈ {4, 8} |
| P4 | Less-filtered manifest | A–D, Full\_Q, Full\_mis (decision 6) |

Within an item, identical `(kept set hash, answering suffix)` pairs reuse one result. For example, S\_Q can equal S\_mis exactly. E5 is the exception: it deliberately runs without reuse, since that is what it tests.

### 7.2 Results

Each unit appends one row to a shard file, flushed per item. Rows include the unit key, the git commit, manifest hash, kept-set hash, per-token log-probabilities for both candidates, M, the greedy ids and text, outcome (correct, alt, other), and the top-two gaps.

Score arrays and kept sets are stored per item as compressed `.npz`. A crash resumes by skipping completed keys. A consolidation step produces one Parquet file per run and fails if any expected unit is missing or duplicated.

### 7.3 Confirmatory guard

`make confirm` refuses to run unless all of these hold:

- The working tree is clean.
- `PREREG.md` is committed under the `prereg-freeze` tag, and the code matches that tag or a documented fix commit.
- Manifest hashes match those recorded in `PREREG.md`.
- The output directory has no completed run of that config.

During a run, the monitor shows only throughput, memory and error counts, never margins or answers. After a run, the screen's Full\_Q rows are compared to the run's Full\_Q rows as a consistency check.

Timing runs use a separate entry point with hashing and extra logging off, as §8 requires.

## 8. Validation

`tests/README.md` is written first and holds the acceptance policy, including decision 3. The spike was exploratory, on a throwaway model with no tolerances applied. The real tests still run for the first time under the written policy.

| Test | Implementation | Runs on |
| --- | --- | --- |
| Level 1 | The spike promoted to pytest: random, concentrated and scored sets, several budgets, 10 generated tokens, limit 1e-5 | Mac |
| Level 1 additions (not in the proposal) | Wrong positions must fail; perturbing dropped entries must change nothing in dense; perturbing kept entries must change the output; K and V gathered with the same indices | Mac |
| Noise floor | Full forward vs prefix-then-suffix on 50 development items. Saved as the reference distribution for E2, E3, E5 | GPU |
| E1 | Exact checks on all development items; window span and composition logged | Tokenizer only, so Mac |
| E2, E3 | As in §11, comparing compact and dense through the same `forward_step` | GPU |
| E4 | SHA-256 over prefix tensors before and after each probe and replay, behind a test flag | GPU |
| E5 | Run B\_c and C\_c with kept indices forced equal | GPU |
| E6 | Pure unit tests, plus coverage plots and radius vs random | Mac |
| E7 | Rerun 20 items; compare rows byte for byte, with deterministic algorithms enabled | GPU |

**Order:** Level 1 → E6 and E1 (no GPU) → real-model integration → noise floor → E2 and E3 on single items → E4, E5, E7 → the full runs on development items once those exist.

**Analysis tests** (added): simulate data with a known effect and known image clustering, then confirm the estimands recover it and the bootstrap's coverage is near nominal. Check E\_harm's exact decomposition numerically, and check Holm against hand-worked examples.

## 9. Analysis

`bootstrap.py` works on per-image sums, so each resample is a weighted sum. 10,000 resamples per estimand take seconds. It reuses one set of resample weights per dataset with seed 0, implements the §10 p-value exactly, and returns the point estimate, 95% percentile interval, 90% interval for equivalence, and the §10 interval reading as an explicit label.

| Family | Module output |
| --- | --- |
| Primary, H1 | S\_H^(mis) per dataset, Holm over 2 |
| Specificity, H2 | T\_match, T\_irr, Holm over 4 |
| H3, H4 | Holm over 4 (decision 7) |
| H5 | OLS with image-clustered standard errors |
| H6 | ERR(S\_Q) − ERR(S\_mis) under all four mapping rules |
| Others | Interval tables, each labelled with its pre-registered family |

The whole analysis runs first on development results and simulated data, so the first confirmatory computation runs tested code. The primary analysis then runs once on 21 November. `make results` rebuilds every table and figure from the Parquet files and fails if any input hash doesn't match.

### Runtime estimate

These are unmeasured guesses that R1 will replace, assuming about 0.1 s for a pass over the full prompt and about 0.15 s for one replay with scoring and decoding.

| Stage | Volume | Estimate |
| --- | --- | --- |
| Eligibility screen | Up to about 20,000 items across both datasets | 2 h |
| P4 core | 2,000 main items, 600 with controls | 1–2 h |
| P5 extensions | 26–30 replays × 2,000 items | 2–3 h |
| Image loading | Adds overhead, depending on the disk | Measure |

Even if these are off by a factor of five, the GPU fits the schedule. That leaves the build and verification work as the real constraint.

## 10. Schedule

Dates follow §12, with two changes. E-tests that need development items run in two passes, and the template is fixed before the screen (decision 4).

| Date | Work | Done when |
| --- | --- | --- |
| 7 Nov | Project setup; `tests/README.md`; Level 1; E1 and E6 scaffolding | Level 1 passes on the Mac |
| 8 Nov | GPU environment; real-model integration; download data; GQA funnel | One item runs end to end; GQA funnel known |
| 9–10 Nov | E2–E5 on single items; selectors; geometry; GQA builder; verification tool | E2 and E3 pass on single items |
| 11 Nov | Runner, store, grid; GQA development items built; start verifying | **Gate P0a:** E1–E7 pass on GQA items available so far |
| 12 Nov | GQA development verified; casing rule and template fixed for GQA; GQA screen; control ordering; pilot starts | D1 for GQA development; screen complete |
| 12–14 Nov | Pilot: B vs C, window composition, overlaps, clustered SD, S1, S2, R1. Verify GQA control subset and audit. TextVQA builder | Runtime confirmed; decision 1 resolved |
| 14 Nov | TextVQA development verified; TextVQA screen; E1–E7 rerun on TextVQA | **Gate P0b** |
| 14–15 Nov | Verify TextVQA control subset and audit; D2 overlays; precision projection | D1, D2, D3 complete |
| 15 Nov | Write and commit `PREREG.md`; tag | **Freeze** |
| 16–20 Nov | P4 confirmatory core, including the less-filtered manifest | All units complete; consolidation passes |
| 21 Nov | Primary analysis, once | Report produced |
| 21–27 Nov | P5 extensions in cut-order priority | Each result labelled |
| 29 Nov | Results freeze | `make results` regenerates everything from a clean checkout |

**Tightest point: 14–15 November.** TextVQA's builder, pilot, screen and 400 verified items all land in about two days, and a builder fix after a failed audit restarts the screen and verification. If TextVQA isn't ready on 15 November, freeze GQA on time. `PREREG.md` would state in advance that TextVQA's protocol is frozen by a dated addendum before any TextVQA confirmatory run and before the 21 November analysis, with Holm still over both datasets. If that also slips, TextVQA follows the proposal's cut order and becomes a labelled follow-up, and Holm is over one test.

## 11. Risks

| Risk | Likelihood | Response |
| --- | --- | --- |
| Real-model path behaves differently from the spike (library's image merge, mask handling, fp16) | Medium | Found on 8 November by end-to-end comparison. Fallback: pin another version or call the decoder layers directly. §11's token-pruning fallback only if cache surgery can't be made correct |
| PW-attn barely changes the kept set | Medium | Rung 1 is measured on 12–14 November; decision 1 |
| E3 fails because of fp16 noise | Medium | fp32 head from the start; never loosen a tolerance. Identify the cause |
| Small GQA funnel | Medium | Known on 8 November; widen to `val_all` before the screen |
| Low TextVQA eligibility | High | Pool splits; report a smaller main set and its precision |
| Audit above 5% error | Medium | Fix the builder and rebuild; the 15 November fallback applies |
| Verification takes longer than planned | High | Tool first; if behind, verify in seed order so partial work stays valid |
| Bug found after freeze | Low | §10 procedure: documented, outcome-blind fix; rerun every affected condition. `grid.py` can list the affected units |
| Library upgrade changes results | Low | Exact pin and lockfile; commit recorded on every row |

## 12. Definition of done

- Level 1 and E1–E7 pass under the policy written before they first ran.
- `PREREG.md` was committed before any confirmatory output, and git history shows it.
- All confirmatory units are present exactly once, with manifest hashes matching the freeze.
- `make results` rebuilds every table from a clean checkout.
- Every result carries its pre-registered family and the claim-ladder rung it supports.
