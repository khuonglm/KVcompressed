# Spec: exploratory follow-ups for the poster (colour search, window/layer ablation), TextVQA as pre-registered, poster figures

Status: ready for agent. Date: 19 November 2026. The poster PDF is due 3 December 2026 (submitted as a PDF, not printed). GPU runs may continue until 2 December 2026, but a result arriving that late can enter the poster only if its figure slot and script were already prepared.

The agent works directly on the GPU server, with the researcher present. The researcher runs the GPU jobs and does every hand-verification decision. The agent writes code, runs CPU commands, and hands over GPU commands with their expected runtime.

## Problem Statement

The study now has three tiers of evidence that agree:

- **Original GQA study (confirmatory):** misleading text steers PW-attn's visual-KV selection toward worse patches. S_H^(mis) = +0.111 at K = 144.
- **Replication on 1,000 fresh items (confirmatory, Holm-final):**
  - Steering replicated at K = 72 (+0.173) and K = 144 (+0.096), and is negligible at K = 288 (equivalence supported).
  - Steering is specific to false content: T_irr = +0.167 at K = 72, and T_match (exploratory, a minimal pair) = +0.143 at K = 72 and +0.117 at K = 144.
- **LLaVA-NeXT (pre-specified, exploratory after the E3 NO-GO):** the same sign and the same dose–response: +0.110 / +0.075 / +0.019 at 12.5% / 25% / 50%.

Three weaknesses remain, and a poster reader or judge will see each one:

1. **Mechanism.** The results show *that* one false word moves the kept set, not *where* it moves it. The working hypothesis is "colour search": the false colour word pulls selection toward patches of other objects that have that colour. It is untested.
2. **Design rule.** Only PW-attn is steered. The question-span window (+0.015, n.s.) and LT-early (−0.075) are not, even though the question span also comes after the description. So "a window after untrusted text is vulnerable" is too simple, and nobody knows which windows or layers carry the steering.
3. **The second dataset.** PREREG section 2 pre-registered TextVQA, and it was never run. A pre-registered arm left unrun is the most visible procedural weakness, and the H1–H4 Holm decisions stay formally "pending" until it runs.

The researcher also needs the poster's figures produced from the stored results, with nothing transcribed by hand.

## Solution

Four pieces of work, all on the server, in this order of priority:

1. **Colour search (CPU, exploratory).** A mechanism analysis on the kept sets the original and replication runs already stored. It uses GQA scene-graph boxes to ask whether S_mis keeps more patches of objects that have the a_alt colour than S_Q does. Its definitions and predictions are written into a new, dated PREREG section and committed *before* anything is computed.
2. **Window/layer ablation (GPU, exploratory).** On the original 1,000 GQA items at K = 72 only, S_H^(mis) is measured for 12 probe configurations: 3 windows × 4 layer bands. The replication and NeXT items are never touched.
3. **TextVQA, exactly as PREREG section 2 says.** Verify to 1,000 passes, freeze stage 3, run once, then the joint `confirmatory report`, which finalizes the H1–H4 Holm decisions. No new code is expected; the reorganized commands already take TextVQA.
4. **Poster figures.** One script that draws every figure from the stored `results.json` files.

The new code changes the package's code identity, so it lands as **code amendment 5**. That means a dated entry, `kvsel reproduce` on every stored report and on GPU rows, and evidence written into the amendment log. It must land **before** TextVQA's stage-3 freeze, so that TextVQA runs on frozen code that already includes it.

## User Stories

### Colour search (mechanism)

1. As the researcher, I want the colour-search definitions, item filter, predictions and inference written as a dated PREREG section and committed before any computation, so that the poster can call the analysis "specified before computation" and not post hoc.
2. As the researcher, I want the colour-search command to refuse while that section still has a draft line or a `[to fill` blank, so that I can't compute it before the plan is final.
3. As the researcher, I want the command to record the sha256 of that PREREG section in its `results.json`, so that a reader can check the plan predates the numbers.
4. As the researcher, I want the analysis to read only stored run folders (the original confirmatory run, the follow-ups' budgets part, and the replication run) plus the manifest and the scene-graph file, with no model and no GPU, so that it runs on CPU while the GPU is busy.
5. As the researcher, I want the analysis to refuse to run if any manifest item's `object_id` is missing from the scene graph of its image, or if its box differs from the item's `evidence_box`, so that a different scene-graph version can't slip in silently.
6. As the researcher, I want only colour items (`attribute_kind = color`) analysed, and their count reported for each run, so that the population is explicit.
7. As the researcher, I want the "alt-colour patches" of an item defined as the patches of every scene-graph object in the same image, other than the target object, whose attributes (normalised as in PREREG 1.3) include a_alt, minus the target's own evidence patches, so that the measure is about patches *elsewhere* in the image.
8. As the researcher, I want the "true-colour patches" defined the same way with a_true, so that the matching description gives a built-in minimal-pair control.
9. As the researcher, I want patches mapped from boxes with the four H6 mapping rules unchanged, with the primary rule as the headline and the other three as sensitivity, so that the mapping is the one already reviewed.
10. As the researcher, I want retention R(S, P) = |S ∩ P| / |P|, with items whose P is empty left out for that measure and their count reported, so that it matches H6's ERR.
11. As the researcher, I want the colour pull Δ_alt^(mis) = R(S_mis, Alt) − R(S_Q, Alt) at every budget each run has, so that I can see whether the false colour word pulls selection toward its colour.
12. As the researcher, I want Δ_true^(match) = R(S_match, True) − R(S_Q, True) and Δ_alt^(match) = R(S_match, Alt) − R(S_Q, Alt) on control items, so that the matching word's pull toward *its* colour is the control.
13. As the researcher, I want the minimal-pair contrast Δ_alt^(mis) − Δ_alt^(match) on control items, so that the effect of swapping one colour word is isolated.
14. As the researcher, I want a per-item OLS of S_H^(mis) on Δ_alt^(mis), with CR1 image-clustered errors (H5's method), at each budget, so that I can see whether pulling toward alt-colour patches goes with lost margin.
15. As the researcher, I want every interval to come from the image-clustered bootstrap (10,000 resamples, seed 0, 95% percentile), with no Holm family and every row labelled exploratory, so that the inference matches the rest of the study without claiming confirmatory status.
16. As the researcher, I want the original and replication results side by side and never pooled, so that the replication acts as an internal check of a mechanism developed on the original items.
17. As the researcher, I want the predictions printed beside each estimate (Δ_alt^(mis) > 0, larger at smaller K; Δ_true^(match) > 0; minimal-pair contrast > 0; positive OLS slope), so that confirmation or failure is obvious.
18. As the researcher, I want `results.json` and `report.md` written atomically to a new output folder, refusing a non-empty one, as the other reports do.

### Window/layer ablation

19. As the researcher, I want the ablation's configurations, item set, budget, outputs and runtime written as a dated PREREG section (the same section as colour search) before it runs, so that its exploratory status is clear and fixed.
20. As the researcher, I want the ablation to run only on the original confirmatory main set (1,000 items), refusing any other set, so that the replication and NeXT items stay clean.
21. As the researcher, I want three windows (the last suffix token, the last 16 suffix tokens, and the question span) crossed with four layer bands (2–8, 9–20, 21–31 and 2–31), so that I can see which window and which layers carry the steering.
22. As the researcher, I want every configuration's scores pooled 3 × 3 within the map and its top-72 taken exactly as PW-attn does, so that the window and layers are the only things that change.
23. As the researcher, I want, per item and configuration, S_Q from the Q probe and S_mis from the QH_mis probe, then cells B_mis (S_mis, QH_mis) and C_mis (S_Q, QH_mis) replayed, so that S_H^(mis) = M(C) − M(B) is measured per configuration.
24. As the researcher, I want one prefill and two probes per item (Q and mis) returning per-layer attention, with every configuration's scores derived from them, so that the run fits one night.
25. As the researcher, I want each distinct kept set replayed once per answering context, so that configurations whose kept sets coincide cost no extra GPU time.
26. As the researcher, I want the configuration "last 16 × layers 2–31" (that is, PW-attn) checked against the follow-ups' stored K = 72 kept sets and margins for B_mis and C_mis, with the run refusing on a mismatch, so that the new probe path is proven to equal the old one on the real model.
27. As the researcher, I want rows to carry the window and the layer band, with the kept set stored, so that the report and later figures can recompute anything.
28. As the researcher, I want the run resumable through the shared item loop, each item's file written atomically, progress-only output, and refusal of item files from another plan, so that a crash or closed tmux session costs nothing.
29. As the researcher, I want the run to print its measured per-item time after the first 25 items, so that I know early whether it fits the night.
30. As the researcher, I want an ablation report with S_H^(mis), the paired behaviour difference (correct, C − B) and the mean Jaccard(S_Q, S_mis) per configuration, intervals only, so that the design-rule panel has numbers.

### TextVQA as pre-registered

31. As the researcher, I want the agent to check first whether the amendment-1 TextVQA confirmatory screen exists on the server and is complete. If it doesn't, the agent hands over the `kvsel confirmatory screen` command for `manifests/textvqa_confirm.jsonl`, so that verification can start.
32. As the researcher, I want to verify TextVQA eligible items in manifest order with `kvsel verify` on the server until 1,000 pass, with instructions to reach the tool from my laptop if it serves a page, so that I can verify without copying folders around.
33. As the researcher, I want a count of passed eligible TextVQA items I can run at any time without the GPU, so that I know how far I am from 1,000.
34. As the researcher, I want the TextVQA stage-3 freeze, run and joint report run with the existing commands and runbook, under PREREG section 2 unchanged, so that nothing about TextVQA is decided after seeing GQA.
35. As the researcher, I want the TextVQA freeze to happen only after code amendment 5 is accepted, so that the run's frozen code identity already includes the new code and needs no later amendment.
36. As the researcher, I want the joint `confirmatory report` (GQA + TextVQA) to print final Holm decisions for H1–H4, and the GQA numbers to reproduce the stored report exactly, so that adding TextVQA can only finalize, not change, GQA's estimates.
37. As the researcher, I want a fallback written down: if 1,000 passes aren't reached by 27 November, TextVQA is dropped as a disclosed deviation (PREREG 2.3 already says the H1 family then becomes GQA alone), and the verification decisions made so far are kept and not used, so that the decision is mechanical.

### Code amendment 5

38. As the researcher, I want the new code to change no output of any existing command, proven by the characterization suite's golden files staying identical, so that the amendment touches nothing reported.
39. As the researcher, I want a dated PREREG entry for code amendment 5 and `kvsel reproduce` run on every stored report (confirmatory, follow-ups, replication and NeXT) and on GPU rows, with its evidence written into the amendment log, so that every finished study's checks still accept the code.
40. As the researcher, I want the probe's per-layer output proven equal to the existing averaged probe on the tiny model, so that the engine change is safe before the GPU check in story 26.

### Poster figures

41. As the researcher, I want one script that draws every poster figure from stored `results.json` files (the original report, follow-ups, replication, NeXT, colour search, ablation and, if run, TextVQA) and records each file's sha256 in a sidecar, so that every number on the poster is traceable.
42. As the researcher, I want a forest plot of S_H^(mis), T_irr and T_match per study at matched budgets: the original, then the replication, then NeXT. NeXT gets hollow markers and an "exploratory" label, so that three tiers read as one picture.
43. As the researcher, I want a dose–response figure of S_H^(mis) against budget (K = 72/144/288, against shares of 12.5/25/50%) for the replication and NeXT, so that the main trend is one glance.
44. As the researcher, I want a behaviour figure in accuracy points first (correct-answer rate, C_mis − B_mis per budget), with nats as a secondary axis or label, so that poster readers don't need to understand nats.
45. As the researcher, I want a colour-search figure: Δ_alt^(mis) against Δ_true^(match) per budget, original beside replication, so that the mechanism reads as a minimal pair.
46. As the researcher, I want an ablation heatmap: windows × layer bands, coloured by S_H^(mis) with the interval in each cell, so that the design rule is visible.
47. As the researcher, I want kept-set maps (S_Q against S_mis overlaid on the image, with the target box and alt-colour boxes drawn) for items chosen by a fixed rule, not by eye: the replication colour items at the 50th and 90th percentiles of S_H^(mis) at K = 72. That way the example images aren't cherry-picked.
48. As the researcher, I want every figure in PDF (vector) and PNG at poster resolution, sized for a portrait A0 poster, so that it drops into the poster without rescaling artefacts.
49. As the researcher, I want the script to skip, with a message, any figure whose input file doesn't exist yet (for example TextVQA), so that figures can be built while runs are still pending.

### Handoff and runbook

50. As the researcher, I want a README section with copy-pasteable commands for colour search, the ablation run and report, the amendment-5 reproduction, the TextVQA steps and the figures, each with its runtime, GPU or not, and what to send back, so that I can run everything from the README.
51. As the researcher, I want a dated work order, with the colour-search plan and the amendment landing before tonight's GPU job, so that GPU nights aren't wasted.

## Implementation Decisions

- **Where the code lives.** A new study package for the exploratory work, next to the existing studies, with one CLI group (proposed name: `explore`) and the subcommands `colour`, `ablation-run` and `ablation-report`. It uses the shared layers (items, verification, model families, engine, selectors, runner and item loop, analysis blocks, protocol) through their public interfaces, and imports no other study's private helpers. Nothing in the existing studies changes.
- **Plan as a PREREG section.** A new **section 6, "Exploratory analyses specified before computation (date)"**, holds both analyses' definitions. It sits outside every hashed section, so no stage-1 section hash changes.
  - Colour search and the ablation both refuse while section 6 has a draft line or a `[to fill` blank, using the protocol's existing draft rule.
  - Their outputs record the sha256 of section 6's text.
  - Neither uses a freeze record. The git commit of section 6 is the timestamp. This is deliberately lighter than a stage-3 freeze because neither analysis has a confirmatory claim.
- **Section 6 must state plainly:**
  - colour search is computed on runs whose comparative outcomes were seen;
  - its predictions were formed after the original and replication outcomes were seen, but before any colour-search quantity was computed;
  - the ablation runs on items whose outcomes were seen.
- **Scene graphs.** `val_sceneGraphs.json` from the official GQA scene-graph download. Its path is an argument and its sha256 is recorded in `results.json`. The integrity check of story 5 runs before any computation.
- **Colour normalisation.** The same normalisation as PREREG 1.3 (lower case, "grey" → "gray"), applied to scene-graph attributes and to a_true/a_alt.
- **Which kept sets.** All are PW-attn without the reserve, and all come from stored rows:
  - S_Q is the kept set of any unit whose selection is Q (cell A or C_c);
  - S_c is the kept set of any unit whose selection is c (B_c or D_c).
  - Original: K = 144 from the confirmatory run (mis on all items; match on control items), and K = 72 and 288 from the follow-ups' budgets part (mis only).
  - Replication: K = 72/144/288 for mis and irr on all items, and match on control items.
  - The analysis checks that the two units sharing a selection carry identical kept sets, and refuses otherwise.
- **Patches.** Boxes map to the 24 × 24 grid with the existing box-to-square conversion and the four H6 mapping rules, unchanged. The target object's primary-rule evidence patches are removed from both the Alt and True sets.
- **Estimands and statistics.**
  - The estimands are those of stories 11 to 14.
  - Intervals come from the existing image-clustered bootstrap.
  - The OLS slope uses the existing clustered-OLS helper.
  - Nothing enters a Holm family.
- **Ablation probe.** The engine's probe gains an option to return per-layer attention, averaged over heads only: an array of layers × suffix tokens × visual entries.
  - The existing averaged output must equal the mean over layers 2–31 of the new output, within float32 tolerance. The default path, and so every existing output, stays byte-identical.
  - Each window/band score is the mean over that band's layers and that window's query positions, then the existing within-map 3 × 3 pooling and `top_k`.
  - The question span is the one the tokens already carry for the existing question-span selector.
- **Ablation units and rows.** Each configuration gives cells B_mis and C_mis at K = 72.
  - Rows carry the selector name (proposed: `attn`), the window (`last1`, `last16` or `question`) and the layer band (`2-8`, `9-20`, `21-31` or `2-31`), plus the stored kept set.
  - Replays are cached by (kept set, answering context) within an item.
  - The PW-attn configuration's kept sets and B_mis/C_mis margins are compared with the follow-ups' stored K = 72 rows for that item. A mismatch stops the run with a refusal that names the item.
- **Ablation item set and output.**
  - The main set is taken exactly as the follow-ups take it, from the confirmatory run folder and its stage-3 record.
  - Output goes to its own folder under `results/explore/ablation/`.
  - The item loop's record hash is the sha256 of section 6, so an item file from an edited plan is refused on resume.
- **Runtime.** It is measured, not assumed: the run prints seconds per item after 25 items. If the projected total exceeds the night, the researcher decides; the agent doesn't cut configurations on its own.
- **TextVQA.** No new code is expected. The agent's job:
  - confirm the screen folder, verification file and image folder exist;
  - provide the count command (the existing replication/NeXT `count` pattern, or a short CLI command reusing the main-set rule if none fits);
  - run the existing stage-3 freeze, run and joint report.
  - If anything in those commands fails on TextVQA under the reorganized code, the fix is a bug fix inside amendment 5's scope, reproduced like the rest, and lands before the TextVQA freeze.
- **Amendment 5 order.**
  1. Write section 6 and the amendment-5 PREREG entry.
  2. Land the code, with the characterization suite green.
  3. Run `kvsel reproduce --reason "code amendment 5: ..."` with `--images` on the GPU.
  4. Commit the filled amendment log.
  5. Only then: the ablation run, colour search and TextVQA's freeze.
- **Figures.** The existing figure script is replaced or extended, reusing its forest-plot helper and colours.
  - It reads `results.json` files only, never item files, except for the kept-set maps, which read the named items' rows and images.
  - It writes `documents/figures/poster/` with PDF and PNG plus a sidecar listing each input file and its sha256.
  - Accessible colours, consistent across figures.
  - The hand-transcribed follow-up JSON in `documents/figures/` is not used for new figures: the follow-ups' own `results.json` is read instead.

## Testing Decisions

**What a good test is here.** As in the reorganization spec: drive the public command line on fixtures and the tiny models, and assert on outputs a reader cares about (files, rows, printed verdicts, exit codes). Expectations are known answers (hand-built fixtures, planted effects), never recomputed by the code under test. One behaviour per test.

**Seams (please confirm):**

1. **The command line on fixtures: the primary seam, as in the reorganization.**
   - *Colour search:* a hand-built manifest, a tiny scene graph and synthetic run folders, with kept sets planted so that Δ_alt^(mis), Δ_true^(match), the contrast and the slope have known values. Tests:
     - it refuses a scene graph whose box differs from an evidence box;
     - it refuses a draft section 6;
     - it refuses mismatched kept sets between units that share a selection;
     - it leaves out items with empty Alt sets and counts them;
     - it refuses a non-empty output folder.
   - *Ablation:* end to end on the tiny LLaVA model on a few items:
     - the exact rows per item (12 configurations × B/C, with window and band);
     - resume skips finished items, and a changed section 6 refuses old item files;
     - the run refuses a set other than the confirmatory main set;
     - the PW-attn consistency check passes against a tiny follow-ups folder and refuses a doctored one;
     - output is progress only.
   - *Ablation report:* a synthetic run folder with planted per-configuration effects gives the right means and intervals.
   - *Figures:* the script runs on fixture `results.json` files, writes every figure whose inputs exist, and skips the others with a message.
2. **The engine's probe (deep module).** On the tiny model, the default probe output equals the per-layer output averaged over layers 2 and up, within tolerance, for both prefix residencies.

**Unchanged guards.**
- The characterization golden files for every existing command stay identical.
- Each existing record's settings still equal what the study declarations produce.
- `kvsel reproduce` passes, on the server, as amendment 5's evidence.

**Prior art.**
- The follow-ups' end-to-end tests on tiny models (per-part units, resume).
- The replication and NeXT report tests with planted effects.
- The H6 evidence-retention tests on hand-built boxes.
- The characterization suite and `kvsel reproduce` from amendment 4.

## Out of Scope

- **Colour search on NeXT.** It needs box-to-patch mapping on the AnyRes layout, which was deliberately not ported (PREREG 4.2). It's a later spec, only if nights remain after the TextVQA run.
- **The ablation on NeXT.** Same condition as above.
- **The referent-following analysis** for irrelevant descriptions. It was demoted because R2 split across budgets.
- **Any audit or re-verification** of the frozen GQA sets (decided 19 November: duplicates of a same-answer referent can't inflate S_H). That goes on the poster as a limitations line.
- **A dense-masked re-check of NeXT's replay.** NeXT stays exploratory either way.
- **The "compression dampens misinformation" panel.** E_harm^(mis) flips sign across models and budgets, so it becomes a one-line caveat.
- **Any change to PREREG sections 1–4, any frozen record, or any reported number.**
- **TextVQA beyond section 2's unit list:** extra budgets, a replication, NeXT.
- **Writing a paper.** The deliverable is a poster PDF.

## Further Notes

**Date order:**

| Date | What |
| --- | --- |
| 19–20 Nov | Section 6, amendment 5 code, CPU tests, `kvsel reproduce` on the GPU. |
| First free GPU night | Ablation run. |
| As soon as the scene graphs are downloaded | Colour search (CPU). |
| Through 27 Nov | TextVQA verification (researcher). Then freeze, run and joint report. |
| 28 Nov – 2 Dec | Figures and poster assembly, with nights spare for NeXT follow-ups if wanted. |
| 3 Dec | Poster PDF submitted. |

**Data still to fetch on the server.** GQA scene graphs: the official `sceneGraphs.zip` from the GQA download page (it holds `val_sceneGraphs.json`). Record its sha256 in the colour-search output.

**Limitations to state on the poster:**
- GQA's attribute labels are incomplete, which biases colour search toward zero.
- Same-answer duplicate referents exist in the verified sets.
- NeXT is exploratory because of E3.
- The mechanism analyses were specified after the main outcomes were seen.

**No issue tracker.** None is configured for this repo, so this spec is saved here, as the earlier specs were. Running `/setup-matt-pocock-skills` would set one up.
