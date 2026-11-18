# Spec: pre-registered GQA replication on fresh items

Status: ready for agent. Date: 15 November 2026. Deadline for the whole study: 3 December 2026, 23:59 KST.

## Problem Statement

The GQA confirmatory run found that misleading text steers PW-attn's visual-KV selection at K = 144: S_H^(mis) = +0.111 nats [+0.080, +0.144]. The follow-ups then suggested the effect grows as the budget shrinks (S_H^(mis) +0.050 at K = 288, +0.219 at K = 72). They also suggested E_harm^(mis) changes sign at K = 72, and that irrelevant text steers selection too (S_H^(irr) +0.041 on the control subset).

Those budget and irrelevant-text findings are exploratory. They were found on the same 1,000 items, after looking at the outcomes, so a reviewer can call them forking paths. The researcher needs them confirmed on items no comparative run has ever touched, under rules written down before any outcome exists. They also need the irrelevant-text threat (the second case in the threat model) tested on every item, not only on the 300-item control subset.

## Solution

A second, pre-registered confirmatory run on GQA, the **replication**, on a fresh set of 1,000 hand-verified items from the same screened pool. These items come after the frozen main set in manifest order and share no image with any item already run.

- **Conditions:** misleading and irrelevant on every item, matching on the first 300.
- **Budgets:** K = 72, 144 and 288.
- **Protocol:** the same model, prompt, surface rule, margin, selector and bootstrap as the frozen study.
- **Hypotheses and family:** stated in a new PREREG section, frozen by the new code's own freeze record before the run.
- **Analysis:** one analysis command produces the pre-registered report, with Holm decisions final at once, because the replication is GQA-only by design.

The researcher's workflow is the one they already know:
1. Count the pool.
2. Review the PREREG section and remove its DRAFT mark.
3. Freeze.
4. Run once on the GPU.
5. Report.

## User Stories

1. As the researcher, I want the replication set chosen by a fixed rule from items already screened and hand-verified, so that I don't have to verify anything again.
2. As the researcher, I want the replication set to start right after the frozen main set's last item in manifest order, so that the two sets are disjoint by construction.
3. As the researcher, I want the selection to refuse to skip an unverified eligible item, as the main-set rule does, so that the set is a valid sample in seeded image order.
4. As the researcher, I want items whose image appears in the frozen main set left out of the replication set, so that the replication is independent at the image level, the unit the bootstrap resamples.
5. As the researcher, I want every item that has a result file in any earlier comparative, pilot or follow-up run folder left out, so that items whose outcomes were already produced (such as the 9 extra items of the first GQA run) can't enter the replication.
6. As the researcher, I want the list of excluded item ids, and the reason for each, recorded in the freeze record, so that a reader can audit the exclusions.
7. As the researcher, I want a no-GPU command that prints how many items qualify and where the next unverified item is, before I freeze, so that I know the set reaches 1,000.
8. As the researcher, I want the selection to report a clear error naming the next item to verify when fewer than 1,000 qualify, so that I know exactly what to do next.
9. As the researcher, I want the first 300 replication items to form the control subset, so that the matching-description condition runs on the same kind of subset as before.
10. As the researcher, I want misleading and irrelevant cells A, B, C and D run on every item at each of K = 72, 144 and 288, so that R1, R2 and R5 have the full sample at every budget.
11. As the researcher, I want Full_Q, Full_mis and Full_irr on every item, so that E_harm can be computed for both conditions.
12. As the researcher, I want matching cells B, C and D plus Full_match on the control subset at each K, so that T_match and S_H^(match) can be reported as exploratory.
13. As the researcher, I want Blind_Q run on every item as the same consistency check, so that the replication repeats the blind-baseline guard.
14. As the researcher, I want the check that Full_Q is still eligible repeated, so that nondeterminism would show up.
15. As the researcher, I want one probe per context per item reused for all three budgets, so that the GPU run stays within a few hours on the 20 GB A100.
16. As the researcher, I want the run to write each item's file only when that item is complete, and to skip finished items on rerun, so that a crash or a closed tmux session never costs work or causes a double run.
17. As the researcher, I want the run to print only progress, never margins or answers, so that I can't peek before the report.
18. As the researcher, I want a freeze command that records hashes of PREREG.md, the stage-1 record, the manifest, the screen flags, the replication set with its verification digests, the exclusions, the settings and all the code (including subpackages), so that any later change is detected.
19. As the researcher, I want the freeze to refuse while the replication section of PREREG.md is still marked DRAFT, so that I review it before freezing.
20. As the researcher, I want the run to refuse if anything recorded at the freeze has changed, so that the replication runs once under the frozen protocol.
21. As the researcher, I want adding the replication code to leave the GQA stage-3 record's code hash unchanged, so that `confirm-report --gqa-record` keeps working without another code amendment.
22. As the researcher, I want R1 (S_H^(mis) ≠ 0 at K = 72 and at K = 144) tested on the full replication set, so that the main effect and its growth at small budgets are confirmed.
23. As the researcher, I want R2 (S_H^(irr) ≠ 0 at K = 72 and at K = 144) tested on the full set, so that irrelevant-text steering is confirmed as its own threat.
24. As the researcher, I want R3 (E_harm^(mis) ≠ 0 at K = 72) tested, so that the sign change seen in the follow-ups is checked on fresh items.
25. As the researcher, I want R5 (T_irr ≠ 0 at K = 72) tested on the full set, so that I learn whether misleading text steers more than irrelevant text.
26. As the researcher, I want R1, R2, R3 and R5 treated as one Holm family of 6 tests at α = 0.05, so that the multiplicity correction is fixed in advance.
27. As the researcher, I want R4 tested as equivalence: S_H^(mis) at K = 288 has a 90% clustered interval inside ±0.10 nats. I want it reported separately, outside the Holm family, so that "the effect vanishes at a generous budget" is either supported or not.
28. As the researcher, I want every interval, p-value and reading computed with the frozen estimand definitions, bootstrap (10,000 resamples, seed 0, image clusters), p-value formula and reading rules, so that the replication can be compared directly with the original.
29. As the researcher, I want exploratory, interval-only rows in the report: S_Q, I, E_harm^(irr) at every K, S_H^(mis) at K = 288, the behaviour rates, Jaccard overlap by K, and T_match and S_H^(match) on the control subset. These let me describe the results without adding tests.
30. As the researcher, I want the report to say whether the original and replication estimates agree, side by side for each K, so that the paper can show both.
31. As the researcher, I want the report to refuse if the run folder holds anything other than the frozen replication set, so that stray files can't change the analysis.
32. As the researcher, I want the report to refuse if the code changed since the replication freeze, unless the change is a recorded code amendment, so that analysis code can't drift after the outcomes exist.
33. As the researcher, I want copy-pasteable commands in the README for count, freeze, run and report, with runtimes and what to send back, so that I can run everything on the server without asking.
34. As the agent building this, I want every new command runnable on CPU against the tiny test models, so that the GPU run is the first real-model run, not the first time the code executes.
35. As a reader of the paper, I want the replication's PREREG section dated and frozen before its run, with its record committed, so that I can verify that the hypotheses predate the data.

## Implementation Decisions

- **Code location.** The replication lives in a new subpackage next to the follow-ups, outside the flat module set that `code_hash_all` hashes, so the GQA stage-3 record's hash stays valid (story 21). Its own freeze record hashes all of the code recursively: the top-level modules, the follow-up subpackage and the replication subpackage.
- **Entry point.** A `python -m` command group with subcommands `count`, `freeze`, `run` and `report`, matching the follow-ups. It stays out of the `kvsel` CLI for the same reason.
- **Reuse, not copies.** Selection reuses the frozen main-set rule, the screen flags loader and the verification decision loader. The run reuses `run_unit`, `contexts_for`, prefill and the PW-attn probe. Analysis reuses `estimands`, `cluster_bootstrap`, `holm`, `margin_variant`, the readings and the interval formatting. Nothing in the frozen modules changes.
- **Selection rule.** This is the first decision to write into PREREG.
  1. Take eligible items (amendment-1 screen flags) in manifest order, starting after the last item of the frozen main set.
  2. Skip items whose image_id appears in the frozen main set, and items with a result file in any listed earlier run folder.
  3. Stop at the first unverified eligible item; failed items are passed over.
  4. Take the first 1,000 that passed. The first 300 form the control subset.

  Verification decisions are bound by item digest as before, so stale decisions count as unverified.
- **Excluded-run folders.** These are given explicitly on the command line (the confirmatory run, the follow-up run, any pilots on confirmatory items). The freeze records them and the ids they excluded.
- **Freeze record.** A separate record for the replication. It holds:
  - PREREG sha256 and the stage-1 record sha256 (amendment 2)
  - manifest sha256 and screen-flags sha256
  - the hash of the replication set with its verification digests
  - the exclusions
  - the recursive code hash
  - the settings: units, budgets, sizes and the hypothesis list

  It refuses to overwrite a different record, as `freeze_comparative` does.
- **Units per item.** Selector: PW-attn only, which is the primary selector.
  - For each K in {72, 144, 288}: A, plus B, C and D for mis and irr.
  - Full_Q, Full_mis and Full_irr, plus Blind_Q.
  - On the control subset, for each K: B, C and D for match, plus Full_match.
  - Left out: question-span selector, reserve variants, uniform grid, LT-early, random sets and restoration. They are exploratory in the original study, and leaving them out keeps the run to about 4 to 5 GPU hours.
- **Result rows.** Rows carry K, so columns are told apart by budget. The report splits rows by K before pivoting, then calls the existing estimand function per K.
- **Probing.** One prefill per item. One PW-attn probe per context (Q, mis, irr, plus match on control items). Kept sets for all three budgets come from the same scores via top-K.
- **Hypotheses (PREREG section 3, new).**
  - The Holm family of 6 at α = 0.05:
    - R1: S_H^(mis) at K = 72 and at K = 144
    - R2: S_H^(irr) at K = 72 and at K = 144
    - R3: E_harm^(mis) at K = 72
    - R5: T_irr at K = 72, all items
  - R4: equivalence of S_H^(mis) at K = 288 with δ = 0.10 nats (90% interval inside ±δ), reported separately.
  - Everything else is interval-only.
- **Definition change to state in PREREG.** T_irr is computed on all 1,000 items in the replication, not on the control subset, because irr now runs on every item.
- **Holm decisions are final when the report runs.** The replication is GQA-only by design, so nothing is pending.
- **Original vs replication.** The report reads the original confirmatory and follow-up result files only to print their estimates beside the replication's, labelled as the original exploratory values. No test pools the two sets.
- **Pool-count command.** The scratch count script (`usable_items.py`) becomes the `count` subcommand of the replication entry point, extended with the image and earlier-run exclusions. It prints counts only, never model outcomes.

## Testing Decisions

- **What a good test is.** It checks behaviour at the command or public-function level on tiny synthetic inputs: which items get selected, which rows exist, what the report computes. It doesn't check internal helpers or exact print text.
- **Proposed seams.** There are four, all at the top of the subpackage; please confirm.
  1. **Selection.** A pure function from (manifest items, screen flags, decisions, frozen main set, excluded ids) to the replication set. Test it on synthetic manifests:
     - It starts after the cutoff.
     - It drops shared images and earlier-run ids.
     - It stops at an unverified item and passes over failed ones.
     - It treats stale digests as unverified.
     - It gives exactly 1,000 items, with a 300-item control subset.
  2. **Freeze and check.** The record round-trips. It refuses with DRAFT present, refuses a different existing record, and refuses to run after a change to PREREG, the manifest, the flags, the decisions or the code.
  3. **Run.** An end-to-end run with the tiny LLaVA model on a handful of items checks four things:
     - The expected unit set per item: main vs control, every K.
     - Resume skips finished items.
     - Nothing beyond progress is printed.
     - The PW-attn kept sets match `top_k` of one probe at each K.
  4. **Report.** A synthetic run folder with planted effects checks four things:
     - The Holm family of 6 gives the right decisions.
     - R4 is equivalent when the planted effect is about 0, and not equivalent when it is large.
     - The report refuses an incomplete folder or extra files.
     - The report refuses changed code unless the change is a recorded amendment.
- **Prior art.** `tests/test_confirm.py` covers selection, freeze and run with tiny models. `tests/test_confirm_analysis.py` covers planted-effect report tests and the amendment chain. `tests/test_followup.py` covers subpackage commands and per-K units.
- **Unchanged guard.** A test that `code_hash_all` still equals the GQA stage-3 record's accepted hash after the subpackage is added. It is skipped if the record is absent.
- **Before handing off.** The full suite passes on CPU, and the README runbook is updated.

## Out of Scope

- LLaVA-NeXT-7B (its own spec; validated by 24 November or dropped).
- The exploratory controls on the original 1,000 items: colour search, referent following, steering maps, answer flips, negation and lexical controls, the hint-masked probe.
- TextVQA, both its original confirmatory run and any TextVQA replication.
- Published-method baselines beyond framing (VisionZip and others).
- Question-span, reserve, uniform-grid, LT-early, random and restoration units on the replication items.
- Any change to the frozen GQA study, its records or its analysis code.
- Multi-turn spillover (dropped).

## Further Notes

- Ordering and integrity:
  - No comparative unit may run on any replication item before the replication freeze is committed.
  - `count` must not load the model.
- If fewer than 1,000 items qualify after the exclusions, the researcher must verify more items in manifest order before freezing. The count is unknown until `count` runs on the server, and it depends on how many images the main set used up.
- The equivalence power projection for R4 should be stated in PREREG from the original follow-up's K = 288 standard error, labelled as a planning number.
- The issue tracker isn't configured for this repo, so this spec is saved here instead of published. Running `/setup-matt-pocock-skills` would set one up.
