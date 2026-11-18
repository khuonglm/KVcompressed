# Spec: Reorganize the codebase into one package with shared protocol, model and analysis layers

Status: ready-for-agent. Date: 18 November 2026 (revised the same day after reading every source, script, record and test file).

Supersedes the NeXT rework tickets of 17 November 2026, whose code changes were reverted. Their NeXT findings are carried into this spec (see "NeXT corrections after the reorganization").

## Problem Statement

The researcher has one study told in four parts: the confirmatory GQA/TextVQA experiment with its development pilot, the follow-ups, the GQA replication, and the second model, LLaVA-NeXT. Each part was added after the previous one was frozen. Frozen code could not change, so each part copied what it needed instead of sharing it, and said so in its docstrings ("mirrors … instead of calling them: those are frozen code"). The codebase now reads as four parallel codebases.

**Freezing and records**
- Freeze logic exists four times: stage 1, the confirmatory stage 3, the replication and NeXT each compute state, write records and check them their own way.
- The code hash exists four times, with four scopes:
  - a hand-kept list of screening modules, with a test to keep the list honest;
  - the top-level modules;
  - the whole package tree;
  - two packages.
- Some checks honour the amendment log and the stage-1 check does not.
- There are three different rules for a draft PREREG section: "DRAFT" anywhere in the preamble; a line starting with DRAFT in section 3; that same line plus "[to fill" blanks in section 4.

**Running**
- The per-unit runner exists twice.
- The probe exists twice: the follow-ups copy it "same operations in the same order" to also read layer 2.
- Kept sets are chosen by four separate dispatchers: development and confirmatory, follow-ups, NeXT, validation.
- There are three unit types and three screens. Seven places run an item loop and write item files, each resuming its own way.
- The verify command tells a development screen from a confirmatory one by reading the first file's format.

**Analysis**
- Behaviour rates, overlap and per-budget estimand blocks are computed three times.
- `jaccard` exists twice, and the two copies disagree on empty sets.
- The newer reports import the older ones' private helpers.
- The confirmatory stage-3 module and the analysis module import each other (one direction is hidden inside a function).

**Models**
- LLaVA-1.5's layout is hard-coded about 56 times (576 tokens, a 24×24 grid). So NeXT re-implements selection, prompts, prefill, the per-unit runner, model files and validation in a separate top-level package.
- The model-family table holds one family. A `--model` option offers that one choice, left over from the removed Qwen support. The engine still documents Qwen's position scheme.

**Command line and errors**
- There are four command-line entry points; three exist "so the frozen code hash doesn't change".
- Library code exits the process in 67 places, so tests drive the command line and match message text.

**Tests**
- Six guard tests exist only to prove that frozen code was not touched:
  - the stage-1 hash still matches its record;
  - the stage-3 hash is still accepted;
  - the replication tree hash is still accepted;
  - the follow-ups and NeXT live outside the hashed files;
  - the screening module list matches the screen's imports;
  - section 1 still matches the stage-1 record.

The cost shows in review. A reader can't check one rule against the pre-registration in one place. Every fix is repeated in several copies, and every new study adds another copy. Two code reviews of the NeXT work kept finding problems whose root cause was this structure. The 3 December deadline leaves time for one reorganization, not for another round of patches.

## Solution

One package, organized by domain concept rather than by the order in which the studies happened. Everything shared lives once:

- items and verification;
- model families, with their visual layouts;
- the replay engine;
- selection;
- units, runs and screens;
- the protocol (freezes, records, code identity);
- analysis building blocks;
- validation;
- one command line.

Each study becomes a small declaration (stages, settings, unit lists, main-set rule, Holm family) plus only the logic unique to it, including how its report lays out its results.

LLaVA-1.5 and LLaVA-NeXT become two implementations of one model-family interface. LLaVA-1.5's layout is one 24×24 map with no newlines; NeXT's is a base map plus an unpadded high-resolution map with newlines. Selection, budgets, validation and overlap work on the layout, so nothing assumes 576.

The reorganization must not move any LLaVA-1.5 result. It is code amendment 4: dated, documented in PREREG, and accepted only with evidence that every model input, every tiny-model output and every stored report is reproduced exactly. A reproduction command that reruns every stored report and compares it does that check, and serves every later amendment too.

NeXT corrections from the two reviews come after the reorganization, as separate changes written into PREREG section 4 before NeXT's stage 1 is frozen again.

## User Stories

### Reading and reviewing

1. As a reviewer checking the code against the pre-registration, I want each study's rules (stages, settings, unit lists, budgets, main-set rule, Holm family, α) declared in one place per study, so that I can compare it with its PREREG section line by line.
2. As a reviewer, I want one freeze mechanism shared by every stage of every study, so that I verify how freezing works once instead of four times.
3. As a reviewer, I want one definition of the code hash and one rule for accepting changed code, so that "the code was frozen" means the same thing for every record.
4. As a reviewer, I want each statistic (estimand blocks, behaviour rates, overlap, Holm, intervals) computed by one function that every report calls, so that two studies can't compute the same estimand two ways.
5. As a reviewer, I want each module to change for one reason (layout, selection, engine, protocol, analysis, one study), so that a diff touching a module tells me what kind of change it is.
6. As a reviewer, I want no module to import another module's private helpers, and no import cycles, so that every dependency I see is a deliberate public interface.
7. As a reviewer, I want the domain vocabulary of PREREG used as names in the code, so that I don't translate between two vocabularies.
8. As a reviewer, I want each name to mean one thing ("pool", "family", "record" and "kept sets" each currently mean two or three things), so that I never have to check which meaning applies.
9. As a reviewer, I want functions short enough to read in one sitting and at one level of detail, so that I can verify each against its rule.
10. As a reviewer, I want documented coding standards in the repository, so that a review can cite a rule instead of a preference.
11. As a reviewer, I want stale text gone (Qwen notes in the engine, a one-family model table, wrapper functions that only forward), so that what I read is what runs.

### Running studies (the researcher)

12. As the researcher, I want one command-line program with a group per study, so that I don't remember four entry points and their differing option names.
13. As the researcher, I want every option with the same meaning to have the same name and default in every command, so that runbooks don't drift.
14. As the researcher, I want every refusal to print one clear message and exit with one exit code, so that the overnight scripts can tell "stop: action needed" from a crash.
15. As the researcher, I want one resumable item loop for every screen and run, so that a crash anywhere resumes the same way.
16. As the researcher, I want a resumed run to refuse an existing item file that was produced under a different record, subset or unit list, so that leftovers from an earlier freeze or a test run can't enter a result.
17. As the researcher, I want freeze records written atomically, so that an interrupted freeze can't leave a truncated record that a rerun then trusts.
18. As the researcher, I want every existing freeze record to remain checkable by the new code, so that the reorganization doesn't strand a finished study.
19. As the researcher, I want a screen folder to say whether it is a development or a confirmatory screen, so that verification never guesses from file contents.
20. As the researcher, I want the README runbooks and the NeXT overnight script updated to the new commands, so that I can run NeXT from the README alone.
21. As the researcher, I want commands for finished studies to keep working for their reports, so that I can regenerate any reported number.
22. As the researcher, I want the GPU memory behaviour NeXT needs (prefix held in CPU memory, head computed in slices without an fp32 copy) available as a configuration of the shared engine, so that NeXT still fits on the 20 GiB slice.

### Reproducing results (the amendment)

23. As the researcher, I want one reproduction command that reruns every stored report from its run folder with the current code and compares it with the stored results, so that every code amendment is proven the same way.
24. As the researcher, I want the list of stored reports and their inputs kept in one committed data file, so that the reproduction command needs no editing for the next amendment.
25. As the researcher, I want the reproduction command to name the first difference when it fails, so that I can tell a real change from noise.
26. As the researcher, I want the reproduction command to write its evidence into the amendment log only when everything reproduced, so that the log never claims evidence that doesn't exist.
27. As the researcher, I want model inputs (prompt strings, prefix and suffix tokens, spans, candidate tokens) and item digests compared on every manifest before and after, so that verification decisions stay bound to the same items.
28. As the researcher, I want the tiny-model outputs of every command compared with copies captured from the code before the reorganization, so that behaviour preservation is checked on every commit, not only at the end.
29. As the researcher, I want a short GPU spot check (a few stored items of each LLaVA-1.5 run, rerun and compared row for row), so that changes to the engine path are verified on the real model.
30. As the researcher, I want the settings each existing record froze recomputed by the new study declarations and compared exactly, so that a declaration can't drift from what was frozen.
31. As the researcher, I want one record-compatibility test that checks every record against the current code through the amendment log, so that the six "nothing frozen was touched" guard tests become one honest check.
32. As the researcher, I want the PREREG amendment placed outside every hashed section, so that no stage-1 section hash changes.

### Models and layouts

33. As the researcher, I want LLaVA-1.5 and LLaVA-NeXT to implement one model-family interface, so that adding a third model means writing one implementation.
34. As the researcher, I want the visual layout (maps with their shapes, newline positions, feature positions) to be an explicit object for both models, so that budgets, pooling, overlap and kept sets work the same way on a 576-token grid and an AnyRes image.
35. As the researcher, I want budgets as a small type (an absolute K or a share of feature tokens), resolved against a layout, so that "K = 144" and "25%" can't be confused.
36. As the researcher, I want one kept-set entry point for every selector (PW-attn, question-span, LT-early, uniform, random, full, coverage reserve), so that adding a selector means one change.
37. As the researcher, I want pooling to stay within each map, top-k ties to go to the lower index, and newlines to be always kept and outside k, all in one selection module, so that both models select by the same code.
38. As the researcher, I want one probe that can also return the layer-2 attention of the last suffix token, so that the follow-ups' LT-early scores don't come from a copied probe.
39. As the researcher, I want the evidence-box mapping to be a function of LLaVA-1.5's fixed-grid layout, used by verification, H6 and restoration, so that model-specific pixels don't leak into shared code.

### Validation

40. As the researcher, I want one validation command with one function per E-check, parameterized by model family, so that LLaVA-1.5 and NeXT are validated by the same code.
41. As the researcher, I want each family's check list, tolerances and GO rule declared with it, so that the criteria in tests/README can be checked against one place.
42. As the researcher, I want validation to print a final GO or NO-GO and exit non-zero on NO-GO, with a test that forces a failure, so that the gate can't pass by construction.

### Developers and agents

43. As an implementing agent, I want the work split so that every intermediate commit keeps the characterization suite green, so that I can stop at any ticket without a broken tree.
44. As an implementing agent, I want a characterization suite at the command line, captured before any code moves, so that I can move code freely and know at once when an output changes.
45. As a maintainer, I want library code to raise one error type for refusals, converted to an exit only by the command line, so that library functions can be tested and composed without catching process exits.
46. As a maintainer, I want pure modules (selection, statistics, layout) to have no model or file access, so that they test in milliseconds.
47. As a maintainer, I want shared test fixtures (one item factory, one tiny model per family, one study "world"), so that tests read as specifications instead of setup.
48. As a maintainer, I want lint rules that catch unused code and import cycles to run in the test command, so that the cleanliness survives after this spec.
49. As a maintainer, I want dead code (paths no command or study reaches) deleted, so that the codebase holds only what the study uses.

## Implementation Decisions

### Domain vocabulary (use these names in code)

- **Item:** one question with a_true, a_alt and descriptions per condition.
- **Condition:** mis, match or irr.
- **Context:** Q or a condition.
- **Cell:** A, B_c, C_c, D_c, or a Full or Blind anchor.
- **Unit:** selector, budget, reserve flag, selection context, answering context, plus the follow-ups' seed, m and arm.
- **Selector:** pw, qspan, lt, uniform, random, restore or full.
- **Budget:** an absolute K or a share. **Kept set:** the visual positions kept, including newlines.
- **Layout:** the maps and newlines of one image's visual span.
- **Prefix and suffix; probe; replay; compact and dense.**
- **Margin:** a score difference under a surface rule. **Estimand.**
- **Main set and control items; screen and eligibility; verification decision.**
- **Stage, record, settings; code identity; amendment and amendment log.**
- **Study:** a pre-registered experiment with its own PREREG section and records: the development pilot, the confirmatory study, the follow-ups, the replication, NeXT.

### Modules (one package; each changes for one reason)

- **Items:**
  - the GQA and TextVQA builders, as pure functions of raw files, version and seed;
  - the helpers they share (the development split, JSONL writing, the description-length rule), which today live in the GQA builder and are imported by TextVQA's;
  - manifest reading;
  - image fetching.
  - The builders' description-length filter tokenizes with LLaVA-1.5's tokenizer, so manifest reproduction needs it.
  - Manifests produced by the builders are byte-identical to the current ones.
- **Verification:**
  - the hand-verification tool and the append-only decision log;
  - the item digest, whose output is unchanged.
  - The page's HTML becomes a package resource instead of a string constant in code.
- **Model families:** one interface, two implementations.
  - The interface covers:
    - model identity, with an optional pinned revision;
    - required files, fetch, and resolution of the local copy;
    - tokenizer and image-processor options;
    - prompt text;
    - tokens for every context, with a prefix shared across contexts;
    - candidate tokens;
    - the layout of an image;
    - prefill;
    - the blank image;
    - numerics (dtype, head precision, head slice size);
    - prefix residency (on the device, or held in CPU memory).
  - LLaVA-1.5 stays unpinned: its records name no revision, and requiring a revision marker would make the existing local copy "incomplete".
  - The package-level constants (model id, image token id, 576, grid side) move into the LLaVA-1.5 implementation; nothing else refers to 576 or 24.
  - The one-entry family table and the `--model` option are replaced by the two-family registry.
- **Layout:** an immutable object listing each map (offset, rows, columns) and the newline positions, with feature positions derived from it.
  - LLaVA-1.5: one 24×24 map, no newlines.
  - NeXT: the base 24×24 map, then the unpadded high-resolution map with a newline ending each row, derived from the model's own AnyRes code and checked against the processor's expansion.
  - The image transform (pad to square, padding colour) and the evidence-box mapping rules are LLaVA-1.5 layout functions. Verification, H6 and the follow-ups' restoration sets call them.
- **Engine:**
  - cache compaction, masks and hashing;
  - the decoder;
  - the prefix;
  - one probe, which optionally also returns the layer-2 attention of the last suffix token;
  - replay, candidate scoring and greedy decoding.
  - **The fp32 head:** computed either from a stored fp32 copy or in row slices of the fp16 head; the slice size is a decoder setting chosen by the family. LLaVA-1.5 keeps its current head path, so its numerics don't change.
  - **Prefix residency:** a setting of the replay. A CPU-held prefix is gathered on the CPU and moved per unit; a device-held prefix is replayed in place.
  - The prefix's separate logical length, kept for Qwen, stays only if NeXT's gathered prefix needs it; its Qwen documentation goes.
- **Selection:** pure NumPy.
  - Window scores (PW-attn), question-span scores, LT-early scores, pooling within each map, top-k with ties to the lower index, the uniform grid, the coverage reserve and seeded random sets.
  - Budgets resolved against a layout. A share that isn't a whole number of feature tokens is refused.
  - One kept-set entry point for every selector. Newlines are always kept and outside k.
  - The follow-ups' restoration sets stay follow-up logic.
  - Jaccard is defined once, with the analysis's rule that two empty sets have overlap 1.
- **Units, runs and screens:**
  - one unit type for every study, and cell naming;
  - the per-unit runner, with one ordering (score, then greedy) for both families;
  - the per-item runner;
  - eligibility;
  - one resumable item loop for every screen and run.
  - **The blind anchor** is a Full unit run on the blank image, labelled "blind" in its row, in one place: today the confirmatory, replication and NeXT loops each relabel it.
  - **Item files** are written atomically and carry the hash of the record they were produced under, their subset and their unit list. On resume, a mismatched file is refused, never overwritten.
  - **Attention arrays** are saved for the studies that save them today (development pilot, confirmatory run).
  - **A screen folder records its kind** (development rows, or confirmatory flags only). Eligibility flags are read from one place.
- **Protocol:** the freeze mechanism, shared by every stage of every study.
  - **A stage declares:**
    - its name and record location;
    - its inputs: whole PREREG file or a named section, manifests, other records, the screen flags, the selected set with verification digests;
    - its settings;
    - the PREREG section whose DRAFT line or "[to fill" blank it refuses. One rule: a line starting with DRAFT.
  - **Freeze:** write-once and atomic (fsynced before the rename). It returns the existing record when the state is equal, code included through the amendment log, and refuses when it differs.
  - **Check:** recomputes the state and returns the record it verified. It refuses with every changed input named. A report-time check doesn't compare PREREG, which may grow after the run; the run-time check refuses any PREREG change (decided 18 November 2026).
  - **Refusals** raise one error type. Only the command line turns it into an exit.
  - **The main-set rules stay study logic,** since each is a distinct pre-registered rule: the confirmatory first 1,000 in manifest order; the replication's fresh items after the main set; NeXT's first 1,000 eligible among passed items.
- **Code identity:** one function, the hash of every source file in the package (`.py`, and `.html` resources) by relative path.
  - Every stage's code check accepts the frozen hash, or any hash reached from it through the amendment log.
  - This includes stage 1, which today compares exactly. The change is required, not optional: the replication's report check calls the stage-1 check, so without it no reorganization could keep the replication report running.
  - Old records keep their old hashes. Amendment 4 adds, for each record's latest accepted hash, one log entry to the new hash, with evidence.
- **Analysis:** building blocks that every report uses.
  - **Statistics:** margins and variants, estimands, image-clustered bootstrap, Holm, interval readings, clustered OLS.
  - **Run tables:** loading a run, wide margins, cell presence, overlap, behaviour rates, paired behaviour differences, consistency checks.
  - **Report blocks:**
    - completeness of a run against its record;
    - the estimand block at one budget;
    - the Holm family;
    - a comparison against a named, hashed source file;
    - the public markdown table and number formatting;
    - one writer for results and markdown.
  - **Each study composes its own results layout and markdown,** because those must stay byte-identical. The confirmatory report's datasets and families, the follow-ups' parts, the replication's budgets and equivalence, and NeXT's shares differ, down to whether the file ends with a newline.
- **Studies:** one module per study (development pilot, confirmatory, follow-ups, replication, NeXT).
  - **Each declares:**
    - its stages, settings and unit lists;
    - its main-set rule, Holm family and α;
    - its report composition.
  - **Each holds only its unique logic:**
    - the follow-ups' parts and restoration sets;
    - the replication's selection of fresh items and its reading against the original estimates;
    - NeXT's main set by share;
    - the development pilot's development-only guard and rehearsal reports.
- **Validation:**
  - one function per E-check, each returning a typed result;
  - parameterized by family, with the check list (E5 for LLaVA-1.5), the budgets or shares, the kept-set kinds, tolerances and GO rule declared per family;
  - both head precisions' noise floors reported for every family;
  - memory and time projections for any family;
  - the handler only prints.
- **Command line:**
  - one program;
  - groups for data, model, verify, validate, reproduce, and one group per study, whose subcommands come from its stages (screen, count, freeze, run, report);
  - shared argument helpers, with one name and default per meaning;
  - one place that turns a refusal into an exit;
  - exit codes: 0 done, 1 error, 2 refusal or stop for action.
- **Reproduction command:**
  - reads a committed list of stored reports: each report's study, inputs, stored results path, and whether comparison is byte-exact or on stored keys;
  - reruns each report into a fresh folder and compares;
  - names the first difference;
  - fills the evidence of pending amendment-log entries only when everything reproduces.
  - It replaces the one-off amendment-3 script.
- **Scripts:**
  - the NeXT overnight script is updated to the new commands;
  - the finished replication's overnight script and the amendment-3 script are removed (git history keeps them);
  - the figures script reads only stored result files, imports nothing from the package, and stays as it is.
- **Coding standards:** a short standards document in the repository, plus lint configuration run by the test command.
  - **Documented rules:**
    - library code doesn't exit;
    - records and item files are written atomically;
    - pure modules don't touch the model or files;
    - each frozen rule cites its PREREG section in its docstring;
    - no private imports across modules and no import cycles;
    - one name per meaning;
    - no wrapper that only forwards.
  - **Lint:** a small rule set covering unused code, import order and modern syntax.

### Record, row and PREREG compatibility

- **Records:** no existing freeze record is rewritten or deleted.
- **Settings:** the new study declarations recompute exactly the settings each record froze, with the same key names:
  - the latest LLaVA-1.5 stage-1 screening settings;
  - the confirmatory stage-3 settings (K = 144; the development pilot's full unit grid on control items, its mis-only grid on main items);
  - the replication settings;
  - the NeXT stage-1 and stage-3 settings.
- **Unit serialization** matches what records and item rows hold today, field for field:
  - with an absolute budget: selector, k, reserve, selection, answering, plus seed, m and arm when the follow-ups set them;
  - with a share: selector, share, selection, answering.
  - The blind unit is recorded with selector "full", as frozen.
- **Rows** keep their schema: the unit's fields, cell, kept (none for Full), scores by candidate and surface form, greedy ids and text, outcome, truncation and top-two gaps; NeXT adds share and layout. Item files keep their top-level fields.
- **PREREG gains one dated section after section 4, "Code amendment 4: reorganization, no rule changed".** It states:
  - the change and its reason;
  - that no confirmatory outcome motivated it;
  - the timing (after the replication report was seen; before NeXT's comparative run);
  - the old-to-new command names;
  - the single code-identity scope and the amendment-log rule now applied to every stage;
  - the evidence.
- Sections 1 and 4.1 are not edited by amendment 4, so no stage-1 section hash changes.

### Outputs that may change

Only these:
- command names and help text;
- validation's report layout and printed text (not a frozen artifact), though every verdict and measured value stays identical on the tiny model;
- the development screen's folder gains its kind marker;
- item files gain their record hash.

Everything else a command writes stays byte-identical.

### NeXT corrections after the reorganization

These are rule or behaviour changes for NeXT only. They land after amendment 4 is accepted, are written into PREREG section 4 as a dated amendment before NeXT's stage 1 is frozen again, and don't change any LLaVA-1.5 output:

- **Prompt tokens:** NeXT's tokens outside the image span equal LLaVA-1.5's. This restores the space token before the newline; the tokenizer option that does it is recorded in the screening settings. E1 compares against LLaVA-1.5's own tokenizer.
- **Image processor:** the slow processor is pinned explicitly and recorded in the screening settings.
- **Full rows:** a Full row's k is the feature count, and every row records its layout with the newline count.
- **Overlap:** computed on feature positions, newlines excluded.
- **Validation:**
  - E3 adds a "tail" kept set (the last k feature tokens), so every share covers the high-resolution map;
  - E4 also hashes the CPU-held prefix;
  - E7 reruns every unit, including the probe-scored ones;
  - the noise floor is reported under both head precisions.
- **Report records:** the LLaVA-1.5 comparison file's path and sha256, and the resample count.
- **Records:** the first NeXT stage-1 and stage-3 records and their outputs are set aside under a superseded location named in the amendment. NeXT is re-frozen and rerun.

## Testing Decisions

### What makes a good test here

- It tests behaviour through a public interface, ideally the command line, and asserts on outputs a reader cares about: files, records, rows, printed verdicts and exit codes.
- It fails if the behaviour it names breaks. A test whose assertion is computed from the output it checks, or that accepts either outcome, is not a test.
- It doesn't re-implement the code under test to compute its expectation; it uses known answers (hand-built fixtures, planted effects, captured outputs).
- One named behaviour per test; shared setup lives in fixtures.

### Seams

1. **The command line on fixture inputs (primary, and the seam for the whole reorganization).**
   - **Before any code moves:** run every command on hand-built fixtures and the tiny models, and capture its outputs as golden files. The commands are build, screen, freeze, run (including the follow-up parts), report, validate and the scriptable parts of verify. The captures include item files, attention arrays, records, results and markdown. Timestamps and code hashes are masked.
   - **After every change:** the same command under its new name on the same inputs must produce identical files and the same exit code, apart from the listed outputs that may change.
2. **The engine's Level-1 invariants (kept, required by the acceptance policy).**
   - compact equals dense with dropped columns masked;
   - logical positions pass and physical positions fail;
   - perturbing dropped entries changes nothing;
   - the shared prefix is never mutated.
   - Run for both families' layouts, including a 4:3 image that gets NeXT's 2×2 grid, and for both prefix residencies.
3. **Real-artifact reproduction (outside the test suite).**
   - The reproduction command on the stored run folders, on the server, without a GPU.
   - The model-input and item-digest comparison on every manifest.
   - The GPU spot check of LLaVA-1.5 rows. The researcher runs these and their output is the amendment's evidence.

Module-level tests survive only for deep modules with public interfaces: selection (E6 selector tests), statistics (simulated data with known answers), layout derivation, the builders' rules on hand-built fixtures, the verification server, image fetching, and the protocol's freeze and check. They are rewritten against the new interfaces. Tests of private helpers are deleted.

### The record-compatibility test and the per-commit gate

The six guard tests compare the code against the real records. They would fail from the first moved file until amendment 4's log entries exist. They are replaced by one parametrized test: every record's code hash reaches the current code through the amendment log, and every record's settings equal the study declaration's.

- **During the reorganization,** the per-commit gate is the characterization suite plus the module tests. The record-compatibility test is marked expected to fail, with the reason "amendment 4 pending".
- **Amendment 4's ticket** writes the log entries, removes that mark, and makes the test required.

### Required tests beyond the golden files

- A forced failure in validation gives NO-GO and exit code 1.
- An item file from another record, with the wrong subset, or missing a unit is refused on resume and in the report.
- An interrupted freeze leaves no record.
- The code hash changes when any source file changes.
- The device-held and CPU-held prefix residencies give identical rows.
- The sliced and copied fp32 heads give equal logits on an fp16 tiny model.
- The probe's layer-2 output equals the follow-ups' current LT-early input.

### Prior art

- **The tiny-model end-to-end pipeline test:** same classes and calls as the GPU run.
- **The confirmatory, follow-up, replication and NeXT end-to-end tests:** freeze, run, resume and report on tiny models.
- **Simulated-data analysis tests:** planted effects with known answers.
- **The Level-1 operator tests**, and the builder tests on hand-built fixtures.
- **The command-line test** that every command parses its help.
- **Amendment 2's evidence:** model inputs identical on all manifests, tiny-model rows byte-identical.
- **Amendment 3's reproduction script:** byte-identical results files.

## Out of Scope

- Any change to a pre-registered rule, estimand, statistical method or tolerance for the LLaVA-1.5 studies. Their results must reproduce exactly.
- Running the TextVQA comparative experiment, and the exploratory controls planned for the original items.
- Pinning LLaVA-1.5's model revision.
- Performance work beyond what NeXT needs to fit on the 20 GiB slice.
- A type checker, packaging for publication, and continuous integration services.
- Rewriting historical documents (proposal, implementation plan, status report) and the historical sections of tests/README (Qwen, closed failures). README, the current parts of tests/README, and PREREG are updated.
- Any change to stored data: run folders, verification decisions, manifests, freeze records, and the leftovers of removed development commands (the s1 manifests and the checks, coverage and v31 result folders).

## Further Notes

- **Order:**
  1. Golden capture first, before anything moves.
  2. The shared layers: layout and model families, engine settings, selection, units, runs and screens, protocol, analysis blocks.
  3. One study at a time onto them.
  4. The single command line, then deletion of the old modules.
  5. Amendment 4: the log entries, the record-compatibility test made required, and the evidence.
  6. The NeXT corrections.
  The characterization suite stays green at every step.
- **Deadlines:** the NeXT validation gate is 24 November 2026 and the study deadline is 3 December 2026, 23:59 KST. The NeXT GPU validation of memory and time doesn't depend on this reorganization and can run on the current code in parallel, to learn early whether an AnyRes cap is needed. Its numbers stay valid if the engine path's rows are unchanged.
- **Risk:** the GPU spot check is the only real-model evidence for the engine path. If an LLaVA-1.5 row differs, that change is reverted for LLaVA-1.5: the family keeps its current setting. The amendment is not accepted with a difference.
- **Smaller option, not chosen:** keep the current packages and extract only the protocol and analysis layers. That removes about half the duplication, but leaves the 576 assumption and the parallel NeXT package in place.
