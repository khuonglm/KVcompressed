---
name: test-failure-triage
description: Use when a validation test fails (Level 1, E1-E7, noise floor, `kvsel validate` output, a selector test) or when a numeric result looks wrong. Enforces the project's acceptance policy - tolerances are never loosened, a failure closes only when its cause is identified and documented.
---

# Triaging a test failure

The acceptance policy is in `tests/README.md`. It was written before the tests first ran, and it is
binding. Read it before doing anything else.

## Rules that are never broken

1. **Never loosen a tolerance after a failure.** Not "temporarily", not "just to see". A failure
   closes only when its cause is identified.
2. Numerical-policy changes (head precision, dtype of a reduction) are allowed during development
   only, need a documented cause, and require rerunning **every** affected test.
3. Nothing may change in response to a confirmatory scientific outcome.
4. Timing runs have instrumentation (hashing, extra logging) off.

## Procedure

1. **Restate the failure exactly**: test id, measured value, limit, which suffix (Q or QH_c), which K,
   which items. Quote the numbers from the output the user sent.
2. **List candidate causes** that would each produce that number. Typical ones in this codebase:
   - physical instead of logical positions in compact replay;
   - a limit taken from the wrong noise floor (for example Q vs QH_mis, the closed E2 failure);
   - fp16 rounding in hidden states vs in the output head;
   - a prefix tensor mutated by a probe or replay (E4);
   - tokenization: candidate tokens not an exact prefix extension, casing changing token count.
3. **Write a diagnostic whose reading is fixed before it runs.** State, in the message to the user,
   what each possible outcome would mean. `kvsel diagnose-e2` is the model: it separates
   full-vs-compact, full-vs-dense and compact-vs-dense, so each cause predicts a different row.
   Build it CPU-testable, then hand it off with the `gpu-handoff` skill.
4. **Only after the cause is confirmed**, change the code or the test's definition. If the test's
   definition changes, the new version must be at least as strict (E2 gained "compact equals dense
   exactly" when its limit moved to the per-suffix noise floor).
5. **Document the closed failure** in `tests/README.md` under a heading
   `## Closed failure: <test>, <run> (<date>)` with Failure, Cause, and Change paragraphs and the
   diagnostic's table.
6. Ask the user to rerun every test the change affects, normally all of `kvsel validate`.

## What to tell the user

Say plainly that the test failed and what the measured value was. Give the diagnostic command and
what each outcome would mean. Do not propose a fix that only makes the number pass.
