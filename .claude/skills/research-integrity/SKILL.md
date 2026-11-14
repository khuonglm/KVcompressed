---
name: research-integrity
description: Use whenever work touches confirmatory items, the freeze, pre-registration, hand verification, eligibility, analysis or reporting of results - or when tempted to choose a selector, casing, K or threshold by looking at outcomes. Keeps development and confirmatory work separate and prevents outcome-driven changes.
---

# Research integrity for this study

The study asks whether unreliable text steers visual-KV selection in LLaVA-1.5-7B. Its claims are
only valid if confirmatory data is untouched until the protocol is frozen. The proposal is
`documents/Idea 1 proposal v3.md`; decisions are in `documents/Idea 1 implementation plan.md` §3.

## Development vs confirmatory

- Every manifest item has `split`: `dev` or a confirmatory value. Manifests are frozen and their
  sha256 is recorded in the funnel JSON. Do not edit a manifest by hand.
- `require_development` in `runner.py` refuses non-dev items. Do not bypass, weaken or add a flag to
  skip it. Confirmatory runs need the freeze guard, which does not exist yet; building it is its own
  task and needs the user's sign-off on the design.
- Development reports support **no claims**. Say so whenever you summarise one.

## Things never to do

- Pick a selector, K, casing rule, threshold or exclusion by which gives the largest effect.
- Inspect comparative (B vs C) outcomes on confirmatory items before the freeze.
- Drop or "repair" items after seeing their outcomes. Items failing hand verification are removed
  and logged, never repaired.
- Show model outcomes in the hand-verification tool. Verification is blind; `verify.py` must keep it so.
- Rewrite or delete records in `verification/` — it is append-only, the latest decision counts.

## The three-stage freeze (plan decision 4)

1. Freeze screening rules: model, prompt, surface forms, normalisation, eligibility, seed.
2. Automated Full_Q-only screen, exposing only ids, eligibility flags and funnel counts.
3. Freeze the comparative experiment, then run it once.

If a screening rule must change after held-out failures were seen, it needs a documented amendment
and a fresh held-out partition.

## When a choice is needed

Settle it on development items, on scientific grounds, write the rule down in the plan (or
`PREREG.md` once it exists) with the date, and only then apply it. If it is the user's scientific
call, ask them rather than deciding.
