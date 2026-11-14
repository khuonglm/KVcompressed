---
name: gpu-handoff
description: Use when a change needs the real LLaVA-1.5-7B model, a GPU, or the full datasets to run or confirm (validate, diagnose-e2, screen, pilot, fetch-model, timing). Covers what to check on CPU first, how to hand the user a runnable command, and what output to ask for back.
---

# Handing a GPU job to the user

The local Mac (16 GB, no NVIDIA GPU) cannot run LLaVA-1.5-7B. The user runs every real-model job
on a 20 GB GPU server and sends the output back. Never try to load the real model locally, start a
14 GB download, or claim a GPU result you have not seen.

## Before handing off

1. Make the code path run on CPU against the tiny models in `tests/conftest.py`
   (`tiny_llama`, `tiny_llava`). Every GPU command should have a CPU test that reaches the same code.
2. Run the suite: `uv run pytest` (about a minute). Tests marked `network` need the tokenizer
   config from the Hub; use `uv run pytest -m "not network"` when offline and say so.
3. If the command is new, add it to the parametrized help test in `tests/test_cli.py`.
4. Make the job safe to stop and rerun: finished items are skipped, results are written with
   `write_json_atomic`, and nothing is overwritten.
5. Make sure the command refuses non-development items (`require_development`).

## The handoff message

Give the user:

- The exact command, copy-pasteable, with the real paths (`manifests/gqa_dev.jsonl`,
  `data/gqa/images`, `results/...`). Suggest `tmux` for anything longer than a few minutes.
- Expected runtime and GPU memory if known (weights about 14.1 GB, plus about 1 GB for caches and
  the fp32 head).
- What to send back: the printed output **and** the result files (for example `results/validate.json`),
  whether it passed or failed.
- If the step belongs in the README runbook, update `README.md` "GPU runbook" in the same change.
  Steps there are numbered and the rule is "stop at the first step that fails, and send me its output".

## When results come back

- Read the actual numbers. Compare against the pass criteria in `tests/README.md`, not against
  what you expected.
- If anything fails, switch to the `test-failure-triage` skill. Do not adjust tolerances.
- Record durable facts learned on the GPU (noise floors, answer casing, eligibility counts) where the
  project keeps them: `tests/README.md` for test outcomes, the implementation plan for decisions.
