# KVcompressed

Does unreliable text steer visual-KV selection in LLaVA-1.5-7B? See
[the proposal](documents/Idea%201%20proposal%20v3.md) and
[the implementation plan](documents/Idea%201%20implementation%20plan.md).

**Current state: the minimal version.** It runs a development pilot on GQA. The runner refuses
non-development items until the freeze guard exists.

## Setup

```bash
uv sync --extra dev
uv run pytest                # CPU only; a minute or less. The first run downloads the tokenizer config.
```

The real model needs about 14.1 GB of GPU memory for weights, plus about 1 GB for caches and the
fp32 output head. A 20 GB GPU is enough.

## GPU runbook

Stop at the first step that fails, and send me its output.

### 1. Get the GQA data

Download only the images the manifests use (1,799 files, about 325 MB), not the 21.8 GB archive:

```bash
uv run kvsel fetch-images manifests/gqa_dev.jsonl manifests/gqa_confirm.jsonl
```

Files go to `data/gqa/images`, and each is checked against the archive's CRC. If it's interrupted or
some files fail, rerun the same command and it fetches only what's missing. Use `--workers` to change
the parallelism; the default is 3 because the server returned 503 errors at 8.

The item manifests are already in `manifests/`, so you don't need the question files on the GPU.
If you rebuild them with different filters, rerun this command for any new images.

### 2. Validate the pipeline

This is the gate: the noise floor plus tests E1–E5 and E7 on the real model in fp16. The criteria
are in [tests/README.md](tests/README.md).

```bash
uv run kvsel validate --items manifests/gqa_dev.jsonl --images data/gqa/images
```

Please send me the output and `results/validate.json` before going further, whether it passes or
fails. It also reports the noise floor under both output heads, which settles whether the fp32
head is worth keeping. Tolerances aren't changed after a failure; the cause gets identified.

### 3. Screen development items

Runs Full_Q on all 600 development candidates and prints how many are eligible and a sample of
greedy answers. Answer casing decides the `--surface` value below.

```bash
uv run kvsel screen --items manifests/gqa_dev.jsonl --images data/gqa/images
```

### 4. Run the pilot

Runs every cell on the first 200 eligible items: A–D, Full anchors, uniform, the reserve variants,
and question-span selection. Safe to stop and rerun, since finished items are skipped.

```bash
uv run kvsel pilot --items manifests/gqa_dev.jsonl --images data/gqa/images --surface lower
```

### 5. Report

```bash
uv run kvsel report results/pilot --surface lower --out results/pilot/report.md
```

Development items only, so nothing in this report supports a claim. Its first use is to settle the
open decisions: rung 1 for PW-attn, answer casing, and the clustered SD for planning.

## Layout

| Path | Contents |
| --- | --- |
| `src/kvsel/cache.py` | Compaction and masks; the only code touching the cache API |
| `src/kvsel/engine.py` | Probe, replay, candidate scoring, greedy decoding |
| `src/kvsel/selectors.py` | PW-attn, question-span, uniform, coverage reserve |
| `src/kvsel/prompts.py` | Template v3.0, prefix/suffix split, candidate tokenization |
| `src/kvsel/gqa.py` | GQA builder |
| `src/kvsel/runner.py` | Work units; one result file per item |
| `src/kvsel/validate.py` | Level 2 acceptance tests |
| `src/kvsel/analysis.py` | Estimands, clustered bootstrap, development report |
| `manifests/` | Built items and the funnel. Rebuild with `kvsel build-gqa` |
| `tests/README.md` | Acceptance policy |
