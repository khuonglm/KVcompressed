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

### 1. Get the model

```bash
uv run kvsel fetch-model
```

This downloads 14.1 GB into `models/llava-1.5-7b-hf`, one connection per weight file. Run it inside
`tmux` or `screen` so it survives a dropped SSH session. Rerun it to resume after an interruption.

Every command then uses that copy and doesn't go online for the model. With no copy there, commands
fall back to the Hub; a partial copy stops with an error, so a 14 GB download can't start by accident.
Use `--model-dir` to point at another location, such as a shared disk.

Plain HTTP is the default because it was about 3x faster than Hugging Face's Xet downloader in a
test from a Mac. To try Xet on your network, run `HF_HUB_DISABLE_XET=0 HF_XET_HIGH_PERFORMANCE=1 uv run kvsel fetch-model`.

### 2. Get the GQA data

Download only the images the manifests use (1,799 files, about 325 MB), not the 21.8 GB archive:

```bash
uv run kvsel fetch-images manifests/gqa_dev.jsonl manifests/gqa_confirm.jsonl
```

Files go to `data/gqa/images`, and each is checked against the archive's CRC. If it's interrupted or
some files fail, rerun the same command and it fetches only what's missing. Use `--workers` to change
the parallelism; the default is 3 because the server returned 503 errors at 8.

The item manifests are already in `manifests/`, so you don't need the question files on the GPU.
If you rebuild them with different filters, rerun this command for any new images. The `gqa-3`
confirmatory pool (9,209 items, 4,429 images) needs about 0.8 GB more.

To rebuild (no GPU; about a minute and 3 GB of memory), keeping the development items unchanged:

```bash
uv run kvsel build-gqa --questions val_all_questions.json --scene-graphs val_sceneGraphs.json \
  --dev-manifest manifests/gqa_dev.jsonl
```

### 3. Validate the pipeline

This is the gate: the noise floor plus tests E1–E5 and E7 on the real model in fp16. The criteria
are in [tests/README.md](tests/README.md).

```bash
uv run kvsel validate --items manifests/gqa_dev.jsonl --images data/gqa/images
```

Please send me the output and `results/validate.json` before going further, whether it passes or
fails. It also reports the noise floor under both output heads, which settles whether the fp32
head is worth keeping. Tolerances aren't changed after a failure; the cause gets identified.

### 4. Screen development items

Runs Full_Q on all 600 development candidates and prints how many are eligible and a sample of
greedy answers. Answer casing decides the `--surface` value below.

```bash
uv run kvsel screen --items manifests/gqa_dev.jsonl --images data/gqa/images
```

### 5. Run the pilot

Runs every cell on the first 200 eligible items: A–D, Full anchors, uniform, the reserve variants,
and question-span selection. Safe to stop and rerun, since finished items are skipped.

```bash
uv run kvsel pilot --items manifests/gqa_dev.jsonl --images data/gqa/images --surface lower
```

### 6. Report

```bash
uv run kvsel report results/pilot --surface lower --out results/pilot/report.md
```

Development items only, so nothing in this report supports a claim. Its first use is to settle the
open decisions: rung 1 for PW-attn, answer casing, and the clustered SD for planning.

## TextVQA runbook

Same rule: stop at the first step that fails, and send me its output. Results go under
`results/textvqa/`, so nothing overwrites the GQA files (`results/validate.json`,
`results/screen_dev`, `results/pilot`).

```bash
mkdir -p results/textvqa
set -o pipefail   # so a failure inside `| tee` still stops the && chain

# 1. Development images only (about 0.4 GB)
uv run kvsel fetch-images manifests/textvqa_dev.jsonl --out data/textvqa/images \
  --url https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip --workers 4

# 2. Validate, then screen only if every test passes (about 35 minutes together)
uv run kvsel validate --items manifests/textvqa_dev.jsonl --images data/textvqa/images \
  --surface capitalized --out results/textvqa/validate.json 2>&1 | tee results/textvqa/validate.log \
&& uv run kvsel screen --items manifests/textvqa_dev.jsonl --images data/textvqa/images \
  --out results/textvqa/screen_dev 2>&1 | tee results/textvqa/screen.log

# 3. No GPU: do scored tokens match generated tokens?
uv run kvsel check-tokenization results/textvqa/screen_dev --items manifests/textvqa_dev.jsonl \
  | tee results/textvqa/check_tokenization.json
```

## Qwen2.5-VL-7B runbook

The second model. It reuses the same manifests and images. Everything it writes goes under
`results/qwen/`. Stop at the first step that fails and send me its output.

```bash
mkdir -p results/qwen/gqa results/qwen/textvqa
set -o pipefail

# 1. Weights: 16.6 GB into models/qwen2.5-vl-7b-instruct (resumes if rerun)
uv run kvsel fetch-model --model qwen

# 2. GQA: validate, then screen only if every test passes
uv run kvsel validate --model qwen --items manifests/gqa_dev.jsonl --images data/gqa/images \
  --surface capitalized --out results/qwen/gqa/validate.json 2>&1 | tee results/qwen/gqa/validate.log \
&& uv run kvsel screen --model qwen --items manifests/gqa_dev.jsonl --images data/gqa/images \
  --out results/qwen/gqa/screen_dev 2>&1 | tee results/qwen/gqa/screen.log

# 3. No GPU: do scored tokens match generated tokens?
uv run kvsel check-tokenization --model qwen results/qwen/gqa/screen_dev --items manifests/gqa_dev.jsonl \
  | tee results/qwen/gqa/check_tokenization.json
```

Qwen runs in bf16 with an fp32 output head (upcast slice by slice) and fp32 attention scores in the
language model; see the acceptance policy in
[tests/README.md](tests/README.md).

## Hand verification

Run the tool on the server and open it through an SSH tunnel from your own machine. Or run it on any
machine that has the repo and the images.

```bash
# On the server: queues eligible development items only
uv run kvsel verify --items manifests/gqa_dev.jsonl --images data/gqa/images \
  --screen results/screen_dev --surface capitalized --limit 200 --decisions verification/gqa_dev.jsonl

# On your own machine
ssh -L 8765:127.0.0.1:8765 root@YOUR_SERVER
# then open http://127.0.0.1:8765
```

| Key | Action |
| --- | --- |
| `y` | Pass |
| `1`–`6` | Fail, with the reason shown on the page |
| `g` | Next view: clean image (each item opens here), box only, then evidence patches under the primary, fallback, centre and expanded rules |
| `n` | Add a note to the next decision |
| `←` `→` / `u` | Move / jump to the next undecided item |

Decisions save immediately to an append-only file, so you can stop any time. Press a key again on an
item to change its decision; the latest one counts. Commit `verification/` regularly: it's the only
record of this work. The page shows no model outcomes.

To rerun the report on items that passed:

```bash
uv run kvsel report results/pilot --surface capitalized --verified verification/gqa_dev.jsonl \
  --out results/pilot/report_verified.md
```

## Prompt test (adoption)

Full cache only, on hand-verified development passes, under four ways of showing the description. The
rule that decides the outcome is in the plan's decision record of 9 November 2026. About 10 minutes
per dataset.

```bash
mkdir -p results/prompt_test
uv run kvsel prompt-test --items manifests/gqa_dev.jsonl --images data/gqa/images \
  --verified verification/gqa_dev.jsonl --out results/prompt_test/gqa.json
uv run kvsel prompt-test --items manifests/textvqa_dev.jsonl --images data/textvqa/images \
  --verified verification/textvqa_dev.jsonl --out results/prompt_test/textvqa.json
```

## Checks before the freeze (S1, R1, E6)

```bash
bash scripts/gpu_s1_r1.sh      # GPU, about 15-20 minutes; results in results/checks/
uv run kvsel coverage --attention results/v31/gqa_pilot/attention   # no GPU; E6 figures in results/coverage/
```

S1 runs 200 random questions of any type with LLaVA's standard prompt, to catch bugs only. R1
times every stage of the pilot workload and projects the confirmatory runtime. The S1 manifests are
rebuilt with `kvsel build-s1` from the raw question files.

## Freeze stage 1 and the confirmatory screen

Rules: `PREREG.md`, section 1. Run in this order and stop at the first step that fails.

1. Review `PREREG.md` and commit it yourself.
2. Freeze the screening rules (no GPU). This writes `freeze/screening.json`; commit that too.
   ```
   uv run kvsel freeze-screening
   ```
3. Screen on the GPU (GQA about 1.6 h, TextVQA about 1.1 h). It writes only eligible flags and resumes
   if rerun. Add `--limit N` to screen only the first N items if not all images are there yet.
   ```
   uv run kvsel screen-confirm --items manifests/gqa_confirm.jsonl --images data/gqa/images --out results/confirm/gqa_screen
   uv run kvsel screen-confirm --items manifests/textvqa_confirm.jsonl --images data/textvqa/images --out results/confirm/textvqa_screen
   ```
4. Copy `results/confirm/*_screen` to the machine you verify on, then verify only eligible items, in
   manifest order, until 1,000 pass per dataset. Earlier decisions carry over.
   ```
   uv run kvsel verify --items manifests/gqa_confirm.jsonl --images data/gqa/images --screen results/confirm/gqa_screen --decisions verification/gqa_confirm.jsonl
   uv run kvsel verify --items manifests/textvqa_confirm.jsonl --images data/textvqa/images --screen results/confirm/textvqa_screen --decisions verification/textvqa_confirm.jsonl
   ```

The screen refuses to run if `PREREG.md` section 1, a manifest or the screening code changed after the freeze.

## Freeze stage 3 and the confirmatory run (GQA first)

1. Optional, development only: share of items answerable with a blank image (about 2 min):
   ```
   uv run kvsel blind-rate --items manifests/gqa_dev.jsonl --images data/gqa/images --verified verification/gqa_dev.jsonl --out results/checks/blind_gqa.json
   ```
2. Review `PREREG.md` section 2, delete its DRAFT line, then freeze (no GPU) and commit `freeze/comparative_gqa.json`:
   ```
   uv run kvsel freeze-comparative --items manifests/gqa_confirm.jsonl --screen results/confirm/gqa_screen --decisions verification/gqa_confirm.jsonl --record freeze/comparative_gqa.json
   ```
3. On the GPU, with the same `PREREG.md`, `freeze/`, screen folder and decisions file (about 3 h):
   ```
   uv run kvsel confirm --items manifests/gqa_confirm.jsonl --screen results/confirm/gqa_screen --decisions verification/gqa_confirm.jsonl --record freeze/comparative_gqa.json --images data/gqa/images --out results/confirm/gqa_run
   ```
   It refuses if anything changed since the freeze. Send back only its printed summary; don't open the item files.

## TextVQA data

The manifests are in `manifests/`. Only the images are needed, not the 7 GB archive: about 0.4 GB for
the 1,320 development images, or about 1.8 GB with the confirmatory images:

```bash
uv run kvsel fetch-images manifests/textvqa_dev.jsonl --out data/textvqa/images \
  --url https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip --workers 4
```

## Layout

| Path | Contents |
| --- | --- |
| `src/kvsel/cache.py` | Compaction and masks; the only code touching the cache API |
| `src/kvsel/engine.py` | Probe, replay, candidate scoring, greedy decoding |
| `src/kvsel/selectors.py` | PW-attn, question-span, uniform, coverage reserve |
| `src/kvsel/prompts.py` | Template v3.0, prefix/suffix split, candidate tokenization |
| `src/kvsel/gqa.py`, `textvqa.py` | Dataset builders |
| `src/kvsel/geometry.py` | Image transform and box-to-grid mapping rules |
| `src/kvsel/verify.py` | Hand-verification tool |
| `src/kvsel/runner.py` | Work units; one result file per item |
| `src/kvsel/validate.py` | Level 2 acceptance tests |
| `src/kvsel/analysis.py` | Estimands, clustered bootstrap, development report |
| `manifests/` | Built items and the funnel. Rebuild with `kvsel build-gqa` |
| `tests/README.md` | Acceptance policy |
