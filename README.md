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
uv run kvsel model fetch
```

This downloads 14.1 GB into `models/llava-1.5-7b-hf`, one connection per weight file. Run it inside
`tmux` or `screen` so it survives a dropped SSH session. Rerun it to resume after an interruption.

Every command then uses that copy and doesn't go online for the model. With no copy there, commands
fall back to the Hub; a partial copy stops with an error, so a 14 GB download can't start by accident.
Use `--model-dir` to point at another location, such as a shared disk.

Plain HTTP is the default because it was about 3x faster than Hugging Face's Xet downloader in a
test from a Mac. To try Xet on your network, run `HF_HUB_DISABLE_XET=0 HF_XET_HIGH_PERFORMANCE=1 uv run kvsel model fetch`.

### 2. Get the GQA data

Download only the images the manifests use (1,799 files, about 325 MB), not the 21.8 GB archive:

```bash
uv run kvsel data fetch-images manifests/gqa_dev.jsonl manifests/gqa_confirm.jsonl
```

Files go to `data/gqa/images`, and each is checked against the archive's CRC. If it's interrupted or
some files fail, rerun the same command and it fetches only what's missing. Use `--workers` to change
the parallelism; the default is 3 because the server returned 503 errors at 8.

The item manifests are already in `manifests/`, so you don't need the question files on the GPU.
If you rebuild them with different filters, rerun this command for any new images. The `gqa-3`
confirmatory pool (9,209 items, 4,429 images) needs about 0.8 GB more.

To rebuild (no GPU; about a minute and 3 GB of memory), keeping the development items unchanged:

```bash
uv run kvsel data build-gqa --questions val_all_questions.json --scene-graphs val_sceneGraphs.json \
  --dev-manifest manifests/gqa_dev.jsonl
```

### 3. Validate the pipeline

This is the gate: the noise floor plus tests E1–E5 and E7 on the real model in fp16. The criteria
are in [tests/README.md](tests/README.md).

```bash
uv run kvsel validate --items manifests/gqa_dev.jsonl --images data/gqa/images
```

Please send me the output and `results/validate_llava.json` before going further, whether it passes or
fails. It also reports the noise floor under both output heads, which settles whether the fp32
head is worth keeping. Tolerances aren't changed after a failure; the cause gets identified.

### 4. Screen development items

Runs Full_Q on all 600 development candidates and prints how many are eligible and a sample of
greedy answers. Answer casing decides the `--surface` value below.

```bash
uv run kvsel pilot screen --items manifests/gqa_dev.jsonl --images data/gqa/images
```

### 5. Run the pilot

Runs every cell on the first 200 eligible items: A–D, Full anchors, uniform, the reserve variants,
and question-span selection. Safe to stop and rerun, since finished items are skipped.

```bash
uv run kvsel pilot run --items manifests/gqa_dev.jsonl --images data/gqa/images --surface lower
```

### 6. Report

The development pilot is analysed as a rehearsal of the confirmatory analysis; nothing in it supports a claim:

```bash
uv run kvsel pilot report --gqa-run results/pilot --gqa-items manifests/gqa_dev.jsonl --out results/pilot/report
```

## TextVQA runbook

Same rule: stop at the first step that fails, and send me its output. Results go under
`results/textvqa/`, so nothing overwrites the GQA files (`results/validate_llava.json`,
`results/screen_dev`, `results/pilot`).

```bash
mkdir -p results/textvqa
set -o pipefail   # so a failure inside `| tee` still stops the && chain

# 1. Development images only (about 0.4 GB)
uv run kvsel data fetch-images manifests/textvqa_dev.jsonl --out data/textvqa/images \
  --url https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip --workers 4

# 2. Validate, then screen only if every test passes (about 35 minutes together)
uv run kvsel validate --items manifests/textvqa_dev.jsonl --images data/textvqa/images \
  --surface capitalized --out results/textvqa/validate.json 2>&1 | tee results/textvqa/validate.log \
&& uv run kvsel pilot screen --items manifests/textvqa_dev.jsonl --images data/textvqa/images \
  --out results/textvqa/screen_dev 2>&1 | tee results/textvqa/screen.log
```

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

## Freeze stage 1 and the confirmatory screen

Rules: `PREREG.md`, section 1, as amended on 10 November (amendment 1: items answerable with a blank
image aren't eligible). The first record, `freeze/screening.json`, is kept; amendment 1's record
supersedes it. Run in this order and stop at the first step that fails.

1. Freeze amendment 1 (no GPU); commit `freeze/screening_amend1.json`:
   ```
   uv run kvsel confirmatory freeze-screening --amends freeze/screening.json --reason "amendment 1: exclude items answerable with a blank image"
   ```
2. Screen on the GPU into a new folder (GQA about 2.5 h). It writes only flags and resumes if rerun.
   `--limit N` screens only the first N items.
   ```
   uv run kvsel confirmatory screen --items manifests/gqa_confirm.jsonl --images data/gqa/images --out results/confirm/gqa_screen_a1
   ```
3. Copy `results/confirm/gqa_screen_a1` to the machine you verify on and verify eligible items in manifest
   order until 1,000 pass. Earlier decisions carry over; only new items are queued.
   ```
   uv run kvsel verify --items manifests/gqa_confirm.jsonl --images data/gqa/images --screen results/confirm/gqa_screen_a1 --decisions verification/gqa_confirm.jsonl
   ```

The screen refuses to run if `PREREG.md` section 1, a manifest or the screening code changed after the freeze.

**Amendment 2 (15 November, code clean-up only).** Supersedes amendment 1; no rule changed (PREREG 1.9).
Freeze it once (no GPU) and commit `freeze/screening_amend2.json`:
```
uv run kvsel confirmatory freeze-screening --amends freeze/screening_amend1.json --reason "amendment 2: code clean-up, no rule changed (PREREG 1.9)"
```

## Freeze stage 3 and the confirmatory run (GQA first)

1. Review `PREREG.md` section 2, delete its DRAFT line, then freeze (no GPU) and commit `freeze/comparative_gqa.json`:
   ```
   uv run kvsel confirmatory freeze --items manifests/gqa_confirm.jsonl --screen results/confirm/gqa_screen_a1 --decisions verification/gqa_confirm.jsonl --record freeze/comparative_gqa.json
   ```
2. On the GPU, with the same `PREREG.md`, `freeze/`, screen folder and decisions file (about 3 h):
   ```
   uv run kvsel confirmatory run --items manifests/gqa_confirm.jsonl --screen results/confirm/gqa_screen_a1 --decisions verification/gqa_confirm.jsonl --record freeze/comparative_gqa.json --images data/gqa/images --out results/confirm/gqa_run
   ```
   It refuses if anything changed since the freeze. Send back only its printed summary; don't open the item files.
3. The pre-registered analysis, once, no GPU. TextVQA's `--textvqa-*` options join later; until then the
   Holm decisions show as pending.
   ```
   uv run kvsel confirmatory report --gqa-run results/confirm/gqa_run --gqa-items manifests/gqa_confirm.jsonl --gqa-record freeze/comparative_gqa.json --out results/confirm/report
   ```
   It refuses if the code changed since the freeze or the run is incomplete.

## Follow-ups (not confirmatory)

PREREG 2.4's follow-ups on the finished GQA run: K = 288 and 72, LT-early, random seeds 0–4 and oracle
restoration. They run as `kvsel followups`. The confirmatory run folder must hold exactly the frozen main set.

1. On the GPU, in `tmux` (about 4–5 h for all parts; `--parts budgets lt random restore` picks some). It resumes if rerun.
   ```
   uv run kvsel followups run --items manifests/gqa_confirm.jsonl --images data/gqa/images --confirm-run results/confirm/gqa_run --record freeze/comparative_gqa.json --out results/followup/gqa
   ```
2. The descriptive report (no GPU):
   ```
   uv run kvsel followups report --run results/followup/gqa --confirm-run results/confirm/gqa_run --items manifests/gqa_confirm.jsonl --out results/followup/gqa/report
   ```

## GQA replication (PREREG section 3)

A second confirmatory run on 1,000 fresh, already-verified GQA items: misleading and irrelevant text on
every item, matching text on the first 300, at K = 72, 144 and 288. It runs as `kvsel replication`.
**No comparative unit may run on any
replication item before step 3's record is committed.** Run in this order and stop at the first step that
fails; send me its output.

`--exclude-runs` must list every folder that holds comparative item files on confirmatory items: the
confirmatory run, the follow-ups, and the folder you moved the first run's 9 extra item files to. Pass the
same list to `count`, `freeze` and `run`, and never put the replication's `--out` inside one of them.

```
EXCLUDE="results/confirm/gqa_run results/followup/gqa <folder with the 9 extra item files>"
```

1. Count the pool (no GPU, seconds). It reads screen flags and decisions only. You want `"replication": 1000`
   and `"next_unverified": null`; otherwise verify the named item and the ones after it, in manifest order.
   ```
   uv run kvsel replication count --items manifests/gqa_confirm.jsonl --screen results/confirm/gqa_screen_a1 --decisions verification/gqa_confirm.jsonl --exclude-runs $EXCLUDE
   ```
2. Review `PREREG.md` section 3 and delete its DRAFT line.
3. Freeze (no GPU) and commit `freeze/replication_gqa.json` together with `PREREG.md` before step 4:
   ```
   uv run kvsel replication freeze --items manifests/gqa_confirm.jsonl --screen results/confirm/gqa_screen_a1 --decisions verification/gqa_confirm.jsonl --exclude-runs $EXCLUDE
   ```
   It refuses while section 3 says DRAFT, if the main set no longer matches the GQA record, if fewer than
   1,000 items qualify, or if a different record already exists. Send me its printed summary.
4. On the GPU, in `tmux` (about 4 h: 25 replays per item, 35 on control items, about 1.3 times the first
   confirmatory run; memory as that run, about 15-16 GB). It refuses if anything changed since the freeze,
   prints progress only, and resumes if rerun.
   ```
   uv run kvsel replication run --items manifests/gqa_confirm.jsonl --screen results/confirm/gqa_screen_a1 --decisions verification/gqa_confirm.jsonl --exclude-runs $EXCLUDE --images data/gqa/images --out results/replication/gqa_run
   ```
   Send me only its final summary (item counts and the two consistency counts); don't open the item files.
5. The pre-registered analysis, once, no GPU (a few minutes):
   ```
   uv run kvsel replication report --items manifests/gqa_confirm.jsonl --screen results/confirm/gqa_screen_a1 --decisions verification/gqa_confirm.jsonl --exclude-runs $EXCLUDE --run results/replication/gqa_run --original-confirm results/confirm/report/results.json --original-followup results/followup/gqa/report/results.json --out results/replication/report
   ```
   It refuses, as `run` does, if anything changed since the freeze (code changes only through
   `freeze/code_amendments.json`; PREREG.md may grow by dated amendments), and refuses missing or extra
   item files. Send me `report.md`.
6. After a code amendment (PREREG 1.9, 3.6, 5), prove it reproduces every stored report and a few items of each
   stored run on the GPU; this writes the pending entries and, only if everything reproduces, their evidence:
   `uv run kvsel reproduce --reason "<the amendment, as PREREG names it>" --images data/gqa/images`. It exits 2
   naming the first difference otherwise. Commit `freeze/code_amendments.json` with the code.

## LLaVA-NeXT (PREREG section 4; validation gate 24 November)

The steering test on a second model, LLaVA-NeXT-7B with AnyRes, on GQA. It runs as `kvsel next`
(and `kvsel model fetch` and `kvsel validate` with `--model llava_next`). Run in this order and stop at the first step that
fails; send me its output. **Never open the item files** under `results/next/screen/items` or
`results/next/gqa_run/items`: send only what each step says.

**No comparative unit may run on NeXT before `freeze/next_gqa.json` and `PREREG.md` are committed.**

**Timing rule.** After the stage-1 freeze (step 5), section 4.1 is not edited, and nothing under
`src/kvsel/` changes without a dated code amendment in `freeze/code_amendments.json`
(PREREG 1.9): the NeXT records hash the package, and `screen`, `freeze`, `run` and `report` refuse code
that the amendments file doesn't list. After the stage-3 freeze (step 8), `PREREG.md` grows only by dated
amendments.

`scripts/overnight_next.sh` runs steps 5-10 in a row and stops after each freeze until you commit (see its
header). Steps 1-4 stay by hand. It checks its inputs first, logs everything to `results/next/overnight.log`,
and is safe to rerun; step 10 needs the replication's report and is skipped once `report.md` exists.

1. Get the model (about 14 GB, in `tmux`; rerun to resume). It downloads the pinned revision into
   `models/llava-v1.6-vicuna-7b-hf`; a partial copy, or one at another revision, stops every NeXT command.
   ```
   uv run kvsel model fetch --model llava_next
   ```
   Send me its last line (`model complete at …, revision …`), or the list it says is still missing.
2. Validate on GQA development items (GPU). It refuses items that are not development items.
   ```
   uv run kvsel validate --model llava_next --items manifests/gqa_dev.jsonl --images data/gqa/images
   ```
   Criteria: [tests/README.md](tests/README.md), "LLaVA-NeXT-7B, Level 2". They don't change after a failure.
3. Send me the whole console output and `results/validate_llava_next.json`, GO or NO-GO. The last line is
   `GO` or `NO-GO` (exit code 1). Its last lines also give the time and memory for the steps below: the peak
   GPU memory of a run, and `projected: screen <= X h, comparative run <= Y h` (upper bounds: every unit
   costed as a Full unit on the longest suffix). Stop here on NO-GO.
4. With me: settle PREREG 4.5, fill in 4.1's model revision and AnyRes cap from step 3's numbers, then
   review section 4 and delete its DRAFT line. Stage 1 refuses while the line or any "[to fill" blank is
   there. No AnyRes cap is implemented yet: if step 3 says one is needed, it needs code before stage 1.
5. Freeze stage 1 (no GPU, seconds) and commit `freeze/next_screening.json` with `PREREG.md` before step 6:
   ```
   uv run kvsel next freeze-screening --items manifests/gqa_confirm.jsonl
   ```
   It refuses while section 4 has its DRAFT line and if a different record exists. Send me its printed summary.
6. Screen (GPU, in `tmux`; at most step 3's projected screen hours, usually far less since it stops at
   1,000 eligible items). It runs Full_Q, then Blind_Q on items that pass, over the passed items in manifest
   order, refuses if anything in the stage-1 record changed, writes flags only, and resumes if rerun.
   ```
   uv run kvsel next screen --items manifests/gqa_confirm.jsonl --decisions verification/gqa_confirm.jsonl --images data/gqa/images --out results/next/screen
   ```
   Send me its final funnel counts.
7. Count (no GPU, seconds). It reads the flags only. You want `"main_set": 1000` and `"short_of_1000": false`.
   If it is short with `"unscreened_passed": 0`, the whole pool is screened and, by PREREG 4.5(d), the main set
   is every eligible item: send me the count before step 8, which freezes that size. If it is short with
   unscreened items left, rerun step 6.
   ```
   uv run kvsel next count --items manifests/gqa_confirm.jsonl --decisions verification/gqa_confirm.jsonl --screen results/next/screen
   ```
8. Freeze stage 3 (no GPU) and commit `freeze/next_gqa.json` with `PREREG.md` before step 9:
   ```
   uv run kvsel next freeze --items manifests/gqa_confirm.jsonl --decisions verification/gqa_confirm.jsonl --screen results/next/screen
   ```
   It refuses while section 4 says DRAFT, if anything in the stage-1 record changed, if fewer items qualify
   than stage 1 froze, or if a different record exists. Send me its printed summary.
9. Run (GPU, in `tmux`; at most step 3's projected comparative hours; memory as step 3's run peak). Each
   item runs 24 units, 34 on the 300 control items, plus Blind_Q. It refuses if anything changed since the
   freeze, prints progress only, and resumes if rerun.
   ```
   uv run kvsel next run --items manifests/gqa_confirm.jsonl --decisions verification/gqa_confirm.jsonl --screen results/next/screen --images data/gqa/images --out results/next/gqa_run
   ```
   Send me only its final summary (the two consistency counts).
10. The pre-registered analysis, once, no GPU (a few minutes). It needs the replication's report
    (`results/replication/report/results.json`), whose estimates it prints beside NeXT's.
    ```
    uv run kvsel next report --items manifests/gqa_confirm.jsonl --decisions verification/gqa_confirm.jsonl --screen results/next/screen --run results/next/gqa_run --llava15 results/replication/report/results.json --out results/next/report
    ```
    It refuses, as `run` does, if anything changed since the freeze (code only through
    `freeze/code_amendments.json`; `PREREG.md` may grow by dated amendments), and refuses missing or extra
    item files. Send me `report.md`.
11. With the script, also send `results/next/overnight.log`.

**Memory.** The weights take about 13.2 GiB of the 20 GiB slice. The prefix cache (up to 1.45 GiB for a
square image) waits in CPU memory, and a unit holds at most two cache copies on the GPU. The peak is expected in prefill (eager attention over the visual span). The number that
counts is the printed run peak (prefill, then a Full unit on the cache); validate's own full forward
passes are longer and don't happen in the real run. If the run peak is above 20 GB, or validate stops
with CUDA out of memory, send me the output up to that point (each item prints its run peak): an AnyRes
cap (the largest grid allowed) then goes into PREREG 4.1 before stage 1, chosen from these numbers only,
and validate reruns under it.

## TextVQA data

The manifests are in `manifests/`. Only the images are needed, not the 7 GB archive: about 0.4 GB for
the 1,320 development images, or about 1.8 GB with the confirmatory images:

```bash
uv run kvsel data fetch-images manifests/textvqa_dev.jsonl --out data/textvqa/images \
  --url https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip --workers 4
```

## Layout

| Path | Contents |
| --- | --- |
| `src/kvsel/cli.py` | The one command line, `kvsel <group> <command>`; exit codes 0 done, 1 error, 2 refusal or stop for action |
| `src/kvsel/items/` | GQA and TextVQA builders, their shared rules, manifest reading, image fetching |
| `src/kvsel/verification/` | Hand-verification tool, decision log, item digest; the page is `page.html` |
| `src/kvsel/models/` | Model families (LLaVA-1.5, LLaVA-NeXT): files, prompts and tokens, candidate tokens, the layout of an image's visual span, prefill, numerics; LLaVA-1.5's image transform and box-to-grid mapping rules |
| `src/kvsel/cache.py`, `engine.py` | Compaction and masks; probe, replay, candidate scoring, greedy decoding |
| `src/kvsel/selectors.py` | Budgets and every selector's kept set, on any layout |
| `src/kvsel/prompts.py`, `normalize.py` | Surface forms and the PW-attn window's composition; answer normalisation |
| `src/kvsel/runner.py`, `runs.py` | The unit, the per-unit and per-item runners, the resumable item loop, screens and eligibility; running a frozen main set |
| `src/kvsel/protocol.py`, `files.py` | Every stage's freeze and check, the code hash and the amendment log; `Refusal`; atomic writes |
| `src/kvsel/reproduce.py`, `stored_reports.json` | `kvsel reproduce`: proving a code amendment against the stored reports and runs |
| `src/kvsel/analysis/` | Statistics, run tables and report blocks every report uses |
| `src/kvsel/validation.py` | Level 2 checks for both families, GO or NO-GO |
| `src/kvsel/studies/` | One package per study: `pilot`, `confirmatory`, `followups`, `replication`, `next` |
| `manifests/` | Built items and the funnel. Rebuild with `kvsel data build-gqa` |
| `CODING_STANDARDS.md` | The code's rules; `tests/test_standards.py` checks the mechanical ones |
| `tests/README.md` | Acceptance policy |
