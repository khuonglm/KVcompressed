#!/usr/bin/env bash
# LLaVA-NeXT on GQA end to end (README, "LLaVA-NeXT"): freeze stage 1 -> screen -> count -> freeze stage 3
# -> run -> report. README steps 1-4 (model fetch, validate, settling section 4) come first, by hand.
#
# Usage, from the repo root, in tmux:
#   bash scripts/overnight_next.sh
#
# Before the first call: settle PREREG 4.5, fill in 4.1's blanks, and delete section 4's DRAFT line.
# The script stops after each freeze until its record and PREREG.md are committed: first
# freeze/next_screening.json (nothing is screened before that), then freeze/next_gqa.json (PREREG 4.4: no
# comparative unit runs before that). Commit, then call it again. A freeze whose record already exists is
# skipped (`screen` and `run` recheck it themselves), the screen is skipped once stage 3 exists, and the
# screen and run resume, and the report runs once (skipped if report.md exists), so rerunning after any
# failure is safe.
# Everything is logged to results/next/overnight.log; stops at the first failing step.
# Exit codes, as kvsel's: 0 done, 1 an error, 2 a stop for action (a commit, a refusal, a missing report).
set -euo pipefail

ITEMS=manifests/gqa_confirm.jsonl
DECISIONS=verification/gqa_confirm.jsonl
IMAGES=data/gqa/images
MODEL=models/llava-v1.6-vicuna-7b-hf  # kvsel.models.llava_next.NEXT.default_dir
LLAVA15=results/replication/report/results.json  # the replication's report, printed beside NeXT's
STAGE1=freeze/next_screening.json
STAGE3=freeze/next_gqa.json
OUT=results/next
SCREEN=$OUT/screen
COMMON=(--items "$ITEMS" --decisions "$DECISIONS" --screen "$SCREEN")

mkdir -p "$OUT"
exec > >(tee -a "$OUT/overnight.log") 2>&1
LOG_WRITER=$!
finish() {  # close the pipe and wait for tee, so the log holds every line before the script exits
  local status=$?
  exec >&- 2>&-
  wait "$LOG_WRITER" 2>/dev/null || true
  exit "$status"
}
trap finish EXIT

for path in "$ITEMS" "$DECISIONS"; do
  [[ -f "$path" ]] || { echo "input file not found: $path"; exit 1; }
done
for folder in "$IMAGES" "$MODEL"; do
  [[ -d "$folder" ]] || { echo "input folder not found: $folder"; exit 1; }
done
step() { echo; echo "== $(date '+%F %T') $*"; }
committed() {  # the record is tracked and neither it nor PREREG.md differs from HEAD
  git ls-files --error-unmatch "$1" >/dev/null 2>&1 && git diff --quiet HEAD -- "$1" PREREG.md
}

step "1/6 freeze stage 1"
if [[ -f "$STAGE1" ]]; then echo "$STAGE1 exists; screen rechecks it"; else
  uv run kvsel next freeze-screening --items "$ITEMS"
fi
committed "$STAGE1" || { echo "STOP: commit $STAGE1 and PREREG.md, then rerun this script."; exit 2; }

step "2/6 screen (GPU; time: validate's projected screen hours, an upper bound)"
if [[ -f "$STAGE3" ]]; then echo "$STAGE3 exists; the screen is finished"; else
  uv run kvsel next screen --items "$ITEMS" --decisions "$DECISIONS" --images "$IMAGES" --out "$SCREEN"
fi

step "3/6 count"
uv run kvsel next count "${COMMON[@]}"

step "4/6 freeze stage 3"
if [[ -f "$STAGE3" ]]; then echo "$STAGE3 exists; run rechecks it"; else
  uv run kvsel next freeze "${COMMON[@]}"
fi
committed "$STAGE3" || { echo "STOP: commit $STAGE3 and PREREG.md, then rerun this script."; exit 2; }

step "5/6 run (GPU; time: validate's projected comparative hours, an upper bound)"
uv run kvsel next run "${COMMON[@]}" --images "$IMAGES" --out "$OUT/gqa_run"

step "6/6 report"
if [[ -f "$OUT/report/report.md" ]]; then echo "$OUT/report/report.md exists; the analysis runs once"; else
  # Checked here, not at the start: the replication's report may not exist yet when stage 1 runs.
  [[ -f "$LLAVA15" ]] || { echo "STOP: $LLAVA15 not found; run the replication's report (README), then rerun."; exit 2; }
  uv run kvsel next report "${COMMON[@]}" --run "$OUT/gqa_run" --llava15 "$LLAVA15" --out "$OUT/report"
fi

step "done: send $OUT/report/report.md and this log ($OUT/overnight.log)"
