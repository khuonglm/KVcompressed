#!/usr/bin/env bash
# GQA replication end to end (README, "GQA replication"): count -> freeze -> run -> report.
#
# Usage, from the repo root, in tmux:
#   EXCLUDE="results/confirm/gqa_run results/followup/gqa <folder with the 9 extra item files>" \
#     bash scripts/overnight_replication.sh
#
# Before the first call: review PREREG.md section 3 and delete its DRAFT line.
# The first call stops after the freeze until freeze/replication_gqa.json and PREREG.md are committed
# (PREREG 3.5: nothing runs on replication items before that). Commit, then call it again; the freeze
# step is idempotent and the run resumes, so rerunning after any failure is safe.
# Everything is logged to results/replication/overnight.log; stops at the first failing step.
set -euo pipefail

: "${EXCLUDE:?set EXCLUDE to the earlier run folders (see the usage line)}"
read -ra EXCLUDE_DIRS <<< "$EXCLUDE"  # split on spaces only, no globbing
for folder in "${EXCLUDE_DIRS[@]}"; do
  [[ -d "$folder" ]] || { echo "EXCLUDE folder not found: $folder" >&2; exit 1; }
done

ITEMS=manifests/gqa_confirm.jsonl
SCREEN=results/confirm/gqa_screen_a1
DECISIONS=verification/gqa_confirm.jsonl
RECORD=freeze/replication_gqa.json
OUT=results/replication
COMMON=(--items "$ITEMS" --screen "$SCREEN" --decisions "$DECISIONS" --exclude-runs "${EXCLUDE_DIRS[@]}")

mkdir -p "$OUT"
exec > >(tee -a "$OUT/overnight.log") 2>&1
step() { echo; echo "== $(date '+%F %T') $*"; }

step "1/4 count"
uv run python -m kvsel.replication count "${COMMON[@]}"

step "2/4 freeze"
uv run python -m kvsel.replication freeze "${COMMON[@]}"
if ! git ls-files --error-unmatch "$RECORD" >/dev/null 2>&1 || ! git diff --quiet HEAD -- "$RECORD" PREREG.md; then
  echo "STOP: commit $RECORD and PREREG.md, then rerun this script."
  exit 2
fi

step "3/4 run (GPU, about 4 h)"
uv run python -m kvsel.replication run "${COMMON[@]}" --images data/gqa/images --out "$OUT/gqa_run"

step "4/4 report"
uv run python -m kvsel.replication report "${COMMON[@]}" --run "$OUT/gqa_run" \
  --original-confirm results/confirm/report/results.json \
  --original-followup results/followup/gqa/report/results.json \
  --out "$OUT/report"

step "done: send $OUT/report/report.md and this log ($OUT/overnight.log)"
