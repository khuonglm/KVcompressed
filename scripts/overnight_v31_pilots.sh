#!/usr/bin/env bash
# Overnight run (9 November 2026), template v3.1, both datasets. Existing commands only.
#   1. Prompt test on GQA and TextVQA (about 10 minutes each). GQA is rerun so its saved summary
#      has the corrected alternative rates.
#   2. Development pilot, verified report and tokenization check (several hours).
#
#   tmux new -s overnight
#   bash scripts/overnight_v31_pilots.sh
#
# Safe to rerun: the pilot skips items that already finished. A failing step doesn't stop the
# others. Everything goes to results/v31/, and the whole output to results/v31/overnight.log.

set -uo pipefail
cd "$(dirname "$0")/.."
OUT=results/v31
mkdir -p "$OUT"
exec > >(tee -a "$OUT/overnight.log") 2>&1
echo "=== start $(date) ==="

# Preflight: stop now rather than fail at 3 am.
uv run python -c 'from kvsel.families import LLAVA; assert LLAVA.template_version == "v3.1", LLAVA.template_version' \
  || { echo "the code is not on template v3.1: update it first"; exit 1; }
for path in manifests/gqa_dev.jsonl manifests/textvqa_dev.jsonl verification/gqa_dev.jsonl \
            verification/textvqa_dev.jsonl results/screen_dev/items results/textvqa/screen_dev/items \
            data/gqa/images data/textvqa/images; do
  [ -e "$path" ] || { echo "missing: $path"; exit 1; }
done

declare -A STATUS

prompt_test() {
  local name=$1 images=$2
  echo "=== $name: prompt test $(date) ==="
  if uv run kvsel prompt-test --items "manifests/${name}_dev.jsonl" --images "$images" \
       --verified "verification/${name}_dev.jsonl" --out "$OUT/${name}_prompt_test.json"; then
    STATUS[${name}_prompt]="done"
  else
    STATUS[${name}_prompt]="FAILED"
  fi
}

pilot() {
  local name=$1 images=$2 screen=$3
  echo "=== $name: pilot $(date) ==="
  uv run kvsel pilot --items "manifests/${name}_dev.jsonl" --images "$images" --screen "$screen" \
      --surface capitalized --limit 200 --out "$OUT/${name}_pilot" \
    || { STATUS[${name}_pilot]="FAILED in pilot"; return; }
  echo "=== $name: report $(date) ==="
  uv run kvsel report "$OUT/${name}_pilot" --surface capitalized \
      --verified "verification/${name}_dev.jsonl" --out "$OUT/${name}_report.md" \
    || { STATUS[${name}_pilot]="FAILED in report"; return; }
  echo "=== $name: tokenization check $(date) ==="
  uv run kvsel check-tokenization "$OUT/${name}_pilot" --items "manifests/${name}_dev.jsonl" \
      > "$OUT/${name}_check_tokenization.json" \
    || { STATUS[${name}_pilot]="FAILED in check-tokenization"; return; }
  STATUS[${name}_pilot]="done"
}

prompt_test gqa data/gqa/images
prompt_test textvqa data/textvqa/images
pilot gqa data/gqa/images results/screen_dev
pilot textvqa data/textvqa/images results/textvqa/screen_dev

echo "=== finished $(date) ==="
for step in gqa_prompt textvqa_prompt gqa_pilot textvqa_pilot; do echo "$step: ${STATUS[$step]}"; done
echo "Review everything in $OUT except the two *_pilot folders:"
echo "  gqa_report.md, textvqa_report.md, *_prompt_test.json, *_check_tokenization.json, overnight.log"
echo "and the verification/ folder (needed for the builder fixes)."
