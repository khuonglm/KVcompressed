#!/usr/bin/env bash
# S1 (standard-prompt accuracy) and R1 (timing and memory) on both datasets. About 15-20 minutes.
#   tmux new -s checks
#   bash scripts/gpu_s1_r1.sh
# A failing step doesn't stop the others. Everything goes to results/checks/.

set -uo pipefail
cd "$(dirname "$0")/.."
OUT=results/checks
mkdir -p "$OUT"
exec > >(tee -a "$OUT/checks.log") 2>&1
echo "=== start $(date) ==="

for path in manifests/s1_gqa.jsonl manifests/s1_textvqa.jsonl manifests/gqa_dev.jsonl manifests/textvqa_dev.jsonl \
            data/gqa/images data/textvqa/images; do
  [ -e "$path" ] || { echo "missing: $path (update the code first)"; exit 1; }
done

declare -A STATUS
run() {
  local name=$1; shift
  echo "=== $name $(date) ==="
  if "$@"; then STATUS[$name]="done"; else STATUS[$name]="FAILED"; fi
}

run s1_gqa uv run kvsel s1 --items manifests/s1_gqa.jsonl --images data/gqa/images --out "$OUT/s1_gqa.json"
run s1_textvqa uv run kvsel s1 --items manifests/s1_textvqa.jsonl --images data/textvqa/images --out "$OUT/s1_textvqa.json"
run r1_gqa uv run kvsel timing --items manifests/gqa_dev.jsonl --images data/gqa/images --limit 20 --out "$OUT/r1_gqa.json"
run r1_textvqa uv run kvsel timing --items manifests/textvqa_dev.jsonl --images data/textvqa/images --limit 20 --out "$OUT/r1_textvqa.json"

echo "=== finished $(date) ==="
for step in s1_gqa s1_textvqa r1_gqa r1_textvqa; do echo "$step: ${STATUS[$step]}"; done
echo "Review everything in $OUT: s1_*.json, r1_*.json, checks.log"
