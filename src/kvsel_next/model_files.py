"""NeXT's local model copy at a pinned revision. kvsel's helpers know only LLaVA-1.5's files and take no revision."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from kvsel.model_files import DOWNLOAD_PATTERNS, WEIGHT_INDEX

from .family import NEXT

# The Hub's main for llava-hf/llava-v1.6-vicuna-7b-hf (last changed 27 January 2025), checked 16 November 2026.
REVISION = "c916e6cdcd760b4cecd1dd4907f84ac649f93b23"
REVISION_FILE = "kvsel_next_revision.txt"  # written by fetch_model; the Hub's own metadata format is internal
# From the repository's file listing at REVISION, besides the weights.
REQUIRED_FILES = (
    "tokenizer.json", "tokenizer_config.json", "tokenizer.model", "special_tokens_map.json", "added_tokens.json",
    "chat_template.json", "config.json", "generation_config.json", "preprocessor_config.json", "processor_config.json",
)


def missing_files(model_dir: Path, weights: bool = True) -> list[str]:
    """Files still absent, and the revision file if it names another revision."""
    model_dir = Path(model_dir)
    required = [*REQUIRED_FILES, REVISION_FILE] + ([WEIGHT_INDEX] if weights else [])
    missing = [name for name in required if not (model_dir / name).is_file()]
    marker = model_dir / REVISION_FILE
    recorded = marker.read_text().strip() if marker.is_file() else REVISION
    if recorded != REVISION:
        missing.append(f"{REVISION_FILE} (holds {recorded}, expected {REVISION})")
    index = model_dir / WEIGHT_INDEX
    if weights and index.is_file():
        shards = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        missing += [name for name in shards if not (model_dir / name).is_file()]
    return missing


def resolve_model(model_dir: Path, weights: bool = True) -> str:
    """The local path if the copy is complete at REVISION, or the Hub id (loaded at REVISION) if nothing is there."""
    model_dir = Path(model_dir)
    if not model_dir.is_dir() or not any(model_dir.iterdir()):
        print(f"no local model at {model_dir}; using {NEXT.model_id}@{REVISION[:7]} from the Hub", file=sys.stderr, flush=True)
        return NEXT.model_id
    missing = missing_files(model_dir, weights)
    if missing:
        raise SystemExit(
            f"the model at {model_dir} is incomplete or at another revision (missing {', '.join(missing)}). "
            f"Run `python -m kvsel_next fetch-model --out {model_dir}` to finish it."
        )
    print(f"using local model at {model_dir}", file=sys.stderr, flush=True)
    return str(model_dir)


def fetch_model(out_dir: Path, workers: int) -> list[str]:
    """Downloads REVISION into ``out_dir``; rerunning resumes. Returns what is still missing."""
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # as kvsel's fetch: plain HTTP was faster and resumes
    from huggingface_hub import snapshot_download

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(NEXT.model_id, revision=REVISION, local_dir=out_dir, allow_patterns=DOWNLOAD_PATTERNS,
                      max_workers=workers)
    (out_dir / REVISION_FILE).write_text(REVISION + "\n")
    return missing_files(out_dir)
