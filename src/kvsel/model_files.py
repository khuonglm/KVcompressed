"""Finds the local model copy, falling back to the Hub only when there is none, and downloads it."""

from __future__ import annotations

import json
import os
from pathlib import Path

from . import MODEL_ID

DEFAULT_MODEL_DIR = Path("models/llava-1.5-7b-hf")
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json")
CONFIG_FILES = ("config.json", "generation_config.json", "preprocessor_config.json", "processor_config.json")
WEIGHT_INDEX = "model.safetensors.index.json"
# Everything needed to run; skips the README and .gitattributes.
DOWNLOAD_PATTERNS = ["*.json", "*.safetensors", "tokenizer.model", "*.jinja"]


def missing_files(model_dir: Path, weights: bool = True) -> list[str]:
    """Files still absent. Hub downloads appear under their final name only once complete."""
    required = [*TOKENIZER_FILES, *CONFIG_FILES]
    if weights:
        required.append(WEIGHT_INDEX)
    missing = [name for name in required if not (model_dir / name).is_file()]
    index = model_dir / WEIGHT_INDEX
    if weights and index.is_file():
        shards = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        missing += [name for name in shards if not (model_dir / name).is_file()]
    return missing


def resolve_model(model_dir: Path = DEFAULT_MODEL_DIR, weights: bool = True) -> str:
    """The local path if the copy is present, or the Hub id if nothing is there.

    An incomplete copy is an error: falling back would silently start a 14 GB download.
    """
    model_dir = Path(model_dir)
    if not model_dir.is_dir() or not any(model_dir.iterdir()):
        print(f"no local model at {model_dir}; using {MODEL_ID} from the Hugging Face Hub", flush=True)
        return MODEL_ID
    missing = missing_files(model_dir, weights)
    if missing:
        raise SystemExit(
            f"the model at {model_dir} is incomplete (missing {', '.join(missing)}). "
            f"Run `kvsel fetch-model --out {model_dir}` to finish it."
        )
    print(f"using local model at {model_dir}", flush=True)
    return str(model_dir)


def fetch_model(out_dir: Path = DEFAULT_MODEL_DIR, workers: int = 8) -> list[str]:
    """Downloads into ``out_dir``. Rerunning resumes and skips finished files."""
    # Plain HTTP was about 3x faster than Xet in a 60 s comparison, and it resumes partial files.
    # Must be set before huggingface_hub is imported. Set HF_HUB_DISABLE_XET=0 to use Xet instead.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import snapshot_download

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(MODEL_ID, local_dir=out_dir, allow_patterns=DOWNLOAD_PATTERNS, max_workers=workers)
    return missing_files(out_dir)
