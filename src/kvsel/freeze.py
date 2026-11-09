"""Stage 1 of the freeze (plan, decision 4): the screening rules, before any confirmatory model output.

`freeze_screening` records hashes of everything the eligibility screen depends on: the screening
section of PREREG.md, the confirmatory manifests, the screening code and the fixed settings below.
`check_screening` refuses to go on if any of them changed since. Changing a screening rule after the
freeze needs a documented amendment and a fresh held-out partition, not a new freeze record.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from .families import LLAVA

# Fixed screening settings. Eligibility: Full_Q's greedy answer is correct and its margin is positive
# under this surface rule (decision 2; greedy answers are mostly capitalized on both datasets).
SCREENING = {
    "model": LLAVA.name,
    "model_id": LLAVA.model_id,
    "template": LLAVA.template_version,
    "head_precision": LLAVA.default_head_precision,
    "surface": "capitalized",
    "eligibility": "Full_Q greedy answer correct and margin(true - alt) > 0",
    # Amendment 1 (10 November 2026): an item that passes the rule above with a blank image instead
    # of the real one is not eligible, since the model answers it without the image.
    "blind_filter": "not eligible if the same rule passes on a blank 336 x 336 image of the padding colour",
    "amendment": 1,
    "max_new_tokens": 16,
    "batch_size": 1,
}

# Modules whose code decides a screen's outcome. A later confirm module must not live in these files,
# or resuming the screen would be refused.
SCREENING_MODULES = (
    "__init__.py", "attention.py", "cache.py", "engine.py", "families.py", "model.py", "normalize.py",
    "prompts.py", "runner.py", "screening.py",
)
PREREG_SECTION = "## 1. Screening rules"
DEFAULT_RECORD = Path("freeze/screening_amend1.json")  # supersedes freeze/screening.json


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_hash() -> str:
    package = Path(__file__).parent
    digest = hashlib.sha256()
    for name in SCREENING_MODULES:
        digest.update(name.encode() + b"\0" + (package / name).read_bytes() + b"\0")
    return digest.hexdigest()


def screening_section(prereg: Path) -> str:
    """The text from the screening heading up to the next level-2 heading."""
    lines = Path(prereg).read_text().splitlines()
    try:
        start = lines.index(PREREG_SECTION)
    except ValueError:
        raise SystemExit(f"{prereg} has no line '{PREREG_SECTION}'") from None
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    return "\n".join(lines[start:end]).strip() + "\n"


def current_state(prereg: Path, manifests: list[Path]) -> dict:
    return {
        "prereg_screening_sha256": hashlib.sha256(screening_section(prereg).encode()).hexdigest(),
        "manifests": {Path(m).name: sha256_file(m) for m in manifests},
        "code_sha256": code_hash(),
        "settings": SCREENING,
    }


def freeze_screening(prereg: Path, manifests: list[Path], record: Path = DEFAULT_RECORD,
                     amends: Path | None = None, reason: str | None = None) -> dict:
    """Writes the stage-1 record. An amendment names the record it supersedes and why; that record stays."""
    if (amends is None) != (reason is None):
        raise SystemExit("an amendment needs both --amends and --reason")
    if amends is not None and not Path(amends).exists():
        raise SystemExit(f"{amends} does not exist")
    for manifest in manifests:
        splits = {json.loads(line).get("split") for line in Path(manifest).read_text().splitlines() if line.strip()}
        if splits != {"confirm"}:
            raise SystemExit(f"{manifest} is not a confirmatory manifest (splits: {sorted(map(str, splits))})")
    state = current_state(prereg, manifests)
    if record.exists():
        frozen = json.loads(record.read_text())
        if {key: frozen.get(key) for key in state} == state:
            return frozen
        raise SystemExit(f"{record} already exists with different contents. A screening rule can't change after "
                         "the freeze without a documented amendment and a fresh held-out partition.")
    payload = {"stage": "screening", "frozen_at": datetime.now(UTC).isoformat(timespec="seconds"), **state}
    if amends is not None:
        payload["amends"] = {"record": str(amends), "sha256": sha256_file(amends), "reason": reason}
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def check_screening(prereg: Path, manifest: Path, record: Path = DEFAULT_RECORD) -> dict:
    """Returns the frozen record if the screen may run on this manifest; otherwise exits saying what changed."""
    if not record.exists():
        raise SystemExit(f"no screening freeze at {record}: run `kvsel freeze-screening` first")
    frozen = json.loads(record.read_text())
    name = Path(manifest).name
    if name not in frozen["manifests"]:
        raise SystemExit(f"{name} was not frozen; frozen manifests: {sorted(frozen['manifests'])}")
    now = current_state(prereg, [manifest])
    changed = []
    if now["prereg_screening_sha256"] != frozen["prereg_screening_sha256"]:
        changed.append(f"the '{PREREG_SECTION}' section of {prereg}")
    if now["manifests"][name] != frozen["manifests"][name]:
        changed.append(f"manifest {manifest}")
    if now["code_sha256"] != frozen["code_sha256"]:
        changed.append(f"screening code ({', '.join(SCREENING_MODULES)})")
    if now["settings"] != frozen["settings"]:
        changed.append("screening settings in kvsel/freeze.py")
    if changed:
        raise SystemExit("Refusing to screen: changed since the freeze: " + "; ".join(changed))
    return frozen
