"""Writing a file atomically: a temporary file in the same folder, flushed to disk, then renamed into place, so an
interrupted write leaves no file (CODING_STANDARDS.md, rule 2)."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def write_atomic(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(handle, "w") as out:
            out.write(text)
            out.flush()
            os.fsync(out.fileno())  # the bytes are on disk before the name points at them
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def write_json(path: Path, payload, indent: int | None = None) -> None:
    """Item files are compact (``indent`` None); records are indented and end with a newline."""
    write_atomic(path, json.dumps(payload) if indent is None else json.dumps(payload, indent=indent) + "\n")
