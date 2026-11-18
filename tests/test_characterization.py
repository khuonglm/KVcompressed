"""Characterization suite: every command on hand-built inputs and the tiny models, compared with golden files.

The golden files under `characterization/golden/` were captured from the code before the reorganization
("Codebase reorganization spec", Testing Decisions), so moving code can't change an output unnoticed. Recapture
only on purpose, and check that the diff is what you meant:

    UPDATE_GOLDEN=1 uv run pytest tests/test_characterization.py

A step captures its exit code, its printed lines and every file it creates or changes in the workspace. Commands
run in process from the workspace with relative paths, in the order of `STEPS`; each step reads what earlier
steps wrote. Attention arrays (`.npz`) are stored unpacked, one `.npy` per array, since a zip archive records
the time it was written; the arrays themselves are compared exactly.

**The mask rule, the only one.** `mask` replaces every ISO timestamp with `<timestamp>`, and the value of every
`code_sha256` key and every `*record_sha256` key with `<code hash>` (a freeze record holds a timestamp and a
code hash, so a hash of a record changes with them). Nothing else is masked. The clock itself is an input:
`time.time` and `time.perf_counter` read 0 during a step, so measured durations are 0.

**Seams the suite replaces** (update them here when the code they name moves): model loading (`kvsel.cli._bundle`
and `kvsel_next.__main__.load_bundle` return the tiny models) and the study sizes in `SIZES` (1,000 items
won't fit a test). The tiny models have their end-of-sequence row tilted so that some greedy answers end
early: without it every answer runs to the 16-token cap and no item is ever eligible.

**Outputs the spec allows to change** get an `Exemption` in `EXEMPTIONS`, filled in by the ticket that changes
them; each names one allowed change, one step and one output path, and either the top-level keys that may
differ or, with no keys, that the whole file may.
"""

from __future__ import annotations

import _thread
import contextlib
import fnmatch
import hashlib
import importlib
import io
import json
import os
import re
import shutil
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

pytestmark = pytest.mark.network

HERE = Path(__file__).parent / "characterization"
INPUTS, GOLDEN = HERE / "inputs", HERE / "golden"

# --- The mask rule -------------------------------------------------------------------------------------------

TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?")
CODE_HASH = re.compile(r'("(?:code_sha256|\w*record_sha256)": ?)"[0-9a-f]{64}"')


def mask(text: str) -> str:
    return CODE_HASH.sub(r'\1"<code hash>"', TIMESTAMP.sub("<timestamp>", text))


# --- Exemptions ----------------------------------------------------------------------------------------------

ALLOWED_CHANGES = ("command names", "help text", "validation report", "screen kind marker", "item record hash")


@dataclass(frozen=True)
class Exemption:
    change: str  # one of ALLOWED_CHANGES
    step: str
    path: str  # a path in the step's capture (glob), for example "files/screen_dev/kind.json" or "command.json"
    keys: tuple[str, ...] = ()  # top-level JSON keys that may differ; none: the whole file may differ


EXEMPTIONS: tuple[Exemption, ...] = ()

# --- Steps ---------------------------------------------------------------------------------------------------

PROGRAMS = {"kvsel": "kvsel.cli", "followup": "kvsel.followup.__main__",
            "replication": "kvsel.replication.__main__", "next": "kvsel_next.__main__"}
SIZES = [("kvsel.confirm", "MAIN_SIZE", 2), ("kvsel.confirm", "CONTROL_SIZE", 1),
         ("kvsel.replication.select", "REPLICATION_SIZE", 2), ("kvsel.replication.select", "CONTROL_SIZE", 1)]
NEXT_SIZES = {"main_size": 2, "control_size": 1}  # items of kvsel_next.screen.SCREENING


@dataclass(frozen=True)
class Step:
    name: str
    program: str
    argv: tuple[str, ...]
    http: str | None = None  # verify: the folder the suite saves the page's responses to


def step(name, program, *argv, http=None):
    return Step(name, program, argv, http)


DEV = ("--items", "gqa_dev.jsonl", "--images", "images", "--device", "cpu")
CONFIRM = ("--prereg", "PREREG.md", "--items", "gqa_confirm.jsonl", "--screen", "screen_confirm",
           "--decisions", "decisions.jsonl", "--stage1", "freeze/screening.json")
COMPARATIVE = (*CONFIRM, "--record", "freeze/comparative_gqa.json")
REPLICATION = (*CONFIRM, "--gqa-record", "freeze/comparative_gqa.json", "--exclude-runs", "earlier_run",
               "--record", "freeze/replication_gqa.json")
NEXT = ("--prereg", "PREREG.md", "--items", "next_confirm.jsonl", "--screen", "next_screen",
        "--decisions", "next_decisions.jsonl", "--stage1", "freeze/next_screening.json", "--record", "freeze/next_gqa.json")
SMALL_VALIDATE = ("--noise-items", "2", "--sparse-items", "1", "--repeat-items", "1")

STEPS = (
    step("build_gqa", "kvsel", "build-gqa", "--questions", "raw/gqa_questions.json", "--scene-graphs",
         "raw/gqa_scene_graphs.json", "--out-dir", "built", "--dev-items", "2"),
    step("build_textvqa", "kvsel", "build-textvqa", "--questions", "raw/textvqa_questions.json", "--ocr",
         "raw/textvqa_ocr.json", "--out-dir", "built", "--dev-items", "1"),
    # Development pilot
    step("screen_dev", "kvsel", "screen", *DEV, "--out", "screen_dev"),
    step("pilot", "kvsel", "pilot", *DEV, "--screen", "screen_dev", "--surface", "capitalized", "--limit", "2",
         "--out", "pilot"),
    step("pilot_report", "kvsel", "confirm-report", "--gqa-run", "pilot", "--gqa-items", "gqa_dev.jsonl",
         "--resamples", "50", "--out", "pilot_report"),
    step("validate", "kvsel", "validate", *DEV, *SMALL_VALIDATE, "--out", "validate/report.json"),
    step("verify_dev", "kvsel", "verify", "--items", "gqa_dev.jsonl", "--images", "images", "--decisions",
         "verify/dev.jsonl", "--screen", "screen_dev", "--verifier", "tester", http="verify/dev_page"),
    # Confirmatory study
    step("freeze_screening", "kvsel", "freeze-screening", "--prereg", "PREREG.md", "--manifests", "gqa_confirm.jsonl",
         "--record", "freeze/screening.json"),
    step("screen_confirm", "kvsel", "screen-confirm", "--items", "gqa_confirm.jsonl", "--images", "images",
         "--device", "cpu", "--prereg", "PREREG.md", "--record", "freeze/screening.json", "--out", "screen_confirm"),
    step("verify_confirm", "kvsel", "verify", "--items", "gqa_confirm.jsonl", "--images", "images", "--decisions",
         "verify/confirm.jsonl", "--screen", "screen_confirm", "--verifier", "tester", http="verify/confirm_page"),
    step("freeze_comparative", "kvsel", "freeze-comparative", *COMPARATIVE),
    step("confirm", "kvsel", "confirm", *COMPARATIVE, "--images", "images", "--device", "cpu", "--out", "confirm_run"),
    step("confirm_report", "kvsel", "confirm-report", "--gqa-run", "confirm_run", "--gqa-items", "gqa_confirm.jsonl",
         "--gqa-record", "freeze/comparative_gqa.json", "--resamples", "50", "--out", "confirm_report"),
    # Follow-ups
    step("followup_run", "followup", "run", "--items", "gqa_confirm.jsonl", "--images", "images", "--confirm-run",
         "confirm_run", "--record", "freeze/comparative_gqa.json", "--device", "cpu", "--out", "followup"),
    step("followup_report", "followup", "report", "--run", "followup", "--confirm-run", "confirm_run", "--items",
         "gqa_confirm.jsonl", "--resamples", "50", "--out", "followup_report"),
    # Replication
    step("replication_count", "replication", "count", *REPLICATION),
    step("replication_freeze", "replication", "freeze", *REPLICATION),
    step("replication_run", "replication", "run", *REPLICATION, "--images", "images", "--device", "cpu",
         "--out", "replication_run"),
    step("replication_report", "replication", "report", *REPLICATION, "--run", "replication_run", "--resamples", "50",
         "--original-confirm", "confirm_report/results.json", "--original-followup", "followup_report/results.json",
         "--out", "replication_report"),
    # LLaVA-NeXT
    step("next_validate", "next", "validate", *DEV, *SMALL_VALIDATE, "--out", "next_validate/report.json"),
    step("next_freeze_screening", "next", "freeze-screening", "--prereg", "PREREG.md", "--items", "next_confirm.jsonl",
         "--record", "freeze/next_screening.json"),
    step("next_screen", "next", "screen", "--prereg", "PREREG.md", "--items", "next_confirm.jsonl", "--decisions",
         "next_decisions.jsonl", "--images", "images", "--device", "cpu", "--record", "freeze/next_screening.json",
         "--out", "next_screen"),
    step("next_count", "next", "count", "--items", "next_confirm.jsonl", "--decisions", "next_decisions.jsonl",
         "--screen", "next_screen"),
    step("next_freeze", "next", "freeze", *NEXT),
    step("next_run", "next", "run", *NEXT, "--images", "images", "--device", "cpu", "--out", "next_run"),
    step("next_report", "next", "report", *NEXT, "--run", "next_run", "--resamples", "50", "--llava15",
         "replication_report/results.json", "--out", "next_report"),
)

# --- Inputs: images and the tiny models ----------------------------------------------------------------------

EOS = 2


def image(k: int) -> Image.Image:
    """Seeded noise; the size varies with k so the images differ in aspect."""
    pixels = np.random.default_rng(k).integers(0, 255, (60 + 20 * (k % 5), 80 + 20 * (k % 3), 3), dtype=np.uint8)
    return Image.fromarray(pixels)


def tilt_end_of_sequence(model, repo: str, revision: str | None, strength: float):
    """Moves the EOS row of the head toward the mean final hidden state of one prompt, by `strength`."""
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(repo, revision=revision)
    inputs = processor(text="USER: <image>\nWhat is this? ASSISTANT:", images=image(1000), return_tensors="pt")
    with torch.no_grad():
        hidden = model(**inputs, output_hidden_states=True).hidden_states[-1][0].mean(0)
        model.lm_head.weight[EOS] += strength * hidden / hidden.norm()
    return model


def tiny_bundles():
    from conftest import make_tiny_llava, make_tiny_llava_next

    from kvsel.model import bundle_from_model
    from kvsel.models import llava15, llava_next

    llava = tilt_end_of_sequence(make_tiny_llava(), "llava-hf/llava-1.5-7b-hf", None, 0.12)
    nxt = tilt_end_of_sequence(make_tiny_llava_next(), "llava-hf/llava-v1.6-vicuna-7b-hf", llava_next.REVISION, 0.16)
    processor = llava_next.NEXT.load_processor()
    return (bundle_from_model(llava, llava15.LLAVA.load_tokenizer(), llava15.LLAVA.load_processor()),
            bundle_from_model(nxt, processor.tokenizer, processor, family=llava_next.NEXT))


def make_workspace(root: Path) -> Path:
    shutil.copytree(INPUTS, root)
    (root / "images").mkdir()
    names = {json.loads(line)["image_file"] for manifest in root.glob("*.jsonl") if "decisions" not in manifest.name
             for line in manifest.read_text().splitlines()}
    for name in sorted(names):
        image(int(name.removeprefix("img").removesuffix(".png"))).save(root / "images" / name)
    return root


# --- Running and capturing a step ----------------------------------------------------------------------------


def invoke(program: str, argv: list[str]) -> tuple[int, str, str]:
    """Runs a command line in process as `python -m` would: exit code, stdout and stderr."""
    module = importlib.import_module(PROGRAMS[program])
    stdout, stderr = io.StringIO(), io.StringIO()
    sys.argv = [program, *argv]
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        try:
            module.main()
            code = 0
        except SystemExit as stop:
            if stop.code is None or isinstance(stop.code, int):
                code = stop.code or 0
            else:
                code = 1
                print(stop.code, file=sys.stderr)
    return code, stdout.getvalue(), stderr.getvalue()


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def use_page(port: int, out: Path, failures: list) -> None:
    """Drives the verification page as a person would, saves each response, then stops the server (Ctrl-C)."""
    base = f"http://127.0.0.1:{port}"

    def fetch(path, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        try:
            with urllib.request.urlopen(urllib.request.Request(base + path, data=data), timeout=10) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            return error.read()

    serving = False
    try:
        for _ in range(200):
            try:
                fetch("/api/state")
                serving = True
                break
            except OSError:
                time.sleep(0.05)
        if not serving:
            raise RuntimeError("the verification page never started")
        out.mkdir(parents=True)
        (out / "page.html").write_bytes(fetch("/"))
        items = fetch("/api/items")
        (out / "items.json").write_bytes(items)
        (out / "state_before.json").write_bytes(fetch("/api/state"))
        first = json.loads(items)[0]["item_id"]
        (out / "decide.json").write_bytes(fetch("/api/decide", {"item_id": first, "decision": "pass", "note": "seen"}))
        (out / "decide_refused.json").write_bytes(fetch("/api/decide", {"item_id": first, "decision": "maybe"}))
        (out / "missing_image.json").write_bytes(fetch("/image/no-such-item"))
    except Exception as error:  # noqa: BLE001  reported by the step, after the server stops
        failures.append(error)
    finally:
        if serving:  # otherwise the command already returned, and Ctrl-C would stop pytest instead
            _thread.interrupt_main()


def run_step(spec: Step) -> tuple[int, str, str]:
    if spec.http is None:
        return invoke(spec.program, list(spec.argv))
    port, failures = free_port(), []
    client = threading.Thread(target=use_page, args=(port, Path(spec.http), failures))
    client.start()
    code, stdout, stderr = invoke(spec.program, [*spec.argv, "--port", str(port)])
    client.join()
    if failures:
        raise failures[0]
    return code, stdout.replace(f"127.0.0.1:{port}", "127.0.0.1:<port>"), stderr  # the port the suite chose


def snapshot(root: Path) -> dict[str, str]:
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file()}


def capture(spec: Step, workspace: Path, out: Path) -> None:
    before = snapshot(workspace)
    code, stdout, stderr = run_step(spec)
    after = snapshot(workspace)
    out.mkdir(parents=True)
    command = {"exit_code": code, "stdout": mask(stdout).splitlines(), "stderr": mask(stderr).splitlines()}
    (out / "command.json").write_text(json.dumps(command, indent=1, ensure_ascii=False) + "\n")
    for name in sorted(name for name, digest in after.items() if before.get(name) != digest):
        source, target = workspace / name, out / "files" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix == ".npz":
            target.mkdir()
            with np.load(source) as arrays:
                for key in arrays.files:
                    np.save(target / f"{key}.npy", arrays[key])
            continue
        data = source.read_bytes()
        try:
            target.write_text(mask(data.decode()))
        except UnicodeDecodeError:
            target.write_bytes(data)


@pytest.fixture(scope="session")
def captures(tmp_path_factory) -> Path:
    """Runs every step once, in order, and returns the folder holding one capture per step."""
    root = tmp_path_factory.mktemp("characterization")
    workspace, out = make_workspace(root / "workspace"), root / "captures"
    llava, nxt = tiny_bundles()
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(workspace)
        patch.setattr(sys, "argv", list(sys.argv))
        patch.setattr("kvsel.cli._bundle", lambda args: llava)
        patch.setattr("kvsel_next.__main__.load_bundle", lambda device, model_dir: nxt)
        for module, name, value in SIZES:
            patch.setattr(importlib.import_module(module), name, value)
        for key, value in NEXT_SIZES.items():
            patch.setitem(importlib.import_module("kvsel_next.screen").SCREENING, key, value)
        patch.setattr(time, "time", lambda: 0.0)
        patch.setattr(time, "perf_counter", lambda: 0.0)
        for spec in STEPS:
            capture(spec, workspace, out / spec.name)
    if os.environ.get("UPDATE_GOLDEN"):
        shutil.rmtree(GOLDEN, ignore_errors=True)
        shutil.copytree(out, GOLDEN)
    return out


# --- Comparing -----------------------------------------------------------------------------------------------


def _short(value) -> str:
    text = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else repr(value)
    return text if len(text) <= 120 else text[:117] + "..."


def json_difference(expected, actual, where: str = "$") -> str | None:
    """The first place two JSON values differ, key order included."""
    if type(expected) is not type(actual):
        return f"{where}: {_short(expected)} != {_short(actual)}"
    if isinstance(expected, dict):
        for key in expected:
            if key not in actual:
                return f"{where}.{key}: missing"
            found = json_difference(expected[key], actual[key], f"{where}.{key}")
            if found:
                return found
        extra = [key for key in actual if key not in expected]
        if extra:
            return f"{where}.{extra[0]}: unexpected"
        if list(expected) != list(actual):
            return f"{where}: key order {list(expected)} != {list(actual)}"
        return None
    if isinstance(expected, list):
        for n, (a, b) in enumerate(zip(expected, actual)):
            found = json_difference(a, b, f"{where}[{n}]")
            if found:
                return found
        if len(expected) != len(actual):
            return f"{where}: {len(expected)} entries != {len(actual)}"
        return None
    return None if expected == actual else f"{where}: {_short(expected)} != {_short(actual)}"


def _parse(data: bytes, name: str):
    if name.endswith(".jsonl"):
        return [json.loads(line) for line in data.decode().splitlines()]
    return json.loads(data)


def _drop(value, keys: set[str]):
    if isinstance(value, list):  # a JSONL file: one object per line
        return [_drop(entry, keys) for entry in value]
    return {key: entry for key, entry in value.items() if key not in keys} if isinstance(value, dict) else value


def content_difference(expected: bytes, actual: bytes, name: str) -> str:
    """Where two different files first differ: a JSON path, an array index, a line, or a byte offset."""
    if name.endswith(".npy"):
        a, b = np.load(io.BytesIO(expected)), np.load(io.BytesIO(actual))
        if a.dtype != b.dtype or a.shape != b.shape:
            return f"array {a.dtype}{a.shape} != {b.dtype}{b.shape}"
        index = np.argwhere(a != b)[0].tolist() if np.any(a != b) else None
        return f"array at {index}: {a[tuple(index)]!r} != {b[tuple(index)]!r}" if index else "same array, other bytes"
    try:
        found = json_difference(_parse(expected, name), _parse(actual, name))
    except (ValueError, UnicodeDecodeError):
        found = None
    if found:
        return found
    try:
        old, new = expected.decode().splitlines(), actual.decode().splitlines()
    except UnicodeDecodeError:
        offset = next((n for n, (a, b) in enumerate(zip(expected, actual)) if a != b), min(len(expected), len(actual)))
        return f"bytes differ from offset {offset}"
    for n, (a, b) in enumerate(zip(old, new), 1):
        if a != b:
            return f"line {n}: {a[:120]!r} != {b[:120]!r}"
    if len(old) != len(new):
        return f"{len(old)} lines != {len(new)}"
    return "same lines, other bytes (line endings or the final newline)"


def first_difference(expected: Path, actual: Path, exemptions=()) -> str | None:
    """None when two captures match; otherwise the first differing file and the first difference inside it."""
    def files(root):
        return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()} if root.is_dir() else set()

    for name in sorted(files(expected) | files(actual)):
        rules = [e for e in exemptions if fnmatch.fnmatchcase(name, e.path)]
        if any(not e.keys for e in rules):
            continue
        if not (actual / name).is_file():
            return f"{name}: missing (only in the golden files)"
        if not (expected / name).is_file():
            return f"{name}: unexpected (not in the golden files)"
        old, new = (expected / name).read_bytes(), (actual / name).read_bytes()
        if rules:
            keys = {key for e in rules for key in e.keys}
            found = json_difference(_drop(_parse(old, name), keys), _drop(_parse(new, name), keys))
            if found:
                return f"{name}: {found}"
        elif old != new:
            return f"{name}: {content_difference(old, new, name)}"
    return None


# --- Tests ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", [s.name for s in STEPS])
def test_command_output_matches_the_golden_files(captures, name):
    assert (GOLDEN / name).is_dir(), f"no golden files for {name}; capture them with UPDATE_GOLDEN=1"
    found = first_difference(GOLDEN / name, captures / name, [e for e in EXEMPTIONS if e.step == name])
    assert found is None, f"{name}: {found}"


def test_exemptions_are_narrow():
    steps = {s.name for s in STEPS}
    for exemption in EXEMPTIONS:
        assert exemption.change in ALLOWED_CHANGES, exemption
        assert exemption.step in steps, exemption
        assert "*" not in exemption.path.split("/")[0] and exemption.path not in ("", "**"), exemption


def test_mask_hides_only_timestamps_and_code_hashes():
    digest = "ab" * 32
    text = (f'{{"frozen_at": "2026-11-18T01:02:03+00:00", "time": "2026-11-01T00:00:00+0000", '
            f'"code_sha256": "{digest}", "stage1_record_sha256": "{digest}", "main_set_sha256": "{digest}"}}')
    assert mask(text) == ('{"frozen_at": "<timestamp>", "time": "<timestamp>", "code_sha256": "<code hash>", '
                          f'"stage1_record_sha256": "<code hash>", "main_set_sha256": "{digest}"}}')


def test_first_difference_names_the_file_and_the_place(tmp_path):
    expected, actual = tmp_path / "expected", tmp_path / "actual"
    for root in (expected, actual):
        (root / "files").mkdir(parents=True)
        (root / "files" / "a.json").write_text('{"x": 1}')
        (root / "files" / "b.md").write_text("one\ntwo\n")
    assert first_difference(expected, actual) is None
    (actual / "files" / "b.md").write_text("one\nTwo\n")
    assert first_difference(expected, actual) == "files/b.md: line 2: 'two' != 'Two'"
    (actual / "files" / "a.json").write_text('{"x": 1, "rows": [1]}')
    assert first_difference(expected, actual) == "files/a.json: $.rows: unexpected"
    assert first_difference(expected, actual, [Exemption("item record hash", "s", "files/a.json", ("rows",))]) \
        == "files/b.md: line 2: 'two' != 'Two'"
    (actual / "files" / "c.txt").write_text("new")
    (actual / "files" / "b.md").write_text("one\ntwo\n")
    assert first_difference(expected, actual, [Exemption("item record hash", "s", "files/a.json", ("rows",))]) \
        == "files/c.txt: unexpected (not in the golden files)"
