"""Proves code amendment 3 (PREREG 3.6) reproduces every stored result, then records the evidence.

    uv run python scripts/verify_amendment3.py

Reruns the three reports with the current code into *_amend3 folders and compares them with the
stored ones: confirm and follow-up results.json must be byte-identical; the replication's must be
identical on every key the frozen code produced (the new code only adds the 3.3 sign-rule reading).
On success the pending entries in freeze/code_amendments.json get their evidence; on failure they
stay pending and the script exits 1 naming the first difference.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from kvsel.confirm import code_hash_all
from kvsel.replication.protocol import code_hash_tree

ROOT = Path(__file__).resolve().parents[1]
AMENDMENTS = ROOT / "freeze" / "code_amendments.json"
ITEMS = "manifests/gqa_confirm.jsonl"
COMMON = ["--items", ITEMS, "--screen", "results/confirm/gqa_screen_a1", "--decisions", "verification/gqa_confirm.jsonl",
          "--exclude-runs", "results/confirm/gqa_run", "results/followup/gqa", "results/confirm/gqa_run_extra"]
REPORTS = {  # stored results.json -> (command, new output folder, exact?)
    "results/confirm/report": (["kvsel", "confirm-report", "--gqa-run", "results/confirm/gqa_run", "--gqa-items", ITEMS,
                                "--gqa-record", "freeze/comparative_gqa.json"], "results/confirm/report_amend3", True),
    "results/followup/gqa/report": (["python", "-m", "kvsel.followup", "report", "--run", "results/followup/gqa",
                                     "--confirm-run", "results/confirm/gqa_run", "--items", ITEMS],
                                    "results/followup/gqa/report_amend3", True),
    "results/replication/report": (["python", "-m", "kvsel.replication", "report", *COMMON, "--run",
                                    "results/replication/gqa_run", "--original-confirm", "results/confirm/report/results.json",
                                    "--original-followup", "results/followup/gqa/report/results.json"],
                                   "results/replication/report_amend3", False),
}
EVIDENCE = ("confirm-report results.json and the follow-up results.json byte-identical to those from the accepted code; "
            "replication results.json identical on every key the frozen code produced (the new code adds the 3.3 "
            "sign-rule reading); produced by scripts/verify_amendment3.py")


def first_difference(old, new, path="") -> str | None:
    """Where `old` is not contained in `new` (dicts by key, everything else by equality)."""
    if isinstance(old, dict) and isinstance(new, dict):
        for key, value in old.items():
            if key not in new:
                return f"{path}/{key} missing"
            found = first_difference(value, new[key], f"{path}/{key}")
            if found:
                return found
        return None
    return None if old == new else f"{path}: {old!r} != {new!r}"


def main() -> None:
    entries = json.loads(AMENDMENTS.read_text())
    pending = [e for e in entries if e["evidence"].startswith("pending")]
    current = {"stage 3 (code_hash_all)": code_hash_all(), "replication (code_hash_tree)": code_hash_tree()}
    if {e["scope"] for e in pending} != set(current):
        raise SystemExit(f"expected one pending entry per scope {sorted(current)}, found {[e['scope'] for e in pending]}")
    for entry in pending:
        if entry["to"] != current[entry["scope"]]:
            raise SystemExit(f"the code changed after the amendment was written ({entry['scope']}); rewrite the entry first")
    for stored, (command, out, exact) in REPORTS.items():
        print(f"== {stored} -> {out}", flush=True)
        subprocess.run(["uv", "run", *command, "--out", out], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        old_text, new_text = (ROOT / stored / "results.json").read_text(), (ROOT / out / "results.json").read_text()
        if exact:
            found = None if old_text == new_text else f"{stored}/results.json is not byte-identical to {out}/results.json"
        else:
            found = first_difference(json.loads(old_text), json.loads(new_text))
        if found:
            raise SystemExit(f"NOT reproduced: {found}. The amendment stays pending; report this difference.")
        print(f"   reproduced {stored}/results.json")
    for entry in pending:
        entry["evidence"] = EVIDENCE
    AMENDMENTS.write_text(json.dumps(entries, indent=2) + "\n")
    print(f"evidence recorded in {AMENDMENTS.relative_to(ROOT)}; commit it with the code")


if __name__ == "__main__":
    sys.exit(main())
