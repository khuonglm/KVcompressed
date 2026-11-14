"""`python -m kvsel.replication count|freeze|run|report`. Kept out of `kvsel.cli` so the frozen code hash doesn't change."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from ..analysis import RESAMPLES

RECORD = Path("freeze/replication_gqa.json")


def cmd_count(args) -> None:
    from ..confirm import load_flags, main_set
    from ..runner import load_items
    from ..verify import load_decisions
    from .select import earlier_run_ids, replication_set

    items = load_items(args.items)
    flags = load_flags(args.screen)
    decisions, stale = load_decisions(args.decisions, items)
    main = main_set(items, args.screen, args.decisions)["main"]
    result = replication_set(items, flags, decisions, main, earlier_run_ids(args.exclude_runs))
    print(json.dumps({"main_set": len(main), "replication": len(result["replication"]),
                      "control": len(result["control"]), "failed_before_cutoff": result["failed"],
                      "excluded": dict(Counter(result["excluded"].values())), "stale_decisions": stale,
                      "next_unverified": result["next_unverified"]}, indent=2))


def _protocol_args(args) -> dict:
    return {"prereg": args.prereg, "manifest": args.items, "screen": args.screen, "decisions": args.decisions,
                "stage1": args.stage1, "gqa_record": args.gqa_record, "exclude_runs": args.exclude_runs, "record": args.record}


def cmd_freeze(args) -> None:
    from .protocol import freeze

    frozen = freeze(**_protocol_args(args))
    print(json.dumps({key: frozen[key] for key in ("frozen_at", "replication_set_sha256", "code_sha256")}
                     | {"replication": len(frozen["replication_ids"]), "control": len(frozen["control_ids"]),
                        "excluded": dict(Counter(frozen["excluded"].values()))}, indent=2))


def cmd_run(args) -> None:
    from ..cli import _bundle
    from ..runner import load_items
    from .protocol import check
    from .run import run_replication

    selection = check(**_protocol_args(args))  # before loading the model
    summary = run_replication(_bundle(args), load_items(args.items), args.images, args.out, selection)
    print(json.dumps(summary, indent=2))


def cmd_report(args) -> None:
    from .report import replication_report

    results = replication_report(args.run, args.record, args.out, args.resamples, args.original_confirm,
                                 args.original_followup)
    print(f"wrote {args.out / 'report.md'} ({results['items']} items)")


def _add_protocol(sub) -> None:
    from ..freeze import DEFAULT_RECORD

    sub.add_argument("--prereg", type=Path, default=Path("PREREG.md"))
    sub.add_argument("--items", type=Path, required=True, help="confirmatory manifest (JSONL)")
    sub.add_argument("--screen", type=Path, required=True, help="the amendment-1 screen folder")
    sub.add_argument("--decisions", type=Path, required=True, help="verification decisions (JSONL)")
    sub.add_argument("--stage1", type=Path, default=DEFAULT_RECORD)
    sub.add_argument("--gqa-record", type=Path, default=Path("freeze/comparative_gqa.json"))
    sub.add_argument("--exclude-runs", type=Path, nargs="*", default=[],
                     help="earlier run folders; any item with a file in them is left out")
    sub.add_argument("--record", type=Path, default=RECORD)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m kvsel.replication", description="GQA replication (PREREG 3)")
    commands = parser.add_subparsers(dest="command", required=True)
    sub = commands.add_parser("count", help="how many fresh items qualify; flags and decisions only. No GPU")
    sub.add_argument("--items", type=Path, required=True, help="confirmatory manifest (JSONL)")
    sub.add_argument("--screen", type=Path, required=True, help="the amendment-1 screen folder")
    sub.add_argument("--decisions", type=Path, required=True, help="verification decisions (JSONL)")
    sub.add_argument("--exclude-runs", type=Path, nargs="*", default=[],
                     help="earlier run folders; any item with a file in them is left out")
    sub.set_defaults(handler=cmd_count)

    sub = commands.add_parser("freeze", help="freeze the replication protocol and set. No GPU")
    _add_protocol(sub)
    sub.set_defaults(handler=cmd_freeze)

    sub = commands.add_parser("run", help="run the frozen replication once (GPU); resumes if rerun")
    _add_protocol(sub)
    sub.add_argument("--images", type=Path, required=True)
    sub.add_argument("--out", type=Path, required=True)
    sub.add_argument("--model-dir", type=Path, default=None)
    sub.add_argument("--device", default="cuda")
    sub.set_defaults(handler=cmd_run, head_precision=None)

    sub = commands.add_parser("report", help="the pre-registered replication analysis. No GPU")
    sub.add_argument("--run", type=Path, required=True)
    sub.add_argument("--record", type=Path, default=RECORD)
    sub.add_argument("--resamples", type=int, default=RESAMPLES)
    sub.add_argument("--original-confirm", type=Path, default=None, help="the GQA confirm-report results.json")
    sub.add_argument("--original-followup", type=Path, default=None, help="the follow-up report results.json")
    sub.add_argument("--out", type=Path, required=True)
    sub.set_defaults(handler=cmd_report)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
