"""`python -m kvsel.followup run|report`. Kept out of `kvsel.cli` so the frozen code hash doesn't change."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..analysis import RESAMPLES
from .run import PARTS


def cmd_run(args) -> None:
    from ..cli import _bundle
    from ..runner import load_items
    from .run import confirm_selection, run_followup

    items = load_items(args.items)
    selection = confirm_selection(args.confirm_run, args.record, items)  # before loading the model
    if not selection["code_unchanged"]:
        print("note: the code differs from the stage-3 record; the confirmatory run used other code", flush=True)
    summary = run_followup(_bundle(args), items, args.items, args.images, args.out, selection, tuple(args.parts))
    print(json.dumps({"confirm_code_unchanged": selection["code_unchanged"], **summary}, indent=2))


def cmd_report(args) -> None:
    from .report import followup_report

    print(json.dumps(followup_report(args.run, args.confirm_run, args.items, args.out, args.resamples), indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m kvsel.followup", description="follow-ups (PREREG 2.4); not confirmatory")
    commands = parser.add_subparsers(dest="command", required=True)

    sub = commands.add_parser("run", help="follow-up units on the confirmatory main set (GPU)")
    sub.add_argument("--items", type=Path, required=True, help="confirmatory manifest (JSONL)")
    sub.add_argument("--images", type=Path, required=True)
    sub.add_argument("--confirm-run", type=Path, required=True, help="the finished confirmatory run directory")
    sub.add_argument("--record", type=Path, required=True, help="its stage-3 record, e.g. freeze/comparative_gqa.json")
    sub.add_argument("--parts", nargs="+", default=list(PARTS), choices=PARTS)
    sub.add_argument("--out", type=Path, required=True, help="one folder per part is made inside")
    sub.add_argument("--model-dir", type=Path, default=None)
    sub.add_argument("--device", default="cuda")
    sub.set_defaults(handler=cmd_run, head_precision=None)

    sub = commands.add_parser("report", help="descriptive analysis of the follow-up runs. No GPU")
    sub.add_argument("--run", type=Path, required=True, help="the follow-up folder given to run --out")
    sub.add_argument("--confirm-run", type=Path, required=True)
    sub.add_argument("--items", type=Path, required=True, help="confirmatory manifest, for evidence boxes")
    sub.add_argument("--resamples", type=int, default=RESAMPLES)
    sub.add_argument("--out", type=Path, required=True)
    sub.set_defaults(handler=cmd_report)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
