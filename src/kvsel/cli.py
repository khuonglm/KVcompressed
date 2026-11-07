"""Command line entry points. ``kvsel <command> --help`` lists each command's options."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .model_files import DEFAULT_MODEL_DIR


def _bundle(args):
    from .model import load_bundle

    return load_bundle(device=args.device, head_precision=args.head_precision, model_dir=args.model_dir)


def cmd_build_gqa(args) -> None:
    from .gqa import build
    from .model import load_tokenizer

    funnel = build(args.questions, args.scene_graphs, args.out_dir, load_tokenizer(args.model_dir), args.dev_items)
    print(json.dumps(funnel, indent=2))


def cmd_fetch_model(args) -> None:
    from .model_files import fetch_model

    missing = fetch_model(args.out, args.workers)
    if missing:
        raise SystemExit(f"still missing {', '.join(missing)}; rerun the command to resume")
    print(f"model complete at {args.out}")


def cmd_fetch_images(args) -> None:
    from .fetch_images import fetch

    summary = fetch(args.manifests, args.out, args.url, args.workers, args.limit)
    print(json.dumps(summary, indent=2))
    if summary["failed"]:
        raise SystemExit(f"{len(summary['failed'])} files failed; rerun the command to retry them")


def cmd_check_serialization(args) -> None:
    from .model import load_tokenizer
    from .runner import load_items
    from .validate import check_serialization

    result = check_serialization(load_tokenizer(args.model_dir), load_items(args.items))
    print(json.dumps(result, indent=2))
    if not result["pass"]:
        raise SystemExit(1)


def cmd_validate(args) -> None:
    from .runner import load_items, require_development
    from .validate import validate

    items = load_items(args.items)
    require_development(items)
    report = validate(_bundle(args), items, args.images, args.noise_items, args.sparse_items, args.repeat_items)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    for name in ("E1", "E2", "E3", "E4", "E5", "E7"):
        print(f"{name}: {'PASS' if report[name]['pass'] else 'FAIL'}")
    print(f"noise floor, fp32 head:   {report['noise_floor']}")
    print(f"noise floor, native head: {report['noise_floor_native_head']}")
    print(f"full report: {args.out}")
    if not report["all_pass"]:
        raise SystemExit(1)


def cmd_screen(args) -> None:
    from .runner import eligible, load_items, run_screen

    items = load_items(args.items)
    run_screen(_bundle(args), items, args.images, args.out)
    for surface in ("lower", "capitalized"):
        results = [json.loads(path.read_text()) for path in (args.out / "items").glob("*.json")]
        passed = sum(eligible(result["rows"][0], surface) for result in results)
        print(f"eligible under `{surface}`: {passed} of {len(results)}")
    answers = [json.loads(path.read_text())["rows"][0]["greedy_text"] for path in (args.out / "items").glob("*.json")]
    print("sample greedy answers:", answers[:20])


def cmd_pilot(args) -> None:
    from .runner import eligible, load_items, pilot_units, run_items

    pool = load_items(args.items)
    chosen = []
    for item in pool:
        path = args.screen / "items" / f"{item['item_id']}.json"
        if path.exists() and eligible(json.loads(path.read_text())["rows"][0], args.surface):
            chosen.append(item)
    chosen = chosen[: args.limit]
    print(f"{len(chosen)} eligible items selected under `{args.surface}`")
    if not chosen:
        raise SystemExit("no eligible items; run `kvsel screen` first")
    run_items(_bundle(args), chosen, args.images, args.out, pilot_units(k=args.k))


def cmd_report(args) -> None:
    from .analysis import report

    text = report(args.run, args.surface, args.resamples)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text)


def main() -> None:
    parser = argparse.ArgumentParser(prog="kvsel", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    def add_model_dir(sub) -> None:
        sub.add_argument(
            "--model-dir", type=Path, default=DEFAULT_MODEL_DIR,
            help="local model copy, used if present; otherwise the Hugging Face Hub (default: %(default)s)",
        )

    def add_model_options(sub) -> None:
        add_model_dir(sub)
        sub.add_argument("--device", default="cuda")
        sub.add_argument("--head-precision", default="fp32", choices=["fp32", "native"])

    def add_item_options(sub) -> None:
        sub.add_argument("--items", type=Path, required=True, help="development manifest (JSONL)")
        sub.add_argument("--images", type=Path, required=True, help="directory holding the image files")

    sub = commands.add_parser("build-gqa", help="build GQA item manifests and print the funnel")
    sub.add_argument("--questions", type=Path, required=True)
    sub.add_argument("--scene-graphs", type=Path, required=True)
    sub.add_argument("--out-dir", type=Path, default=Path("manifests"))
    sub.add_argument("--dev-items", type=int, default=600, help="candidates before eligibility and verification")
    add_model_dir(sub)
    sub.set_defaults(run=cmd_build_gqa)

    sub = commands.add_parser("fetch-model", help="download the model for local use; resumes if rerun")
    sub.add_argument("--out", type=Path, default=DEFAULT_MODEL_DIR)
    sub.add_argument("--workers", type=int, default=8)
    sub.set_defaults(run=cmd_fetch_model)

    sub = commands.add_parser("fetch-images", help="download only the images a manifest needs")
    sub.add_argument("manifests", type=Path, nargs="+")
    sub.add_argument("--out", type=Path, default=Path("data/gqa/images"))
    sub.add_argument("--url", default="https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip")
    sub.add_argument("--workers", type=int, default=3, help="the server returns 503 errors at 8")
    sub.add_argument("--limit", type=int, help="fetch at most this many files, to try it out")
    sub.set_defaults(run=cmd_fetch_images)

    sub = commands.add_parser("check-serialization", help="test E1; tokenizer only, no GPU")
    sub.add_argument("--items", type=Path, required=True)
    add_model_dir(sub)
    sub.set_defaults(run=cmd_check_serialization)

    sub = commands.add_parser("validate", help="noise floor and tests E1-E5, E7 on the real model")
    add_item_options(sub)
    add_model_options(sub)
    sub.add_argument("--noise-items", type=int, default=50)
    sub.add_argument("--sparse-items", type=int, default=20, help="items for E3-E5")
    sub.add_argument("--repeat-items", type=int, default=5, help="items for E7")
    sub.add_argument("--out", type=Path, default=Path("results/validate.json"))
    sub.set_defaults(run=cmd_validate)

    sub = commands.add_parser("screen", help="Full_Q on development items, for eligibility")
    add_item_options(sub)
    add_model_options(sub)
    sub.add_argument("--out", type=Path, default=Path("results/screen_dev"))
    sub.set_defaults(run=cmd_screen)

    sub = commands.add_parser("pilot", help="the full grid of cells on eligible development items")
    add_item_options(sub)
    add_model_options(sub)
    sub.add_argument("--screen", type=Path, default=Path("results/screen_dev"))
    sub.add_argument("--surface", default="lower", choices=["lower", "capitalized"])
    sub.add_argument("--limit", type=int, default=200)
    sub.add_argument("--k", type=int, default=144, choices=[288, 144, 72])
    sub.add_argument("--out", type=Path, default=Path("results/pilot"))
    sub.set_defaults(run=cmd_pilot)

    sub = commands.add_parser("report", help="estimands and tables from a run directory")
    sub.add_argument("run", type=Path)
    sub.add_argument("--surface", default="lower", choices=["lower", "capitalized", "logsum"])
    sub.add_argument("--resamples", type=int, default=10_000)
    sub.add_argument("--out", type=Path)
    sub.set_defaults(run=cmd_report)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
