"""Command line entry points. ``kvsel <command> --help`` lists each command's options."""

from __future__ import annotations

import argparse
import getpass
import json
from pathlib import Path

from .families import FAMILIES, LLAVA, get_family
from .freeze import DEFAULT_RECORD


def _family(args):
    return get_family(getattr(args, "model", LLAVA.name))


def _bundle(args):
    from .model import load_bundle

    return load_bundle(device=args.device, head_precision=args.head_precision, model_dir=args.model_dir, family=_family(args))


def cmd_build_gqa(args) -> None:
    from .gqa import build
    from .model import load_tokenizer

    funnel = build(args.questions, args.scene_graphs, args.out_dir, load_tokenizer(args.model_dir), args.dev_items,
                   args.dev_manifest)
    print(json.dumps(funnel, indent=2))


def cmd_build_textvqa(args) -> None:
    from .model import load_tokenizer
    from .textvqa import build

    funnel = build(args.questions, args.ocr, args.out_dir, load_tokenizer(args.model_dir), args.dev_items, args.dev_manifest)
    print(json.dumps(funnel, indent=2))


def cmd_fetch_model(args) -> None:
    from .model_files import fetch_model

    family = _family(args)
    out = args.out or family.default_dir
    missing = fetch_model(out, args.workers, family.model_id)
    if missing:
        raise SystemExit(f"still missing {', '.join(missing)}; rerun the command to resume")
    print(f"model complete at {out}")


def cmd_fetch_images(args) -> None:
    from .fetch_images import fetch

    summary = fetch(args.manifests, args.out, args.url, args.workers, args.limit)
    print(json.dumps(summary, indent=2))
    if summary["failed"]:
        raise SystemExit(f"{len(summary['failed'])} files failed; rerun the command to retry them")


def cmd_validate(args) -> None:
    from .runner import load_items, require_development
    from .validate import validate

    items = load_items(args.items)
    require_development(items)
    from . import validate as validate_module

    validate_module.SURFACE = args.surface
    report = validate(_bundle(args), items, args.images, args.noise_items, args.sparse_items, args.repeat_items)
    report["surface"] = args.surface
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    for name in ("E1", "E2", "E3", "E4", "E5", "E7"):
        print(f"{name}: {'PASS' if report[name]['pass'] else 'FAIL'}")
    print(f"noise floor, {report['head_precision']} head:   {report['noise_floor']}")
    print(f"noise floor, QH_mis:      {report['noise_floor_mis']}")
    print(f"noise floor, native head: {report['noise_floor_native_head']}")
    print(f"full report: {args.out}")
    if not report["all_pass"]:
        raise SystemExit(1)


def cmd_screen(args) -> None:
    from .runner import eligible, load_items, run_screen

    items = load_items(args.items)
    run_screen(_bundle(args), items, args.images, args.out)
    for surface in ("lower", "capitalized", "upper"):
        results = [json.loads(path.read_text()) for path in (args.out / "items").glob("*.json")]
        passed = sum(eligible(result["rows"][0], surface) for result in results)
        print(f"eligible under `{surface}`: {passed} of {len(results)}")
    answers = [json.loads(path.read_text())["rows"][0]["greedy_text"] for path in (args.out / "items").glob("*.json")]
    print("sample greedy answers:", answers[:20])


def cmd_freeze_screening(args) -> None:
    from .freeze import freeze_screening

    print(json.dumps(freeze_screening(args.prereg, args.manifests, args.record, args.amends, args.reason), indent=2))


def cmd_screen_confirm(args) -> None:
    from .freeze import check_screening
    from .runner import load_items
    from .screening import run_confirm_screen

    frozen = check_screening(args.prereg, args.items, args.record)  # before loading the model
    items = load_items(args.items)[: args.limit]
    funnel = run_confirm_screen(_bundle(args), items, args.images, args.out, frozen)
    print(json.dumps({key: funnel[key] for key in ("screened", "eligible", "not_eligible")}, indent=2))


def _confirm_paths(args) -> tuple:
    return args.prereg, args.items, args.screen, args.decisions, args.record, args.stage1


def cmd_freeze_comparative(args) -> None:
    from .confirm import freeze_comparative

    payload = freeze_comparative(*_confirm_paths(args))
    print(json.dumps({key: value for key, value in payload.items() if key != "settings"}, indent=2))


def cmd_confirm(args) -> None:
    from .confirm import check_comparative, main_set, run_confirm
    from .runner import load_items

    check_comparative(*_confirm_paths(args))  # before loading the model
    items = load_items(args.items)
    selection = main_set(items, args.screen, args.decisions)
    print(json.dumps(run_confirm(_bundle(args), items, args.images, args.out, selection), indent=2))


def cmd_confirm_report(args) -> None:
    from .confirm_analysis import confirm_report

    runs, records = {}, {}
    for dataset in ("gqa", "textvqa"):
        run, items, record = (getattr(args, f"{dataset}_{part}") for part in ("run", "items", "record"))
        if run is None:
            continue
        if items is None:
            raise SystemExit(f"--{dataset}-items is needed with --{dataset}-run")
        runs[dataset] = (run, items)
        if record is not None:
            records[dataset] = record
    if not runs:
        raise SystemExit("give at least one run, for example --gqa-run results/confirm/gqa_run")
    confirm_report(runs, args.out, args.resamples, records)
    print((args.out / "report.md").read_text())


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


def _is_confirm_screen(screen: Path) -> bool:
    """A confirmatory screen stores only eligibility flags; a development screen stores full rows."""
    first = next((screen / "items").glob("*.json"), None)
    return first is not None and "eligible" in json.loads(first.read_text())


def cmd_verify(args) -> None:
    from .runner import eligible, load_items
    from .verify import build_queue, load_decisions, make_server, summarize

    items = load_items(args.items)
    eligible_ids = None
    if args.screen and _is_confirm_screen(args.screen):
        from .screening import eligible_ids as confirm_eligible

        eligible_ids = confirm_eligible(args.screen)
    elif args.screen:
        eligible_ids = set()
        for item in items:
            path = args.screen / "items" / f"{item['item_id']}.json"
            if path.exists() and eligible(json.loads(path.read_text())["rows"][0], args.surface):
                eligible_ids.add(item["item_id"])
    queue = build_queue(items, eligible_ids, args.limit)
    server = make_server(queue, args.images, args.decisions, args.verifier, args.port)
    print(f"{len(queue)} items queued; decisions go to {args.decisions}")
    print(f"open http://127.0.0.1:{server.server_address[1]}  (Ctrl-C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    print(json.dumps(summarize(load_decisions(args.decisions, queue)[0], queue), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(prog="kvsel", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    def add_model_dir(sub, choose_model: bool = True) -> None:
        if choose_model:
            sub.add_argument("--model", default=LLAVA.name, choices=sorted(FAMILIES), help="model family (default: %(default)s)")
        sub.add_argument(
            "--model-dir", type=Path, default=None,
            help="local model copy, used if present; otherwise the Hugging Face Hub "
            f"(default: {LLAVA.default_dir})",
        )

    def add_model_options(sub) -> None:
        add_model_dir(sub)
        sub.add_argument("--device", default="cuda")
        sub.add_argument("--head-precision", default=None, choices=["fp32", "native"],
                         help="output head precision (default: fp32)")

    def add_item_options(sub) -> None:
        sub.add_argument("--items", type=Path, required=True, help="development manifest (JSONL)")
        sub.add_argument("--images", type=Path, required=True, help="directory holding the image files")

    sub = commands.add_parser("build-gqa", help="build GQA item manifests and print the funnel")
    sub.add_argument("--questions", type=Path, required=True)
    sub.add_argument("--scene-graphs", type=Path, required=True)
    sub.add_argument("--out-dir", type=Path, default=Path("manifests"))
    sub.add_argument("--dev-items", type=int, default=600, help="candidates before eligibility and verification")
    sub.add_argument("--dev-manifest", type=Path, default=None,
                     help="keep these development items unchanged; confirmatory items avoid their images")
    add_model_dir(sub, choose_model=False)
    sub.set_defaults(handler=cmd_build_gqa)

    sub = commands.add_parser("build-textvqa", help="build TextVQA item manifests and print the funnel")
    sub.add_argument("--questions", type=Path, nargs="+", required=True)
    sub.add_argument("--ocr", type=Path, nargs="+", required=True)
    sub.add_argument("--out-dir", type=Path, default=Path("manifests"))
    sub.add_argument("--dev-items", type=int, default=1500, help="candidates before eligibility and verification")
    sub.add_argument("--dev-manifest", type=Path, default=None,
                     help="keep these development items unchanged; confirmatory items avoid their images")
    add_model_dir(sub, choose_model=False)
    sub.set_defaults(handler=cmd_build_textvqa)

    sub = commands.add_parser("fetch-model", help="download the model for local use; resumes if rerun")
    sub.add_argument("--model", default=LLAVA.name, choices=sorted(FAMILIES))
    sub.add_argument("--out", type=Path, default=None, help="default: the family's models/ directory")
    sub.add_argument("--workers", type=int, default=8)
    sub.set_defaults(handler=cmd_fetch_model)

    sub = commands.add_parser("fetch-images", help="download only the images a manifest needs")
    sub.add_argument("manifests", type=Path, nargs="+")
    sub.add_argument("--out", type=Path, default=Path("data/gqa/images"))
    sub.add_argument("--url", default="https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip")
    sub.add_argument("--workers", type=int, default=3, help="the server returns 503 errors at 8")
    sub.add_argument("--limit", type=int, help="fetch at most this many files, to try it out")
    sub.set_defaults(handler=cmd_fetch_images)

    sub = commands.add_parser("validate", help="noise floor and tests E1-E5, E7 on the real model")
    add_item_options(sub)
    add_model_options(sub)
    sub.add_argument("--noise-items", type=int, default=50)
    sub.add_argument("--sparse-items", type=int, default=20, help="items for E3-E5")
    sub.add_argument("--repeat-items", type=int, default=5, help="items for E7")
    sub.add_argument("--surface", default="capitalized", choices=["lower", "capitalized", "upper"],
                     help="surface form for the margin checks (default: %(default)s)")
    sub.add_argument("--out", type=Path, default=Path("results/validate.json"))
    sub.set_defaults(handler=cmd_validate)

    sub = commands.add_parser("screen", help="Full_Q on development items, for eligibility")
    add_item_options(sub)
    add_model_options(sub)
    sub.add_argument("--out", type=Path, default=Path("results/screen_dev"))
    sub.set_defaults(handler=cmd_screen)

    sub = commands.add_parser("freeze-screening", help="freeze stage 1: record hashes of the screening rules. No GPU")
    sub.add_argument("--prereg", type=Path, default=Path("PREREG.md"))
    sub.add_argument("--manifests", type=Path, nargs="+",
                     default=[Path("manifests/gqa_confirm.jsonl"), Path("manifests/textvqa_confirm.jsonl")])
    sub.add_argument("--record", type=Path, default=DEFAULT_RECORD)
    sub.add_argument("--amends", type=Path, default=None, help="the stage-1 record this amendment supersedes")
    sub.add_argument("--reason", default=None, help="why the screening rules changed")
    sub.set_defaults(handler=cmd_freeze_screening)

    sub = commands.add_parser("screen-confirm", help="eligibility screen on confirmatory items; writes flags only (GPU)")
    sub.add_argument("--items", type=Path, required=True, help="confirmatory manifest (JSONL)")
    sub.add_argument("--images", type=Path, required=True, help="directory holding the image files")
    add_model_dir(sub, choose_model=False)
    sub.add_argument("--device", default="cuda")
    sub.add_argument("--prereg", type=Path, default=Path("PREREG.md"))
    sub.add_argument("--record", type=Path, default=DEFAULT_RECORD)
    sub.add_argument("--limit", type=int, default=None, help="screen only the first N items in manifest order")
    sub.add_argument("--out", type=Path, required=True)
    sub.set_defaults(handler=cmd_screen_confirm, head_precision=None)

    def add_confirm_paths(sub) -> None:
        sub.add_argument("--items", type=Path, required=True, help="confirmatory manifest (JSONL)")
        sub.add_argument("--screen", type=Path, required=True, help="confirmatory screen directory")
        sub.add_argument("--decisions", type=Path, required=True, help="verification decisions (JSONL)")
        sub.add_argument("--prereg", type=Path, default=Path("PREREG.md"))
        sub.add_argument("--stage1", type=Path, default=DEFAULT_RECORD)
        sub.add_argument("--record", type=Path, required=True, help="for example freeze/comparative_gqa.json")

    sub = commands.add_parser("freeze-comparative", help="freeze stage 3 for one dataset: record hashes and the main set. No GPU")
    add_confirm_paths(sub)
    sub.set_defaults(handler=cmd_freeze_comparative)

    sub = commands.add_parser("confirm", help="the confirmatory run, once, on the frozen main set (GPU)")
    add_confirm_paths(sub)
    sub.add_argument("--images", type=Path, required=True, help="directory holding the image files")
    add_model_dir(sub, choose_model=False)
    sub.add_argument("--device", default="cuda")
    sub.add_argument("--out", type=Path, required=True)
    sub.set_defaults(handler=cmd_confirm, head_precision=None)

    sub = commands.add_parser("confirm-report", help="the pre-registered analysis of confirmatory runs, Holm across datasets. No GPU")
    for dataset in ("gqa", "textvqa"):
        sub.add_argument(f"--{dataset}-run", type=Path, default=None, help=f"{dataset} run directory")
        sub.add_argument(f"--{dataset}-items", type=Path, default=None, help=f"{dataset} manifest, for evidence boxes")
        sub.add_argument(f"--{dataset}-record", type=Path, default=None, help="stage-3 record; checks the run is complete")
    sub.add_argument("--resamples", type=int, default=10_000)
    sub.add_argument("--out", type=Path, required=True)
    sub.set_defaults(handler=cmd_confirm_report)

    sub = commands.add_parser("pilot", help="the full grid of cells on eligible development items")
    add_item_options(sub)
    add_model_options(sub)
    sub.add_argument("--screen", type=Path, default=Path("results/screen_dev"))
    sub.add_argument("--surface", default="lower", choices=["lower", "capitalized", "upper"])
    sub.add_argument("--limit", type=int, default=200)
    sub.add_argument("--k", type=int, default=144, choices=[288, 144, 72])
    sub.add_argument("--out", type=Path, default=Path("results/pilot"))
    sub.set_defaults(handler=cmd_pilot)

    sub = commands.add_parser("verify", help="hand-verify items in a local web page")
    add_item_options(sub)
    sub.add_argument("--decisions", type=Path, required=True, help="append-only JSONL, for example verification/gqa_dev.jsonl")
    sub.add_argument("--screen", type=Path, help="queue only items eligible in this screen directory (development or confirmatory)")
    sub.add_argument("--surface", default="capitalized", choices=["lower", "capitalized", "upper"])
    sub.add_argument("--limit", type=int)
    sub.add_argument("--verifier", default=getpass.getuser())
    sub.add_argument("--port", type=int, default=8765)
    sub.set_defaults(handler=cmd_verify)

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
