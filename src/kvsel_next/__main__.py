"""`python -m kvsel_next fetch-model|validate|freeze-screening|screen|count|freeze|run|report`. ``python -m kvsel_next <command> --help`` lists each command's options."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kvsel.models.llava_next import NEXT

MEMORY_LIMIT_GB = 20  # the A100 slice; above it, PREREG 4.1's AnyRes cap is needed


def cmd_fetch_model(args) -> None:
    missing = NEXT.fetch(args.out, args.workers)
    if missing:
        raise SystemExit(f"still missing {', '.join(missing)}; rerun the command to resume")
    print(f"model complete at {args.out}, revision {NEXT.revision}")


def load_bundle(device: str, model_dir: Path | None):
    """The seam tests replace with the tiny model; importing torch models lazily keeps `--help` fast."""
    from kvsel.model import load_bundle as load

    return load(device, model_dir=model_dir, family=NEXT)


def cmd_validate(args) -> None:
    from kvsel.runner import load_items, require_development

    from .validate import CHECKS, validate

    items = load_items(args.items)
    require_development(items)  # before loading the model
    report = validate(load_bundle(args.device, args.model_dir), items, args.images, args.noise_items,
                      args.sparse_items, args.repeat_items, args.surface)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    for name in CHECKS:
        print(f"{name}: {'PASS' if report[name]['pass'] else 'FAIL'}")
    print(f"noise floor, Q:   {report['noise_floor']}")
    print(f"noise floor, mis: {report['noise_floor_mis']}")
    layouts, memory, timing = report["layouts"], report["memory"], report["time"]
    print(f"visual tokens: {layouts['visual_tokens_min']}-{layouts['visual_tokens_max']}, grids {layouts['grids']}")
    if memory["run_peak_gb"] is None:
        print("peak GPU memory: n/a (not on CUDA)")
    else:
        print(f"peak GPU memory, run: {memory['run_peak_gb']:.2f} GB at {memory['at_visual_tokens']} visual tokens "
              f"(prefill {memory['prefill_peak_gb']:.2f} GB; validate's own full passes {memory['validate_peak_gb']:.2f} GB)")
    if memory["run_peak_gb"] is not None and memory["run_peak_gb"] > MEMORY_LIMIT_GB:
        print(f"  over {MEMORY_LIMIT_GB} GB: an AnyRes cap must be set in PREREG 4.1 before stage 1")
    print(f"time per item: {timing['seconds']:.1f} s in validate; prefill {timing['prefill_seconds']:.2f} s, "
          f"probe {timing['probe_seconds']:.2f} s, Full unit {timing['unit_seconds']:.2f} s")
    print(f"projected: screen <= {timing['screen_hours_upper']:.1f} h, comparative run <= {timing['comparative_hours_upper']:.1f} h")
    print(f"model {report['model']} at revision {report['revision']}")
    print(f"full report: {args.out}")
    print("GO" if report["go"] else "NO-GO")
    if not report["go"]:
        raise SystemExit(1)


def refuse_missing_images(items: list[dict], images_dir: Path) -> None:
    missing = [item["image_file"] for item in items if not (images_dir / item["image_file"]).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} images are missing from {images_dir} (first: {missing[0]})")


def cmd_freeze_screening(args) -> None:
    from .screen import freeze_screening

    frozen = freeze_screening(args.prereg, args.items, args.record)
    print(json.dumps({key: frozen[key] for key in ("frozen_at", "prereg_section_sha256", "code_sha256")}, indent=2))


def cmd_screen(args) -> None:
    from kvsel.runner import load_items

    from .screen import check_screening, pool, run_screen

    frozen = check_screening(args.prereg, args.items, args.record)  # before loading the model
    items = pool(load_items(args.items), args.decisions)
    refuse_missing_images(items, args.images)
    funnel = run_screen(load_bundle(args.device, args.model_dir), items, args.images, args.out, frozen["settings"])
    print(json.dumps(funnel, indent=2))


def cmd_count(args) -> None:
    from kvsel.runner import load_items

    from .screen import count

    print(json.dumps(count(load_items(args.items), args.decisions, args.screen), indent=2))


def _protocol(args) -> dict:
    return {"prereg": args.prereg, "manifest": args.items, "screen_dir": args.screen, "decisions": args.decisions,
            "stage1": args.stage1, "record": args.record}


def cmd_freeze(args) -> None:
    from .comparative import freeze

    frozen = freeze(**_protocol(args))
    print(json.dumps({key: frozen[key] for key in ("frozen_at", "main_set_sha256", "code_sha256")}
                     | {"main": len(frozen["main_ids"]), "control": len(frozen["control_ids"])}, indent=2))


def cmd_run(args) -> None:
    from kvsel.runner import load_items

    from .comparative import check, run_comparative

    selection = check(**_protocol(args))  # before loading the model
    items = load_items(args.items)
    by_id = {item["item_id"]: item for item in items}
    refuse_missing_images([by_id[i] for i in selection["main_ids"]], args.images)
    summary = run_comparative(load_bundle(args.device, args.model_dir), items, args.images, args.out, selection)
    print(json.dumps(summary, indent=2))


def cmd_report(args) -> None:
    from .comparative import check
    from .report import report

    check(**_protocol(args), prereg_frozen=False)
    found = report(args.run, args.record, args.out, args.llava15, args.resamples)
    print(f"wrote {args.out / 'report.md'} ({found['items']} items)")


def _add_protocol(sub) -> None:
    sub.add_argument("--prereg", type=Path, default=Path("PREREG.md"))
    sub.add_argument("--items", type=Path, required=True, help="confirmatory manifest (JSONL)")
    sub.add_argument("--screen", type=Path, required=True, help="the NeXT screen folder")
    sub.add_argument("--decisions", type=Path, required=True, help="verification decisions (JSONL)")
    sub.add_argument("--stage1", type=Path, default=Path("freeze/next_screening.json"))
    sub.add_argument("--record", type=Path, default=Path("freeze/next_gqa.json"))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m kvsel_next", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    sub = commands.add_parser("fetch-model", help="download LLaVA-NeXT at the pinned revision; resumes if rerun")
    sub.add_argument("--out", type=Path, default=NEXT.default_dir)
    sub.add_argument("--workers", type=int, default=8)
    sub.set_defaults(handler=cmd_fetch_model)

    sub = commands.add_parser("validate", help="noise floor and E1-E4, E7 on the real model; prints GO or NO-GO")
    sub.add_argument("--items", type=Path, required=True, help="development manifest (JSONL)")
    sub.add_argument("--images", type=Path, required=True, help="directory holding the image files")
    sub.add_argument("--model-dir", type=Path, default=None, help=f"local model copy (default: {NEXT.default_dir})")
    sub.add_argument("--device", default="cuda")
    sub.add_argument("--noise-items", type=int, default=50)
    sub.add_argument("--sparse-items", type=int, default=20, help="items for E3 and E4")
    sub.add_argument("--repeat-items", type=int, default=5, help="items for E7")
    sub.add_argument("--surface", default="capitalized", choices=["lower", "capitalized", "upper"])
    sub.add_argument("--out", type=Path, default=Path("results/next_validate.json"))
    sub.set_defaults(handler=cmd_validate)

    record = Path("freeze/next_screening.json")  # screen.RECORD; not imported, to keep `--help` fast
    sub = commands.add_parser("freeze-screening", help="write the NeXT stage-1 record (PREREG 4.4). No GPU")
    sub.add_argument("--prereg", type=Path, default=Path("PREREG.md"))
    sub.add_argument("--items", type=Path, required=True, help="confirmatory manifest (JSONL)")
    sub.add_argument("--record", type=Path, default=record)
    sub.set_defaults(handler=cmd_freeze_screening)

    sub = commands.add_parser("screen", help="NeXT eligibility screen on passed items (GPU); flags only, resumes if rerun")
    sub.add_argument("--prereg", type=Path, default=Path("PREREG.md"))
    sub.add_argument("--items", type=Path, required=True, help="confirmatory manifest (JSONL)")
    sub.add_argument("--decisions", type=Path, required=True, help="verification decisions (JSONL)")
    sub.add_argument("--images", type=Path, required=True)
    sub.add_argument("--out", type=Path, required=True, help="screen folder")
    sub.add_argument("--record", type=Path, default=record)
    sub.add_argument("--model-dir", type=Path, default=None)
    sub.add_argument("--device", default="cuda")
    sub.set_defaults(handler=cmd_screen)

    sub = commands.add_parser("count", help="main-set and control sizes from the screen's flags. No GPU")
    sub.add_argument("--items", type=Path, required=True, help="confirmatory manifest (JSONL)")
    sub.add_argument("--decisions", type=Path, required=True, help="verification decisions (JSONL)")
    sub.add_argument("--screen", type=Path, required=True, help="screen folder")
    sub.set_defaults(handler=cmd_count)

    sub = commands.add_parser("freeze", help="write the NeXT stage-3 record (PREREG 4.4). No GPU")
    _add_protocol(sub)
    sub.set_defaults(handler=cmd_freeze)

    sub = commands.add_parser("run", help="run the frozen NeXT comparison once (GPU); resumes if rerun")
    _add_protocol(sub)
    sub.add_argument("--images", type=Path, required=True)
    sub.add_argument("--out", type=Path, required=True)
    sub.add_argument("--model-dir", type=Path, default=None)
    sub.add_argument("--device", default="cuda")
    sub.set_defaults(handler=cmd_run)

    sub = commands.add_parser("report", help="the pre-registered NeXT analysis (PREREG 4.3); rechecks the freeze. No GPU")
    _add_protocol(sub)
    sub.add_argument("--run", type=Path, required=True)
    sub.add_argument("--resamples", type=int, default=10_000)  # kvsel.analysis.RESAMPLES; pandas would slow `--help`
    sub.add_argument("--llava15", type=Path, required=True,
                     help="the replication report's results.json, printed beside NeXT's estimates (PREREG 4.3)")
    sub.add_argument("--out", type=Path, required=True)
    sub.set_defaults(handler=cmd_report)

    args = parser.parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
