"""The command line: `kvsel <group> <command>`. `kvsel <group> <command> --help` lists each command's options.

Groups: data, model, verify, validate, reproduce, and one per study (pilot, confirmatory, followups, replication, next), whose
commands are its stages. Exit codes: 0 done; 1 an error, or a validation NO-GO; 2 a refusal, a stop for action.
A refusal is turned into its exit here and nowhere else.
"""

from __future__ import annotations

import argparse
import getpass
import json
import sys
import time
from collections import Counter
from datetime import date
from functools import partial
from pathlib import Path

from . import reproduce
from .analysis.statistics import RESAMPLES, SURFACE
from .items import gqa, images, textvqa
from .items.manifest import load_items
from .model import Bundle, load_bundle
from .models import FAMILIES
from .models.family import Family
from .models.llava15 import LLAVA
from .models.llava_next import NEXT
from .prompts import SURFACE_FORMS
from .protocol import FrozenInputs, Refusal, fill_evidence, sha256_file, write_pending
from .runner import eligible, eligible_ids, require_images, screen_flags
from .runs import MainSet
from .studies import pilot
from .studies.confirmatory import comparative, report, screening
from .studies.followups import report as followup_report
from .studies.followups import run as followups
from .studies.next import comparative as next_comparative
from .studies.next import report as next_report
from .studies.next import screening as next_screening
from .studies.replication import report as replication_report
from .studies.replication import run as replication
from .studies.replication import select
from .validation import summary, validate, write
from .verification import build_queue, load_decisions, make_server, summarize

PREREG = Path("PREREG.md")
PLANS = {LLAVA.name: pilot.VALIDATION, NEXT.name: next_comparative.VALIDATION}


def bundle(args, family: Family) -> Bundle:
    """The model behind every GPU command; the seam tests replace with the tiny models."""
    return load_bundle(args.device, getattr(args, "head_precision", None), args.model_dir, family)


def _json(value) -> None:
    print(json.dumps(value, indent=2))


def _images_first(items: list[dict], main_set: MainSet, args) -> None:
    """The run's images, checked before the model loads (the run checks them again)."""
    by_id = {item["item_id"]: item for item in items}
    require_images([by_id[i] for i in main_set.ids], args.images)


def _frozen_inputs(args) -> FrozenInputs:
    return FrozenInputs(args.prereg, args.items, args.screen, args.decisions, args.stage1, args.record)


def stage_printer():
    """Prints each validation stage with the minutes since the call."""
    start = time.time()
    return lambda line: print(f"[{(time.time() - start) / 60:5.1f} min] {line}", flush=True)


# --- data, model, verify, validate --------------------------------------------------------------------------


def cmd_build_gqa(args) -> None:
    _json(gqa.build(args.questions, args.scene_graphs, args.out_dir, LLAVA.load_tokenizer(args.model_dir),
                    args.dev_items, args.dev_manifest))


def cmd_build_textvqa(args) -> None:
    _json(textvqa.build(args.questions, args.ocr, args.out_dir, LLAVA.load_tokenizer(args.model_dir), args.dev_items,
                        args.dev_manifest))


def cmd_fetch_images(args) -> None:
    result = images.fetch(args.manifests, args.out, args.url, args.workers, args.limit)
    _json(result)
    if result["failed"]:
        raise SystemExit(f"{len(result['failed'])} files failed; rerun the command to retry them")


def cmd_fetch_model(args) -> None:
    family = FAMILIES[args.model]
    out = args.out or family.default_dir
    missing = family.fetch(out, args.workers)
    if missing:
        raise SystemExit(f"still missing {', '.join(missing)}; rerun the command to resume")
    print(f"model complete at {out}" + (f", revision {family.revision}" if family.revision else ""))


def cmd_verify(args) -> None:
    items = load_items(args.items)
    ids = set(eligible_ids(args.screen, items, args.surface)) if args.screen else None
    queue = build_queue(items, ids, args.limit)
    server = make_server(queue, args.images, args.decisions, args.verifier, args.port)
    print(f"{len(queue)} items queued; decisions go to {args.decisions}")
    print(f"open http://127.0.0.1:{server.server_address[1]}  (Ctrl-C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    _json(summarize(load_decisions(args.decisions, queue)[0], queue))


def cmd_validate(args) -> None:
    items = load_items(args.items)
    pilot.require_development(items)  # before loading the model
    out = args.out or Path(f"results/validate_{args.model}.json")
    found = validate(bundle(args, FAMILIES[args.model]), items, args.images, PLANS[args.model], args.noise_items,
                     args.sparse_items, args.repeat_items, args.surface, progress=stage_printer())
    write(found, out)
    print("\n".join(summary(found, out)))
    if not found.go:
        raise SystemExit(1)


# --- The development pilot ----------------------------------------------------------------------------------


def cmd_pilot_screen(args) -> None:
    pilot.run_screen(bundle(args, LLAVA), load_items(args.items), args.images, args.out)
    results = [json.loads(path.read_text()) for path in (args.out / "items").glob("*.json")]
    for surface in SURFACE_FORMS:
        print(f"eligible under `{surface}`: {sum(eligible(r['rows'][0], surface) for r in results)} of {len(results)}")
    print("sample greedy answers:", [r["rows"][0]["greedy_text"] for r in results][:20])


def cmd_pilot_run(args) -> None:
    pool = load_items(args.items)
    ids = set(eligible_ids(args.screen, pool, args.surface))
    chosen = [item for item in pool if item["item_id"] in ids][: args.limit]
    print(f"{len(chosen)} eligible items selected under `{args.surface}`")
    if not chosen:
        raise Refusal("no eligible items; run `kvsel pilot screen` first")
    pilot.run_items(bundle(args, LLAVA), chosen, args.images, args.out, pilot.pilot_units(k=args.k))


def cmd_report(args, records: bool) -> None:
    """The confirmatory report; without records, on pilot runs, whose item files carry no subset (a rehearsal)."""
    runs, frozen = {}, {}
    for dataset in report.DATASETS:
        run, items = getattr(args, f"{dataset}_run"), getattr(args, f"{dataset}_items")
        if run is None:
            continue
        if items is None:
            raise Refusal(f"--{dataset}-items is needed with --{dataset}-run")
        runs[dataset] = (run, items)
        if records and getattr(args, f"{dataset}_record") is not None:
            frozen[dataset] = getattr(args, f"{dataset}_record")
    if not runs:
        raise Refusal("give at least one run, for example --gqa-run results/confirm/gqa_run")
    report.confirm_report(runs, args.out, args.resamples, frozen)
    print((args.out / "report.md").read_text())


# --- The confirmatory study ---------------------------------------------------------------------------------


def cmd_confirmatory_freeze_screening(args) -> None:
    _json(screening.freeze_screening(args.prereg, args.manifests, args.record, args.amends, args.reason))


def cmd_confirmatory_screen(args) -> None:
    frozen = screening.check_screening(args.prereg, args.items, args.record)  # before loading the model
    items = load_items(args.items)[: args.limit]
    require_images(items, args.images, "; fetch them or screen fewer items with --limit")  # before loading the model
    funnel = screening.run_confirm_screen(bundle(args, LLAVA), items, args.images, args.out, frozen,
                                          sha256_file(args.record))
    _json({key: funnel[key] for key in ("screened", "eligible", "not_eligible")})


def cmd_confirmatory_freeze(args) -> None:
    payload = comparative.freeze_comparative(_frozen_inputs(args))
    _json({key: value for key, value in payload.items() if key != "settings"})


def cmd_confirmatory_run(args) -> None:
    given = _frozen_inputs(args)
    comparative.check_comparative(given)  # before loading the model
    items, main_set = load_items(args.items), comparative.frozen_main_set(given)
    _images_first(items, main_set, args)
    _json(comparative.run_confirm(bundle(args, LLAVA), items, args.images, args.out, main_set, sha256_file(args.record)))


# --- The follow-ups -----------------------------------------------------------------------------------------


def cmd_followups_run(args) -> None:
    items = load_items(args.items)
    main_set, code_unchanged = followups.confirm_selection(args.confirm_run, args.record, items)  # before the model
    if not code_unchanged:
        print("note: the code differs from the stage-3 record; the confirmatory run used other code", flush=True)
    _images_first(items, main_set, args)
    found = followups.run_followup(bundle(args, LLAVA), items, args.items, args.images, args.out, main_set,
                                   sha256_file(args.record), tuple(args.parts))
    _json({"confirm_code_unchanged": code_unchanged, **found})


def cmd_followups_report(args) -> None:
    _json(followup_report.followup_report(args.run, args.confirm_run, args.items, args.out, args.resamples))


# --- The replication ----------------------------------------------------------------------------------------


def _replication(args) -> tuple:
    return _frozen_inputs(args), args.gqa_record, args.exclude_runs


def cmd_replication_count(args) -> None:
    items = load_items(args.items)
    decisions, stale = load_decisions(args.decisions, items)
    main = comparative.main_set(items, args.screen, args.decisions)["main"]
    found = select.replication_set(items, screen_flags(args.screen), decisions, main,
                                   select.earlier_run_ids(args.exclude_runs))
    _json({"main_set": len(main), "replication": len(found["replication"]), "control": len(found["control"]),
           "failed_before_cutoff": found["failed"], "excluded": dict(Counter(found["excluded"].values())),
           "stale_decisions": stale, "next_unverified": found["next_unverified"]})


def cmd_replication_freeze(args) -> None:
    frozen = replication.freeze_replication(*_replication(args))
    _json({key: frozen[key] for key in ("frozen_at", "replication_set_sha256", "code_sha256")}
          | {"replication": len(frozen["replication_ids"]), "control": len(frozen["control_ids"]),
             "excluded": dict(Counter(frozen["excluded"].values()))})


def cmd_replication_run(args) -> None:
    main_set = replication.frozen_main_set(replication.check_replication(*_replication(args)))  # before the model
    items = load_items(args.items)
    _images_first(items, main_set, args)
    _json(replication.run_replication(bundle(args, LLAVA), items, args.images, args.out, main_set,
                                      sha256_file(args.record)))


def cmd_replication_report(args) -> None:
    replication.check_replication(*_replication(args), report=True)
    found = replication_report.replication_report(args.run, args.record, args.out, args.resamples,
                                                  args.original_confirm, args.original_followup)
    print(f"wrote {args.out / 'report.md'} ({found['items']} items)")


# --- LLaVA-NeXT ---------------------------------------------------------------------------------------------


def cmd_next_freeze_screening(args) -> None:
    frozen = next_screening.freeze_screening(args.prereg, args.items, args.record)
    _json({key: frozen[key] for key in ("frozen_at", "prereg_section_sha256", "code_sha256")})


def cmd_next_screen(args) -> None:
    frozen = next_screening.check_screening(args.prereg, args.items, args.record)  # before loading the model
    items = next_screening.pool(load_items(args.items), args.decisions)
    require_images(items, args.images)  # before loading the model
    _json(next_screening.run_screen(bundle(args, NEXT), items, args.images, args.out, frozen["settings"],
                                    sha256_file(args.record)))


def cmd_next_count(args) -> None:
    _json(next_screening.count(load_items(args.items), args.decisions, args.screen))


def cmd_next_freeze(args) -> None:
    frozen = next_comparative.freeze_comparative(_frozen_inputs(args))
    _json({key: frozen[key] for key in ("frozen_at", "main_set_sha256", "code_sha256")}
          | {"main": len(frozen["main_ids"]), "control": len(frozen["control_ids"])})


def cmd_next_run(args) -> None:
    main_set = next_comparative.frozen_main_set(next_comparative.check_comparative(_frozen_inputs(args)))  # before the model
    items = load_items(args.items)
    _images_first(items, main_set, args)
    _json(next_comparative.run_comparative(bundle(args, NEXT), items, args.images, args.out, main_set,
                                           sha256_file(args.record)))


def cmd_next_report(args) -> None:
    next_comparative.check_comparative(_frozen_inputs(args), report=True)
    found = next_report.report(args.run, args.record, args.out, args.llava15, args.resamples)
    print(f"wrote {args.out / 'report.md'} ({found['items']} items)")


# --- Proving a code amendment ------------------------------------------------------------------------------


def cmd_reproduce(args) -> None:
    """PREREG 1.9: the stored reports, and with --images a few items of each stored run, rerun and compared."""
    if args.out.exists() and any(args.out.iterdir()):  # a rerun must not resume, and so reuse, older results
        raise Refusal(f"{args.out} is not empty; give a new --out")
    reports, runs = reproduce.load_list(args.list)
    if args.reason:
        pending = write_pending(args.freeze, args.reason, date.today().isoformat())
        print(f"{len(pending)} pending entries in {args.freeze / 'code_amendments.json'}", flush=True)
    found = reproduce.reproduce_reports(reports, args.out, run)
    rows = 0
    if args.images:
        differences, rows = reproduce.reproduce_rows(bundle(args, LLAVA), runs, args.images, args.out, args.rows)
        found += differences
    if found:
        raise Refusal(f"NOT reproduced ({len(found)} differences; pending entries stay pending). First: {found[0]}")
    if not args.images:
        print("reports reproduced; evidence is recorded only with the row check too (--images, on the GPU)")
        return
    evidence = (f"`kvsel reproduce` on {date.today().isoformat()}: {', '.join(r.name for r in reports)} results.json "
                f"reproduced ({', '.join(f'{r.name} by {r.compare}' for r in reports)}); {rows} stored run items rerun "
                "and identical row for row")
    filled = fill_evidence(args.freeze, evidence)
    print(f"everything reproduced; evidence recorded in {filled} pending entries" if filled
          else "everything reproduced; no pending amendment entries")


# --- Arguments: one name and one default per meaning --------------------------------------------------------


def add_model(sub, family: Family | None, head: bool = False) -> None:
    """The model copy and device; ``family`` None: the command takes `--model`. ``head``: also the output head's
    precision, offered only off the frozen path."""
    where = family.default_dir if family else "the family's folder under models/"
    sub.add_argument("--model-dir", type=Path, default=None,
                     help=f"local model copy, used if present; otherwise the Hugging Face Hub (default: {where})")
    sub.add_argument("--device", default="cuda")
    if head:
        default = family.default_head_precision if family else "the family's"
        sub.add_argument("--head-precision", default=None, choices=["fp32", "native"],
                         help=f"output head precision (default: {default})")


def add_items(sub, what: str, images_needed: bool = True) -> None:
    sub.add_argument("--items", type=Path, required=True, help=f"{what} manifest (JSONL)")
    if images_needed:
        sub.add_argument("--images", type=Path, required=True, help="directory holding the image files")


def add_surface(sub) -> None:
    sub.add_argument("--surface", default=SURFACE, choices=SURFACE_FORMS, help="surface rule (default: %(default)s)")


def add_resamples(sub) -> None:
    sub.add_argument("--resamples", type=int, default=RESAMPLES)


def add_family(sub) -> None:
    sub.add_argument("--model", default=LLAVA.name, choices=sorted(FAMILIES), help="model family (default: %(default)s)")


def add_frozen_inputs(sub, stage1: Path, record: Path | None) -> None:
    """The confirmatory manifest and what a stage-3 freeze holds: PREREG, the screen, the decisions, the records."""
    add_items(sub, "confirmatory", images_needed=False)
    sub.add_argument("--screen", type=Path, required=True, help="screen folder")
    sub.add_argument("--decisions", type=Path, required=True, help="verification decisions (JSONL)")
    sub.add_argument("--prereg", type=Path, default=PREREG)
    sub.add_argument("--stage1", type=Path, default=stage1)
    if record is None:
        sub.add_argument("--record", type=Path, required=True, help="for example freeze/comparative_gqa.json")
    else:
        sub.add_argument("--record", type=Path, default=record)


def add_run(sub, family: Family) -> None:
    sub.add_argument("--images", type=Path, required=True, help="directory holding the image files")
    sub.add_argument("--out", type=Path, required=True)
    add_model(sub, family)


def add_build(sub, dev_items: int) -> None:
    sub.add_argument("--out-dir", type=Path, default=Path("manifests"))
    sub.add_argument("--dev-items", type=int, default=dev_items, help="candidates before eligibility and verification")
    sub.add_argument("--dev-manifest", type=Path, default=None,
                     help="keep these development items unchanged; confirmatory items avoid their images")
    sub.add_argument("--model-dir", type=Path, default=None, help=f"LLaVA-1.5's tokenizer (default: {LLAVA.default_dir})")


def add_report_runs(sub, records: bool) -> None:
    for dataset in report.DATASETS:
        sub.add_argument(f"--{dataset}-run", type=Path, default=None, help=f"{dataset} run folder")
        sub.add_argument(f"--{dataset}-items", type=Path, default=None, help=f"{dataset} manifest, for evidence boxes")
        if records:
            sub.add_argument(f"--{dataset}-record", type=Path, default=None, help="stage-3 record; checks the run is complete")
    add_resamples(sub)
    sub.add_argument("--out", type=Path, required=True)


def add_replication(sub) -> None:
    add_frozen_inputs(sub, screening.RECORD, replication.RECORD)
    sub.add_argument("--gqa-record", type=Path, default=Path("freeze/comparative_gqa.json"))
    sub.add_argument("--exclude-runs", type=Path, nargs="*", default=[],
                     help="earlier run folders; any item with a file in them is left out")


def command(commands, name: str, help: str, handler, *adders):
    """A subcommand: each adder, called with the parser, adds its options."""
    sub = commands.add_parser(name, help=help)
    for add in adders:
        add(sub)
    sub.set_defaults(handler=handler)
    return sub


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kvsel", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    groups = parser.add_subparsers(dest="group", required=True)

    def group(name: str, help: str):
        return groups.add_parser(name, help=help).add_subparsers(dest="command", required=True)

    data = group("data", "build item manifests and fetch their images")
    sub = command(data, "build-gqa", "build GQA item manifests and print the funnel", cmd_build_gqa,
                  partial(add_build, dev_items=600))
    sub.add_argument("--questions", type=Path, required=True)
    sub.add_argument("--scene-graphs", type=Path, required=True)
    sub = command(data, "build-textvqa", "build TextVQA item manifests and print the funnel", cmd_build_textvqa,
                  partial(add_build, dev_items=1500))
    sub.add_argument("--questions", type=Path, nargs="+", required=True)
    sub.add_argument("--ocr", type=Path, nargs="+", required=True)
    sub = command(data, "fetch-images", "download only the images a manifest needs", cmd_fetch_images)
    sub.add_argument("manifests", type=Path, nargs="+")
    sub.add_argument("--out", type=Path, default=Path("data/gqa/images"))
    sub.add_argument("--url", default="https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip")
    sub.add_argument("--workers", type=int, default=3, help="the server returns 503 errors at 8")
    sub.add_argument("--limit", type=int, help="fetch at most this many files, to try it out")

    sub = command(group("model", "the model files"), "fetch", "download a model family for local use; resumes if rerun",
                  cmd_fetch_model, add_family)
    sub.add_argument("--out", type=Path, default=None, help="default: the family's models/ directory")
    sub.add_argument("--workers", type=int, default=8)

    sub = command(groups, "verify", "hand-verify items in a local web page", cmd_verify,
                  partial(add_items, what="item"), add_surface)
    sub.add_argument("--decisions", type=Path, required=True, help="append-only JSONL, for example verification/gqa_dev.jsonl")
    sub.add_argument("--screen", type=Path, help="queue only items eligible in this screen folder (development or confirmatory)")
    sub.add_argument("--limit", type=int)
    sub.add_argument("--verifier", default=getpass.getuser())
    sub.add_argument("--port", type=int, default=8765)

    sub = command(groups, "validate", "noise floor and the Level 2 checks on the real model; prints GO or NO-GO",
                  cmd_validate, add_family, partial(add_items, what="development"), partial(add_model, family=None, head=True),
                  add_surface)
    sub.add_argument("--noise-items", type=int, default=50)
    sub.add_argument("--sparse-items", type=int, default=20, help="items for E3-E5")
    sub.add_argument("--repeat-items", type=int, default=5, help="items for E7")
    sub.add_argument("--out", type=Path, default=None, help="default: results/validate_<model>.json")

    study = group("pilot", "the development pilot: screen, run and rehearsal report")
    development = (partial(add_items, what="development"), partial(add_model, family=LLAVA, head=True))
    sub = command(study, "screen", "Full_Q on development items, for eligibility", cmd_pilot_screen, *development)
    sub.add_argument("--out", type=Path, default=Path("results/screen_dev"))
    sub = command(study, "run", "the full grid of cells on eligible development items", cmd_pilot_run, *development,
                  add_surface)
    sub.add_argument("--screen", type=Path, default=Path("results/screen_dev"))
    sub.add_argument("--limit", type=int, default=200)
    sub.add_argument("--k", type=int, default=144, choices=[288, 144, 72])
    sub.add_argument("--out", type=Path, default=Path("results/pilot"))
    command(study, "report", "the confirmatory analysis on pilot runs, as a rehearsal. No GPU", partial(cmd_report, records=False),
            partial(add_report_runs, records=False))

    study = group("confirmatory", "the confirmatory study (PREREG 1-2)")
    sub = command(study, "freeze-screening", "freeze stage 1: record hashes of the screening rules. No GPU",
                  cmd_confirmatory_freeze_screening)
    sub.add_argument("--prereg", type=Path, default=PREREG)
    sub.add_argument("--manifests", type=Path, nargs="+",
                     default=[Path("manifests/gqa_confirm.jsonl"), Path("manifests/textvqa_confirm.jsonl")])
    sub.add_argument("--record", type=Path, default=screening.RECORD)
    sub.add_argument("--amends", type=Path, default=None, help="the stage-1 record this amendment supersedes")
    sub.add_argument("--reason", default=None, help="why the screening rules changed")
    sub = command(study, "screen", "eligibility screen on confirmatory items; writes flags only (GPU)",
                  cmd_confirmatory_screen, partial(add_items, what="confirmatory"), partial(add_model, family=LLAVA))
    sub.add_argument("--prereg", type=Path, default=PREREG)
    sub.add_argument("--record", type=Path, default=screening.RECORD)
    sub.add_argument("--limit", type=int, default=None, help="screen only the first N items in manifest order")
    sub.add_argument("--out", type=Path, required=True)
    stage3 = partial(add_frozen_inputs, stage1=screening.RECORD, record=None)
    command(study, "freeze", "freeze stage 3 for one dataset: record hashes and the main set. No GPU",
            cmd_confirmatory_freeze, stage3)
    command(study, "run", "the confirmatory run, once, on the frozen main set (GPU)", cmd_confirmatory_run, stage3,
            partial(add_run, family=LLAVA))
    command(study, "report", "the pre-registered analysis of confirmatory runs, Holm across datasets. No GPU",
            partial(cmd_report, records=True), partial(add_report_runs, records=True))

    study = group("followups", "the follow-ups (PREREG 2.4); not confirmatory")
    sub = command(study, "run", "follow-up units on the confirmatory main set (GPU)", cmd_followups_run,
                  partial(add_items, what="confirmatory", images_needed=False), partial(add_run, family=LLAVA))
    sub.add_argument("--confirm-run", type=Path, required=True, help="the finished confirmatory run folder")
    sub.add_argument("--record", type=Path, required=True, help="its stage-3 record, for example freeze/comparative_gqa.json")
    sub.add_argument("--parts", nargs="+", default=list(followups.PARTS), choices=followups.PARTS)
    sub = command(study, "report", "descriptive analysis of the follow-up runs. No GPU", cmd_followups_report,
                  partial(add_items, what="confirmatory", images_needed=False), add_resamples)
    sub.add_argument("--run", type=Path, required=True, help="the follow-up folder given to run --out")
    sub.add_argument("--confirm-run", type=Path, required=True)
    sub.add_argument("--out", type=Path, required=True)

    study = group("replication", "the GQA replication (PREREG 3)")
    command(study, "count", "how many fresh items qualify; flags and decisions only. No GPU", cmd_replication_count,
            add_replication)
    command(study, "freeze", "freeze the replication protocol and set. No GPU", cmd_replication_freeze, add_replication)
    command(study, "run", "run the frozen replication once (GPU); resumes if rerun", cmd_replication_run,
            add_replication, partial(add_run, family=LLAVA))
    sub = command(study, "report", "the pre-registered replication analysis; rechecks the freeze. No GPU",
                  cmd_replication_report, add_replication, add_resamples)
    sub.add_argument("--run", type=Path, required=True)
    sub.add_argument("--original-confirm", type=Path, default=None, help="the GQA confirmatory report's results.json")
    sub.add_argument("--original-followup", type=Path, default=None, help="the follow-up report's results.json")
    sub.add_argument("--out", type=Path, required=True)

    study = group("next", "LLaVA-NeXT on GQA (PREREG 4)")
    sub = command(study, "freeze-screening", "write the NeXT stage-1 record (PREREG 4.4). No GPU",
                  cmd_next_freeze_screening, partial(add_items, what="confirmatory", images_needed=False))
    sub.add_argument("--prereg", type=Path, default=PREREG)
    sub.add_argument("--record", type=Path, default=next_screening.RECORD)
    sub = command(study, "screen", "NeXT eligibility screen on passed items (GPU); flags only, resumes if rerun",
                  cmd_next_screen, partial(add_items, what="confirmatory"), partial(add_model, family=NEXT))
    sub.add_argument("--prereg", type=Path, default=PREREG)
    sub.add_argument("--decisions", type=Path, required=True, help="verification decisions (JSONL)")
    sub.add_argument("--out", type=Path, required=True, help="screen folder")
    sub.add_argument("--record", type=Path, default=next_screening.RECORD)
    sub = command(study, "count", "main-set and control sizes from the screen's flags. No GPU", cmd_next_count,
                  partial(add_items, what="confirmatory", images_needed=False))
    sub.add_argument("--decisions", type=Path, required=True, help="verification decisions (JSONL)")
    sub.add_argument("--screen", type=Path, required=True, help="screen folder")
    stage3 = partial(add_frozen_inputs, stage1=next_screening.RECORD, record=next_comparative.RECORD)
    command(study, "freeze", "write the NeXT stage-3 record (PREREG 4.4). No GPU", cmd_next_freeze, stage3)
    command(study, "run", "run the frozen NeXT comparison once (GPU); resumes if rerun", cmd_next_run, stage3,
            partial(add_run, family=NEXT))
    sub = command(study, "report", "the pre-registered NeXT analysis (PREREG 4.3); rechecks the freeze. No GPU",
                  cmd_next_report, stage3, add_resamples)
    sub.add_argument("--run", type=Path, required=True)
    sub.add_argument("--llava15", type=Path, required=True,
                     help="the replication report's results.json, printed beside NeXT's estimates (PREREG 4.3)")
    sub.add_argument("--out", type=Path, required=True)
    sub = command(groups, "reproduce", "prove a code amendment: rerun the stored reports (and run items) and compare",
                  cmd_reproduce)
    sub.add_argument("--list", type=Path, default=reproduce.LIST, help="the committed list of stored reports and runs")
    sub.add_argument("--freeze", type=Path, default=Path("freeze"), help="the folder of records and the amendment log")
    sub.add_argument("--reason", default=None,
                     help="write a pending amendment entry, with this reason, for every record the current code no longer matches")
    sub.add_argument("--out", type=Path, default=Path("results/reproduce"), help="a new or empty folder")
    sub.add_argument("--images", type=Path, default=None, help="also rerun stored run items (GPU); needed for the evidence")
    sub.add_argument("--rows", type=int, default=3, help="main and control items rerun per run")
    add_model(sub, LLAVA)
    return parser


class Refused(SystemExit):
    """The exit of a refusal: code 2, with the reason, which is printed to stderr."""

    def __init__(self, reason: str):
        super().__init__(2)
        self.reason = reason

    def __str__(self) -> str:
        return self.reason


def run(argv: list[str] | None = None) -> None:
    """Parses and runs one command; a refusal propagates."""
    args = build_parser().parse_args(argv)
    args.handler(args)


def main(argv: list[str] | None = None) -> None:
    try:
        run(argv)
    except Refusal as refusal:
        print(f"refused: {refusal}", file=sys.stderr)
        raise Refused(str(refusal)) from None


if __name__ == "__main__":
    main()
