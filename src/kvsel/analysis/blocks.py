"""Report blocks every study composes into its own results layout and markdown."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..protocol import sha256_file
from .statistics import RESAMPLES, bootstrap, estimands, holm
from .tables import PRIMARY, has_cells, paired_behaviour, wide_margins

# --- Results -------------------------------------------------------------------------------------------------


def check_complete(run_dir: Path, found: set[str], expected: set[str], what: str, rerun: str) -> None:
    """Refuse a run missing frozen items or holding items outside the frozen set."""
    missing, extra = expected - found, found - expected
    if missing:
        raise SystemExit(f"{run_dir} is missing {len(missing)} {what}; rerun {rerun} to finish it")
    if extra:
        raise SystemExit(f"{run_dir} has {len(extra)} extra item files outside the frozen set; move them out")


def budget_results(rows: list[dict], conditions: tuple[str, ...], resamples: int = RESAMPLES) -> dict:
    """Every estimand at one budget, and paired behaviour differences (shares). Match cells exist on control items
    only, so match terms are control-only."""
    wide, images = wide_margins(rows, "primary")
    per = {c: estimands(wide, PRIMARY, c) for c in conditions if has_cells(wide, PRIMARY, c)}
    block = {f"{name}^({c})": bootstrap(series, images, resamples)
             for c, values in per.items() for name, series in values.items()}
    for other in ("irr", "match"):
        if other in per:
            block[f"T_{other}"] = bootstrap(per["mis"]["S_H"] - per[other]["S_H"], images, resamples)
    paired = {f"{result}: C_{c} − B_{c}": bootstrap(difference, images, resamples)
              for c in per for result, difference in paired_behaviour(rows, c).items()}
    return {"margins": block, "behaviour_paired": paired}


def holm_family(tests: dict[str, dict], alpha: float) -> dict:
    """One Holm family over bootstrap results, on their two-sided p-values."""
    return {"alpha": alpha, "tests": tests, "reject": holm({t: r["p_two_sided"] for t, r in tests.items()}, alpha)}


def source_file(path: Path) -> dict:
    """A results file another report is compared against: its name, its sha256 and its contents."""
    return {"path": str(path), "sha256": sha256_file(Path(path)), "data": json.loads(Path(path).read_text())}


# --- Markdown ------------------------------------------------------------------------------------------------


def row(result: dict, nats: bool = True, digits: int = 3) -> str:
    """One table row. The δ-based reading applies to margins in nats only."""
    if result.get("estimate") is None:
        return "— | — | — | — | — |"
    low, high = result["ci95"]
    reading = result.get("reading", "") if nats else ""
    return (f"{result['estimate']:+.{digits}f} | [{low:+.{digits}f}, {high:+.{digits}f}] | {result['p_two_sided']:.4f} | "
            f"{reading} | {result['items']} / {result['images']} |")


def table(title: str, block: dict, unit: str, nats: bool) -> list[str]:
    reading = "Reading" if nats else "—"
    return [f"### {title}", "", f"| Estimand | Mean ({unit}) | 95% interval | Two-sided p | {reading} | Items / images |",
            "| --- | ---: | --- | ---: | --- | ---: |"] + [f"| {name} | {row(result, nats)}" for name, result in block.items()] + [""]


def interval(result: dict) -> str:
    low, high = result["ci95"]
    return f"{result['estimate']:+.3f} [{low:+.3f}, {high:+.3f}]"


def rates_table(rates: dict[str, dict]) -> list[str]:
    """Behaviour rates as a table, without a title."""
    return (["| Run | Correct | Alternative | Other |", "| --- | ---: | ---: | ---: |"]
            + [f"| {name} | {r['correct']:.3f} | {r['alt']:.3f} | {r['other']:.3f} |" for name, r in rates.items()])


# --- Writing -------------------------------------------------------------------------------------------------


def plain(value):
    """JSON fallback for NumPy scalars, tuples and sets."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple | set):
        return list(value)
    raise TypeError(f"not serialisable: {type(value)}")


def write_report(out_dir: Path, results: dict, markdown: str, trailing_newline: bool = False) -> None:
    """`results.json` (indent 2) and `report.md`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=plain) + ("\n" if trailing_newline else ""))
    (out_dir / "report.md").write_text(markdown)
