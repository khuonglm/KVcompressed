"""Level 2 checks on the real model, for any model family, ending in a mechanical GO or NO-GO.

The criteria are tests/README.md's ("Level 2: LLaVA pipeline" and "LLaVA-NeXT, Level 2"), written before each family's
GPU run and not changed after a failure. Each E-check is one function returning a typed result. What differs between
families (E5 or not, E3's budgets and kept-set kinds, E7's units, the projection) is declared in a `Plan`.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .cache import entries_hash
from .engine import Decoder, greedy, greedy_releasing, probe, release_cached_memory, replay, score_sequence
from .model import Bundle, full_forward_logits
from .models.family import Tokens, visual_span
from .models.layout import Layout
from .prompts import surface_forms, window_composition
from .runner import NO_CONTEXT, Q, Unit, open_image, run_item, run_unit
from .selectors import Budget, kept_set

SURFACE = "capitalized"  # the default; `validate --surface` sets it per run
SAMPLE_PER_COMPARISON = 20_000  # logit differences sampled per comparison for the 99th percentiles
MARGIN_LIMIT = 0.01  # E3: mean |margin difference| in nats
GREEDY_STEPS = 10  # E3: tokens generated per kept set
MEMORY_LIMIT_GB = 20  # the A100 slice
KINDS = ("random", "concentrated", "scored")  # E3's kept-set kinds

# --- What a family runs ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Projection:
    """A study's sizes, for upper bounds on its screen and run time: every unit is costed as a Full unit."""

    pool: int  # items screened
    main: int
    control: int
    main_units: tuple[Unit, ...]
    control_units: tuple[Unit, ...]


@dataclass(frozen=True)
class Plan:
    """A family's validation: E3's budgets and kept-set kinds, E5's budget (None: no E5), E7's units."""

    budgets: tuple[Budget, ...]
    kinds: tuple[str, ...]
    units: tuple[Unit, ...]
    placebo: Budget | None = None
    projection: Projection | None = None

    @property
    def checks(self) -> tuple[str, ...]:
        """The checks whose passes make GO."""
        return ("E1", "E2", "E3", "E4", *(("E5",) if self.placebo else ()), "E7")


# --- Results --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Serialization:
    """E1: shared prefix and question tokens, candidate tokens in context, and a visual span the layout explains."""

    passed: bool
    items: int
    failures: list[str]
    window_composition: dict[str, int]
    visual_tokens_min: int | None
    visual_tokens_max: int | None
    grids: dict[str, int]  # the last map's rows x columns


@dataclass(frozen=True)
class Floor:
    """One full forward pass vs prefix-then-suffix with the full cache: the noise every other limit comes from."""

    items: int
    logit_p99: float
    logit_max: float
    margin_mean_abs: float
    margin_max_abs: float


@dataclass(frozen=True)
class SuffixFidelity:
    passed: bool
    max_logit_diff: float
    limit: float
    compact_vs_dense_max: float


@dataclass(frozen=True)
class Fidelity:
    """E2: compact with every visual entry kept, on each suffix, vs one full pass and that suffix's own floor."""

    passed: bool
    by_suffix: dict[str, SuffixFidelity]


@dataclass(frozen=True)
class Mismatch:
    item: str
    context: str
    budget: str
    kind: str
    step: int
    gap: float
    near_tie: bool


@dataclass(frozen=True)
class SparseEquivalence:
    """E3: compact vs dense on sparse kept sets."""

    passed: bool
    logit_p99: float
    p99_limit: float
    margin_mean_abs: float
    margin_limit: float
    greedy_mismatches: list[Mismatch]
    greedy_mismatches_not_near_tie: int
    comparisons: int


@dataclass(frozen=True)
class Hashes:
    """E4: the shared prefix is unchanged by every probe and replay."""

    passed: bool
    failures: list[str]


@dataclass(frozen=True)
class Placebo:
    """E5: the same kept indices and suffix through the B and C code path."""

    passed: bool
    max_logit_diff: float
    different_answers: int


@dataclass(frozen=True)
class Determinism:
    """E7: rerunning an item reproduces its rows and probe attention."""

    passed: bool
    different_items: list[str]


@dataclass(frozen=True)
class ItemCost:
    """One item's memory (None off CUDA) and time: prefill, one probe, and a Full unit on the longest suffix."""

    item_id: str
    visual_tokens: int
    grid: list[int]
    prefill_peak_gb: float | None
    run_peak_gb: float | None
    prefill_seconds: float
    probe_seconds: float
    unit_seconds: float


@dataclass(frozen=True)
class Report:
    model: str
    revision: str | None
    head_precision: str
    surface: str
    device: str
    noise_floor: Floor  # Q suffix, the bundle's head
    noise_floor_mis: Floor
    noise_floor_native_head: Floor
    checks: dict[str, Serialization | Fidelity | SparseEquivalence | Hashes | Placebo | Determinism]
    costs: list[ItemCost]
    hours: dict[str, float] | None  # projected upper bounds, when the plan has a projection
    go: bool

    def to_json(self) -> dict:
        return dataclasses.asdict(self)


# --- Shared steps ---------------------------------------------------------------------------------------------


def sample_differences(difference: torch.Tensor, generator: np.random.Generator) -> np.ndarray:
    values = difference.abs().flatten().cpu().numpy()
    if values.size <= SAMPLE_PER_COMPARISON:
        return values
    return values[generator.integers(0, values.size, SAMPLE_PER_COMPARISON)]


def margin_incremental(bundle: Bundle, state, prompt: str, a_true: str, a_alt: str, surface: str) -> float:
    total = {}
    for name, answer in (("true", a_true), ("alt", a_alt)):
        ids = bundle.family.candidate_ids(bundle.tokenizer, prompt, surface_forms(answer)[surface])
        total[name] = sum(score_sequence(bundle.decoder, state, ids))
    return total["true"] - total["alt"]


def margin_full_forward(bundle: Bundle, tokens: Tokens, vision: dict, a_true: str, a_alt: str, surface: str) -> float:
    total = {}
    base = tokens.prefix_ids + tokens.suffix_ids
    for name, answer in (("true", a_true), ("alt", a_alt)):
        ids = bundle.family.candidate_ids(bundle.tokenizer, tokens.prompt, surface_forms(answer)[surface])
        release_cached_memory(bundle.device)  # before each uncached full pass
        logits = full_forward_logits(bundle, base + ids[:-1], vision, len(base) - 1)
        log_probs = torch.log_softmax(logits[0], dim=-1)
        total[name] = sum(float(log_probs[i, token]) for i, token in enumerate(ids))
    return total["true"] - total["alt"]


def concentrated_set(layout: Layout, k: int) -> np.ndarray:
    """A validation kept-set kind, not a selector: the first k feature tokens, plus every newline."""
    return np.union1d(layout.features[:k], np.array(layout.newlines, dtype=np.int64))


def validation_set(layout: Layout, budget: Budget, kind: str, seed: int, attention: np.ndarray | None = None) -> np.ndarray:
    """One of E3's kept sets: random, concentrated or probe-scored, with k = round(share x feature tokens) for a
    share (validation sets need not be a whole share), plus every newline."""
    k = budget.k if budget.k is not None else round(budget.share * len(layout.features))
    if kind == "concentrated":
        return concentrated_set(layout, k)
    if kind == "random":
        return kept_set(layout, "random", Budget(k=k), rng=np.random.default_rng(seed))
    if kind == "scored":
        return kept_set(layout, "pw", Budget(k=k), attention=attention)
    raise ValueError(f"unknown kept-set kind {kind!r}")


def _label(budget: Budget) -> str:
    return f"k={budget.k}" if budget.k is not None else f"share={budget.share}"


def _prefilled(bundle: Bundle, item: dict, image):
    """(tokens by context, model inputs, prefix, layout): one prefill, shared by every context."""
    tokens, inputs = bundle.family.contexts(bundle, item, image)
    prefix, layout = bundle.family.prefill(bundle, tokens[Q].prefix_ids, inputs)
    return tokens, inputs, prefix, layout


def _replayed(bundle: Bundle, prefix, kept: torch.Tensor, suffix: torch.Tensor, mode: str, prompt: str, item: dict,
              surface: str):
    """(suffix logits, margin, greedy) for one replay. Greedy frees the state as it steps, so with a CPU-held prefix
    the device holds at most two cache copies."""
    state, logits = replay(bundle.decoder, prefix, kept, suffix, mode)
    margin = margin_incremental(bundle, state, prompt, item["a_true"], item["a_alt"], surface)
    box = [state]
    del state
    return logits, margin, greedy_releasing(bundle.decoder, box, max_new=GREEDY_STEPS, stop_at_eos=False)


def native_head(bundle: Bundle) -> Bundle:
    old = bundle.decoder
    return dataclasses.replace(bundle, decoder=Decoder(old.language_model, old.lm_head, "native", old.stop_ids,
                                                       old.head_slice_rows))


# --- The checks -----------------------------------------------------------------------------------------------


def serialization(bundle: Bundle, items: list[dict], images_dir: Path) -> Serialization:
    """E1 on every item."""
    family, failures, compositions, totals, grids = bundle.family, [], Counter(), [], Counter()
    for item in items:
        try:
            tokens, inputs = family.contexts(bundle, item, open_image(images_dir, item))
            if len({tuple(t.prefix_ids) for t in tokens.values()}) != 1:
                raise ValueError("prefix differs between contexts")
            if len({tuple(t.suffix_ids[slice(*t.question_span)]) for t in tokens.values()}) != 1:
                raise ValueError("question tokens differ between contexts")
            start, stop = visual_span(tokens[Q].prefix_ids, family.image_token_id)
            layout = family.layout(bundle.model.config, inputs)
            if stop - start != layout.total:
                raise ValueError(f"span has {stop - start} tokens, layout {layout.total}")
            for answer in (item["a_true"], item["a_alt"]):
                for surface in surface_forms(answer).values():
                    for t in tokens.values():
                        family.candidate_ids(bundle.tokenizer, t.prompt, surface)
        except ValueError as error:
            failures.append(f"{item['item_id']}: {error}")
            continue
        for name, t in tokens.items():
            composition = window_composition(t)
            compositions[f"{name}: q={composition['question']:.3f} d={composition['description']:.3f}"] += 1
        totals.append(layout.total)
        grids[f"{layout.maps[-1].rows}x{layout.maps[-1].cols}"] += 1
    return Serialization(not failures, len(items), failures[:50], dict(compositions),
                         min(totals, default=None), max(totals, default=None), dict(grids.most_common()))


def noise_floor(bundle: Bundle, loaded: list[tuple[dict, object]], context: str, surface: str) -> Floor:
    """Full forward vs prefix-then-suffix with the full cache, on one suffix."""
    generator = np.random.default_rng(0)
    samples, maxima, margins = [], [], []
    for item, image in loaded:
        tokens, inputs, prefix, layout = _prefilled(bundle, item, image)
        t, vision = tokens[context], bundle.family.device_inputs(bundle, inputs)
        release_cached_memory(bundle.device)
        full = full_forward_logits(bundle, t.prefix_ids + t.suffix_ids, vision, len(t.prefix_ids))
        state, incremental = replay(bundle.decoder, prefix, torch.arange(layout.total), torch.tensor([t.suffix_ids]),
                                    "dense")
        difference = full - incremental
        samples.append(sample_differences(difference, generator))
        maxima.append(float(difference.abs().max()))
        margin = margin_incremental(bundle, state, t.prompt, item["a_true"], item["a_alt"], surface)
        del state
        margins.append(abs(margin_full_forward(bundle, t, vision, item["a_true"], item["a_alt"], surface) - margin))
    return Floor(len(loaded), float(np.percentile(np.concatenate(samples), 99)), float(max(maxima)),
                 float(np.mean(margins)), float(np.max(margins)))


def full_fidelity(bundle: Bundle, loaded, floors: dict[str, Floor]) -> Fidelity:
    """E2. Compact must also equal dense with nothing dropped exactly: the part the noise floor cannot show, since
    with nothing dropped the two compute the same thing."""
    by_suffix = {}
    for name, floor in floors.items():
        worst_full, worst_dense = 0.0, 0.0
        for item, image in loaded:
            tokens, inputs, prefix, layout = _prefilled(bundle, item, image)
            t, keep_all = tokens[name], torch.arange(layout.total)
            release_cached_memory(bundle.device)
            full = full_forward_logits(bundle, t.prefix_ids + t.suffix_ids, bundle.family.device_inputs(bundle, inputs),
                                       len(t.prefix_ids))
            suffix = torch.tensor([t.suffix_ids])
            compact = replay(bundle.decoder, prefix, keep_all, suffix, "compact")[1]
            dense = replay(bundle.decoder, prefix, keep_all, suffix, "dense")[1]
            worst_full = max(worst_full, float((full - compact).abs().max()))
            worst_dense = max(worst_dense, float((compact - dense).abs().max()))
        by_suffix[name] = SuffixFidelity(worst_full <= floor.logit_max and worst_dense == 0, worst_full, floor.logit_max,
                                         worst_dense)
    return Fidelity(all(result.passed for result in by_suffix.values()), by_suffix)


def sparse_equivalence(bundle: Bundle, loaded, floor: Floor, plan: Plan, surface: str) -> SparseEquivalence:
    """E3 on Q and QH_mis, at each budget and kept-set kind, with ``GREEDY_STEPS`` generated tokens."""
    generator = np.random.default_rng(1)
    samples, margins, mismatches = [], [], []
    near_tie_limit = 2 * floor.logit_max
    for position, (item, image) in enumerate(loaded):
        tokens, _, prefix, layout = _prefilled(bundle, item, image)
        for name in (Q, "mis"):
            t, suffix = tokens[name], torch.tensor([tokens[name].suffix_ids])
            attention = probe(bundle.decoder, prefix, suffix)
            for budget in plan.budgets:
                for kind in plan.kinds:
                    kept = torch.from_numpy(validation_set(layout, budget, kind, position, attention))
                    logits_c, margin_c, greedy_c = _replayed(bundle, prefix, kept, suffix, "compact", t.prompt, item, surface)
                    logits_d, margin_d, greedy_d = _replayed(bundle, prefix, kept, suffix, "dense", t.prompt, item, surface)
                    samples.append(sample_differences(logits_c - logits_d, generator))
                    margins.append(abs(margin_c - margin_d))
                    for step, (logit_c, logit_d) in enumerate(zip(greedy_c.step_logits, greedy_d.step_logits)):
                        samples.append(sample_differences(logit_c - logit_d, generator))
                        if greedy_c.token_ids[step] != greedy_d.token_ids[step]:
                            gap = min(greedy_c.top_two_gaps[step], greedy_d.top_two_gaps[step])
                            mismatches.append(Mismatch(item["item_id"], name, _label(budget), kind, step, gap,
                                                       gap < near_tie_limit))
                            break
    p99 = float(np.percentile(np.concatenate(samples), 99)) if samples else float("nan")
    margin = float(np.mean(margins)) if margins else float("nan")
    real = sum(not m.near_tie for m in mismatches)
    return SparseEquivalence(p99 <= 2 * floor.logit_p99 and margin < MARGIN_LIMIT and not real, p99, 2 * floor.logit_p99,
                             margin, MARGIN_LIMIT, mismatches, real, len(margins))


def cache_hashes(bundle: Bundle, loaded, plan: Plan) -> Hashes:
    """E4: the prefix hashed before and after each probe and each compact and dense replay of E3's kept sets."""
    failures = []
    for position, (item, image) in enumerate(loaded):
        tokens, _, prefix, layout = _prefilled(bundle, item, image)
        before = entries_hash(prefix.entries)
        for name in (Q, "mis"):
            suffix = torch.tensor([tokens[name].suffix_ids])
            attention = probe(bundle.decoder, prefix, suffix)
            if entries_hash(prefix.entries) != before:
                failures.append(f"{item['item_id']}: changed by probe ({name})")
            for budget in plan.budgets:
                for kind in plan.kinds:
                    kept = torch.from_numpy(validation_set(layout, budget, kind, position, attention))
                    for mode in ("compact", "dense"):
                        replay(bundle.decoder, prefix, kept, suffix, mode)
                        if entries_hash(prefix.entries) != before:
                            failures.append(f"{item['item_id']}: changed by {mode} replay ({name})")
    return Hashes(not failures, failures[:20])


def placebo(bundle: Bundle, loaded, floor: Floor, budget: Budget) -> Placebo:
    """E5 on QH_mis with the probe-scored kept set."""
    worst, different_answers = 0.0, 0
    for position, (item, image) in enumerate(loaded):
        tokens, _, prefix, layout = _prefilled(bundle, item, image)
        suffix = torch.tensor([tokens["mis"].suffix_ids])
        kept = torch.from_numpy(validation_set(layout, budget, "scored", position, probe(bundle.decoder, prefix, suffix)))
        state_b, logits_b = replay(bundle.decoder, prefix, kept, suffix)
        state_c, logits_c = replay(bundle.decoder, prefix, kept.clone(), suffix)
        worst = max(worst, float((logits_b - logits_c).abs().max()))
        different_answers += greedy(bundle.decoder, state_b).token_ids != greedy(bundle.decoder, state_c).token_ids
    return Placebo(worst <= floor.logit_max and different_answers == 0, worst, different_answers)


def _fingerprint(bundle: Bundle, item: dict, image, units: tuple[Unit, ...]) -> str:
    result, attention = run_item(bundle, item, image, list(units))
    arrays = {name: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest() for name, a in sorted(attention.items())}
    return json.dumps({"result": result, "attention": arrays}, sort_keys=True)


def determinism(bundle: Bundle, loaded, units: tuple[Unit, ...]) -> Determinism:
    """E7 with every unit of the plan."""
    different = [item["item_id"] for item, image in loaded
                 if _fingerprint(bundle, item, image, units) != _fingerprint(bundle, item, image, units)]
    return Determinism(not different, different)


# --- Memory and time ------------------------------------------------------------------------------------------


def _reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _peak_gb(device: torch.device) -> float | None:
    return torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None


def _timed(device: torch.device, function: Callable):
    _reset_peak(device)
    start = time.perf_counter()
    result = function()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return result, time.perf_counter() - start, _peak_gb(device)


def cost(bundle: Bundle, item: dict, image, progress: Callable[[str], None]) -> ItemCost:
    """What the run spends on one item: prefill, a probe, and a Full unit on the longest suffix (an upper bound)."""
    device = bundle.device
    tokens, inputs = bundle.family.contexts(bundle, item, image)
    (prefix, layout), prefill_seconds, prefill_peak = _timed(
        device, lambda: bundle.family.prefill(bundle, tokens[Q].prefix_ids, inputs))
    grid = [layout.maps[-1].rows, layout.maps[-1].cols]
    progress(f"  {item['item_id']}: {layout.total} visual tokens, grid {grid[0]}x{grid[1]}")  # before any OOM
    _, probe_seconds, probe_peak = _timed(device, lambda: probe(bundle.decoder, prefix,
                                                                 torch.tensor([tokens["mis"].suffix_ids])))
    longest = max(tokens, key=lambda name: len(tokens[name].suffix_ids))
    full = Unit("full", layout.total, False, NO_CONTEXT, longest)
    _, unit_seconds, unit_peak = _timed(device, lambda: run_unit(bundle, item, full, prefix,
                                                                 np.arange(layout.total), tokens))
    peaks = [p for p in (prefill_peak, probe_peak, unit_peak) if p is not None]
    return ItemCost(item["item_id"], layout.total, grid, prefill_peak, max(peaks) if peaks else None, prefill_seconds,
                    probe_seconds, unit_seconds)


def projected_hours(costs: list[ItemCost], projection: Projection) -> dict[str, float]:
    """Upper bounds: every unit costs a Full unit, and the screen runs Full_Q and Blind_Q on every pool item."""
    prefill, probe_, unit = (float(np.mean([getattr(c, key) for c in costs]))
                             for key in ("prefill_seconds", "probe_seconds", "unit_seconds"))
    blind = prefill + unit
    probes = {u.selection for u in projection.main_units if u.selector == "pw"}
    extra_probes = {u.selection for u in projection.control_units if u.selector == "pw"} - probes
    extra_units = len(projection.control_units) - len(projection.main_units)
    run = (projection.main * (prefill + len(probes) * probe_ + len(projection.main_units) * unit + blind)
           + projection.control * (len(extra_probes) * probe_ + extra_units * unit))
    return {"screen_hours_upper": projection.pool * 2 * blind / 3600, "run_hours_upper": run / 3600}


# --- The whole validation -------------------------------------------------------------------------------------


def validate(bundle: Bundle, items: list[dict], images_dir: Path, plan: Plan, noise_items: int = 50,
             sparse_items: int = 20, repeat_items: int = 5, surface: str = SURFACE,
             progress: Callable[[str], None] = lambda line: None) -> Report:
    """E1 on every item; the noise floors, E2 and costs on the first ``noise_items``; E3-E5 on the first
    ``sparse_items``; E7 on the first ``repeat_items``."""
    loaded = [(item, open_image(images_dir, item)) for item in items[:noise_items]]
    sparse = loaded[:sparse_items]
    head = bundle.decoder.head_precision
    checks: dict = {}
    progress(f"E1: serialization and layouts, {len(items)} items")
    checks["E1"] = serialization(bundle, items, images_dir)
    progress(f"noise floor, {head} head, Q suffix, {len(loaded)} items")
    floor = noise_floor(bundle, loaded, Q, surface)
    progress(f"noise floor, {head} head, QH_mis suffix, {len(loaded)} items")
    floor_mis = noise_floor(bundle, loaded, "mis", surface)
    if head == "native":
        floor_native = floor
    else:
        progress(f"noise floor, native head, Q suffix, {len(loaded)} items")
        floor_native = noise_floor(native_head(bundle), loaded, Q, surface)
    progress(f"E2: full fidelity, both suffixes, {len(loaded)} items")
    checks["E2"] = full_fidelity(bundle, loaded, {Q: floor, "mis": floor_mis})
    progress(f"E3: sparse equivalence, {len(sparse)} items (the slowest check)")
    checks["E3"] = sparse_equivalence(bundle, sparse, floor, plan, surface)
    progress(f"E4: prefix hashes, {len(sparse)} items")
    checks["E4"] = cache_hashes(bundle, sparse, plan)
    if plan.placebo:
        progress(f"E5: placebo, {len(sparse)} items")
        checks["E5"] = placebo(bundle, sparse, floor, plan.placebo)
    progress(f"E7: determinism, {len(loaded[:repeat_items])} items, {len(plan.units)} units each")
    checks["E7"] = determinism(bundle, loaded[:repeat_items], plan.units)
    progress(f"memory and time, {len(loaded)} items")
    costs = [cost(bundle, item, image, progress) for item, image in loaded]
    hours = projected_hours(costs, plan.projection) if plan.projection and costs else None
    progress("done")
    return Report(bundle.family.model_id, bundle.family.revision, head, surface, str(bundle.device), floor, floor_mis,
                  floor_native, checks, costs, hours, all(checks[name].passed for name in plan.checks))


def summary(report: Report, out: Path) -> list[str]:
    """The lines a validate command prints after the report is written, ending in GO or NO-GO."""
    lines = [f"{name}: {'PASS' if check.passed else 'FAIL'}" for name, check in report.checks.items()]
    for label, floor in (("Q", report.noise_floor), ("QH_mis", report.noise_floor_mis),
                         ("Q, native head", report.noise_floor_native_head)):
        lines.append(f"noise floor, {label}: logit p99 {floor.logit_p99:.4g}, max {floor.logit_max:.4g}; "
                     f"margin mean {floor.margin_mean_abs:.4g}, max {floor.margin_max_abs:.4g}")
    e1 = report.checks["E1"]
    lines.append(f"visual tokens: {e1.visual_tokens_min}-{e1.visual_tokens_max}, grids {e1.grids}")
    peaks = [(c.run_peak_gb, c.visual_tokens) for c in report.costs if c.run_peak_gb is not None]
    if not peaks:
        lines.append("peak GPU memory: n/a (not on CUDA)")
    else:
        peak, tokens = max(peaks)
        lines.append(f"peak GPU memory, run: {peak:.2f} GB at {tokens} visual tokens")
        if peak > MEMORY_LIMIT_GB:
            lines.append(f"  over the {MEMORY_LIMIT_GB} GB GPU slice: the run would not fit")
    if report.costs:
        mean = {key: float(np.mean([getattr(c, key) for c in report.costs]))
                for key in ("prefill_seconds", "probe_seconds", "unit_seconds")}
        lines.append(f"time per item: prefill {mean['prefill_seconds']:.2f} s, probe {mean['probe_seconds']:.2f} s, "
                     f"Full unit {mean['unit_seconds']:.2f} s")
    if report.hours:
        lines.append(f"projected: screen <= {report.hours['screen_hours_upper']:.1f} h, "
                     f"run <= {report.hours['run_hours_upper']:.1f} h")
    lines.append(f"model {report.model}" + (f" at revision {report.revision}" if report.revision else ""))
    lines.append(f"full report: {out}")
    lines.append("GO" if report.go else "NO-GO")
    return lines


def write(report: Report, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_json(), indent=2))
