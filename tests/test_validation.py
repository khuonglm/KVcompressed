"""Validation for both families: kept-set kinds, the projection, each E-check on the tiny models, and the GO rule."""

import json

import numpy as np
import pytest
import torch
from PIL import Image

from kvsel import cli, validation
from kvsel.model import bundle_from_model
from kvsel.models.llava15 import LAYOUT, LLAVA
from kvsel.models.llava_next import NEXT, anyres_layout
from kvsel.runner import open_image
from kvsel.selectors import Budget
from kvsel.studies.next.comparative import SHARES
from kvsel.studies.next.comparative import VALIDATION as NEXT_PLAN
from kvsel.studies.pilot import VALIDATION as LLAVA15_PLAN
from kvsel.validation import (
    ItemCost,
    cache_hashes,
    concentrated_set,
    determinism,
    full_fidelity,
    noise_floor,
    placebo,
    projected_hours,
    sparse_equivalence,
    validation_set,
)

# --- Kept-set kinds and the projection (pure) -----------------------------------------------------------------


@pytest.mark.parametrize("kind", ["random", "concentrated"])
def test_share_sets_hold_round_k_features_and_every_newline(kind):
    layout = anyres_layout(24, 24, 44)
    for share in SHARES:
        kept = validation_set(layout, Budget(share=share), kind, 0)
        assert set(layout.newlines) <= set(kept.tolist())
        assert len(kept) == round(share * len(layout.features)) + len(layout.newlines) == len(set(kept.tolist()))
        assert np.all(np.diff(kept) > 0)


def test_concentrated_is_the_first_k_features():
    assert concentrated_set(LAYOUT, 72).tolist() == list(range(72))


def test_an_unknown_kind_is_refused():
    with pytest.raises(ValueError, match="unknown kept-set kind"):
        validation_set(LAYOUT, Budget(k=72), "uniform", 0)


def test_each_family_declares_its_checks():
    assert LLAVA15_PLAN.checks == ("E1", "E2", "E3", "E4", "E5", "E7")
    assert NEXT_PLAN.checks == ("E1", "E2", "E3", "E4", "E7")


def cost(prefill, probe, unit):
    return ItemCost("i", 576, [24, 24], None, None, prefill, probe, unit)


def test_next_projection_counts_match_prereg_42():
    """By hand from 4.2: main items probe Q, mis, irr and run 3 shares x 7 cells + 3 Full = 24 units, plus Blind_Q;
    control items also probe match and run 3 x 3 match cells + Full_match = 10 more. Costs in seconds isolate each."""
    projection = NEXT_PLAN.projection
    assert (projection.pool, projection.main, projection.control) == (2335, 1000, 300)
    assert projected_hours([cost(0, 1, 0)], projection)["run_hours_upper"] * 3600 == pytest.approx(1000 * 3 + 300 * 1)
    assert projected_hours([cost(0, 0, 1)], projection)["run_hours_upper"] * 3600 == pytest.approx(1000 * 25 + 300 * 10)
    assert projected_hours([cost(1, 0, 0)], projection)["screen_hours_upper"] * 3600 == pytest.approx(2335 * 2)


# --- The checks on the tiny models ----------------------------------------------------------------------------


def make_item(index, split="dev"):
    return {
        "item_id": f"q{index}", "image_id": f"img{index}", "image_file": f"img{index}.jpg", "dataset": "gqa",
        "split": split, "question": "What color is the bus?", "a_true": "red", "a_alt": "blue",
        "descriptions": {
            "mis": "The bus in this image is blue.", "match": "The bus in this image is red.",
            "irr": "The road in this image is gray.",
        },
    }


def write_dataset(directory, items, sizes):
    generator = np.random.default_rng(0)
    for item, (width, height) in zip(items, sizes):
        Image.fromarray(generator.integers(0, 255, (height, width, 3), dtype=np.uint8)).save(directory / item["image_file"])
    manifest = directory / "items.jsonl"
    manifest.write_text("".join(json.dumps(item) + "\n" for item in items))
    return manifest


@pytest.fixture(scope="module")
def llava(tiny_llava):
    return bundle_from_model(tiny_llava, LLAVA.load_tokenizer(), LLAVA.load_processor())


@pytest.fixture(scope="module")
def llava_next(tiny_llava_next):
    processor = NEXT.load_processor()
    return bundle_from_model(tiny_llava_next, processor.tokenizer, processor, family=NEXT)


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    directory = tmp_path_factory.mktemp("images")
    items = [make_item(i) for i in range(3)]
    manifest = write_dataset(directory, items, [(560, 300), (300, 560), (560, 300)])
    return items, directory, manifest


@pytest.fixture(scope="module")
def loaded(dataset):
    items, directory, _ = dataset
    return [(item, open_image(directory, item)) for item in items]


@pytest.mark.network
@pytest.mark.parametrize(("family", "plan"), [("llava", LLAVA15_PLAN), ("llava_next", NEXT_PLAN)])
def test_each_check_passes_on_the_tiny_model(family, plan, request, loaded):
    bundle = request.getfixturevalue(family)
    floor = noise_floor(bundle, loaded[:2], "Q", "capitalized")
    assert floor.logit_max < 1e-3 and floor.margin_max_abs < 1e-2
    floors = {"Q": floor, "mis": noise_floor(bundle, loaded[:2], "mis", "capitalized")}
    fidelity = full_fidelity(bundle, loaded[:2], floors)
    assert set(fidelity.by_suffix) == {"Q", "mis"}
    assert all(s.max_logit_diff < 1e-3 and s.compact_vs_dense_max == 0 for s in fidelity.by_suffix.values())
    sparse = sparse_equivalence(bundle, loaded[:1], floor, plan, "capitalized")
    assert sparse.margin_mean_abs < 0.01 and sparse.greedy_mismatches_not_near_tie == 0
    assert sparse.comparisons == 2 * len(plan.budgets) * len(plan.kinds)
    assert cache_hashes(bundle, loaded[:1], plan).passed
    if plan.placebo:
        assert placebo(bundle, loaded[:1], floor, plan.placebo).max_logit_diff == 0
    assert determinism(bundle, loaded[:1], plan.units[:3]).passed


@pytest.mark.network
@pytest.mark.parametrize("family", ["llava", "llava_next"])
@pytest.mark.parametrize(("culprit", "expected"), [("probe", "probe"), ("replay", "compact replay"),
                                                   ("replay", "dense replay")])
def test_e4_catches_an_engine_call_that_changes_the_prefix(family, culprit, expected, request, loaded, monkeypatch):
    bundle = request.getfixturevalue(family)
    real = getattr(validation, culprit)

    def tampering(decoder, prefix, *args, **kwargs):
        if culprit == "probe" or (args[2] if len(args) > 2 else "compact") in expected:
            with torch.inference_mode():
                prefix.entries[0][0].add_(1.0)
        return real(decoder, prefix, *args, **kwargs)

    monkeypatch.setattr(validation, culprit, tampering)
    plan = validation.Plan(budgets=(Budget(k=72),), kinds=("random",), units=())
    hashes = cache_hashes(bundle, loaded[:1], plan)
    assert not hashes.passed and any(f"changed by {expected}" in failure for failure in hashes.failures)


# --- The command: GO or NO-GO ---------------------------------------------------------------------------------


def passing_e3(*args, **kwargs):
    return validation.SparseEquivalence(True, 0.0, 0.0, 0.0, 0.01, [], 0, 1)


def failing_e4(*args, **kwargs):
    return validation.Hashes(False, ["q0: changed by probe (Q)"])


def run(main, arguments) -> int:
    try:
        main(arguments)
    except SystemExit as stop:
        return stop.code
    return 0


@pytest.mark.network
@pytest.mark.parametrize("family", ["llava", "llava_next"])
@pytest.mark.parametrize("forced", ["go", "no-go"])
def test_the_command_prints_each_verdict_and_exits_by_the_go_rule(family, forced, request, dataset, tmp_path, monkeypatch,
                                                                  capsys):
    """The tiny models fail E3 on their own (an fp32 noise floor of 0 leaves no room), so GO forces E3 to pass and
    NO-GO additionally forces E4 to fail."""
    tiny = request.getfixturevalue(family)
    monkeypatch.setattr(cli, "bundle", lambda args, model: tiny)
    monkeypatch.setattr(validation, "sparse_equivalence", passing_e3)
    if forced == "no-go":
        monkeypatch.setattr(validation, "cache_hashes", failing_e4)
    _, directory, manifest = dataset
    out = tmp_path / "report.json"
    code = run(cli.main, ["validate", "--model", family, "--items", str(manifest), "--images", str(directory), "--device", "cpu",
                      "--out", str(out), "--noise-items", "2", "--sparse-items", "1", "--repeat-items", "1"])
    printed = capsys.readouterr().out.splitlines()
    report = json.loads(out.read_text())
    assert report["go"] == (forced == "go") and code == (0 if forced == "go" else 1)
    assert printed[-1] == ("GO" if forced == "go" else "NO-GO")
    assert ("E4: FAIL" in printed) == (forced == "no-go") and "E3: PASS" in printed
    assert {c["item_id"] for c in report["costs"]} == {"q0", "q1"} and report["noise_floor_native_head"]["items"] == 2


@pytest.mark.parametrize("model", ["llava", "llava_next"])
def test_validate_refuses_items_that_are_not_development_items(model, tmp_path, monkeypatch):
    manifest = write_dataset(tmp_path, [make_item(0), make_item(1, "confirm")], [(64, 48)] * 2)
    monkeypatch.setattr(cli, "bundle", lambda *_: pytest.fail("loaded the model before refusing"))
    with pytest.raises(SystemExit, match="not development items") as stop:
        cli.main(["validate", "--model", model, "--items", str(manifest), "--images", str(tmp_path), "--device", "cpu"])
    assert stop.value.code == 2
