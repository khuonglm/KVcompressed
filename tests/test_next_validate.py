"""NeXT fetch-model (fake Hub download) and the whole validate command on the tiny NeXT model on CPU."""

import json

import numpy as np
import pytest
from PIL import Image

from kvsel.model_files import WEIGHT_INDEX
from kvsel_next import __main__ as cli
from kvsel_next.layout import Layout
from kvsel_next.model_files import REQUIRED_FILES, REVISION, REVISION_FILE, fetch_model, missing_files, resolve_model
from kvsel_next.selection import SHARES
from kvsel_next.validate import _projection_sizes, sparse_set

SHARDS = ["model-00001-of-00003.safetensors", "model-00002-of-00003.safetensors", "model-00003-of-00003.safetensors"]


def populate(directory, shards=SHARDS):
    directory.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_FILES:
        (directory / name).write_text("{}")
    (directory / WEIGHT_INDEX).write_text(json.dumps({"weight_map": {str(i): s for i, s in enumerate(SHARDS)}}))
    for name in shards:
        (directory / name).write_bytes(b"x")


# --- fetch-model -----------------------------------------------------------------------------------


def test_fetch_pins_the_revision_and_records_it(tmp_path, monkeypatch):
    calls = {}

    def fake_download(repo_id, **options):
        calls.update(repo_id=repo_id, **options)
        populate(tmp_path)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_download)
    assert fetch_model(tmp_path, 4) == []
    assert calls["repo_id"] == "llava-hf/llava-v1.6-vicuna-7b-hf" and calls["revision"] == REVISION
    assert (tmp_path / REVISION_FILE).read_text().strip() == REVISION
    assert resolve_model(tmp_path) == str(tmp_path)


def test_fetch_reports_a_missing_shard(tmp_path, monkeypatch):
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda repo_id, **_: populate(tmp_path, SHARDS[:2]))
    assert fetch_model(tmp_path, 4) == [SHARDS[2]]
    with pytest.raises(SystemExit, match=SHARDS[2]):
        resolve_model(tmp_path)


def test_copy_at_another_revision_or_without_a_record_is_refused(tmp_path):
    populate(tmp_path)
    with pytest.raises(SystemExit, match=REVISION_FILE):
        resolve_model(tmp_path)
    (tmp_path / REVISION_FILE).write_text("0" * 40)
    assert any("expected" in name for name in missing_files(tmp_path))
    with pytest.raises(SystemExit, match="another revision"):
        resolve_model(tmp_path)


def test_absent_copy_falls_back_to_the_hub(tmp_path):
    assert resolve_model(tmp_path / "nothing") == "llava-hf/llava-v1.6-vicuna-7b-hf"


# --- kept sets -------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["random", "concentrated"])
def test_kept_sets_hold_k_features_and_every_newline(kind):
    layout = Layout(24, (24, 44))
    features = layout.total - len(layout.newlines)
    for share in SHARES:
        kept = sparse_set(layout, share, kind, 0)
        assert set(layout.newlines) <= set(kept.tolist())
        assert len(kept) == round(share * features) + len(layout.newlines) == len(set(kept.tolist()))
        assert np.all(np.diff(kept) > 0)


def test_projection_counts_match_prereg_42():
    """By hand from 4.2: main items probe Q, mis, irr and run 3 shares x 7 cells + 3 Full = 24 units;
    control items also probe match and run 3 x 3 match cells + Full_match = 10 more."""
    assert _projection_sizes() == (1000, 300, 3, 24, 1, 10)


# --- validate --------------------------------------------------------------------------------------


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


def test_validate_refuses_items_that_are_not_development_items(tmp_path, monkeypatch):
    manifest = write_dataset(tmp_path, [make_item(0), make_item(1, "confirm")], [(64, 48)] * 2)
    monkeypatch.setattr(cli, "load_bundle", lambda *_: pytest.fail("loaded the model before refusing"))
    with pytest.raises(SystemExit, match="not development items"):
        cli.main(["validate", "--items", str(manifest), "--images", str(tmp_path), "--device", "cpu"])


@pytest.mark.network
def test_whole_validate_command_on_tiny_model(tmp_path, monkeypatch, capsys, tiny_llava_next):
    from kvsel.model import bundle_from_model
    from kvsel_next.family import NEXT
    from kvsel_next.model import load_processor

    processor = load_processor()
    bundle = bundle_from_model(tiny_llava_next, processor.tokenizer, processor, family=NEXT)
    monkeypatch.setattr(cli, "load_bundle", lambda device, model_dir: bundle)
    items = [make_item(i) for i in range(3)]
    manifest = write_dataset(tmp_path, items, [(560, 300), (300, 560), (560, 300)])
    out = tmp_path / "report.json"
    arguments = ["validate", "--items", str(manifest), "--images", str(tmp_path), "--device", "cpu", "--out", str(out),
                 "--noise-items", "2", "--sparse-items", "1", "--repeat-items", "1"]
    try:
        cli.main(arguments)
        exit_code = 0
    except SystemExit as error:
        exit_code = error.code
    printed = capsys.readouterr().out.splitlines()
    report = json.loads(out.read_text())

    assert report["revision"] == REVISION and report["surface"] == "capitalized"
    assert report["layouts"] == {"visual_tokens_min": 1656, "visual_tokens_max": 1676, "grids": {"24x44": 2, "44x24": 1}}
    assert [r["visual_tokens"] for r in report["items"]] == [1656, 1676]
    assert report["memory"]["run_peak_gb"] is None  # CPU
    assert report["time"]["screen_hours_upper"] > 0 and report["time"]["comparative_hours_upper"] > 0
    assert report["E3"]["comparisons"] == 2 * len(SHARES) * 2  # Q and mis, 3 shares, random and concentrated
    for name in ("E1", "E2", "E4", "E7"):
        assert report[name]["pass"], name
    assert report["E2"]["by_suffix"]["Q"]["compact_vs_dense_max"] == 0
    assert [line.split(":")[0] for line in printed if line.endswith(("PASS", "FAIL"))] == ["E1", "E2", "E3", "E4", "E7"]
    assert printed[-1] == ("GO" if report["go"] else "NO-GO")
    assert exit_code == (0 if report["go"] else 1)
