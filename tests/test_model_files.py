"""Local-first model resolution. No network: the Hub download is replaced with a fake."""

import json

import pytest

from kvsel import MODEL_ID
from kvsel.model_files import REQUIRED_FILES, WEIGHT_INDEX, fetch_model, missing_files, resolve_model

SHARDS = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]


def populate(directory, shards=SHARDS):
    directory.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_FILES[MODEL_ID]:
        (directory / name).write_text("{}")
    (directory / WEIGHT_INDEX).write_text(json.dumps({"weight_map": {"a": SHARDS[0], "b": SHARDS[1], "c": SHARDS[1]}}))
    for name in shards:
        (directory / name).write_bytes(b"x")


def test_complete_local_copy_is_preferred(tmp_path):
    populate(tmp_path / "model")
    assert resolve_model(tmp_path / "model") == str(tmp_path / "model")


def test_absent_directory_falls_back_to_the_hub(tmp_path):
    assert resolve_model(tmp_path / "nothing") == MODEL_ID


def test_empty_directory_falls_back_to_the_hub(tmp_path):
    assert resolve_model(tmp_path) == MODEL_ID


def test_missing_shard_is_an_error_rather_than_a_silent_download(tmp_path):
    populate(tmp_path, shards=SHARDS[:1])
    with pytest.raises(SystemExit, match="model-00002-of-00002"):
        resolve_model(tmp_path)


def test_tokenizer_commands_accept_a_copy_without_weights(tmp_path):
    populate(tmp_path, shards=[])
    (tmp_path / WEIGHT_INDEX).unlink()
    assert resolve_model(tmp_path, weights=False) == str(tmp_path)
    assert missing_files(tmp_path) == [WEIGHT_INDEX]


def test_fetch_reports_what_is_still_missing(tmp_path, monkeypatch):
    calls = {}

    def fake_download(repo_id, **options):
        calls.update(repo_id=repo_id, **options)
        populate(tmp_path, shards=SHARDS[:1])

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_download)
    assert fetch_model(tmp_path, workers=4) == [SHARDS[1]]
    assert calls["repo_id"] == MODEL_ID and calls["local_dir"] == tmp_path and calls["max_workers"] == 4
    assert "merges.txt" in calls["allow_patterns"]
    assert "*.safetensors" in calls["allow_patterns"]

