"""Local-first model resolution. No network: the Hub download is replaced with a fake."""

import json

import pytest

from kvsel.models.family import REVISION_FILE, WEIGHT_INDEX
from kvsel.models.llava15 import LLAVA

SHARDS = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]


def populate(directory, shards=SHARDS):
    directory.mkdir(parents=True, exist_ok=True)
    for name in LLAVA.required_files:
        (directory / name).write_text("{}")
    (directory / WEIGHT_INDEX).write_text(json.dumps({"weight_map": {"a": SHARDS[0], "b": SHARDS[1], "c": SHARDS[1]}}))
    for name in shards:
        (directory / name).write_bytes(b"x")


def test_complete_local_copy_is_preferred(tmp_path):
    populate(tmp_path / "model")
    assert LLAVA.resolve(tmp_path / "model") == str(tmp_path / "model")


def test_unpinned_copy_needs_no_revision_file(tmp_path):
    """LLaVA-1.5 is unpinned: its records name no revision, and the existing local copy has no revision file."""
    populate(tmp_path)
    assert LLAVA.revision is None and not (tmp_path / REVISION_FILE).exists()
    assert LLAVA.missing_files(tmp_path) == []
    assert LLAVA.resolve(tmp_path) == str(tmp_path)


def test_absent_directory_falls_back_to_the_hub(tmp_path):
    assert LLAVA.resolve(tmp_path / "nothing") == LLAVA.model_id


def test_empty_directory_falls_back_to_the_hub(tmp_path):
    assert LLAVA.resolve(tmp_path) == LLAVA.model_id


def test_missing_shard_is_an_error_rather_than_a_silent_download(tmp_path):
    populate(tmp_path, shards=SHARDS[:1])
    with pytest.raises(SystemExit, match="model-00002-of-00002"):
        LLAVA.resolve(tmp_path)


def test_tokenizer_commands_accept_a_copy_without_weights(tmp_path):
    populate(tmp_path, shards=[])
    (tmp_path / WEIGHT_INDEX).unlink()
    assert LLAVA.resolve(tmp_path, weights=False) == str(tmp_path)
    assert LLAVA.missing_files(tmp_path) == [WEIGHT_INDEX]


def test_fetch_reports_what_is_still_missing(tmp_path, monkeypatch):
    calls = {}

    def fake_download(repo_id, **options):
        calls.update(repo_id=repo_id, **options)
        populate(tmp_path, shards=SHARDS[:1])

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_download)
    assert LLAVA.fetch(tmp_path, 4) == [SHARDS[1]]
    assert calls["repo_id"] == LLAVA.model_id and calls["local_dir"] == tmp_path and calls["max_workers"] == 4
    assert "merges.txt" in calls["allow_patterns"]
    assert "*.safetensors" in calls["allow_patterns"]
    assert "revision" not in calls and not (tmp_path / REVISION_FILE).exists()

