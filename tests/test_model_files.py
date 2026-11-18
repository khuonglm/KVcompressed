"""Local-first model resolution. No network: the Hub download is replaced with a fake."""

import json

import pytest

from kvsel.models.family import REVISION_FILE, WEIGHT_INDEX
from kvsel.models.llava15 import LLAVA
from kvsel.models.llava_next import NEXT, REVISION
from kvsel.protocol import Refusal

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
    with pytest.raises(Refusal, match="model-00002-of-00002"):
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



# --- LLaVA-NeXT: pinned to a revision ------------------------------------------------------------------------

NEXT_SHARDS = ["model-00001-of-00003.safetensors", "model-00002-of-00003.safetensors", "model-00003-of-00003.safetensors"]


def populate_next(directory, shards=NEXT_SHARDS):
    directory.mkdir(parents=True, exist_ok=True)
    for name in NEXT.required_files:
        (directory / name).write_text("{}")
    (directory / WEIGHT_INDEX).write_text(json.dumps({"weight_map": {str(i): s for i, s in enumerate(NEXT_SHARDS)}}))
    for name in shards:
        (directory / name).write_bytes(b"x")


def test_fetch_pins_the_revision_and_records_it(tmp_path, monkeypatch):
    calls = {}

    def fake_download(repo_id, **options):
        calls.update(repo_id=repo_id, **options)
        populate_next(tmp_path)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_download)
    assert NEXT.fetch(tmp_path, 4) == []
    assert calls["repo_id"] == "llava-hf/llava-v1.6-vicuna-7b-hf" and calls["revision"] == REVISION
    assert (tmp_path / REVISION_FILE).read_text().strip() == REVISION
    assert NEXT.resolve(tmp_path) == str(tmp_path)


def test_next_fetch_reports_a_missing_shard(tmp_path, monkeypatch):
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda repo_id, **_: populate_next(tmp_path, NEXT_SHARDS[:2]))
    assert NEXT.fetch(tmp_path, 4) == [NEXT_SHARDS[2]]
    with pytest.raises(Refusal, match=NEXT_SHARDS[2]):
        NEXT.resolve(tmp_path)


def test_copy_at_another_revision_or_without_a_record_is_refused(tmp_path):
    populate_next(tmp_path)
    with pytest.raises(Refusal, match=REVISION_FILE):
        NEXT.resolve(tmp_path)
    (tmp_path / REVISION_FILE).write_text("0" * 40)
    assert any("expected" in name for name in NEXT.missing_files(tmp_path))
    with pytest.raises(Refusal, match="another revision"):
        NEXT.resolve(tmp_path)


def test_absent_next_copy_falls_back_to_the_hub(tmp_path):
    assert NEXT.resolve(tmp_path / "nothing") == "llava-hf/llava-v1.6-vicuna-7b-hf"
