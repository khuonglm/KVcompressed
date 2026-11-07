"""Remote extraction, against a local zip standing in for the server."""

import io
import json
import zipfile
from pathlib import Path

import pytest

from kvsel.fetch_images import RangeFile, extract_member, fetch_one, wanted_files


def make_archive(compression):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression) as archive:
        archive.writestr("images/a.jpg", b"first image" * 100)
        archive.writestr("images/b.jpg", b"second image" * 50)
    return buffer.getvalue()


def reader_for(blob):
    return lambda start, stop: blob[start:stop]


@pytest.mark.parametrize("compression", [zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED])
def test_extracts_each_member_exactly(compression):
    blob = make_archive(compression)
    read_range = reader_for(blob)
    members = zipfile.ZipFile(RangeFile(read_range, len(blob))).infolist()
    assert {m.filename: extract_member(read_range, m) for m in members} == {
        "images/a.jpg": b"first image" * 100,
        "images/b.jpg": b"second image" * 50,
    }


def test_corrupt_data_is_rejected():
    blob = make_archive(zipfile.ZIP_STORED)
    info = zipfile.ZipFile(io.BytesIO(blob)).getinfo("images/a.jpg")
    corrupt = bytearray(blob)
    corrupt[info.header_offset + 30 + len(info.filename) + 5] ^= 0xFF
    with pytest.raises(ValueError, match="CRC"):
        extract_member(reader_for(bytes(corrupt)), info)


def test_fetch_one_retries_then_writes(tmp_path, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    blob = make_archive(zipfile.ZIP_STORED)
    info = zipfile.ZipFile(io.BytesIO(blob)).getinfo("images/a.jpg")
    calls = []

    def flaky(start, stop):
        calls.append(start)
        if len(calls) == 1:
            raise OSError("connection reset")
        return blob[start:stop]

    target = tmp_path / "a.jpg"
    assert fetch_one(flaky, info, target) == 1100
    assert target.read_bytes() == b"first image" * 100
    assert not list(tmp_path.glob("*.tmp")) and len(calls) == 2


def test_wanted_files_deduplicates_across_manifests(tmp_path):
    first, second = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    first.write_text(json.dumps({"image_file": "1.jpg"}) + "\n" + json.dumps({"image_file": "2.jpg"}) + "\n")
    second.write_text(json.dumps({"image_file": "2.jpg"}) + "\n")
    assert wanted_files([Path(first), Path(second)]) == ["1.jpg", "2.jpg"]
