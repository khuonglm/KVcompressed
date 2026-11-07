"""Downloads only the images a manifest needs from inside a remote zip, using HTTP range requests."""

from __future__ import annotations

import json
import os
import struct
import time
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

GQA_IMAGES_URL = "https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip"
_LOCAL_HEADER_SIZE = 30
_LOCAL_EXTRA_ALLOWANCE = 1024
_ATTEMPTS = 7
_TIMEOUT_SECONDS = 60

ReadRange = Callable[[int, int], bytes]


def http_range_reader(url: str) -> ReadRange:
    def read_range(start: int, stop: int) -> bytes:
        """Bytes in [start, stop); the server may return fewer at the end of the file."""
        request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{stop - 1}"})
        with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
            if response.status != 206:
                raise RuntimeError(f"server ignored the range request (status {response.status})")
            return response.read()

    return read_range


class RangeFile:
    """The minimal seekable file ZipFile needs to read a central directory."""

    def __init__(self, read_range: ReadRange, size: int):
        self.read_range, self.size, self.position = read_range, size, 0

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = 0) -> int:
        self.position = {0: offset, 1: self.position + offset, 2: self.size + offset}[whence]
        return self.position

    def read(self, count: int = -1) -> bytes:
        stop = self.size if count < 0 else min(self.size, self.position + count)
        if stop <= self.position:
            return b""
        data = self.read_range(self.position, stop)
        self.position += len(data)
        return data


def extract_member(read_range: ReadRange, info: zipfile.ZipInfo) -> bytes:
    """One range request per file: local header plus data, checked against the archive's CRC."""
    stop = info.header_offset + _LOCAL_HEADER_SIZE + len(info.filename.encode()) + _LOCAL_EXTRA_ALLOWANCE + info.compress_size
    block = read_range(info.header_offset, stop)
    name_length, extra_length = struct.unpack("<HH", block[26:30])
    start = _LOCAL_HEADER_SIZE + name_length + extra_length
    compressed = block[start : start + info.compress_size]
    if len(compressed) != info.compress_size:
        raise ValueError(f"short read for {info.filename}")
    if info.compress_type == zipfile.ZIP_STORED:
        data = compressed
    elif info.compress_type == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(compressed, -15)
    else:
        raise ValueError(f"unsupported compression {info.compress_type} for {info.filename}")
    if zlib.crc32(data) != info.CRC:
        raise ValueError(f"CRC mismatch for {info.filename}")
    return data


def wanted_files(manifests: list[Path]) -> list[str]:
    names: set[str] = set()
    for manifest in manifests:
        with open(manifest) as handle:
            names.update(json.loads(line)["image_file"] for line in handle if line.strip())
    return sorted(names)


def fetch_one(read_range: ReadRange, info: zipfile.ZipInfo, target: Path) -> int:
    for attempt in range(_ATTEMPTS):
        try:
            data = extract_member(read_range, info)
            break
        except (OSError, ValueError, RuntimeError):
            if attempt == _ATTEMPTS - 1:
                raise
            time.sleep(2**attempt)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, target)
    return len(data)


def fetch(manifests: list[Path], out_dir: Path, url: str = GQA_IMAGES_URL, workers: int = 3, limit: int | None = None) -> dict:
    """Skips files already present, so an interrupted run resumes. Raises if any file is still missing."""
    out_dir.mkdir(parents=True, exist_ok=True)
    names = wanted_files(manifests)
    missing_files = [name for name in names if not (out_dir / name).exists()]
    pending = missing_files[:limit]
    summary = {"wanted": len(names), "already_present": len(names) - len(missing_files), "fetched": 0, "bytes": 0, "failed": []}
    if not pending:
        return summary
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=_TIMEOUT_SECONDS) as response:
        size = int(response.headers["Content-Length"])
    read_range = http_range_reader(url)
    print("reading the archive's file listing...", flush=True)
    members = {Path(info.filename).name: info for info in zipfile.ZipFile(RangeFile(read_range, size)).infolist()}
    missing = [name for name in pending if name not in members]
    if missing:
        raise SystemExit(f"{len(missing)} images are not in the archive (first: {missing[0]})")
    start = time.time()
    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(fetch_one, read_range, members[name], out_dir / name): name for name in pending}
        for done, future in enumerate(as_completed(futures), 1):
            try:
                summary["bytes"] += future.result()
                summary["fetched"] += 1
            except Exception as error:
                summary["failed"].append(f"{futures[future]}: {error}")
            if done % 100 == 0 or done == len(pending):
                print(f"{done}/{len(pending)} files, {summary['bytes'] / 1e6:.0f} MB, {time.time() - start:.0f} s", flush=True)
    return summary
