"""Hand-verification tool: a small local web page that records one decision per item.

Decisions go to an append-only JSONL file, and the latest decision for an item wins. Each records a
hash of the item, so a decision can't silently carry over to a rebuilt item. The page never shows
model outcomes.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path

from ..models.llava15 import MAPPING_RULES, PAD_COLOR, box_to_square, evidence_patches, pad_offsets

FAILURE_REASONS = {
    "1": ("statement_not_false", "The misleading statement isn't false"),
    "2": ("a_true_wrong", "a_true isn't what the image shows"),
    "3": ("referent_not_unique", "The object isn't unique in the image"),
    "4": ("a_alt_acceptable", "a_alt is an acceptable answer"),
    "5": ("wording", "The wording isn't a clean minimal pair, or reads badly"),
    "6": ("bad_box", "The evidence box is wrong"),
}
PASS = "pass"
VALID_DECISIONS = {PASS, *(code for code, _ in FAILURE_REASONS.values())}


def item_digest(item: dict) -> str:
    """Hash of everything but the split, so moving an item between splits keeps its decision."""
    content = {key: value for key, value in item.items() if key != "split"}
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def load_decisions(path: Path, items: list[dict] | None = None) -> tuple[dict[str, dict], int]:
    """Latest decision per item, and how many items' latest decisions were for an older item version."""
    latest: dict[str, dict] = {}
    if Path(path).exists():
        with open(path) as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    latest[record["item_id"]] = record
    if items is None:
        return latest, 0
    digests = {item["item_id"]: item_digest(item) for item in items}
    current = {key: value for key, value in latest.items() if digests.get(key) == value["digest"]}
    stale = sum(1 for key, value in latest.items() if key in digests and digests[key] != value["digest"])
    return current, stale


def append_decision(path: Path, item: dict, decision: str, verifier: str, note: str = "") -> dict:
    if decision not in VALID_DECISIONS:
        raise ValueError(f"unknown decision {decision!r}")
    record = {
        "item_id": item["item_id"],
        "decision": decision,
        "verifier": verifier,
        "note": note,
        "digest": item_digest(item),
        "builder": item.get("builder"),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    return record


def summarize(decisions: dict[str, dict], queue: list[dict]) -> dict:
    queued = {item["item_id"] for item in queue}
    relevant = [record for item_id, record in decisions.items() if item_id in queued]
    counts = Counter(record["decision"] for record in relevant)
    return {
        "queue": len(queue),
        "decided": len(relevant),
        "remaining": len(queue) - len(relevant),
        "passed": counts[PASS],
        "failed": len(relevant) - counts[PASS],
        "failure_reasons": {key: value for key, value in sorted(counts.items()) if key != PASS},
    }


def build_queue(items: list[dict], eligible_ids: set[str] | None = None, limit: int | None = None) -> list[dict]:
    """Manifest order, which is the seeded image order, so partial work stays a valid sample."""
    queue = [item for item in items if eligible_ids is None or item["item_id"] in eligible_ids]
    return queue[:limit] if limit else queue


def view_of(item: dict) -> dict:
    """What the page shows for an item, including its evidence patches under every mapping rule."""
    size = item["image_size"]
    box = item["evidence_box"]
    boxes = box if isinstance(box, list) else [box]
    width, height = size["width"], size["height"]
    squares = [box_to_square(one, width, height) for one in boxes]
    return {
        "item_id": item["item_id"],
        "image_id": item["image_id"],
        "question": item["question"],
        "object": item.get("object", ""),
        "a_true": item["a_true"],
        "a_alt": item["a_alt"],
        "answers": item.get("answers", []),
        "descriptions": item["descriptions"],
        "image_size": size,
        "pad": pad_offsets(width, height),
        "boxes": squares,
        "patches": {rule: sorted({p for s in squares for p in evidence_patches(s, rule)}) for rule in MAPPING_RULES},
    }


def make_server(queue: list[dict], images_dir: Path, decisions_path: Path, verifier: str, port: int) -> ThreadingHTTPServer:
    by_id = {item["item_id"]: item for item in queue}
    page = files(__package__).joinpath("page.html").read_text()
    page = page.replace("__PAD__", json.dumps(PAD_COLOR)).replace("__REASONS__", json.dumps(FAILURE_REASONS))

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload, status: int = 200) -> None:
            self._send(json.dumps(payload).encode(), "application/json", status)

        def _state(self) -> dict:
            decisions, stale = load_decisions(decisions_path, queue)
            return {"summary": summarize(decisions, queue), "stale": stale, "decisions": decisions}

        def do_GET(self) -> None:
            if self.path == "/":
                self._send(page.encode(), "text/html; charset=utf-8")
            elif self.path == "/api/items":
                self._json([view_of(item) for item in queue])
            elif self.path == "/api/state":
                self._json(self._state())
            elif self.path.startswith("/image/"):
                # Looked up by item id, so a request can only ever reach a queued item's image.
                item = by_id.get(self.path.removeprefix("/image/"))
                path = images_dir / item["image_file"] if item else None
                if path is None or not path.is_file():
                    self._json({"error": "image not found"}, 404)
                else:
                    self._send(path.read_bytes(), "image/jpeg")
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self) -> None:
            if self.path != "/api/decide":
                self._json({"error": "not found"}, 404)
                return
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            item = by_id.get(payload.get("item_id"))
            if item is None or payload.get("decision") not in VALID_DECISIONS:
                self._json({"error": "unknown item or decision"}, 400)
                return
            append_decision(decisions_path, item, payload["decision"], verifier, payload.get("note", ""))
            self._json(self._state())

    # Local only: reach it from another machine through an SSH tunnel.
    return ThreadingHTTPServer(("127.0.0.1", port), Handler)

