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
from pathlib import Path

from .geometry import MAPPING_RULES, PAD_COLOR, box_to_square, evidence_patches, pad_offsets

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


def passed_items(path: Path) -> set[str]:
    return {item_id for item_id, record in load_decisions(path)[0].items() if record["decision"] == PASS}


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
    page = PAGE.replace("__PAD__", json.dumps(PAD_COLOR)).replace("__REASONS__", json.dumps(FAILURE_REASONS))

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


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Item Verification</title>
<style>
:root { --bg:#f6f6f4; --panel:#fff; --ink:#1f2328; --muted:#646b73; --line:#d9dbde; --pass:#17693a; --fail:#a52b22; --accent:#0b5cad; }
@media (prefers-color-scheme: dark) { :root { --bg:#16181b; --panel:#212429; --ink:#e8eaed; --muted:#a2a9b1; --line:#3a3e44; --pass:#5ccb8a; --fail:#f08a80; --accent:#7ab7f5; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.45 system-ui,sans-serif; }
header { display:flex; flex-wrap:wrap; gap:16px; align-items:baseline; padding:10px 16px; border-bottom:1px solid var(--line); background:var(--panel); }
h1 { font-size:16px; margin:0; }
#progress { color:var(--muted); font-variant-numeric:tabular-nums; }
main { display:flex; flex-wrap:wrap; gap:16px; padding:16px; align-items:flex-start; }
#view { position:relative; flex:0 1 672px; min-width:0; }
canvas { display:block; width:100%; max-width:672px; height:auto; border:1px solid var(--line); }
aside { flex:1 1 340px; min-width:0; background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:14px 16px; }
.label { font-size:12px; color:var(--muted); margin-top:12px; }
.value { font-size:16px; overflow-wrap:anywhere; }
.big { font-size:20px; font-weight:600; }
.pair { display:flex; gap:24px; flex-wrap:wrap; }
#status { margin-top:12px; font-weight:600; min-height:22px; }
.pass { color:var(--pass); } .fail { color:var(--fail); }
button { font:inherit; color:inherit; background:transparent; border:1px solid var(--line); border-radius:5px; padding:6px 10px; cursor:pointer; text-align:left; }
button:hover, button:focus-visible { border-color:var(--accent); outline:none; }
#actions { display:grid; gap:6px; margin-top:12px; }
#pass { border-color:var(--pass); color:var(--pass); font-weight:600; }
kbd { font:12px ui-monospace,monospace; border:1px solid var(--line); border-radius:3px; padding:0 4px; margin-right:6px; }
#nav { display:flex; flex-wrap:wrap; gap:6px; margin-top:12px; }
#rule { color:var(--muted); font-size:13px; margin-top:6px; }
#stale { color:var(--fail); }
</style></head><body>
<header><h1>Item Verification</h1><span id="progress"></span><span id="stale"></span></header>
<main>
  <div id="view"><canvas id="canvas" width="672" height="672"></canvas><div id="rule"></div></div>
  <aside>
    <div class="label" id="position"></div>
    <div class="label">Question</div><div class="value big" id="question"></div>
    <div class="pair">
      <div><div class="label">a_true</div><div class="value big pass" id="a_true"></div></div>
      <div><div class="label">a_alt</div><div class="value big fail" id="a_alt"></div></div>
      <div id="object-block"><div class="label">Object</div><div class="value big" id="object"></div></div>
    </div>
    <div id="answers-block"><div class="label">Human answers</div><div class="value" id="answers"></div></div>
    <div class="label">Misleading</div><div class="value" id="mis"></div>
    <div class="label">Matching</div><div class="value" id="match"></div>
    <div class="label">Irrelevant</div><div class="value" id="irr"></div>
    <div id="status"></div>
    <div id="actions"><button id="pass"><kbd>y</kbd>Pass</button></div>
    <div id="nav">
      <button id="prev"><kbd>&larr;</kbd>Back</button><button id="next"><kbd>&rarr;</kbd>Forward</button>
      <button id="undecided"><kbd>u</kbd>Next undecided</button><button id="cycle"><kbd>g</kbd>Next overlay</button>
      <button id="note"><kbd>n</kbd>Note</button>
    </div>
  </aside>
</main>
<script>
const PAD = __PAD__, REASONS = __REASONS__, RULES = ["none", "primary", "fallback", "center", "expanded"];
const $ = (id) => document.getElementById(id);
let items = [], decisions = {}, index = 0, ruleIndex = 1, note = "", image = new Image();
for (const [key, [code, label]] of Object.entries(REASONS)) {
  const button = document.createElement("button");
  const kbd = document.createElement("kbd"); kbd.textContent = key;
  button.append(kbd, label);
  button.addEventListener("click", () => decide(code));
  $("actions").append(button);
}
function draw() {
  const item = items[index]; if (!item) return;
  const canvas = $("canvas"), ctx = canvas.getContext("2d"), scale = canvas.width / 336;
  ctx.fillStyle = `rgb(${PAD.join(",")})`; ctx.fillRect(0, 0, canvas.width, canvas.height);
  if (image.complete && image.naturalWidth) {
    const size = item.image_size, side = Math.max(size.width, size.height), s = canvas.width / side;
    ctx.drawImage(image, item.pad[0] * s, item.pad[1] * s, size.width * s, size.height * s);
  }
  const rule = RULES[ruleIndex];
  if (rule !== "none") {
    ctx.fillStyle = "rgba(255, 170, 0, 0.35)";
    for (const patch of item.patches[rule]) {
      ctx.fillRect((patch % 24) * 14 * scale, Math.floor(patch / 24) * 14 * scale, 14 * scale, 14 * scale);
    }
  }
  ctx.lineWidth = 3;
  for (const [x0, y0, x1, y1] of item.boxes) {
    ctx.strokeStyle = "#000"; ctx.strokeRect(x0 * scale, y0 * scale, (x1 - x0) * scale, (y1 - y0) * scale);
    ctx.strokeStyle = "#ffe600"; ctx.lineWidth = 1.5; ctx.strokeRect(x0 * scale, y0 * scale, (x1 - x0) * scale, (y1 - y0) * scale); ctx.lineWidth = 3;
  }
  $("rule").textContent = rule === "none"
    ? "Overlay: box only" : `Overlay: ${rule} rule, ${item.patches[rule].length} patches`;
}
function render() {
  const item = items[index]; if (!item) return;
  $("position").textContent = `Item ${index + 1} of ${items.length} · ${item.item_id} · image ${item.image_id}`;
  for (const key of ["question", "a_true", "a_alt", "object"]) $(key).textContent = item[key];
  for (const key of ["mis", "match", "irr"]) $(key).textContent = item.descriptions[key];
  $("answers-block").hidden = !item.answers.length;
  $("object-block").hidden = !item.object;
  $("answers").textContent = item.answers.join(", ");
  const decision = decisions[item.item_id], status = $("status");
  status.className = decision ? (decision.decision === "pass" ? "pass" : "fail") : "";
  status.textContent = decision ? `Recorded: ${decision.decision}${decision.note ? ` (${decision.note})` : ""}` : "Not decided";
  image = new Image(); image.onload = draw; image.onerror = () => { $("rule").textContent = "Image failed to load"; };
  image.src = `/image/${encodeURIComponent(item.item_id)}`;
  draw();
}
function applyState(state) {
  decisions = state.decisions;
  const s = state.summary;
  $("progress").textContent = `${s.decided} of ${s.queue} decided · ${s.passed} passed · ${s.failed} failed`;
  $("stale").textContent = state.stale ? `${state.stale} decisions are for older item versions` : "";
}
function nextUndecided() {
  for (let step = 1; step <= items.length; step++) {
    const candidate = (index + step) % items.length;
    if (!decisions[items[candidate].item_id]) return candidate;
  }
  return index;
}
async function decide(decision) {
  const item = items[index];
  const response = await fetch("/api/decide", { method: "POST", body: JSON.stringify({ item_id: item.item_id, decision, note }) });
  if (!response.ok) { $("status").textContent = "Could not save this decision"; $("status").className = "fail"; return; }
  applyState(await response.json());
  note = ""; index = nextUndecided(); render();
}
function move(step) { index = (index + step + items.length) % items.length; note = ""; render(); }
function cycle() { ruleIndex = (ruleIndex + 1) % RULES.length; draw(); }
function addNote() { note = prompt("Note for this item's decision", note) || ""; }
$("pass").addEventListener("click", () => decide("pass"));
$("prev").addEventListener("click", () => move(-1));
$("next").addEventListener("click", () => move(1));
$("undecided").addEventListener("click", () => { index = nextUndecided(); render(); });
$("cycle").addEventListener("click", cycle);
$("note").addEventListener("click", addNote);
document.addEventListener("keydown", (event) => {
  if (event.metaKey || event.ctrlKey || event.altKey) return;
  if (event.key === "y") decide("pass");
  else if (REASONS[event.key]) decide(REASONS[event.key][0]);
  else if (event.key === "ArrowLeft") move(-1);
  else if (event.key === "ArrowRight") move(1);
  else if (event.key === "u") { index = nextUndecided(); render(); }
  else if (event.key === "g") cycle();
  else if (event.key === "n") addNote();
});
(async () => {
  items = await (await fetch("/api/items")).json();
  applyState(await (await fetch("/api/state")).json());
  if (!items.length) { $("position").textContent = "The queue is empty"; return; }
  index = decisions[items[0].item_id] ? nextUndecided() : 0;
  render();
})();
</script></body></html>
"""
