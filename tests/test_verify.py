"""Evidence mapping, decision records and the verification server, over real local HTTP."""

import json
import threading
import urllib.error
import urllib.request

import pytest
from PIL import Image

from kvsel.geometry import box_to_square, evidence_patches, pad_to_square
from kvsel.verify import (
    FAILURE_REASONS, append_decision, build_queue, load_decisions, make_server, passed_items, summarize, view_of,
)


def make_item(item_id="q1", image_file="a.jpg", a_alt="blue"):
    return {
        "item_id": item_id, "image_id": "img", "image_file": image_file, "dataset": "gqa", "split": "dev",
        "builder": "gqa-2", "question": "What color is the bus?", "object": "bus", "a_true": "red", "a_alt": a_alt,
        "descriptions": {"mis": "m", "match": "c", "irr": "i"},
        "evidence_box": {"x": 10, "y": 20, "w": 30, "h": 40}, "image_size": {"width": 200, "height": 100},
    }


def test_box_maps_through_padding_and_scale():
    # 200 x 100 pads to 200 x 200 with 50 pixels above; the scale to 336 is 1.68.
    x0, y0, x1, y1 = box_to_square({"x": 10, "y": 20, "w": 30, "h": 40}, 200, 100)
    assert (x0, y0, x1, y1) == pytest.approx((16.8, 117.6, 67.2, 184.8))


def test_box_mapping_agrees_with_the_padded_image_pixels():
    image = Image.new("RGB", (200, 100), (0, 0, 0))
    image.paste((255, 255, 255), (10, 20, 40, 60))
    resized = pad_to_square(image).resize((336, 336), Image.NEAREST)
    x0, y0, x1, y1 = box_to_square({"x": 10, "y": 20, "w": 30, "h": 40}, 200, 100)
    assert resized.getpixel((int(x0) + 2, int(y0) + 2)) == (255, 255, 255)
    assert resized.getpixel((int(x1) + 2, int(y1) + 2)) == (0, 0, 0)


def test_primary_rule_needs_a_quarter_of_the_patch():
    assert evidence_patches((0, 0, 14, 3.5), "primary") == [0]  # exactly 25% of patch 0
    assert evidence_patches((0, 0, 14, 3.5), "center") == []  # its centre is outside the box


def test_tiny_box_uses_the_fallback():
    tiny = (20.0, 20.0, 21.0, 21.0)  # far under 25% of any patch
    assert evidence_patches(tiny, "primary") == evidence_patches(tiny, "fallback") == [25]


def test_center_and_expanded_rules():
    box = (14.0, 14.0, 28.0, 28.0)  # exactly grid cell (1, 1)
    assert evidence_patches(box, "center") == [25]
    assert evidence_patches(box, "primary") == [25]
    assert evidence_patches(box, "expanded") == [0, 1, 2, 24, 25, 26, 48, 49, 50]


def test_rules_stay_on_the_grid():
    box = (-30.0, -30.0, 400.0, 400.0)
    for rule in ("primary", "fallback", "center", "expanded"):
        patches = evidence_patches(box, rule)
        assert patches and min(patches) >= 0 and max(patches) <= 575


def test_latest_decision_wins_and_failed_items_are_not_passed(tmp_path):
    path = tmp_path / "d.jsonl"
    append_decision(path, make_item("a"), "pass", "me")
    append_decision(path, make_item("b"), "bad_box", "me")
    append_decision(path, make_item("c"), "pass", "me")
    append_decision(path, make_item("c"), "wording", "me")
    assert passed_items(path) == {"a"}
    assert sum(1 for _ in open(path)) == 4  # append-only: nothing is overwritten


def test_rebuilt_item_invalidates_its_old_decision(tmp_path):
    path = tmp_path / "d.jsonl"
    append_decision(path, make_item("a", a_alt="blue"), "pass", "me")
    decisions, stale = load_decisions(path, [make_item("a", a_alt="green")])
    assert decisions == {} and stale == 1


def test_moving_an_item_between_splits_keeps_its_decision(tmp_path):
    path = tmp_path / "d.jsonl"
    append_decision(path, make_item("a"), "pass", "me")
    decisions, stale = load_decisions(path, [{**make_item("a"), "split": "confirm"}])
    assert set(decisions) == {"a"} and stale == 0


def test_unknown_decision_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        append_decision(tmp_path / "d.jsonl", make_item(), "maybe", "me")


def test_queue_filters_to_eligible_keeps_order_and_limits():
    items = [make_item(name) for name in "abcd"]
    assert [i["item_id"] for i in build_queue(items, {"d", "b", "a"}, limit=2)] == ["a", "b"]
    assert len(build_queue(items)) == 4


def test_summary_counts_only_queued_items():
    queue = [make_item("a"), make_item("b"), make_item("c")]
    decisions = {"a": {"decision": "pass"}, "b": {"decision": "bad_box"}, "z": {"decision": "pass"}}
    assert summarize(decisions, queue) == {
        "queue": 3, "decided": 2, "remaining": 1, "passed": 1, "failed": 1, "failure_reasons": {"bad_box": 1},
    }


def test_view_contains_no_model_outcomes():
    view = view_of({**make_item(), "greedy_text": "leak", "margin": 1.0})
    assert "greedy_text" not in view and "margin" not in view
    assert view["pad"] == (0, 50) and set(view["patches"]) == {"primary", "fallback", "center", "expanded"}


@pytest.fixture
def server(tmp_path):
    Image.new("RGB", (200, 100), (10, 20, 30)).save(tmp_path / "a.jpg")
    (tmp_path / "secret.jpg").write_bytes(b"not in the queue")
    queue = [make_item("q1"), make_item("q2", image_file="missing.jpg")]
    instance = make_server(queue, tmp_path, tmp_path / "decisions.jsonl", "tester", 0)
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{instance.server_address[1]}", tmp_path
    instance.shutdown()


def get(url):
    with urllib.request.urlopen(url) as response:
        return response.status, response.read()


def post(url, payload):
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST")
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read())


def test_server_binds_to_localhost_only(server):
    assert server[0].startswith("http://127.0.0.1:")


def test_page_and_items_load(server):
    base, _ = server
    status, body = get(base + "/")
    assert status == 200 and b"Item Verification" in body and b"__PAD__" not in body and b"__REASONS__" not in body
    assert [item["item_id"] for item in json.loads(get(base + "/api/items")[1])] == ["q1", "q2"]


def test_decision_is_saved_and_returned(server):
    base, directory = server
    state = post(base + "/api/decide", {"item_id": "q1", "decision": "pass", "note": "clear"})
    assert state["summary"]["passed"] == 1 and state["decisions"]["q1"]["verifier"] == "tester"
    record = json.loads((directory / "decisions.jsonl").read_text())
    assert record["note"] == "clear" and record["digest"]


@pytest.mark.parametrize("payload", [{"item_id": "nope", "decision": "pass"}, {"item_id": "q1", "decision": "maybe"}])
def test_bad_decisions_are_refused(server, payload):
    with pytest.raises(urllib.error.HTTPError) as error:
        post(server[0] + "/api/decide", payload)
    assert error.value.code == 400


def test_images_are_served_only_for_queued_items(server):
    base, _ = server
    assert get(base + "/image/q1")[0] == 200
    for path in ("/image/secret.jpg", "/image/..%2Fsecret.jpg", "/image/q2"):  # q2's file is absent
        with pytest.raises(urllib.error.HTTPError) as error:
            get(base + path)
        assert error.value.code == 404


def test_failure_reason_keys_are_single_digits():
    assert all(key.isdigit() and len(key) == 1 for key in FAILURE_REASONS)
