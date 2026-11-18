"""Units, the resumable item loop and screen kinds (no model: the loop is given what each item produces)."""

import json

import pytest

from kvsel.protocol import Refusal
from kvsel.runner import (
    DEVELOPMENT,
    FLAGS,
    ItemFile,
    Unit,
    eligible_ids,
    item_loop,
    mark_screen,
    row_units,
    write_flags,
)

PW, FULL = Unit("pw", 144, False, "Q", "Q"), Unit("full", 576, False, "none", "Q")


def test_unit_records_match_what_records_and_rows_hold():
    assert PW.record() == {"selector": "pw", "k": 144, "reserve": False, "selection": "Q", "answering": "Q"}
    lt = Unit("lt", 144, False, "Q", "Q", followup=True)
    assert lt.record() == {**PW.record(), "selector": "lt", "seed": None, "m": None, "arm": None}
    share = Unit("pw", None, False, "mis", "Q", share=0.25)
    assert share.record() == {"selector": "pw", "share": 0.25, "selection": "mis", "answering": "Q"}
    assert row_units([PW], FULL)[-1] == {**FULL.record(), "selector": "blind"}  # the blind anchor keeps selector full
    with pytest.raises(ValueError):
        Unit("pw", 144, False, "Q", "Q", share=0.25)


def run(tmp_path, items, record="rec", subset="main", units=(PW, FULL)):
    """Runs the loop over ``items`` and returns the ids it produced."""
    produced = []

    def files(item):
        return [ItemFile(tmp_path / "items" / f"{item['item_id']}.json", subset, row_units(list(units)))]

    def produce(item, missing):
        produced.append(item["item_id"])
        rows = [{**unit.record(), "cell": "x"} for unit in units]
        return [({"item_id": item["item_id"], "subset": subset, "rows": rows}, None)]

    results = [result for _, _, (result,) in item_loop(items, record, files, produce)]
    return produced, results


ITEMS = [{"item_id": "a"}, {"item_id": "b"}]


def test_finished_items_are_resumed_and_carry_the_record_hash(tmp_path):
    produced, results = run(tmp_path, ITEMS[:1])
    assert produced == ["a"] and results[0]["record_sha256"] == "rec"
    produced, results = run(tmp_path, ITEMS)
    assert produced == ["b"] and [r["item_id"] for r in results] == ["a", "b"]


@pytest.mark.parametrize(("change", "named"), [
    ({"record": "other"}, "record rec (expected other)"),
    ({"subset": "control"}, "subset main (expected control)"),
    ({"units": (PW, FULL, Unit("pw", 72, False, "Q", "Q"))}, "missing unit {'selector': 'pw', 'k': 72"),
    ({"units": (PW,)}, "unexpected unit {'selector': 'full'"),
])
def test_a_file_from_another_record_subset_or_unit_list_is_refused_and_kept(tmp_path, change, named):
    run(tmp_path, ITEMS[:1])
    path = tmp_path / "items" / "a.json"
    before = path.read_bytes()
    with pytest.raises(Refusal, match="item a: ") as refused:
        run(tmp_path, ITEMS, **change)
    assert named in str(refused.value) and path.read_bytes() == before
    assert not (tmp_path / "items" / "b.json").exists()  # refused before running anything


def screens(tmp_path):
    """A development screen with full rows, and a flags screen with eligible.jsonl but no item files at all."""
    development, flags = tmp_path / "dev", tmp_path / "flags"
    mark_screen(development, DEVELOPMENT)
    (development / "items").mkdir()
    for item_id, outcome in (("a", "correct"), ("b", "alt")):
        row = {"outcome": outcome, "scores": {"true": {"capitalized": [-0.1]}, "alt": {"capitalized": [-2.0]}}}
        (development / "items" / f"{item_id}.json").write_text(json.dumps({"item_id": item_id, "rows": [row]}))
    mark_screen(flags, FLAGS)
    write_flags(flags, [{"item_id": "a", "eligible": False}, {"item_id": "b", "eligible": True}])
    return development, flags


def test_eligibility_follows_the_kind_a_screen_records(tmp_path):
    development, flags = screens(tmp_path)
    assert eligible_ids(development, ITEMS, "capitalized") == ["a"]
    assert eligible_ids(flags, ITEMS, "capitalized") == ["b"]  # nothing to guess from: the folder has no item files
    (development / "kind.json").unlink()
    with pytest.raises(Refusal, match="kind.json is missing"):
        eligible_ids(development, ITEMS, "capitalized")
    with pytest.raises(Refusal, match="holds a flags screen"):
        mark_screen(flags, DEVELOPMENT)
