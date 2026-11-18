"""NeXT stage 1: freeze-screening, screen and count (PREREG 4.1 and 4.4)."""

import json

import pytest

from kvsel import cli
from kvsel.studies.next import screening as screen
from kvsel.verification import append_decision

SECTION_4 = (
    "## 3. GQA replication\n\nOld text.\n\n"
    "## 4. Second model\n\nDRAFT (date): delete after review.\n\n"
    "A paragraph that says the word DRAFT in the middle of a line.\n\n"
    "### 4.1 Screening rules\n\n- Model: NeXT.\n\n"
    "### 4.2 Configuration\n\n- Shares.\n\n"
    "## 5. Later\n"
)


def make_item(index):
    return {
        "item_id": f"q{index}", "image_id": f"img{index}", "image_file": f"img{index}.png", "dataset": "gqa",
        "split": "confirm", "question": "What color is the bus?", "a_true": "red", "a_alt": "blue",
        "descriptions": {"mis": "The bus in this image is blue."},
    }


@pytest.fixture
def world(tmp_path):
    prereg = tmp_path / "PREREG.md"
    prereg.write_text(SECTION_4)
    items = [make_item(i) for i in range(8)]
    manifest = tmp_path / "gqa_confirm.jsonl"
    manifest.write_text("".join(json.dumps(item) + "\n" for item in items))
    decisions = tmp_path / "decisions.jsonl"
    for item in items:
        append_decision(decisions, item, "pass" if item["item_id"] not in ("q1", "q4") else "wording", "tester")
    stale = dict(items[6], question="Edited after verification?")  # q6's decision is for an older version
    manifest.write_text("".join(json.dumps(stale if item is items[6] else item) + "\n" for item in items))
    return {"prereg": prereg, "items": manifest, "decisions": decisions, "record": tmp_path / "freeze" / "next.json",
            "out": tmp_path / "screen", "images": tmp_path}


def undraft(world):
    world["prereg"].write_text(SECTION_4.replace("DRAFT (date): delete after review.\n\n", ""))


def freeze(world):
    cli.main(["next", "freeze-screening", "--prereg", str(world["prereg"]), "--items", str(world["items"]),
              "--record", str(world["record"])])
    return json.loads(world["record"].read_text())


# --- freeze-screening ------------------------------------------------------------------------------


def test_freeze_refuses_while_section_4_has_its_draft_line(world):
    with pytest.raises(SystemExit, match="DRAFT"):
        freeze(world)
    assert not world["record"].exists()


def test_freeze_refuses_a_blank_left_to_fill(world):
    undraft(world)
    world["prereg"].write_text(world["prereg"].read_text().replace("- Model: NeXT.", "- Model: NeXT at **[to fill before stage 1]**."))
    with pytest.raises(SystemExit, match="1 blank"):
        freeze(world)
    assert not world["record"].exists()


def test_a_mention_of_the_word_is_not_the_draft_line(world):
    undraft(world)
    assert freeze(world)["stage"] == "next_screening"


def test_section_hash_runs_from_the_41_heading_to_the_next_heading(world):
    undraft(world)
    first = freeze(world)["prereg_section_sha256"]
    world["prereg"].write_text(world["prereg"].read_text().replace("- Shares.", "- Other shares."))
    assert screen.rules(world["prereg"]).sha256() == first
    world["prereg"].write_text(world["prereg"].read_text().replace("- Model: NeXT.", "- Model: other."))
    assert screen.rules(world["prereg"]).sha256() != first


def test_record_holds_manifest_code_and_settings_with_the_revision(world):
    from kvsel.models.llava_next import REVISION

    undraft(world)
    record = freeze(world)
    assert set(record["manifest"]) == {"gqa_confirm.jsonl"}
    assert len(record["code_sha256"]) == 64
    assert record["settings"]["revision"] == REVISION and record["settings"]["surface"] == "capitalized"


def test_rerun_unchanged_returns_the_same_record_and_a_different_one_is_refused(world):
    undraft(world)
    first = freeze(world)
    assert freeze(world) == first
    world["prereg"].write_text(world["prereg"].read_text().replace("- Model: NeXT.", "- Model: other."))
    with pytest.raises(SystemExit, match="already exists"):
        freeze(world)
    assert json.loads(world["record"].read_text()) == first


def fake_bundle():
    from types import SimpleNamespace

    from kvsel.models.llava_next import NEXT

    return SimpleNamespace(family=NEXT, decoder=SimpleNamespace(head_precision=NEXT.default_head_precision))


# --- screen and count ------------------------------------------------------------------------------

POOL = ["q0", "q2", "q3", "q5", "q7"]  # passed, in manifest order: q1 and q4 failed, q6's decision is stale
FLAG_KEYS = {"item_id", "full_q_eligible", "blind_eligible", "eligible", "record_sha256"}

def screen_args(world):
    return ["next", "screen", "--prereg", str(world["prereg"]), "--items", str(world["items"]), "--decisions",
            str(world["decisions"]), "--images", str(world["images"]), "--out", str(world["out"]),
            "--record", str(world["record"]), "--device", "cpu"]


def count_of(world, capsys):
    capsys.readouterr()
    cli.main(["next", "count", "--items", str(world["items"]), "--decisions", str(world["decisions"]),
              "--screen", str(world["out"])])
    return json.loads(capsys.readouterr().out)


@pytest.fixture
def frozen_world(world, monkeypatch):
    from PIL import Image

    undraft(world)
    freeze(world)
    for i in range(8):
        Image.new("RGB", (64, 48)).save(world["images"] / f"img{i}.png")
    monkeypatch.setattr(cli, "bundle", lambda args, family: fake_bundle())
    return world


def fake_screen_item(screened, eligible_ids, stop_after=None):
    def screen_item(bundle, item, image, *rule):
        if stop_after is not None and len(screened) == stop_after:
            raise KeyboardInterrupt  # a simulated stop
        screened.append(item["item_id"])
        ok = item["item_id"] in eligible_ids
        return {"item_id": item["item_id"], "full_q_eligible": ok, "blind_eligible": False if ok else None, "eligible": ok}

    return screen_item


@pytest.mark.parametrize("change", ["section", "manifest", "settings", "code"])
def test_screen_refuses_before_loading_the_model_if_anything_changed(frozen_world, monkeypatch, change):
    world = frozen_world
    if change == "section":
        world["prereg"].write_text(world["prereg"].read_text().replace("- Model: NeXT.", "- Model: other."))
    elif change == "manifest":
        world["items"].write_text(world["items"].read_text() + "\n")
    elif change == "settings":
        monkeypatch.setitem(screen.SCREENING, "surface", "lower")
    else:
        monkeypatch.setattr("kvsel.protocol.code_identity", lambda: "0" * 64)
    monkeypatch.setattr(cli, "bundle", lambda *_: pytest.fail("loaded the model before refusing"))
    with pytest.raises(SystemExit, match="changed since"):
        cli.main(screen_args(world))


def test_screen_runs_the_passed_pool_in_manifest_order_writing_flags_only(frozen_world, monkeypatch, capsys):
    screened = []
    monkeypatch.setattr(screen, "screen_item", fake_screen_item(screened, {"q2", "q5"}))
    cli.main(screen_args(frozen_world))
    assert screened == POOL
    for path in (frozen_world["out"] / "items").glob("*.json"):
        assert set(json.loads(path.read_text())) == FLAG_KEYS
    flags = [json.loads(line) for line in (frozen_world["out"] / "eligible.jsonl").read_text().splitlines()]
    assert [f["item_id"] for f in flags] == POOL and [f["item_id"] for f in flags if f["eligible"]] == ["q2", "q5"]
    assert not list(frozen_world["out"].glob("*.tmp"))
    assert count_of(frozen_world, capsys) == {"main_set": 2, "control": 2, "short_of_1000": True, "unscreened_passed": 0}


def test_screen_stops_once_enough_items_are_eligible(frozen_world, monkeypatch, capsys):
    screened = []
    monkeypatch.setitem(screen.SCREENING, "main_size", 2)
    monkeypatch.setattr(screen, "screen_item", fake_screen_item(screened, {"q0", "q3", "q5"}))
    frozen_world["record"].unlink()
    freeze(frozen_world)  # the settings changed, so refreeze
    cli.main(screen_args(frozen_world))
    assert screened == ["q0", "q2", "q3"]
    assert count_of(frozen_world, capsys) == {"main_set": 2, "control": 2, "short_of_1000": False, "unscreened_passed": 2}


def test_screen_resumes_after_a_stop_without_redoing_items(frozen_world, monkeypatch):
    screened = []
    monkeypatch.setattr(screen, "screen_item", fake_screen_item(screened, {"q2"}, stop_after=2))
    with pytest.raises(KeyboardInterrupt):
        cli.main(screen_args(frozen_world))
    assert sorted(p.stem for p in (frozen_world["out"] / "items").glob("*.json")) == ["q0", "q2"]
    monkeypatch.setattr(screen, "screen_item", fake_screen_item(screened, {"q2"}))
    cli.main(screen_args(frozen_world))
    assert screened == POOL


def test_count_works_mid_screen_and_never_loads_the_model(frozen_world, monkeypatch, capsys):
    monkeypatch.setattr(screen, "screen_item", fake_screen_item([], {"q0"}, stop_after=2))
    with pytest.raises(KeyboardInterrupt):
        cli.main(screen_args(frozen_world))
    monkeypatch.setattr(cli, "bundle", lambda *_: pytest.fail("count loaded the model"))
    assert count_of(frozen_world, capsys) == {"main_set": 1, "control": 1, "short_of_1000": True, "unscreened_passed": 3}



def test_count_reads_the_flags_the_freeze_reads(frozen_world, monkeypatch, capsys):
    from kvsel.items.manifest import load_items
    from kvsel.runner import screen_flags
    from kvsel.studies.next import comparative

    monkeypatch.setattr(screen, "screen_item", fake_screen_item([], {"q2", "q5"}))
    cli.main(screen_args(frozen_world))
    for path in (frozen_world["out"] / "items").glob("*.json"):
        path.unlink()  # item files are not a source of eligibility; eligible.jsonl, which stage 3 hashes, is
    chosen, final = comparative.main_set(load_items(frozen_world["items"]), screen_flags(frozen_world["out"]),
                                         frozen_world["decisions"], 1000)
    assert count_of(frozen_world, capsys)["main_set"] == len(chosen) == 2 and final

# --- on the tiny NeXT model ------------------------------------------------------------------------


def test_blank_has_the_real_images_size_and_the_padding_colour():
    from PIL import Image

    from kvsel.models.llava15 import PAD_COLOR
    from kvsel.models.llava_next import NEXT

    blank = NEXT.blank_image(Image.new("RGB", (560, 300), (1, 2, 3)))
    assert blank.size == (560, 300) and blank.getcolors() == [(560 * 300, PAD_COLOR)]


@pytest.mark.network
def test_end_to_end_on_the_tiny_model_with_a_resume(frozen_world, monkeypatch, tiny_llava_next):
    import numpy as np
    from PIL import Image

    from kvsel.model import bundle_from_model
    from kvsel.models.llava_next import NEXT

    processor = NEXT.load_processor()
    bundle = bundle_from_model(tiny_llava_next, processor.tokenizer, processor, family=NEXT)
    monkeypatch.setattr(cli, "bundle", lambda args, family: bundle)
    generator = np.random.default_rng(0)
    for i in range(8):
        Image.fromarray(generator.integers(0, 255, (300, 560, 3), dtype=np.uint8)).save(frozen_world["images"] / f"img{i}.png")
    # Random weights never answer correctly; let every Full_Q pass so Blind_Q runs too, on the real model.
    rows = []
    monkeypatch.setattr("kvsel.runner.eligible", lambda row, surface: rows.append(row) or True)
    real = screen.screen_item

    def stops_after_two_items(*args):
        if len(rows) == 4:
            raise KeyboardInterrupt
        return real(*args)

    monkeypatch.setattr(screen, "screen_item", stops_after_two_items)
    with pytest.raises(KeyboardInterrupt):
        cli.main(screen_args(frozen_world))
    monkeypatch.setattr(screen, "screen_item", real)
    cli.main(screen_args(frozen_world))
    assert len(rows) == 2 * len(POOL)  # Full_Q and Blind_Q per item, none redone after the stop
    # 560 x 300 gives 1656 visual tokens (worked in test_next.ASPECTS); the 336 x 336 blank would give another count.
    assert all(row["k"] == 1656 for row in rows)
    for path in (frozen_world["out"] / "items").glob("*.json"):
        record = json.loads(path.read_text())
        assert set(record) == FLAG_KEYS
        assert record["full_q_eligible"] and record["blind_eligible"] and not record["eligible"]
