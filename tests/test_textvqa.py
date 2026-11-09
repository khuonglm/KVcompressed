"""TextVQA builder rules on a small hand-built fixture."""

import json

import pytest

from kvsel.textvqa import (
    build, build_items, character_class, choose_alternative, find_spans, modal_answer, normalize_text,
)


def info(word, x=0.1, y=0.1, w=0.1, h=0.1, yaw=0):
    return {"word": word, "bounding_box": {"top_left_x": x, "top_left_y": y, "width": w, "height": h, "rotation": 0, "roll": 0, "pitch": 0, "yaw": yaw}}


def question(answers, text="what is the brand?", question_id=1):
    return {"question": text, "image_id": "img", "image_width": 1000, "image_height": 500, "answers": answers, "question_id": question_id}


OCR = {"img": {"image_id": "img", "ocr_info": [
    info("Canon", x=0.2), info("Nikon", x=0.5), info("Sony", x=0.7), info("Fuji", x=0.8), info("12", x=0.9),
]}}


def test_normalization_keeps_inner_punctuation():
    assert normalize_text("  Coca-Cola. ") == "coca-cola"
    assert normalize_text("7:30!") == "7:30"
    assert normalize_text("(A) ,") == "a"


def test_character_classes():
    assert (character_class("canon"), character_class("2011"), character_class("r2d2")) == ("alpha", "digit", "mixed")


def test_modal_answer_counts_votes_and_rejects_ties():
    assert modal_answer(["Canon", "canon", "CANON", "nikon"]) == ("canon", 3)
    assert modal_answer(["canon", "nikon"]) is None
    assert modal_answer([]) is None


def test_spans_need_complete_contiguous_tokens():
    tokens = ["new", "york", "city", "new", "york"]
    assert find_spans(tokens, "new york") == [(0, 2), (3, 5)]
    assert find_spans(tokens, "york city") == [(1, 3)]
    assert find_spans(tokens, "yor") == []  # a substring of a token never matches
    assert find_spans(tokens, "new city") == []  # not contiguous


def test_item_rules():
    items, funnel = build_items([question(["canon"] * 6 + ["nikon"] * 2 + ["x", "y"])], OCR)
    assert len(items) == 1
    item = items[0]
    assert item["a_true"] == "canon" and item["item_id"] == "textvqa_1"
    assert item["a_alt"] in {"sony", "fuji"}  # nikon is an accepted answer; 12 has another class
    assert item["descriptions"]["match"] == "The relevant text in this image reads 'canon'."
    assert item["descriptions"]["mis"] == f"The relevant text in this image reads '{item['a_alt']}'."
    assert item["descriptions"]["irr"].startswith("Other text in this image reads '")
    assert item["a_alt"] not in item["descriptions"]["irr"]
    assert "nikon" not in item["descriptions"]["irr"]
    assert item["evidence_box"] == [pytest.approx({"x": 200, "y": 50, "w": 100, "h": 50})]
    assert funnel["questions"] == 1 and funnel["description_lengths_within_3_tokens"] == 1


def test_too_few_votes_is_excluded():
    assert build_items([question(["canon"] * 4 + ["a", "b", "c"])], OCR)[0] == []
    assert len(build_items([question(["canon"] * 5 + ["a", "b", "c"])], OCR)[0]) == 1


def test_answer_only_inside_a_longer_token_is_excluded():
    ocr = {"img": {"image_id": "img", "ocr_info": [info("Canonball"), info("Sony"), info("Fuji")]}}
    assert build_items([question(["canon"] * 5)], ocr)[0] == []


def test_repeated_answer_uses_the_union_of_locations():
    ocr = {"img": {"image_id": "img", "ocr_info": [info("Canon", x=0.1), info("Sony"), info("Canon", x=0.6), info("Fuji"), info("Pixel")]}}
    item = build_items([question(["canon"] * 5)], ocr)[0][0]
    assert item["repeated_answer"] and len(item["evidence_box"]) == 2


def test_alternative_must_appear_exactly_once():
    tokens = ["canon", "sony", "sony"]
    assert choose_alternative("k", "canon", tokens, {"canon"}, "what?", {0}) is None


def test_alternative_length_within_two():
    tokens = ["canon", "extraordinarily"]
    assert choose_alternative("k", "canon", tokens, {"canon"}, "what?", {0}) is None


def test_alternative_not_part_of_an_accepted_answer_or_the_question():
    tokens = ["canon", "digital", "brand"]
    assert choose_alternative("k", "canon", tokens, {"canon", "canon digital"}, "what brand?", {0}) is None


def test_fragments_of_an_answer_are_not_alternatives():
    assert choose_alternative("k", "toyota", ["toyota", "yota"], {"toyota"}, "what?", {0}) is None
    assert choose_alternative("k", "cat", ["cat", "cats"], {"cat"}, "what?", {0}) is None


def test_irrelevant_string_does_not_overlap_the_alternative():
    from kvsel.textvqa import choose_irrelevant

    assert choose_irrelevant("k", ["canon", "android", "andr"], {"canon", "android"}, {0}) is None


def test_rotated_box_is_flagged():
    ocr = {"img": {"image_id": "img", "ocr_info": [info("Canon", yaw=0.4), info("Sony"), info("Fuji"), info("Pixel")]}}
    assert build_items([question(["canon"] * 5)], ocr)[0][0]["rotated_box"] is True


def test_build_is_deterministic():
    questions = [question(["canon"] * 5, question_id=1), question(["sony"] * 5, question_id=2)]
    assert build_items(questions, OCR) == build_items(list(reversed(questions)), OCR)


def test_frozen_dev_manifest_is_kept_and_its_images_avoided(tmp_path):
    other = {**question(["canon"] * 6, question_id=2), "image_id": "other"}
    (tmp_path / "q.json").write_text(json.dumps({"data": [question(["canon"] * 6), other]}))
    ocr = {"data": [OCR["img"], {**OCR["img"], "image_id": "other"}]}
    (tmp_path / "ocr.json").write_text(json.dumps(ocr))
    dev = tmp_path / "frozen_dev.jsonl"
    dev_text = json.dumps({"item_id": "old", "image_id": "img", "builder": "textvqa-2", "split": "dev"}) + "\n"
    dev.write_text(dev_text)
    out = tmp_path / "out"
    funnel = build([tmp_path / "q.json"], [tmp_path / "ocr.json"], out, None, 1500, dev)
    assert (out / "textvqa_dev.jsonl").read_text() == dev_text
    confirm = [json.loads(line) for line in (out / "textvqa_confirm.jsonl").read_text().splitlines()]
    assert [(item["item_id"], item["split"]) for item in confirm] == [("textvqa_2", "confirm")]
    assert funnel["dev_frozen_from"]["builder"] == "textvqa-2" and funnel["modal_answer_has_5_votes"] == 2
