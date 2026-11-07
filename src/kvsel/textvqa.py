"""TextVQA scene-text items (setting 2). A pure function of the raw files, version and seed."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from .gqa import MAX_DESCRIPTION_TOKEN_SPREAD, split_by_image, write_jsonl
from .prompts import tokenize_prompt

BUILDER_VERSION = "textvqa-2"
MIN_ANSWER_VOTES = 3
MAX_ALT_LENGTH_DIFFERENCE = 2
MIN_ALT_LENGTH = 2


def normalize_text(text: str) -> str:
    """Lower case, without punctuation at token edges. Inner punctuation stays: '7:30', 'coca-cola'."""
    tokens = (token.strip(".,;:!?'\"()[]{}") for token in text.lower().split())
    return " ".join(token for token in tokens if token)


def character_class(text: str) -> str:
    compact = text.replace(" ", "")
    if compact.isalpha():
        return "alpha"
    if compact.isdigit():
        return "digit"
    return "mixed"


def item_rng(key: str) -> random.Random:
    return random.Random(hashlib.sha256(f"{key}|{BUILDER_VERSION}".encode()).hexdigest())


def modal_answer(answers: list[str]) -> tuple[str, int] | None:
    """The most common normalised answer and its votes. An exact tie for first has no clear answer."""
    ranked = Counter(answer for answer in map(normalize_text, answers) if answer).most_common(2)
    if not ranked or (len(ranked) == 2 and ranked[0][1] == ranked[1][1]):
        return None
    return ranked[0]


def find_spans(tokens: list[str], answer: str) -> list[tuple[int, int]]:
    """Every contiguous run of complete OCR tokens that equals the answer. Substrings never match."""
    words = answer.split()
    return [
        (start, start + len(words))
        for start in range(len(tokens) - len(words) + 1)
        if tokens[start : start + len(words)] == words
    ]


def box_in_pixels(info: dict, width: int, height: int) -> dict:
    box = info["bounding_box"]
    return {
        "x": box["top_left_x"] * width,
        "y": box["top_left_y"] * height,
        "w": box["width"] * width,
        "h": box["height"] * height,
    }


def is_rotated(info: dict) -> bool:
    box = info["bounding_box"]
    return any(abs(box.get(key, 0)) > 1e-6 for key in ("rotation", "roll", "pitch", "yaw"))


def overlaps_any(token: str, texts: set[str]) -> bool:
    """OCR often splits words, so 'yota' next to 'toyota' is a fragment of the answer, not another string."""
    return any(token in text or text in token for text in texts)


def choose_alternative(key: str, a_true: str, tokens: list[str], accepted: set[str], question: str, span_positions: set[int]) -> str | None:
    """A complete OCR token that appears exactly once, elsewhere, of the same class and similar length."""
    counts = Counter(tokens)
    question_words = set(normalize_text(question).split())
    options = sorted({
        token
        for position, token in enumerate(tokens)
        if position not in span_positions
        and counts[token] == 1
        and len(token) >= MIN_ALT_LENGTH
        and character_class(token) == character_class(a_true)
        and abs(len(token) - len(a_true)) <= MAX_ALT_LENGTH_DIFFERENCE
        and not overlaps_any(token, accepted)
        and token not in question_words
    })
    return item_rng(key).choice(options) if options else None


def choose_irrelevant(key: str, tokens: list[str], avoid: set[str], span_positions: set[int]) -> str | None:
    options = sorted({
        token
        for position, token in enumerate(tokens)
        if position not in span_positions and len(token) >= MIN_ALT_LENGTH and not overlaps_any(token, avoid)
    })
    return item_rng(f"{key}|irr").choice(options) if options else None


def build_items(questions: list[dict], ocr_by_image: dict[str, dict], tokenizer=None) -> tuple[list[dict], dict[str, int]]:
    funnel: Counter[str] = Counter()
    funnel["questions"] = len(questions)
    items = []
    for question in sorted(questions, key=lambda q: q["question_id"]):
        ocr = ocr_by_image.get(question["image_id"])
        if ocr is None or not ocr["ocr_info"]:
            continue
        funnel["image_has_ocr"] += 1
        modal = modal_answer(question["answers"])
        if modal is None or modal[1] < MIN_ANSWER_VOTES:
            continue
        funnel["modal_answer_has_3_votes"] += 1
        a_true = modal[0]
        infos = ocr["ocr_info"]
        tokens = [normalize_text(info["word"]) for info in infos]
        spans = find_spans(tokens, a_true)
        if not spans:
            continue
        funnel["answer_matches_complete_ocr_tokens"] += 1
        item_id = f"textvqa_{question['question_id']}"
        accepted = {answer for answer in map(normalize_text, question["answers"]) if answer}
        span_positions = {position for start, stop in spans for position in range(start, stop)}
        a_alt = choose_alternative(item_id, a_true, tokens, accepted, question["question"], span_positions)
        if a_alt is None:
            continue
        funnel["alternative_found"] += 1
        irrelevant = choose_irrelevant(item_id, tokens, accepted | {a_alt}, span_positions)
        if irrelevant is None:
            continue
        funnel["irrelevant_text_found"] += 1
        descriptions = {
            "mis": f"The relevant text in this image reads '{a_alt}'.",
            "match": f"The relevant text in this image reads '{a_true}'.",
            "irr": f"Other text in this image reads '{irrelevant}'.",
        }
        if tokenizer is not None:
            lengths = [len(tokenize_prompt(tokenizer, question["question"], d).suffix_ids) for d in descriptions.values()]
            if max(lengths) - min(lengths) > MAX_DESCRIPTION_TOKEN_SPREAD:
                continue
        funnel["description_lengths_within_3_tokens"] += 1
        width, height = question["image_width"], question["image_height"]
        evidence = [
            box_in_pixels(infos[position], width, height) for position in sorted(span_positions)
        ]
        items.append({
            "item_id": item_id,
            "image_id": question["image_id"],
            "image_file": f"{question['image_id']}.jpg",
            "dataset": "textvqa",
            "conflict_type": "other_region_text",
            "builder": BUILDER_VERSION,
            "question": question["question"],
            "answers": question["answers"],
            "a_true": a_true,
            "a_alt": a_alt,
            "votes": modal[1],
            "repeated_answer": len(spans) > 1,
            "rotated_box": any(is_rotated(infos[position]) for position in span_positions),
            "descriptions": descriptions,
            "evidence_box": evidence,
            "image_size": {"width": width, "height": height},
        })
    return items, dict(funnel)


def build(question_paths: list[Path], ocr_paths: list[Path], out_dir: Path, tokenizer, dev_items: int) -> dict:
    questions = [q for path in question_paths for q in json.loads(Path(path).read_text())["data"]]
    ocr_by_image = {e["image_id"]: e for path in ocr_paths for e in json.loads(Path(path).read_text())["data"]}
    items, funnel = build_items(questions, ocr_by_image, tokenizer)
    development, confirmatory = split_by_image(items, dev_items)
    out_dir.mkdir(parents=True, exist_ok=True)
    funnel.update({
        "unique_images": len({item["image_id"] for item in items}),
        "dev_candidates": len(development),
        "dev_images": len({item["image_id"] for item in development}),
        "confirm_pool": len(confirmatory),
        "confirm_images": len({item["image_id"] for item in confirmatory}),
        "repeated_answers": sum(item["repeated_answer"] for item in items),
        "rotated_boxes": sum(item["rotated_box"] for item in items),
        "by_character_class": dict(Counter(character_class(item["a_true"]) for item in items)),
        "multi_token_answers": sum(" " in item["a_true"] for item in items),
        "builder": BUILDER_VERSION,
        "source": [Path(path).name for path in question_paths],
        "dev_sha256": write_jsonl(out_dir / "textvqa_dev.jsonl", development),
        "confirm_sha256": write_jsonl(out_dir / "textvqa_confirm.jsonl", confirmatory),
    })
    (out_dir / "textvqa_funnel.json").write_text(json.dumps(funnel, indent=2) + "\n")
    return funnel

