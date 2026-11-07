"""GQA direct-attribute items (setting 1). A pure function of the raw files, version and seed."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

from .normalize import normalize
from .prompts import tokenize_prompt

BUILDER_VERSION = "gqa-2"
SPLIT_SEED = 0
# Closed vocabulary for a_true, a_alt and irrelevant values, so a wrong value is clearly wrong. Raw
# dataset answers also include activities ("brushing teeth"), vague values ("dark") and near
# neighbours ("tan", "beige", "light blue"), which would make a_alt arguably acceptable.
VOCABULARY = {
    "color": ("black", "blue", "brown", "gray", "green", "orange", "pink", "purple", "red", "white", "yellow"),
    "material": ("brick", "concrete", "glass", "leather", "metal", "plastic", "porcelain", "wood"),
}
ATTRIBUTE_KINDS = tuple(VOCABULARY)
MAX_DESCRIPTION_TOKEN_SPREAD = 3
_SELECT = re.compile(r"^(?P<name>.+) \((?P<id>\d+)\)$")


def item_rng(item_id: str) -> random.Random:
    """Choices for an item do not depend on which other items exist or on build order."""
    return random.Random(hashlib.sha256(f"{item_id}|{BUILDER_VERSION}".encode()).hexdigest())


def parse_program(question: dict) -> tuple[str, str, str] | None:
    """Returns (object name, object id, attribute kind) for exactly select -> query(color|material)."""
    semantic = question["semantic"]
    if [step["operation"] for step in semantic] != ["select", "query"]:
        return None
    kind = semantic[1]["argument"]
    match = _SELECT.match(semantic[0]["argument"])
    if kind not in ATTRIBUTE_KINDS or match is None:
        return None
    return match["name"], match["id"], kind


def looks_plural(name: str) -> bool:
    return name.endswith("s") and not name.endswith(("ss", "us", "is"))


def verb_for(question: dict) -> str:
    """The dataset's own wording settles agreement for the queried object."""
    text = f" {question['question']} {question['fullAnswer']} ".lower()
    return "are" if " are " in text else "is"


def boxes_overlap(a: dict, b: dict) -> bool:
    return a["x"] < b["x"] + b["w"] and b["x"] < a["x"] + a["w"] and a["y"] < b["y"] + b["h"] and b["y"] < a["y"] + a["h"]


def choose_alternative(item_id: str, a_true: str, attributes: list[str], question_text: str, values) -> str | None:
    """Not the true value, not any attribute of this object, not a word in the question."""
    excluded = {normalize(a_true), *(normalize(attribute) for attribute in attributes)}
    question_words = set(normalize(question_text).split())
    options = [v for v in values if normalize(v) not in excluded and not set(normalize(v).split()) & question_words]
    return item_rng(item_id).choice(options) if options else None


def choose_irrelevant(item_id: str, object_id: str, graph: dict, values, avoid: set[str]) -> tuple[str, str] | None:
    """Another uniquely named singular object with a known value of the same kind, not overlapping."""
    target = graph["objects"][object_id]
    names = Counter(obj["name"] for obj in graph["objects"].values())
    allowed = {normalize(v): v for v in values}
    options = []
    for other_id in sorted(graph["objects"]):
        other = graph["objects"][other_id]
        if other_id == object_id or names[other["name"]] != 1 or other["name"] == target["name"] or looks_plural(other["name"]):
            continue
        if boxes_overlap(target, other):
            continue
        for attribute in sorted(other["attributes"]):
            key = normalize(attribute)
            if key in allowed and key not in avoid:
                options.append((other["name"], attribute))
    return item_rng(f"{item_id}|irr").choice(options) if options else None


def build_items(questions: dict, graphs: dict, tokenizer=None) -> tuple[list[dict], dict[str, int]]:
    funnel: Counter[str] = Counter()
    funnel["questions"] = len(questions)
    items = []
    for item_id in sorted(questions):
        question = questions[item_id]
        parsed = parse_program(question)
        if parsed is None:
            continue
        funnel["program_select_query_color_or_material"] += 1
        name, object_id, kind = parsed
        graph = graphs.get(question["imageId"])
        if graph is None or object_id not in graph["objects"]:
            continue
        funnel["object_in_scene_graph"] += 1
        target = graph["objects"][object_id]
        if sum(obj["name"] == target["name"] for obj in graph["objects"].values()) != 1:
            continue
        funnel["unique_referent_in_scene_graph"] += 1
        a_true = question["answer"]
        if a_true not in VOCABULARY[kind]:
            continue
        funnel["answer_in_closed_vocabulary"] += 1
        if normalize(a_true) not in {normalize(attribute) for attribute in target["attributes"]}:
            continue
        funnel["answer_is_attribute_of_object"] += 1
        a_alt = choose_alternative(item_id, a_true, target["attributes"], question["question"], VOCABULARY[kind])
        if a_alt is None:
            continue
        funnel["alternative_found"] += 1
        irrelevant = choose_irrelevant(item_id, object_id, graph, VOCABULARY[kind], {normalize(a_true), normalize(a_alt)})
        if irrelevant is None:
            continue
        funnel["irrelevant_object_found"] += 1
        verb = verb_for(question)
        descriptions = {
            "mis": f"The {target['name']} in this image {verb} {a_alt}.",
            "match": f"The {target['name']} in this image {verb} {a_true}.",
            "irr": f"The {irrelevant[0]} in this image is {irrelevant[1]}.",
        }
        if tokenizer is not None:
            lengths = [len(tokenize_prompt(tokenizer, question["question"], d).suffix_ids) for d in descriptions.values()]
            if max(lengths) - min(lengths) > MAX_DESCRIPTION_TOKEN_SPREAD:
                continue
        funnel["description_lengths_within_3_tokens"] += 1
        items.append({
            "item_id": item_id,
            "image_id": question["imageId"],
            "image_file": f"{question['imageId']}.jpg",
            "dataset": "gqa",
            "conflict_type": "same_object_attribute",
            "builder": BUILDER_VERSION,
            "attribute_kind": kind,
            "question": question["question"],
            "object": target["name"],
            "object_id": object_id,
            "a_true": a_true,
            "a_alt": a_alt,
            "descriptions": descriptions,
            "evidence_box": {key: target[key] for key in ("x", "y", "w", "h")},
            "image_size": {"width": graph["width"], "height": graph["height"]},
        })
    return items, dict(funnel)


def split_by_image(items: list[dict], dev_items: int) -> tuple[list[dict], list[dict]]:
    """Whole images go to one split. Items within a split follow the shuffled image order."""
    by_image: dict[str, list[dict]] = {}
    for item in items:
        by_image.setdefault(item["image_id"], []).append(item)
    image_ids = sorted(by_image)
    random.Random(SPLIT_SEED).shuffle(image_ids)
    development, confirmatory = [], []
    for image_id in image_ids:
        target = development if len(development) < dev_items else confirmatory
        target.extend({**item, "split": "dev" if target is development else "confirm"} for item in by_image[image_id])
    return development, confirmatory


def write_jsonl(path: Path, items: list[dict]) -> str:
    text = "".join(json.dumps(item, sort_keys=True) + "\n" for item in items)
    path.write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


def build(questions_path: Path, graphs_path: Path, out_dir: Path, tokenizer, dev_items: int) -> dict:
    questions = json.loads(Path(questions_path).read_text())
    graphs = json.loads(Path(graphs_path).read_text())
    items, funnel = build_items(questions, graphs, tokenizer)
    development, confirmatory = split_by_image(items, dev_items)
    out_dir.mkdir(parents=True, exist_ok=True)
    funnel.update({
        "unique_images": len({item["image_id"] for item in items}),
        "dev_candidates": len(development),
        "dev_images": len({item["image_id"] for item in development}),
        "confirm_pool": len(confirmatory),
        "confirm_images": len({item["image_id"] for item in confirmatory}),
        "by_kind": dict(Counter(item["attribute_kind"] for item in items)),
        "builder": BUILDER_VERSION,
        "source": Path(questions_path).name,
        "dev_sha256": write_jsonl(out_dir / "gqa_dev.jsonl", development),
        "confirm_sha256": write_jsonl(out_dir / "gqa_confirm.jsonl", confirmatory),
    })
    (out_dir / "gqa_funnel.json").write_text(json.dumps(funnel, indent=2) + "\n")
    return funnel
