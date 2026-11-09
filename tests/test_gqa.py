"""GQA builder rules on a small hand-built fixture."""

import json

from kvsel.gqa import VOCABULARY, build, build_items, iter_json_object, load_questions, parse_program, split_by_image


def obj(name, attributes, x=0, y=0, w=10, h=10):
    return {"name": name, "attributes": attributes, "x": x, "y": y, "w": w, "h": h, "relations": []}


def question(answer, program_object="bus (1)", query="color", image="img", text="What color is the bus?", full="The bus is red."):
    return {
        "semantic": [
            {"operation": "select", "argument": program_object, "dependencies": []},
            {"operation": "query", "argument": query, "dependencies": [0]},
        ],
        "question": text, "fullAnswer": full, "imageId": image, "answer": answer,
    }


GRAPH = {
    "img": {
        "width": 100, "height": 100,
        "objects": {
            "1": obj("bus", ["red", "large"]),
            "2": obj("road", ["gray"], x=50, y=50),
            "3": obj("sky", ["blue"], x=5, y=5),  # overlaps the bus, so unusable as irrelevant
            "4": obj("door", ["green"], x=70, y=70),
            "5": obj("wheels", ["black"], x=30, y=30),  # plural, so unusable as irrelevant
        },
    },
    "dup": {"width": 100, "height": 100, "objects": {"1": obj("bus", ["red"]), "2": obj("bus", ["blue"], x=50)}},
}


def test_parse_program_accepts_only_the_exact_shape():
    assert parse_program(question("red")) == ("bus", "1", "color")
    assert parse_program(question("car", query="name")) is None
    with_relate = question("red")
    with_relate["semantic"].insert(1, {"operation": "relate", "argument": "x", "dependencies": [0]})
    assert parse_program(with_relate) is None
    assert parse_program(question("red", program_object="bus (1,2)")) is None


def test_item_rules():
    items, funnel = build_items({"q": question("red")}, GRAPH)
    assert len(items) == 1
    item = items[0]
    assert item["a_true"] == "red" and item["a_alt"] in VOCABULARY["color"] and item["a_alt"] != "red"
    assert item["descriptions"]["mis"] == f"The bus in this image is {item['a_alt']}."
    assert item["descriptions"]["match"] == "The bus in this image is red."
    assert item["descriptions"]["irr"] in {"The road in this image is gray.", "The door in this image is green."}
    assert item["a_alt"] not in item["descriptions"]["irr"]
    assert item["evidence_box"] == {"x": 0, "y": 0, "w": 10, "h": 10}
    assert funnel["questions"] == 1


def test_plural_verb_follows_the_dataset_wording():
    question_ = question("red", text="What color are the bus?", full="They are red.")
    assert build_items({"q": question_}, GRAPH)[0][0]["descriptions"]["match"] == "The bus in this image are red."


def test_answers_outside_the_closed_vocabulary_are_excluded():
    graph = {"img": {**GRAPH["img"], "objects": {**GRAPH["img"]["objects"], "1": obj("bus", ["brushing teeth", "tan"])}}}
    for answer in ("brushing teeth", "tan"):
        assert build_items({"q": question(answer)}, graph)[0] == []


def test_alternative_never_an_attribute_of_the_object():
    graph = {"img": {**GRAPH["img"], "objects": {**GRAPH["img"]["objects"], "1": obj("bus", ["red", "pink"])}}}
    for index in range(50):
        for item in build_items({f"q{index}": question("red")}, graph)[0]:
            assert item["a_alt"] != "pink"


def test_non_unique_referent_is_excluded():
    items, funnel = build_items({"q": question("red", image="dup")}, GRAPH)
    assert items == [] and funnel["program_select_query_color_or_material"] == 1
    assert "unique_referent_in_scene_graph" not in funnel


def test_answer_not_on_the_object_is_excluded():
    assert build_items({"q": question("purple")}, GRAPH)[0] == []


def test_build_is_deterministic():
    questions = {"q": question("red"), "r": question("red")}
    assert build_items(questions, GRAPH) == build_items(dict(reversed(list(questions.items()))), GRAPH)


def test_split_keeps_images_whole_and_labels_items():
    items = [{"item_id": str(i), "image_id": str(i // 3)} for i in range(30)]
    development, confirmatory = split_by_image(items, 6)
    assert not {i["image_id"] for i in development} & {i["image_id"] for i in confirmatory}
    assert len(development) == 6 and len(development) + len(confirmatory) == 30
    assert {i["split"] for i in development} == {"dev"} and {i["split"] for i in confirmatory} == {"confirm"}


def test_streaming_reader_matches_json_loads():
    data = {"a": question("red"), "b": {"x": [1, {"y": "}"}], "s": "a, b: c"}, "c": []}
    for text in (json.dumps(data), json.dumps(data, indent=2), "{}", " { } "):
        assert dict(iter_json_object(text)) == json.loads(text)


def test_load_questions_keeps_usable_programs_and_counts_all(tmp_path):
    path = tmp_path / "q.json"
    path.write_text(json.dumps({"q": question("red"), "r": question("car", query="name")}))
    kept, total = load_questions(path)
    assert set(kept) == {"q"} and total == 2


def test_one_item_per_object_and_attribute():
    questions = {"q2": question("red", text="Which color is the bus?"), "q1": question("red")}
    items, funnel = build_items(questions, GRAPH)
    assert [item["item_id"] for item in items] == ["q1"]
    assert funnel["description_lengths_within_3_tokens"] == 2 and funnel["one_item_per_object_and_attribute"] == 1


def test_frozen_dev_manifest_is_kept_and_its_images_avoided(tmp_path):
    graphs = {
        "img": GRAPH["img"],
        "other": {**GRAPH["img"]},
    }
    questions = {"q": question("red"), "r": question("red", image="other")}
    (tmp_path / "q.json").write_text(json.dumps(questions))
    (tmp_path / "g.json").write_text(json.dumps(graphs))
    dev = tmp_path / "frozen_dev.jsonl"
    dev_text = json.dumps({"item_id": "old", "image_id": "img", "builder": "gqa-2", "split": "dev"}) + "\n"
    dev.write_text(dev_text)
    out = tmp_path / "out"
    funnel = build(tmp_path / "q.json", tmp_path / "g.json", out, None, 600, dev)
    assert (out / "gqa_dev.jsonl").read_text() == dev_text
    confirm = [json.loads(line) for line in (out / "gqa_confirm.jsonl").read_text().splitlines()]
    assert [(item["item_id"], item["split"]) for item in confirm] == [("r", "confirm")]
    assert funnel["dev_candidates"] == 1 and funnel["confirm_pool"] == 1 and funnel["questions"] == 2
    assert funnel["dev_frozen_from"]["builder"] == "gqa-2"
