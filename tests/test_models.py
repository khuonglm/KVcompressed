"""Model families and layouts: the layout object, the registry, and model inputs unchanged on every manifest."""

import hashlib
import json
import re
from pathlib import Path

import pytest

from kvsel.items.manifest import load_items
from kvsel.models import FAMILIES
from kvsel.models.layout import Layout, Map
from kvsel.models.llava15 import LAYOUT, LLAVA, MAPPING_RULES, box_to_square, evidence_patches
from kvsel.models.llava_next import NEXT, anyres_layout
from kvsel.prompts import surface_forms

ROOT = Path(__file__).parents[1]
MANIFESTS = ROOT / "manifests"


# --- Layout ----------------------------------------------------------------------------------------------------


def test_llava_15_layout_is_one_24x24_map_without_newlines():
    assert LAYOUT == Layout((Map(0, 24, 24),)) and LAYOUT.total == 576 and LAYOUT.newlines == ()
    assert LAYOUT.features.tolist() == list(range(576))
    assert LAYOUT.positions(0)[1, 0] == 24


def test_feature_positions_skip_the_newlines():
    layout = anyres_layout(2, 2, 3)  # base 0-3, then 4 5 6 | 7, 8 9 10 | 11
    assert layout.newlines == (7, 11) and layout.total == 12
    assert layout.features.tolist() == [0, 1, 2, 3, 4, 5, 6, 8, 9, 10]
    assert layout.positions(1).tolist() == [[4, 5, 6], [8, 9, 10]]


@pytest.mark.parametrize("maps,newlines,match", [
    ((Map(0, 2, 2), Map(5, 1, 2)), (), "starts at 5"),  # the second map starts right after the first
    ((Map(0, 2, 2),), (5,), "inside the span"),  # 5 tokens: 0-4
    ((Map(0, 2, 2),), (2, 1), "sorted"),
])
def test_inconsistent_layouts_are_refused(maps, newlines, match):
    with pytest.raises(ValueError, match=match):
        Layout(maps, newlines)


def test_registry_holds_both_families():
    assert FAMILIES == {"llava": LLAVA, "llava_next": NEXT}
    assert LLAVA.revision is None and NEXT.revision


def test_only_llava_15s_implementation_names_its_grid():
    """576 and 24 live in `kvsel.models.llava15`; every other module asks the layout or imports the constants."""
    offenders = [
        f"{path.relative_to(ROOT)}:{number}"
        for path in sorted((ROOT / "src").rglob("*.py")) if path.name != "llava15.py"
        for number, line in enumerate(path.read_text().splitlines(), 1) if re.search(r"\b(576|24)\b", line)
    ]
    assert offenders == []


# --- Evidence patches, unchanged on every confirmatory item with a box -----------------------------------------

# sha256 of one JSON line [item_id, {rule: sorted patches}] per boxed item in manifest order, captured from
# kvsel.geometry before ticket 03.
PATCH_DIGESTS = {
    "gqa_confirm": (9209, "b0ca22fed9784504fee4d4a93667795d80dd9110f45d208cd52661b2b2493f54"),
    "textvqa_confirm": (5514, "f2de444ecfede92487d5975c018a5b6105a78caf6ea190da9cb9512518916722"),
}


def patches_line(item: dict) -> str:
    boxes = item["evidence_box"] if isinstance(item["evidence_box"], list) else [item["evidence_box"]]
    size = item["image_size"]
    squares = [box_to_square(box, size["width"], size["height"]) for box in boxes]
    patches = {rule: sorted({p for square in squares for p in evidence_patches(square, rule)}) for rule in MAPPING_RULES}
    return json.dumps([item["item_id"], patches]) + "\n"


@pytest.mark.parametrize("name", PATCH_DIGESTS)
def test_evidence_patches_are_unchanged(name):
    boxed = [item for item in load_items(MANIFESTS / f"{name}.jsonl") if item.get("evidence_box")]
    digest = hashlib.sha256("".join(map(patches_line, boxed)).encode()).hexdigest()
    assert (len(boxed), digest) == PATCH_DIGESTS[name]


# --- Model inputs, unchanged on every item of the four manifests -----------------------------------------------

# sha256 of one JSON line per item in manifest order: per context, the prompt string, prefix and suffix tokens,
# question and description spans, and every candidate's tokens; captured from kvsel.prompts before ticket 03.
INPUT_DIGESTS = {
    "gqa_dev": "a9ea7fc60c9c5e1258430be28456d6a6a79809a4eac27a956ad68b11ed3042df",
    "gqa_confirm": "20a37e2da0f85c5b6fdf5d3ba189e6955f9d922a20e187d1285c3d935346e652",
    "textvqa_dev": "0fad9436d0087c8606a7b0ac7840cb958f6d231ee9ce09554dab3d605ec8d528",
    "textvqa_confirm": "3ad27e5a60ef1d05e532000336d1c7ec413ed1d1541c51c1bcba319ce5b3d6cc",
}


def inputs_line(tokenizer, item: dict) -> str:
    contexts = []
    for name, tokens in LLAVA.tokens(tokenizer, item).items():
        candidates = {answer: {form: LLAVA.candidate_ids(tokenizer, tokens.prompt, surface)
                               for form, surface in surface_forms(item[answer]).items()}
                      for answer in ("a_true", "a_alt")}
        contexts.append([name, tokens.prompt, tokens.prefix_ids, tokens.suffix_ids, tokens.question_span,
                         tokens.description_span, candidates])
    return json.dumps([item["item_id"], contexts]) + "\n"


@pytest.mark.network
@pytest.mark.slow  # about six minutes for all four manifests
@pytest.mark.parametrize("name", INPUT_DIGESTS)
def test_model_inputs_are_unchanged(name):
    tokenizer = LLAVA.load_tokenizer()
    items = load_items(MANIFESTS / f"{name}.jsonl")
    lines = "".join(inputs_line(tokenizer, item) for item in items)
    assert hashlib.sha256(lines.encode()).hexdigest() == INPUT_DIGESTS[name]

