"""E1: serialization, on the real tokenizer."""

import pytest

from kvsel.prompts import (
    AUTHORITY, build_prompt, candidate_ids, surface_forms, tokenize_prompt, window_composition,
)

pytestmark = pytest.mark.network


@pytest.fixture(scope="module")
def tokenizer():
    from kvsel.model import load_tokenizer

    return load_tokenizer()


QUESTIONS = [
    ("What color is the bus?", "The bus in this image is blue."),
    ("What material is the large table in the kitchen made of?", "The table in this image is metal."),
    ("what is the brand of this camera?", "The relevant text in this image reads 'nikon'."),
]


@pytest.mark.parametrize("question,description", QUESTIONS)
def test_prefix_identical_with_and_without_description(tokenizer, question, description):
    plain = tokenize_prompt(tokenizer, question)
    with_description = tokenize_prompt(tokenizer, question, description)
    assert plain.prefix_ids == with_description.prefix_ids
    assert len(plain.prefix_ids) == 581
    assert plain.prefix_ids[0] == 1


@pytest.mark.parametrize("question,description", QUESTIONS)
def test_question_span_is_the_same_question_in_both_suffixes(tokenizer, question, description):
    plain = tokenize_prompt(tokenizer, question)
    with_description = tokenize_prompt(tokenizer, question, description)
    span = lambda t: t.suffix_ids[t.question_span[0] : t.question_span[1]]
    assert span(plain) == span(with_description)
    assert tokenizer.decode(span(plain)).strip() == question


@pytest.mark.parametrize("question,description", QUESTIONS)
def test_window_is_entirely_fixed_instruction(tokenizer, question, description):
    for tokens in (tokenize_prompt(tokenizer, question), tokenize_prompt(tokenizer, question, description)):
        assert window_composition(tokens) == {"question": 0, "description": 0, "instruction": 1}
        assert tokenizer.decode(tokens.suffix_ids[-16:]).strip().startswith("Answer the question")


def test_template_round_trip(tokenizer):
    tokens = tokenize_prompt(tokenizer, *QUESTIONS[0])
    text = tokenizer.decode(tokens.suffix_ids)
    assert "Description: The bus in this image is blue." in text and AUTHORITY in text
    assert text.rstrip().endswith("ASSISTANT:")


@pytest.mark.parametrize("answer", ["red", "dark brown", "2011", "coca-cola", "New York"])
def test_candidate_tokens_end_with_eos(tokenizer, answer):
    prompt = build_prompt("What is it?")
    for surface in surface_forms(answer).values():
        ids = candidate_ids(tokenizer, prompt, surface)
        assert ids[-1] == 2 and len(ids) >= 2


def test_casing_changes_tokens(tokenizer):
    prompt = build_prompt("What is it?")
    assert len(candidate_ids(tokenizer, prompt, "canon")) == 2
    assert len(candidate_ids(tokenizer, prompt, "Canon")) == 3
