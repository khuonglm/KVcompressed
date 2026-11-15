"""Prompt tokenization through the processor, which expands the image token for each image."""

from __future__ import annotations

from dataclasses import replace

from kvsel import NUM_VISUAL
from kvsel.families import LLAVA
from kvsel.prompts import Tokens
from kvsel.prompts import tokenize_prompt as llava_tokens
from kvsel.runner import Q

from .family import NEXT
from .layout import visual_span


def tokenize_prompt(processor, image, question: str, description: str | None = None) -> tuple[Tokens, dict]:
    """Returns (tokens, processor inputs). Spans come from kvsel's tokenize_prompt on the same text, same tokenizer."""
    reference = llava_tokens(processor.tokenizer, question, description, LLAVA)
    prompt = NEXT.build_prompt(question, description)
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    ids = inputs["input_ids"][0].tolist()
    start, stop = visual_span(ids, NEXT.image_token_id)
    if ids[:start] != reference.prefix_ids[:-NUM_VISUAL] or ids[stop:] != reference.suffix_ids:
        raise ValueError("the processor's expansion changed the tokens outside the image span")
    return replace(reference, prompt=prompt, prefix_ids=ids[:stop]), inputs


def contexts_for(processor, item: dict, image) -> tuple[dict[str, Tokens], dict]:
    """Tokens for Q and each description condition, all sharing Q's prefix, and Q's processor inputs."""
    tokens, inputs = {}, None
    for name, description in ((Q, None), *item["descriptions"].items()):
        tokens[name], context_inputs = tokenize_prompt(processor, image, item["question"], description)
        inputs = inputs or context_inputs
        if tokens[name].prefix_ids != tokens[Q].prefix_ids:
            raise ValueError(f"the {name} prefix differs from Q's")
    return tokens, inputs
