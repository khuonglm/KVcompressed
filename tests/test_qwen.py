"""Qwen2.5-VL family: M-RoPE positions, compact vs dense, prompts and the full pipeline.

Uses random weights in fp32 with the real tokenizer and image processor. The central check is that
prefix-then-replay equals the library's own full forward pass, which computes M-RoPE positions itself.
"""

import json
from dataclasses import replace

import numpy as np
import pytest
import torch
from PIL import Image

from kvsel.cache import entries_hash
from kvsel.engine import Decoder, advance, probe, replay
from kvsel.families import QWEN
from kvsel.model import bundle_from_model, full_forward_logits, load_image_processor, load_tokenizer, pixel_values, prefill
from kvsel.prompts import candidate_ids, tokenize_prompt, window_composition
from kvsel.runner import pilot_units, run_item
from kvsel.selectors import pw_attn, top_k
from kvsel.validate import check_serialization, validate

pytestmark = pytest.mark.network
LIMIT = 1e-5
QUESTION, DESCRIPTION = "What color is the bus?", "The bus in this image is blue."
GENERATED = [151644, 785, 1894, 374]  # arbitrary vocabulary ids


@pytest.fixture(scope="module")
def bundle(tiny_qwen):
    return bundle_from_model(tiny_qwen, load_tokenizer(family=QWEN), load_image_processor(family=QWEN), family=QWEN)


@pytest.fixture(scope="module")
def image():
    pixels = np.random.default_rng(0).integers(0, 255, (120, 200, 3), dtype=np.uint8)
    return Image.fromarray(pixels)


@pytest.fixture(scope="module")
def setup(bundle, image):
    tokens = tokenize_prompt(bundle.tokenizer, QUESTION, DESCRIPTION, QWEN)
    inputs = pixel_values(bundle, image)
    return tokens, inputs, prefill(bundle, tokens.prefix_ids, inputs)


def test_bundle_uses_the_qwen_family_defaults(bundle):
    decoder = bundle.decoder
    assert decoder.head_precision == "fp32" and decoder.fp32_head is None  # sliced, no 2.2 GB copy
    assert decoder.stop_ids == (151645, 151643)


def test_sliced_fp32_head_matches_a_full_fp32_copy(tiny_qwen):
    """Same arithmetic as the copy, and unlike the bf16 head its logits are not on a 0.125 grid."""
    head = torch.nn.Linear(64, 151936, bias=False)
    head.weight.data = tiny_qwen.lm_head.weight.data.to(torch.bfloat16)
    language_model = tiny_qwen.model.language_model
    hidden = (torch.randn(3, 64) * 8).to(torch.bfloat16)
    sliced = Decoder(language_model, head, "fp32", fp32_head_copy=False).logits(hidden)
    copied = Decoder(language_model, head, "fp32", fp32_head_copy=True).logits(hidden)
    native = Decoder(language_model, head, "native").logits(hidden)
    assert sliced.dtype == torch.float32 and sliced.shape == (3, 151936)
    assert torch.allclose(sliced, copied, atol=1e-5, rtol=0)
    assert not torch.equal(sliced, native)


def test_prefix_layout_and_logical_start(setup):
    tokens, inputs, prefix = setup
    assert inputs["image_grid_thw"].tolist() == [[1, 48, 48]]
    assert prefix.num_nonvisual == 15 and prefix.num_visual == 576 and prefix.length == 591
    assert prefix.logical_start == 15 + 24  # M-RoPE spends 24 positions on the 24 x 24 grid


def test_replay_matches_one_full_forward_pass(bundle, setup):
    tokens, inputs, prefix = setup
    ids = tokens.prefix_ids + tokens.suffix_ids + GENERATED
    full = full_forward_logits(bundle, ids, inputs, len(tokens.prefix_ids))
    state, suffix_logits = replay(bundle.decoder, prefix, torch.arange(576), torch.tensor([tokens.suffix_ids]), "dense")
    _, generated = advance(bundle.decoder, state, GENERATED)
    assert (full - torch.cat([suffix_logits, generated], dim=1)).abs().max() < LIMIT


def test_control_physical_positions_are_detected(bundle, setup):
    tokens, inputs, prefix = setup
    wrong = replace(prefix, logical_length=None)  # the LLaVA convention: start at the entry count
    full = full_forward_logits(bundle, tokens.prefix_ids + tokens.suffix_ids, inputs, len(tokens.prefix_ids))
    _, logits = replay(bundle.decoder, wrong, torch.arange(576), torch.tensor([tokens.suffix_ids]), "dense")
    assert (full - logits).abs().max() > LIMIT


@pytest.mark.parametrize("k", [288, 144, 72])
@pytest.mark.parametrize("kind", ["random", "concentrated", "scored"])
def test_compact_matches_dense(bundle, setup, k, kind):
    tokens, _, prefix = setup
    suffix = torch.tensor([tokens.suffix_ids])
    kept = {
        "random": lambda: torch.randperm(576, generator=torch.Generator().manual_seed(k))[:k],
        "concentrated": lambda: torch.arange(k),
        "scored": lambda: torch.from_numpy(top_k(pw_attn(probe(bundle.decoder, prefix, suffix)), k)),
    }[kind]()
    results = []
    for mode in ("compact", "dense"):
        state, logits = replay(bundle.decoder, prefix, kept, suffix, mode)
        _, generated = advance(bundle.decoder, state, GENERATED)
        results.append(torch.cat([logits, generated], dim=1))
    assert (results[0] - results[1]).abs().max() < LIMIT


def test_prefix_is_not_mutated(bundle, setup):
    tokens, _, prefix = setup
    before = entries_hash(prefix.entries)
    probe(bundle.decoder, prefix, torch.tensor([tokens.suffix_ids]))
    replay(bundle.decoder, prefix, torch.arange(144), torch.tensor([tokens.suffix_ids]), "compact")
    assert entries_hash(prefix.entries) == before


def test_prompt_follows_the_official_chat_template(bundle):
    from huggingface_hub import hf_hub_download

    template = json.load(open(hf_hub_download(QWEN.model_id, "chat_template.json")))["chat_template"]
    text = QWEN.user_text(QUESTION, DESCRIPTION)
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
    official = bundle.tokenizer.apply_chat_template(messages, chat_template=template, add_generation_prompt=True, tokenize=False)
    ours = QWEN.build_prompt(QUESTION, DESCRIPTION)
    assert ours == official.replace("<|image_pad|>", "<|image_pad|>" * 576)


def test_spans_and_window(bundle):
    plain = tokenize_prompt(bundle.tokenizer, QUESTION, family=QWEN)
    described = tokenize_prompt(bundle.tokenizer, QUESTION, DESCRIPTION, QWEN)
    span = lambda t: t.suffix_ids[t.question_span[0]:t.question_span[1]]
    assert span(plain) == span(described) and bundle.tokenizer.decode(span(plain)).strip() == QUESTION
    assert plain.prefix_ids == described.prefix_ids and plain.description_span is None
    assert bundle.tokenizer.decode(described.suffix_ids[slice(*described.description_span)]).startswith("Description: ")
    # Qwen merges '?' with the newline, so the window's first token is the question's last.
    assert window_composition(described) == {"question": 1 / 16, "description": 0, "instruction": 15 / 16}


def test_candidates_have_no_leading_space_and_end_with_im_end(bundle):
    prompt = QWEN.build_prompt(QUESTION)
    ids = candidate_ids(bundle.tokenizer, prompt, "Pink", QWEN)
    assert bundle.tokenizer.decode(ids[:-1]) == "Pink" and ids[-1] == 151645


def make_item(index):
    return {
        "item_id": f"q{index}", "image_id": f"img{index}", "image_file": f"img{index}.jpg", "dataset": "gqa",
        "split": "dev", "question": QUESTION, "a_true": "red", "a_alt": "blue",
        "descriptions": {"mis": DESCRIPTION, "match": "The bus in this image is red.", "irr": "The road in this image is gray."},
    }


def test_full_grid_runs_through_the_runner(bundle, image):
    result, attention = run_item(bundle, make_item(0), image, pilot_units())
    assert result["model"] == "qwen" and result["template"] == "v3.0-qwen"
    assert set(attention) == {"Q", "mis", "match", "irr"}
    for row in result["rows"]:
        assert set(row["scores"]["true"]) == {"lower", "capitalized", "upper"}
        if row["selector"] != "full":
            assert len(row["kept"]) == 144


def test_validate_runs_on_qwen(bundle, image, tmp_path):
    items = [make_item(i) for i in range(2)]
    for item in items:
        image.save(tmp_path / item["image_file"])
    report = validate(bundle, items, tmp_path, noise_items=2, sparse_items=1, repeat_items=1)
    assert report["model"] == "qwen" and report["noise_floor_native_head"] == report["noise_floor"]
    assert report["E1"]["pass"] and report["E4"]["pass"] and report["E7"]["pass"]
    assert report["E2"]["by_suffix"]["Q"]["compact_vs_dense_max"] == 0


@pytest.mark.parametrize("manifest", ["manifests/gqa_dev.jsonl", "manifests/textvqa_dev.jsonl"])
def test_serialization_on_real_items(bundle, manifest):
    with open(manifest) as handle:
        items = [json.loads(line) for _, line in zip(range(60), handle)]
    result = check_serialization(bundle.tokenizer, items, QWEN)
    assert result["pass"], result["failures"][:3]
