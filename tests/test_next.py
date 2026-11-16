"""LLaVA-NeXT prefix, layout and Level-1 invariants on a tiny NeXT model with the real processor."""

import numpy as np
import pytest
import torch
from PIL import Image

from kvsel import IMAGE_TOKEN
from kvsel.cache import compact, dropped_columns, entries_hash
from kvsel.engine import Prefix, advance, greedy, probe, replay
from kvsel.families import LLAVA
from kvsel_next.family import NEXT
from kvsel_next.model import load_processor, prefill
from kvsel_next.prompts import contexts_for, tokenize_prompt
from kvsel_next.selection import SHARES, kept_sets

pytestmark = pytest.mark.network

ITEM = {
    "question": "What color is the bus?",
    "descriptions": {
        "mis": "The bus in this image is blue.", "match": "The bus in this image is red.",
        "irr": "The road in this image is gray.",
    },
}
# (width, height) -> high-resolution grid after unpadding and total visual tokens, worked by hand:
# 560 x 300 picks the 336 x 672 canvas (h x w), a 24 x 48 map; scale 24 / 300 makes it 44.8 wide, which
# transformers truncates (int(round(44.8, 7))) to 44, so padding is 2 a side: 576 + 24 * 45 = 1656.
# 300 x 560 is the transpose: 576 + 44 * 25 = 1676.
# (Images smaller than every canvas tie and all take the first, 336 x 672, whatever their aspect.)
ASPECTS = {"wide": ((560, 300), (24, 44), 1656), "tall": ((300, 560), (44, 24), 1676)}


def image(size):
    return Image.fromarray(np.random.default_rng(0).integers(0, 255, (size[1], size[0], 3), dtype=np.uint8))


@pytest.fixture(scope="module")
def processor():
    return load_processor()


@pytest.fixture(scope="module")
def bundle(tiny_llava_next, processor):
    from kvsel_next.model import make_bundle

    return make_bundle(tiny_llava_next, processor)


@pytest.fixture(scope="module", params=sorted(ASPECTS))
def aspect(request, bundle, processor):
    size, grid, total = ASPECTS[request.param]
    tokens, inputs = contexts_for(processor, ITEM, image(size))
    prefix, layout = prefill(bundle, tokens["Q"].prefix_ids, inputs)
    return {"grid": grid, "total": total, "tokens": tokens, "inputs": inputs, "prefix": prefix, "layout": layout}


# --- Prompt and visual span ------------------------------------------------------------------------


@pytest.mark.parametrize("description", [None, *ITEM["descriptions"].values()])
def test_prompt_text_is_llava_15_template_apart_from_the_image_expansion(description):
    next_before, next_after = NEXT.build_prompt(ITEM["question"], description).split(IMAGE_TOKEN)
    llava = LLAVA.build_prompt(ITEM["question"], description)
    assert llava.startswith(next_before) and llava.endswith(next_after)
    assert llava == next_before + IMAGE_TOKEN * 576 + next_after
    assert NEXT.template_version == "v3.1"


def test_processor_text_after_the_image_span_is_llava_15s(processor):
    """Decoded from the processor's own expansion, for Q and each description condition."""
    tokens, _ = contexts_for(processor, ITEM, image((560, 300)))
    for name, description in (("Q", None), *ITEM["descriptions"].items()):
        after = LLAVA.build_prompt(ITEM["question"], description).split(IMAGE_TOKEN)[-1]
        assert processor.tokenizer.decode(tokens[name].suffix_ids) == after


def test_next_tokenizer_drops_the_space_token_before_the_newline(processor):
    """The text matches LLaVA-1.5's; NeXT's own tokenizer starts the suffix at the newline, one token shorter."""
    tokens, _ = tokenize_prompt(processor, image((560, 300)), ITEM["question"], ITEM["descriptions"]["mis"])
    assert processor.tokenizer.convert_ids_to_tokens(tokens.suffix_ids[:2]) == ["<0x0A>", "Un"]
    assert processor.tokenizer.decode(tokens.suffix_ids[slice(*tokens.question_span)]).strip() == ITEM["question"]


def test_every_context_shares_the_question_only_prefix(aspect):
    tokens = aspect["tokens"]
    assert set(tokens) == {"Q", "mis", "match", "irr"}
    assert all(t.prefix_ids == tokens["Q"].prefix_ids for t in tokens.values())
    assert all(t.suffix_ids != tokens["Q"].suffix_ids for name, t in tokens.items() if name != "Q")


def test_prefix_is_text_then_the_whole_visual_span(aspect):
    ids, prefix = aspect["tokens"]["Q"].prefix_ids, aspect["prefix"]
    assert prefix.num_visual == aspect["total"] and prefix.length == len(ids)
    assert ids[prefix.num_nonvisual:] == [NEXT.image_token_id] * aspect["total"]
    assert NEXT.image_token_id not in ids[: prefix.num_nonvisual]


# --- Layout ----------------------------------------------------------------------------------------


def test_layout_record(aspect):
    layout = aspect["layout"]
    rows, cols = aspect["grid"]
    assert (layout.base, layout.grid, layout.total) == (24, aspect["grid"], aspect["total"])
    assert layout.newlines == tuple(576 + r * (cols + 1) + cols for r in range(rows))
    positions = layout.positions()
    assert len(positions) == aspect["total"]
    assert positions[0] == ("base", 0, 0) and positions[575] == ("base", 23, 23)
    assert positions[576] == ("high", 0, 0) and positions[576 + cols] == ("newline", 0, cols)
    assert positions[-2] == ("high", rows - 1, cols - 1) and positions[-1] == ("newline", rows - 1, cols)


def test_newline_positions_hold_the_models_newline_embedding(aspect, bundle):
    inputs, layout = aspect["inputs"], aspect["layout"]
    with torch.inference_mode():
        features = bundle.model.model.get_image_features(inputs["pixel_values"], inputs["image_sizes"])[0]
    newline = bundle.model.model.image_newline
    is_newline = (features == newline).all(dim=1)
    assert features.shape[0] == layout.total
    assert is_newline.nonzero().squeeze(1).tolist() == list(layout.newlines)


def test_layout_that_disagrees_with_the_span_is_refused(bundle, processor):
    tokens, inputs = tokenize_prompt(processor, image((560, 300)), ITEM["question"])
    inputs["image_sizes"] = torch.tensor([[560, 300]])  # the tall image's layout: 1676 tokens, span has 1656
    with pytest.raises(ValueError, match="1656.*1676"):
        prefill(bundle, tokens.prefix_ids, inputs)


def test_non_contiguous_or_truncated_span_is_refused(bundle, processor):
    tokens, inputs = tokenize_prompt(processor, image((560, 300)), ITEM["question"])
    ids = tokens.prefix_ids
    broken = ids[:-10] + [29871] + ids[-9:]
    with pytest.raises(ValueError, match="contiguous"):
        prefill(bundle, broken, inputs)
    with pytest.raises(ValueError, match="end with"):
        prefill(bundle, ids + [29871], inputs)


# --- Level 1 on the NeXT layout (same controls as tests/test_level1.py) ----------------------------

LIMIT = 1e-5
GENERATED = torch.randint(0, 500, (10,), generator=torch.Generator().manual_seed(4)).tolist()


def level1_kept(layout, share, kind, decoder=None, prefix=None, suffix=None):
    """Feature tokens chosen by ``kind`` plus every newline, as NeXT's selection keeps them."""
    newlines = torch.tensor(layout.newlines)
    features = torch.tensor([i for i in range(layout.total) if i not in set(layout.newlines)])
    k = round(share * len(features))
    if kind == "random":
        chosen = features[torch.randperm(len(features), generator=torch.Generator().manual_seed(5))[:k]]
    elif kind == "concentrated":
        chosen = features[:k]
    elif kind == "tail":
        chosen = features[-k:]
    else:  # scored: NeXT's PW-attn selection, which keeps the newlines itself
        return torch.from_numpy(kept_sets(layout, probe(decoder, prefix, suffix))[share])
    return torch.cat([chosen, newlines])


def suffix_of(aspect, context="mis"):
    return torch.tensor([aspect["tokens"][context].suffix_ids])


def all_logits(decoder, prefix, kept, suffix, mode):
    state, logits = replay(decoder, prefix, kept, suffix, mode)
    _, generated = advance(decoder, state, GENERATED)
    return torch.cat([logits, generated], dim=1)


def perturbed(prefix, columns):
    generator = torch.Generator().manual_seed(8)
    entries = []
    for key, value in prefix.entries:
        key, value = key.clone(), value.clone()
        key[:, :, columns] += 5 * torch.randn(key[:, :, columns].shape, generator=generator)
        value[:, :, columns] += 5 * torch.randn(value[:, :, columns].shape, generator=generator)
        entries.append((key, value))
    return Prefix(tuple(entries), prefix.num_nonvisual, prefix.num_visual)


@pytest.mark.parametrize("share", SHARES)
@pytest.mark.parametrize("kind", ["random", "concentrated", "tail", "scored"])
def test_compact_matches_dense(bundle, aspect, share, kind):
    decoder, prefix, suffix = bundle.decoder, aspect["prefix"], suffix_of(aspect)
    kept = level1_kept(aspect["layout"], share, kind, decoder, prefix, suffix)
    difference = all_logits(decoder, prefix, kept, suffix, "compact") - all_logits(decoder, prefix, kept, suffix, "dense")
    assert difference.abs().max() < LIMIT


def test_control_physical_positions_are_detected(bundle, aspect):
    decoder, prefix, suffix = bundle.decoder, aspect["prefix"], suffix_of(aspect)
    kept = level1_kept(aspect["layout"], 0.25, "random")
    entries = compact(prefix.entries, kept, prefix.num_nonvisual)
    physical = entries[0][0].shape[2]
    hidden, _, _ = decoder.step(suffix, entries, physical, physical)  # logical start is wrong on purpose
    _, reference = replay(decoder, prefix, kept, suffix, "dense")
    assert (decoder.logits(hidden) - reference).abs().max() > LIMIT


def test_control_dropped_entries_cannot_leak(bundle, aspect):
    decoder, prefix, suffix = bundle.decoder, aspect["prefix"], suffix_of(aspect)
    kept = level1_kept(aspect["layout"], 0.25, "random")
    columns = dropped_columns(kept, prefix.num_nonvisual, prefix.num_visual)
    changed = all_logits(decoder, perturbed(prefix, columns), kept, suffix, "dense")
    assert torch.equal(changed, all_logits(decoder, prefix, kept, suffix, "dense"))


def test_control_kept_entries_matter(bundle, aspect):
    decoder, prefix, suffix = bundle.decoder, aspect["prefix"], suffix_of(aspect)
    kept = level1_kept(aspect["layout"], 0.25, "random")
    changed = all_logits(decoder, perturbed(prefix, prefix.num_nonvisual + kept), kept, suffix, "dense")
    assert (changed - all_logits(decoder, prefix, kept, suffix, "dense")).abs().max() > LIMIT


def test_prefix_is_not_mutated(bundle, aspect):
    decoder, prefix, suffix = bundle.decoder, aspect["prefix"], suffix_of(aspect)
    before = entries_hash(prefix.entries)
    probe(decoder, prefix, suffix)
    for mode in ("compact", "dense"):
        state, _ = replay(decoder, prefix, level1_kept(aspect["layout"], 0.25, "random"), suffix, mode)
        greedy(decoder, state, max_new=3)
    assert entries_hash(prefix.entries) == before


def test_one_probe_per_context_gives_every_shares_kept_set(bundle, aspect):
    decoder, prefix, layout = bundle.decoder, aspect["prefix"], aspect["layout"]
    for context in ("Q", "mis", "match", "irr"):
        suffix = suffix_of(aspect, context)
        sets = kept_sets(layout, probe(decoder, prefix, suffix))
        assert set(sets) == set(SHARES)
        for kept in sets.values():
            state, _ = replay(decoder, prefix, torch.from_numpy(kept), suffix, "compact")
            assert state.physical_next == prefix.num_nonvisual + len(kept) + suffix.shape[1]


# --- Numerics and load checks ----------------------------------------------------------------------


def test_numerics_match_llava_15():
    assert NEXT.dtype == torch.float16 and NEXT.default_head_precision == "fp32"


def test_bundle_has_an_fp32_head(bundle, aspect):
    """fp32 logits from the fp32-cast hidden state, through the sliced head (no fp32 copy kept)."""
    assert bundle.decoder.head_precision == "fp32" and bundle.decoder.fp32_head is None
    hidden = torch.randn(1, 2, bundle.model.config.text_config.hidden_size, dtype=bundle.decoder.dtype)
    assert bundle.decoder.logits(hidden).dtype == torch.float32


def test_non_eager_attention_is_refused(tiny_llava_next, processor):
    import copy

    from kvsel_next.model import make_bundle  # what load_bundle uses

    model = copy.deepcopy(tiny_llava_next)
    model.config.text_config._attn_implementation = "sdpa"
    with pytest.raises(RuntimeError, match="eager"):
        make_bundle(model, processor)


# --- memory: same outputs with fewer copies (20 GiB GPU) ------------------------------------------


def test_run_unit_rows_equal_kvsels(bundle, aspect):
    """The NeXT run_unit scores before greedy so it holds one cache copy fewer; its row is kvsel's exactly."""
    from kvsel.runner import NO_CONTEXT, Unit
    from kvsel.runner import run_unit as kvsel_run_unit
    from kvsel_next.model import run_unit

    item = {**ITEM, "a_true": "red", "a_alt": "blue"}
    total = aspect["layout"].total
    kept = np.sort(np.random.default_rng(0).choice(total, total // 4, replace=False))
    for unit, keep in ((Unit("full", total, False, NO_CONTEXT, "Q"), np.arange(total)),
                       (Unit("pw", len(kept), False, "mis", "mis"), kept)):
        assert run_unit(bundle, item, unit, aspect["prefix"], keep, aspect["tokens"]) == \
            kvsel_run_unit(bundle, item, unit, aspect["prefix"], keep, aspect["tokens"])


def test_logits_without_the_fp32_head_copy_are_identical(tiny_llava_next, processor):
    """kvsel's bundle keeps an fp32 head copy; NeXT's upcasts 4,096-row slices on each call, never making the copy."""
    from kvsel.model import bundle_from_model
    from kvsel_next.model import make_bundle

    with_copy = bundle_from_model(tiny_llava_next, processor.tokenizer, processor, family=NEXT)
    without = make_bundle(tiny_llava_next, processor)
    assert without.decoder.fp32_head is None and with_copy.decoder.fp32_head is not None
    assert without.decoder.HEAD_ROWS_PER_SLICE == 4096 and without.decoder.head_precision == "fp32"
    hidden = torch.randn(1, 5, tiny_llava_next.config.text_config.hidden_size, dtype=without.decoder.dtype)
    assert torch.equal(with_copy.decoder.logits(hidden), without.decoder.logits(hidden))


def test_prefill_returns_the_prefix_in_cpu_memory(aspect):
    assert all(k.device.type == "cpu" and v.device.type == "cpu" for k, v in aspect["prefix"].entries)


def test_replay_offloaded_equals_kvsels_compact_replay(bundle, aspect):
    """Same logits, same state entries and positions as kvsel's compact replay, for full and sparse kept sets."""
    from kvsel_next.model import on_device, replay_offloaded

    prefix, suffix = aspect["prefix"], torch.tensor([aspect["tokens"]["mis"].suffix_ids])
    total = aspect["layout"].total
    for kept in (torch.arange(total), torch.from_numpy(np.sort(np.random.default_rng(1).choice(total, total // 8, replace=False)))):
        state_e, logits_e = replay(bundle.decoder, on_device(prefix, bundle.device), kept, suffix, "compact")
        state_o, logits_o = replay_offloaded(bundle.decoder, prefix, kept, suffix)
        assert torch.equal(logits_e, logits_o)
        assert (state_e.logical_next, state_e.physical_next, state_e.dropped) == (state_o.logical_next, state_o.physical_next, None)
        assert all(torch.equal(a, b) for layer_e, layer_o in zip(state_e.entries, state_o.entries) for a, b in zip(layer_e, layer_o))
        assert torch.equal(state_e.last_logits, state_o.last_logits)
