"""Level 1: operator correctness on a tiny fp32 Llama. See tests/README.md."""

import numpy as np
import pytest
import torch

from kvsel.cache import compact, dropped_columns, entries_hash
from kvsel.engine import Prefix, State, advance, greedy, probe, replay, score_sequence
from kvsel.selectors import pw_attn, top_k

LIMIT = 1e-5
SUFFIX = torch.randint(0, 500, (1, 50), generator=torch.Generator().manual_seed(3))
GENERATED = torch.randint(0, 500, (10,), generator=torch.Generator().manual_seed(4)).tolist()


def all_logits(decoder, prefix, kept, mode="compact"):
    state, logits = replay(decoder, prefix, kept, SUFFIX, mode)
    _, generated = advance(decoder, state, GENERATED)
    return torch.cat([logits, generated], dim=1)


def random_set(k, seed):
    return torch.randperm(576, generator=torch.Generator().manual_seed(seed))[:k]


def scored_set(decoder, prefix, k):
    return torch.from_numpy(top_k(pw_attn(probe(decoder, prefix, SUFFIX)), k))


@pytest.mark.parametrize("k", [288, 144, 72])
@pytest.mark.parametrize("kind", ["random", "concentrated", "tail", "scored"])
def test_compact_matches_dense(tiny_decoder, tiny_prefix, k, kind):
    kept = {
        "random": lambda: random_set(k, 5),
        "concentrated": lambda: torch.arange(k),
        "tail": lambda: torch.arange(576 - k, 576),
        "scored": lambda: scored_set(tiny_decoder, tiny_prefix, k),
    }[kind]()
    compact_logits = all_logits(tiny_decoder, tiny_prefix, kept, "compact")
    dense_logits = all_logits(tiny_decoder, tiny_prefix, kept, "dense")
    assert (compact_logits - dense_logits).abs().max() < LIMIT


def test_incremental_generation_steps_match_dense(tiny_decoder, tiny_prefix):
    """One token at a time, as greedy decoding runs, rather than one batch of generated tokens."""
    kept = random_set(144, 6)
    states = [replay(tiny_decoder, tiny_prefix, kept, SUFFIX, mode)[0] for mode in ("compact", "dense")]
    for token in GENERATED:
        states = [advance(tiny_decoder, state, [token])[0] for state in states]
        assert (states[0].last_logits - states[1].last_logits).abs().max() < LIMIT


def test_control_physical_positions_are_detected(tiny_decoder, tiny_prefix):
    kept = random_set(144, 7)
    entries = compact(tiny_prefix.entries, kept, tiny_prefix.num_nonvisual)
    physical = entries[0][0].shape[2]
    hidden, _, _ = tiny_decoder.step(SUFFIX, entries, physical, physical)  # logical start is wrong on purpose
    wrong = tiny_decoder.logits(hidden)
    _, reference = replay(tiny_decoder, tiny_prefix, kept, SUFFIX, "dense")
    assert (wrong - reference).abs().max() > LIMIT


def perturbed(prefix, columns):
    generator = torch.Generator().manual_seed(8)
    entries = []
    for key, value in prefix.entries:
        key, value = key.clone(), value.clone()
        key[:, :, columns] += 5 * torch.randn(key[:, :, columns].shape, generator=generator)
        value[:, :, columns] += 5 * torch.randn(value[:, :, columns].shape, generator=generator)
        entries.append((key, value))
    return Prefix(tuple(entries), prefix.num_nonvisual, prefix.num_visual)


def test_control_dropped_entries_cannot_leak(tiny_decoder, tiny_prefix):
    kept = random_set(144, 9)
    columns = dropped_columns(kept, tiny_prefix.num_nonvisual, tiny_prefix.num_visual)
    changed = all_logits(tiny_decoder, perturbed(tiny_prefix, columns), kept, "dense")
    assert torch.equal(changed, all_logits(tiny_decoder, tiny_prefix, kept, "dense"))


def test_control_kept_entries_matter(tiny_decoder, tiny_prefix):
    kept = random_set(144, 10)
    changed = all_logits(tiny_decoder, perturbed(tiny_prefix, tiny_prefix.num_nonvisual + kept), kept, "dense")
    assert (changed - all_logits(tiny_decoder, tiny_prefix, kept, "dense")).abs().max() > LIMIT


def test_prefix_is_not_mutated(tiny_decoder, tiny_prefix):
    before = entries_hash(tiny_prefix.entries)
    probe(tiny_decoder, tiny_prefix, SUFFIX)
    state, _ = replay(tiny_decoder, tiny_prefix, random_set(144, 11), SUFFIX, "compact")
    greedy(tiny_decoder, state, max_new=3)
    replay(tiny_decoder, tiny_prefix, random_set(144, 11), SUFFIX, "dense")
    assert entries_hash(tiny_prefix.entries) == before


def test_key_and_value_gather_the_same_entries(tiny_prefix):
    kept = random_set(72, 12)
    entries = compact(tiny_prefix.entries, kept, tiny_prefix.num_nonvisual)
    index = torch.cat([torch.arange(5), 5 + kept.sort().values])
    for (key, value), (full_key, full_value) in zip(entries, tiny_prefix.entries):
        assert torch.equal(key, full_key[:, :, index]) and torch.equal(value, full_value[:, :, index])


def test_probe_shape_and_probability_bound(tiny_decoder, tiny_prefix):
    attention = probe(tiny_decoder, tiny_prefix, SUFFIX)
    assert attention.shape == (50, 576)
    assert np.all(attention >= 0) and attention.sum(axis=1).max() <= 1 + 1e-5


def test_score_sequence_matches_manual_chain(tiny_decoder, tiny_prefix):
    state, _ = replay(tiny_decoder, tiny_prefix, random_set(144, 13), SUFFIX)
    tokens = [7, 8, 9]
    expected = [torch.log_softmax(state.last_logits, -1)[7].item()]
    current = state
    for previous, token in zip(tokens[:-1], tokens[1:]):
        current, _ = advance(tiny_decoder, current, [previous])
        expected.append(torch.log_softmax(current.last_logits, -1)[token].item())
    assert score_sequence(tiny_decoder, state, tokens) == pytest.approx(expected, abs=1e-5)


def test_scoring_leaves_state_reusable(tiny_decoder, tiny_prefix):
    state, _ = replay(tiny_decoder, tiny_prefix, random_set(144, 14), SUFFIX)
    assert isinstance(state, State)
    first = score_sequence(tiny_decoder, state, [5, 6, 2])
    score_sequence(tiny_decoder, state, [50, 60, 70, 2])
    assert score_sequence(tiny_decoder, state, [5, 6, 2]) == first


def test_greedy_reports_truncation(tiny_decoder, tiny_prefix):
    state, _ = replay(tiny_decoder, tiny_prefix, random_set(144, 15), SUFFIX)
    result = greedy(tiny_decoder, state, max_new=3, stop_at_eos=False)
    assert len(result.token_ids) == 3 and result.truncated and len(result.top_two_gaps) == 3
