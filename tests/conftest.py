import pytest
import torch
from transformers import CLIPVisionConfig, LlamaConfig, LlamaForCausalLM, LlavaConfig, LlavaForConditionalGeneration

from kvsel.cache import entries_from_cache
from kvsel.engine import Decoder, Prefix


@pytest.fixture(scope="session")
def tiny_llama():
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=500, hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
        max_position_embeddings=4096, attn_implementation="eager",
    )
    return LlamaForCausalLM(config).eval()


ACCELERATOR = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else None


@pytest.fixture(scope="session", params=["device", "cpu"])
def residency(request):
    """Where the prefix lives relative to the decoder. The prefix fixtures stay in CPU memory, so a CPU-held prefix
    means a decoder on an accelerator."""
    if request.param == "cpu" and ACCELERATOR is None:
        pytest.skip("a CPU-held prefix needs a decoder on an accelerator")
    return request.param


@pytest.fixture(scope="session")
def tiny_decoder(tiny_llama, residency):
    import copy

    model = tiny_llama if residency == "device" else copy.deepcopy(tiny_llama).to(ACCELERATOR)
    return Decoder(model.model, model.lm_head)


@pytest.fixture(scope="session")
def tiny_prefix(tiny_llama):
    torch.manual_seed(1)
    ids = torch.randint(0, 500, (1, 581))
    with torch.inference_mode():
        output = tiny_llama.model(input_ids=ids, use_cache=True)
    return Prefix(entries_from_cache(output.past_key_values), 5, 576)


def make_tiny_llava():
    """Same classes as the real model, random weights, full 576-token image grid."""
    torch.manual_seed(2)
    text = LlamaConfig(
        vocab_size=32064, hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4,
        max_position_embeddings=4096,
    )
    vision = CLIPVisionConfig(
        hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2, image_size=336, patch_size=14,
    )
    config = LlavaConfig(
        vision_config=vision, text_config=text, image_token_index=32000, vision_feature_layer=-2,
        vision_feature_select_strategy="default", attn_implementation="eager",
    )
    return LlavaForConditionalGeneration(config).eval()


@pytest.fixture(scope="session")
def tiny_llava():
    return make_tiny_llava()


def make_tiny_llava_next():
    """LLaVA-NeXT classes with random weights: 24 x 24 base tile, AnyRes on with the real pinpoints."""
    from transformers import LlavaNextConfig, LlavaNextForConditionalGeneration

    torch.manual_seed(3)
    text = LlamaConfig(
        vocab_size=32064, hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4,
        max_position_embeddings=4096,
    )
    vision = CLIPVisionConfig(
        hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2, image_size=336, patch_size=14,
    )
    config = LlavaNextConfig(
        vision_config=vision, text_config=text, image_token_index=32000, vision_feature_layer=-2,
        vision_feature_select_strategy="default", attn_implementation="eager",
        image_grid_pinpoints=[[336, 672], [672, 336], [672, 672], [1008, 336], [336, 1008]],
    )
    return LlavaNextForConditionalGeneration(config).eval()


@pytest.fixture(scope="session")
def tiny_llava_next():
    return make_tiny_llava_next()
