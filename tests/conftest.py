import pytest
import torch
from transformers import CLIPVisionConfig, LlamaConfig, LlavaConfig, LlavaForConditionalGeneration, LlamaForCausalLM

from kvsel.engine import Decoder, Prefix
from kvsel.cache import cache_from_entries, entries_from_cache


@pytest.fixture(scope="session")
def tiny_llama():
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=500, hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
        max_position_embeddings=4096, attn_implementation="eager",
    )
    return LlamaForCausalLM(config).eval()


@pytest.fixture(scope="session")
def tiny_decoder(tiny_llama):
    return Decoder(tiny_llama.model, tiny_llama.lm_head)


@pytest.fixture(scope="session")
def tiny_prefix(tiny_llama):
    torch.manual_seed(1)
    ids = torch.randint(0, 500, (1, 581))
    with torch.inference_mode():
        output = tiny_llama.model(input_ids=ids, use_cache=True)
    return Prefix(entries_from_cache(output.past_key_values), 5, 576)


@pytest.fixture(scope="session")
def tiny_llava():
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
def tiny_qwen():
    """Qwen2.5-VL classes with random weights: M-RoPE, merged 2 x 2 patches, the real vocabulary size."""
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

    torch.manual_seed(3)
    config = Qwen2_5_VLConfig(
        text_config=dict(
            vocab_size=151936, hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
            num_key_value_heads=2, max_position_embeddings=4096, rope_scaling={"type": "mrope", "mrope_section": [2, 3, 3]},
        ),
        vision_config=dict(
            depth=2, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64, patch_size=14,
            spatial_merge_size=2, temporal_patch_size=2, window_size=112, fullatt_block_indexes=[1], in_channels=3,
        ),
        image_token_id=151655, video_token_id=151656, vision_start_token_id=151652, vision_end_token_id=151653,
        attn_implementation="eager",
    )
    return Qwen2_5_VLForConditionalGeneration(config).eval()
