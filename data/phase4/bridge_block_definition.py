"""Qwen-Image 2.1 block contract for an unconditioned tokenwise bridge."""
from types import SimpleNamespace
from torch import nn


class QwenImage21BridgeBlock(nn.Module):
    def __init__(self, bridge, processor):
        super().__init__()
        self.bridge = bridge
        # Parent forward inspects this processor to choose mask construction.
        # This contains no attention projections and performs no attention.
        self.attn = SimpleNamespace(processor=processor)

    def forward(self, hidden_states, modulation, rotary_emb=None,
                attention_mask=None, target_token_mask=None, layer_cache=None,
                kv_cache_mode=None, cache_write_slice=None, segments=None,
                key_valid=None):
        if kv_cache_mode not in (None, 'extract', 'cached'):
            raise ValueError('Unknown KV cache mode.')
        # No K/V is required in this slot. Subsequent surviving blocks build
        # their own prefix K/V from the transformed prefix during extraction.
        return self.bridge(hidden_states)


def insert_bridge(transformer, layer, bridge, cache_class):
    """Replace one slot; caller must discard all prior caches and use returned cache."""
    if not 0 <= layer < len(transformer.transformer_blocks):
        raise IndexError('Bridge slot is outside transformer depth.')
    if bridge.down.in_features != transformer.config.num_attention_heads * transformer.config.attention_head_dim:
        raise ValueError('Bridge width differs from transformer width.')
    old = transformer.transformer_blocks[layer]
    reference = next(transformer.parameters())
    bridge = bridge.to(device=reference.device, dtype=reference.dtype)
    wrapper = QwenImage21BridgeBlock(bridge, old.attn.processor)
    wrapper.train(transformer.training)
    transformer.transformer_blocks[layer] = wrapper
    return old, cache_class(len(transformer.transformer_blocks))
