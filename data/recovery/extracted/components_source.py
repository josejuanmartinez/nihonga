"""FP32 master adapters and functional prefix-cache checkpointing for the pilot."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class MasterBridge(nn.Module):
    def __init__(self, width=4096, rank=256):
        super().__init__()
        self.down = nn.Linear(width, rank, bias=False, dtype=torch.float32)
        self.up = nn.Linear(rank, width, bias=False, dtype=torch.float32)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states):
        dtype = hidden_states.dtype
        residual = F.linear(hidden_states, self.down.weight.to(dtype))
        residual = F.gelu(residual)
        residual = F.linear(residual, self.up.weight.to(dtype))
        return hidden_states + residual


class MasterLoRALinear(nn.Module):
    def __init__(self, base, rank=8, alpha=8):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.down = nn.Linear(base.in_features, rank, bias=False,
                              device=base.weight.device, dtype=torch.float32)
        self.up = nn.Linear(rank, base.out_features, bias=False,
                            device=base.weight.device, dtype=torch.float32)
        nn.init.zeros_(self.up.weight)
        self.scale = alpha / rank

    def forward(self, value):
        delta = F.linear(value, self.down.weight.to(value.dtype))
        delta = F.linear(delta, self.up.weight.to(value.dtype))
        return self.base(value) + delta * self.scale


class FunctionalCheckpointBlock(nn.Module):
    """Return prefix K/V as checkpoint outputs; snapshot cached K/V as inputs.

    Recomputations receive a private cache. They cannot mutate the cache used
    by another stage, and prefix gradients remain connected to its extraction.
    """
    def __init__(self, inner, cache_layer_class):
        super().__init__()
        self.inner = inner
        self.cache_layer_class = cache_layer_class

    @property
    def attn(self):
        return self.inner.attn

    def forward(self, hidden_states, modulation, rotary_emb=None,
                attention_mask=None, target_token_mask=None, layer_cache=None,
                kv_cache_mode=None, cache_write_slice=None, segments=None,
                key_valid=None):
        common = dict(attention_mask=attention_mask,
                      target_token_mask=target_token_mask,
                      kv_cache_mode=kv_cache_mode,
                      cache_write_slice=cache_write_slice,
                      segments=segments, key_valid=key_valid)
        if not torch.is_grad_enabled():
            return self.inner(hidden_states, modulation, rotary_emb=rotary_emb,
                              layer_cache=layer_cache, **common)

        if kv_cache_mode == 'extract':
            def extract(h, m, r):
                private = self.cache_layer_class()
                output = self.inner(h, m, rotary_emb=r, layer_cache=private, **common)
                k, v = private.get()
                return output, k, v
            output, k, v = checkpoint(extract, hidden_states, modulation,
                                      rotary_emb, use_reentrant=False)
            layer_cache.store(k, v)
            return output

        if kv_cache_mode == 'cached':
            prefix_k, prefix_v = layer_cache.get()
            def cached(h, m, r, k, v):
                private = self.cache_layer_class()
                private.store(k, v)
                return self.inner(h, m, rotary_emb=r, layer_cache=private, **common)
            return checkpoint(cached, hidden_states, modulation, rotary_emb,
                              prefix_k, prefix_v, use_reentrant=False)

        def uncached(h, m, r):
            return self.inner(h, m, rotary_emb=r, layer_cache=None, **common)
        return checkpoint(uncached, hidden_states, modulation, rotary_emb,
                          use_reentrant=False)


def attach_adapters(model, bridge_weights, insert_bridge, cache_class,
                    cache_layer_class, rank=8, alpha=8):
    bridge = MasterBridge()
    bridge.load_state_dict(bridge_weights)
    removed, _ = insert_bridge(model, 5, bridge, cache_class)
    # insert_bridge matches inference dtype; restore FP32 optimizer masters.
    # Reload the original FP32 checkpoint after the temporary BF16 conversion.
    bridge.float()
    bridge.load_state_dict(bridge_weights)
    del removed
    adapters = []
    for slot, block in enumerate(model.transformer_blocks):
        if slot == 5:
            continue
        for name in ('to_q', 'to_k', 'to_v'):
            setattr(block.attn, name, MasterLoRALinear(getattr(block.attn, name), rank, alpha))
            adapters.append(f'transformer_blocks.{slot}.attn.{name}')
        block.attn.to_out[0] = MasterLoRALinear(block.attn.to_out[0], rank, alpha)
        adapters.append(f'transformer_blocks.{slot}.attn.to_out.0')
        model.transformer_blocks[slot] = FunctionalCheckpointBlock(block, cache_layer_class)
    model.eval()
    model.gradient_checkpointing = False
    trainable = {name: param for name, param in model.named_parameters() if param.requires_grad}
    return trainable, adapters
