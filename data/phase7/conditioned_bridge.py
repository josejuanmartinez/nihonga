import torch
from torch import nn
from types import SimpleNamespace


class ConditionedBridge(nn.Module):
    """Residual bottleneck bridge that can read the model's shared timestep modulation.

    The transformer computes one modulation tensor per timestep, [attn scale, attn gate, MLP scale, MLP gate],
    shared by every block. A conditioned bridge mirrors a block's MLP branch:
    h + tanh(gate) * up(gelu(down(LN(h) * (1 + scale)))), with scale and gate learned per-channel mixes of the
    shared values. Unconditioned, it is the original Phase 4 bridge: h + up(gelu(down(h))).
    """

    def __init__(self, hidden_dim=4096, rank=256, conditioned=True, eps=1e-6):
        super().__init__()
        self.conditioned = conditioned
        self.down = nn.Linear(hidden_dim, rank, bias=False)
        self.up = nn.Linear(rank, hidden_dim, bias=False)
        nn.init.zeros_(self.up.weight)
        if conditioned:
            self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=eps)
            # Start from the block's own MLP modulation (weight 1 on the MLP terms, 0 on the attention terms).
            mix = torch.zeros(2, hidden_dim)
            mix[1] = 1
            self.scale_mix = nn.Parameter(mix.clone())
            self.gate_mix = nn.Parameter(mix.clone())
            self.scale_bias = nn.Parameter(torch.zeros(hidden_dim))
            self.gate_bias = nn.Parameter(torch.zeros(hidden_dim))

    def scale_gate(self, modulation_rows):
        scale1, gate1, scale2, gate2 = modulation_rows.chunk(4, dim=-1)
        scale = self.scale_mix[0] * scale1 + self.scale_mix[1] * scale2 + self.scale_bias
        gate = self.gate_mix[0] * gate1 + self.gate_mix[1] * gate2 + self.gate_bias
        return scale, gate

    def forward(self, hidden_states, modulation_rows=None):
        if not self.conditioned:
            return hidden_states + self.up(nn.functional.gelu(self.down(hidden_states)))
        scale, gate = self.scale_gate(modulation_rows)
        update = self.up(nn.functional.gelu(self.down(self.norm(hidden_states) * (1 + scale))))
        return hidden_states + torch.tanh(gate) * update


class LocalConditionedBridge(ConditionedBridge):
    """Conditioned bridge plus a depthwise 3x3 convolution over the image-token grid inside the bottleneck.

    The convolution starts at zero (identity start) and is applied to target-image tokens only; prompt tokens have
    no grid and pass through the per-token path.
    """
    local = True

    def __init__(self, hidden_dim=4096, rank=1024, eps=1e-6):
        super().__init__(hidden_dim, rank, True, eps)
        self.mix = nn.Parameter(torch.zeros(rank, 9))

    def forward(self, hidden_states, modulation_rows=None, image_mask=None, grid=None):
        scale, gate = self.scale_gate(modulation_rows)
        z = self.down(self.norm(hidden_states) * (1 + scale))
        if grid is not None:
            height, width = grid
            image = z[:, image_mask]
            batch, tokens, rank = image.shape
            planes = image.transpose(1, 2).reshape(batch, rank, height, width)
            mixed = nn.functional.conv2d(planes, self.mix.view(rank, 1, 3, 3).to(planes.dtype), padding=1, groups=rank)
            z = z.clone()
            z[:, image_mask] = image + mixed.reshape(batch, rank, tokens).transpose(1, 2)
        return hidden_states + torch.tanh(gate) * self.up(nn.functional.gelu(z))

    def forward_neighborhood(self, neighborhood, modulation_rows):
        """Training path: neighborhood is (tokens, 9, hidden) in 3x3 raster order, zeros outside the grid."""
        scale, gate = self.scale_gate(modulation_rows)
        z = self.down(self.norm(neighborhood) * (1 + scale.unsqueeze(1)))
        z = z[:, 4] + (z * self.mix.t().unsqueeze(0)).sum(1)
        return neighborhood[:, 4] + torch.tanh(gate) * self.up(nn.functional.gelu(z))


class AttentionConditionedBridge(ConditionedBridge):
    """Conditioned bridge with a small attention branch in front of its MLP, mirroring a block.

    Target-image tokens attend to the prompt prefix and all image tokens with the model's RoPE and key masks; prefix
    tokens only take the MLP path. The branch keeps its own prefix K/V in the layer cache of the slot it replaces.
    The output projection starts at zero, so a fresh branch leaves the conditioned MLP bridge unchanged.
    """
    attention = True

    def __init__(self, hidden_dim=4096, rank=1024, heads=4, head_dim=128, eps=1e-6):
        super().__init__(hidden_dim, rank, True, eps)
        inner = heads * head_dim
        self.heads, self.head_dim, self.eps = heads, head_dim, eps
        self.to_q = nn.Linear(hidden_dim, inner, bias=False)
        self.to_k = nn.Linear(hidden_dim, inner, bias=False)
        self.to_v = nn.Linear(hidden_dim, inner, bias=False)
        self.to_out = nn.Linear(inner, hidden_dim, bias=False)
        nn.init.zeros_(self.to_out.weight)
        self.norm_q = nn.Parameter(torch.ones(head_dim))
        self.norm_k = nn.Parameter(torch.ones(head_dim))
        # Attention modulation starts from the block's own attention terms (weight 1 on scale1/gate1).
        mix = torch.zeros(2, hidden_dim)
        mix[0] = 1
        self.attn_scale_mix = nn.Parameter(mix.clone())
        self.attn_gate_mix = nn.Parameter(mix.clone())
        self.attn_scale_bias = nn.Parameter(torch.zeros(hidden_dim))
        self.attn_gate_bias = nn.Parameter(torch.zeros(hidden_dim))

    def _rms(self, x, weight):
        return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps).to(x.dtype) * weight

    @staticmethod
    def _rope(x, freqs):
        rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2)) * freqs.unsqueeze(1)
        return torch.view_as_real(rotated).flatten(3).type_as(x)

    def attend(self, hidden_states, modulation_rows, rotary_emb, target_count, layer_cache, kv_cache_mode,
               cache_write_slice, key_mask):
        scale1, gate1, scale2, gate2 = modulation_rows.chunk(4, dim=-1)
        scale = self.attn_scale_mix[0] * scale1 + self.attn_scale_mix[1] * scale2 + self.attn_scale_bias
        gate = self.attn_gate_mix[0] * gate1 + self.attn_gate_mix[1] * gate2 + self.attn_gate_bias
        x = self.norm(hidden_states) * (1 + scale)
        q, k, v = (proj(x).unflatten(-1, (self.heads, self.head_dim)) for proj in (self.to_q, self.to_k, self.to_v))
        q, k = self._rms(q, self.norm_q), self._rms(k, self.norm_k)
        if rotary_emb is not None:
            q, k = self._rope(q, rotary_emb), self._rope(k, rotary_emb)
        if layer_cache is not None:
            if kv_cache_mode == 'extract' and cache_write_slice is not None:
                layer_cache.store(k[:, cache_write_slice].clone(), v[:, cache_write_slice].clone())
            elif kv_cache_mode == 'cached':
                cached_k, cached_v = layer_cache.get()
                k, v = torch.cat([cached_k, k], dim=1), torch.cat([cached_v, v], dim=1)
        prefix = hidden_states.shape[1] - target_count
        out = nn.functional.scaled_dot_product_attention(
            q[:, prefix:].transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=key_mask)
        update = torch.tanh(gate[:, prefix:]) * self.to_out(out.transpose(1, 2).flatten(2))
        return torch.cat([hidden_states[:, :prefix], hidden_states[:, prefix:] + update], dim=1)


def modulation_rows(modulation, target_token_mask):
    """Per-token modulation rows, selected exactly as the transformer block does."""
    if target_token_mask is None:
        return modulation.unsqueeze(1)
    real, zero = modulation[:-1].unsqueeze(1), modulation[-1:].unsqueeze(0)
    return torch.where(target_token_mask.view(1, -1, 1), real, zero)


class ConditionedBridgeBlock(nn.Module):
    def __init__(self, bridge, processor):
        super().__init__()
        self.bridge = bridge
        # Parent forward inspects this processor to choose mask construction; no attention is performed here.
        self.attn = SimpleNamespace(processor=processor)

    def forward(self, hidden_states, modulation, rotary_emb=None, attention_mask=None, target_token_mask=None,
                layer_cache=None, kv_cache_mode=None, cache_write_slice=None, segments=None, key_valid=None):
        if kv_cache_mode not in (None, 'extract', 'cached'):
            raise ValueError('Unknown KV cache mode.')
        rows = modulation_rows(modulation, target_token_mask) if self.bridge.conditioned else None
        dtype = hidden_states.dtype
        bridge_dtype = next(self.bridge.parameters()).dtype
        if getattr(self.bridge, 'attention', False):
            # Small bridges may keep FP32 weights for training; compute in their dtype and return the model dtype.
            target_count = hidden_states.shape[1] if target_token_mask is None else int(target_token_mask.sum())
            key_mask = key_valid[:, None, None, :] if key_valid is not None else attention_mask
            h = hidden_states.to(bridge_dtype)
            r = rows.to(bridge_dtype)
            h = self.bridge.attend(h, r, rotary_emb, target_count, layer_cache, kv_cache_mode, cache_write_slice, key_mask)
            return self.bridge(h, r).to(dtype)
        if bridge_dtype != dtype:
            return self.bridge(hidden_states.to(bridge_dtype), None if rows is None else rows.to(bridge_dtype)).to(dtype)
        if getattr(self.bridge, 'local', False):
            mask = target_token_mask if target_token_mask is not None else torch.ones(
                hidden_states.shape[1], dtype=torch.bool, device=hidden_states.device)
            height, width = self.grid
            if height * width != int(mask.sum()):
                raise RuntimeError('Target image grid does not match the image-token count.')
            return self.bridge(hidden_states, rows, mask, (height, width))
        return self.bridge(hidden_states, rows)


def insert_conditioned_bridge(transformer, layer, bridge, keep_dtype=False):
    """Replace one slot; callers must use fresh KV caches afterwards. keep_dtype keeps FP32 bridge weights."""
    old = transformer.transformer_blocks[layer]
    reference = next(transformer.parameters())
    bridge = bridge.to(device=reference.device) if keep_dtype else bridge.to(device=reference.device, dtype=reference.dtype)
    wrapper = ConditionedBridgeBlock(bridge, old.attn.processor)
    wrapper.train(transformer.training)
    transformer.transformer_blocks[layer] = wrapper
    if getattr(bridge, 'local', False) and not getattr(transformer, '_bridge_grid_hook', False):
        # The block never sees img_shapes; record the target image's (height, width) in tokens on every call.
        def record_grid(module, args, kwargs):
            frames, height, width = kwargs['img_shapes'][0][-1]
            for block in module.transformer_blocks:
                if isinstance(block, ConditionedBridgeBlock):
                    block.grid = (int(height), int(width))
        transformer.register_forward_pre_hook(record_grid, with_kwargs=True)
        transformer._bridge_grid_hook = True
    return old
