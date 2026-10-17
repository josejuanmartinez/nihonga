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
        if getattr(self.bridge, 'local', False):
            mask = target_token_mask if target_token_mask is not None else torch.ones(
                hidden_states.shape[1], dtype=torch.bool, device=hidden_states.device)
            side = int(round(int(mask.sum()) ** 0.5))
            if side * side != int(mask.sum()):
                raise RuntimeError('Local mixing expects a square image-token grid.')
            return self.bridge(hidden_states, rows, mask, (side, side))
        return self.bridge(hidden_states, rows)


def insert_conditioned_bridge(transformer, layer, bridge):
    """Replace one slot; callers must use fresh KV caches afterwards."""
    old = transformer.transformer_blocks[layer]
    reference = next(transformer.parameters())
    bridge = bridge.to(device=reference.device, dtype=reference.dtype)
    wrapper = ConditionedBridgeBlock(bridge, old.attn.processor)
    wrapper.train(transformer.training)
    transformer.transformer_blocks[layer] = wrapper
    return old
