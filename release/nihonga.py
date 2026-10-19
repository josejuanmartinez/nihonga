"""Nihonga v1: Qwen-Image-2.1 with two transformer blocks replaced by small bridges and a fused 10-step LoRA.

Built with Qwen. Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, Copyright (c) 2026 Hangzhou Tongyi
Laboratory Technology Co., Ltd. All Rights Reserved. This file is new code by the Nihonga project; the model weights it
loads are a modified version of the Qwen-Image-2.1 transformer (non-commercial use only, see LICENSE and NOTICE).

Usage:
    from nihonga import load_pipeline, nihonga_sigmas
    pipe = load_pipeline("path-or-repo-of-nihonga-v1")             # route A: NihongaScheduler, num_inference_steps=10
    image = pipe("a red fox in snow", width=1024, height=1024, num_inference_steps=10, true_cfg_scale=1.0).images[0]

Route B (plain sigmas, stock scheduler made neutral) is shown in the README and in `use_plain_sigmas` below.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

SCHEDULE_POINTS = (0, 1, 3, 5, 10, 15, 20, 25, 30, 35)   # points of the stock 40-step schedule; the end (0) is appended
BASE_STEPS = 40
BASE_MODEL = 'Qwen/Qwen-Image-2.1'


# ---------------------------------------------------------------------------------------------------------------------
# Bridges: the modules that replace transformer blocks 5 and 20.
# ---------------------------------------------------------------------------------------------------------------------
class ConditionedBridge(nn.Module):
    """Residual bottleneck that reads the transformer's shared timestep modulation:
    h + tanh(gate) * up(gelu(down(LN(h) * (1 + scale)))), with scale and gate learned mixes of the shared values."""

    def __init__(self, hidden_dim=4096, rank=1024, eps=1e-6):
        super().__init__()
        self.conditioned = True
        self.down = nn.Linear(hidden_dim, rank, bias=False)
        self.up = nn.Linear(rank, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=eps)
        self.scale_mix = nn.Parameter(torch.zeros(2, hidden_dim))
        self.gate_mix = nn.Parameter(torch.zeros(2, hidden_dim))
        self.scale_bias = nn.Parameter(torch.zeros(hidden_dim))
        self.gate_bias = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, hidden_states, modulation_rows):
        scale1, gate1, scale2, gate2 = modulation_rows.chunk(4, dim=-1)
        scale = self.scale_mix[0] * scale1 + self.scale_mix[1] * scale2 + self.scale_bias
        gate = self.gate_mix[0] * gate1 + self.gate_mix[1] * gate2 + self.gate_bias
        update = self.up(nn.functional.gelu(self.down(self.norm(hidden_states) * (1 + scale))))
        return hidden_states + torch.tanh(gate) * update


class AttentionConditionedBridge(ConditionedBridge):
    """Conditioned bridge with a small attention branch (8 heads of 128) in front of its MLP: target-image tokens attend
    to the prompt prefix and all image tokens with the model's RoPE; the branch keeps its own prefix K/V cache."""
    attention = True

    def __init__(self, hidden_dim=4096, rank=1024, heads=8, head_dim=128, eps=1e-6):
        super().__init__(hidden_dim, rank, eps)
        inner = heads * head_dim
        self.heads, self.head_dim, self.eps = heads, head_dim, eps
        self.to_q = nn.Linear(hidden_dim, inner, bias=False)
        self.to_k = nn.Linear(hidden_dim, inner, bias=False)
        self.to_v = nn.Linear(hidden_dim, inner, bias=False)
        self.to_out = nn.Linear(inner, hidden_dim, bias=False)
        self.norm_q = nn.Parameter(torch.ones(head_dim))
        self.norm_k = nn.Parameter(torch.ones(head_dim))
        self.attn_scale_mix = nn.Parameter(torch.zeros(2, hidden_dim))
        self.attn_gate_mix = nn.Parameter(torch.zeros(2, hidden_dim))
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


class BridgeBlock(nn.Module):
    """Takes the place of a transformer block; same call signature."""

    def __init__(self, bridge, processor):
        super().__init__()
        self.bridge = bridge
        self.attn = SimpleNamespace(processor=processor)   # the parent forward inspects it to build masks

    def forward(self, hidden_states, modulation, rotary_emb=None, attention_mask=None, target_token_mask=None,
                layer_cache=None, kv_cache_mode=None, cache_write_slice=None, segments=None, key_valid=None):
        if kv_cache_mode not in (None, 'extract', 'cached'):
            raise ValueError('Unknown KV cache mode.')
        rows = modulation_rows(modulation, target_token_mask)
        dtype = hidden_states.dtype
        bridge_dtype = next(self.bridge.parameters()).dtype
        target_count = hidden_states.shape[1] if target_token_mask is None else int(target_token_mask.sum())
        key_mask = key_valid[:, None, None, :] if key_valid is not None else attention_mask
        h, r = hidden_states.to(bridge_dtype), rows.to(bridge_dtype)
        h = self.bridge.attend(h, r, rotary_emb, target_count, layer_cache, kv_cache_mode, cache_write_slice, key_mask)
        return self.bridge(h, r).to(dtype)


# ---------------------------------------------------------------------------------------------------------------------
# Loading.
# ---------------------------------------------------------------------------------------------------------------------
def _local_path(path_or_repo, revision=None):
    path = Path(path_or_repo)
    if path.exists():
        return path
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(str(path_or_repo), revision=revision))


def load_transformer(path_or_repo, torch_dtype=torch.bfloat16, device=None, revision=None):
    """The Nihonga v1 transformer: the Qwen-Image-2.1 architecture with blocks 5 and 20 replaced by bridges, and the
    10-step LoRA already fused into the attention weights."""
    from accelerate import init_empty_weights
    from diffusers import QwenImage21Transformer2DModel
    from safetensors.torch import load_file
    root = _local_path(path_or_repo, revision) / 'transformer'
    nihonga = json.loads((root / 'nihonga_config.json').read_text(encoding='utf-8'))
    config = QwenImage21Transformer2DModel.load_config(root)
    with init_empty_weights():
        model = QwenImage21Transformer2DModel.from_config(config)
    for slot in nihonga['bridge_slots']:
        spec = nihonga['bridge']
        with init_empty_weights():
            bridge = AttentionConditionedBridge(config['num_attention_heads'] * config['attention_head_dim'],
                                                spec['rank'], spec['heads'], spec['head_dim'])
        model.transformer_blocks[slot] = BridgeBlock(bridge, model.transformer_blocks[slot].attn.processor)
    index = json.loads((root / 'diffusion_pytorch_model.safetensors.index.json').read_text(encoding='utf-8'))
    state = {}
    for shard in sorted(set(index['weight_map'].values())):
        state.update(load_file(str(root / shard)))
    model.load_state_dict(state, strict=True, assign=True)
    model.to(dtype=torch_dtype)
    if device is not None:
        model.to(device)
    return model.eval()


# ---------------------------------------------------------------------------------------------------------------------
# The 10-step schedule.
# ---------------------------------------------------------------------------------------------------------------------
def _stock_scheduler_class():
    from diffusers import FlowMatchEulerDiscreteScheduler
    return FlowMatchEulerDiscreteScheduler


def _shift_mu(config, width, height):
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_shift
    return calculate_shift((height // 16) * (width // 16), config.get('base_image_seq_len', 256), config.get('max_image_seq_len', 4096),
                           config.get('base_shift', 0.5), config.get('max_shift', 1.15))


def nihonga_sigmas(scheduler_config, width, height, points=SCHEDULE_POINTS, base_steps=BASE_STEPS):
    """Route B: the ten sigma values for an image size, taken from the stock 40-step schedule (shift and terminal
    stretch included). Pass them as pipe(..., sigmas=...) with a neutral scheduler (see use_plain_sigmas)."""
    reference = _stock_scheduler_class().from_config(scheduler_config)
    reference.set_timesteps(sigmas=np.linspace(1.0, 1 / base_steps, base_steps), mu=_shift_mu(reference.config, width, height))
    return [float(reference.sigmas[i]) for i in points]


def neutral_scheduler(scheduler_config):
    """Route B: a stock scheduler that passes given sigmas through unchanged (no shift, no terminal stretch)."""
    return _stock_scheduler_class().from_config(scheduler_config, use_dynamic_shifting=False, shift=1.0, shift_terminal=None)


def _make_scheduler_class():
    base = _stock_scheduler_class()

    class NihongaScheduler(base):
        """Route A: builds the stock 40-step schedule for the image size and keeps the ten Nihonga points.
        Use with num_inference_steps=10; any sigmas the pipeline proposes are replaced."""

        def set_timesteps(self, num_inference_steps=None, device=None, sigmas=None, mu=None, timesteps=None):
            super().set_timesteps(sigmas=np.linspace(1.0, 1 / BASE_STEPS, BASE_STEPS), device=device, mu=mu)
            keep = list(SCHEDULE_POINTS) + [BASE_STEPS]
            self.sigmas = self.sigmas[keep].clone()
            self.timesteps = (self.sigmas[:-1] * self.config.num_train_timesteps).to(self.sigmas.device)
            self.num_inference_steps = len(SCHEDULE_POINTS)

    import inspect
    NihongaScheduler.set_timesteps.__signature__ = inspect.signature(base.set_timesteps)
    return NihongaScheduler


def nihonga_scheduler(scheduler_config):
    """Route A: the NihongaScheduler built from the stock scheduler config."""
    return _make_scheduler_class().from_config(scheduler_config)


def load_pipeline(path_or_repo, base=BASE_MODEL, base_revision=None, torch_dtype=torch.bfloat16, device='cuda', revision=None):
    """The full pipeline: the Nihonga transformer, the unchanged Qwen-Image-2.1 text encoder, VAE and processor, and the
    NihongaScheduler (route A). Generate with num_inference_steps=10 and true_cfg_scale=1.0."""
    from diffusers import QwenImage21Pipeline
    transformer = load_transformer(path_or_repo, torch_dtype=torch_dtype, revision=revision)
    pipe = QwenImage21Pipeline.from_pretrained(base, revision=base_revision, transformer=transformer, torch_dtype=torch_dtype)
    pipe.nihonga_stock_scheduler_config = dict(pipe.scheduler.config)
    pipe.scheduler = nihonga_scheduler(pipe.nihonga_stock_scheduler_config)
    return pipe.to(device) if device else pipe


def use_plain_sigmas(pipe):
    """Switch a pipeline to route B. Returns the stock scheduler config to pass to nihonga_sigmas:
        config = use_plain_sigmas(pipe)
        image = pipe(prompt, width=w, height=h, sigmas=nihonga_sigmas(config, w, h), true_cfg_scale=1.0).images[0]"""
    if not hasattr(pipe, 'nihonga_stock_scheduler_config'):
        pipe.nihonga_stock_scheduler_config = dict(pipe.scheduler.config)
    pipe.scheduler = neutral_scheduler(pipe.nihonga_stock_scheduler_config)
    return pipe.nihonga_stock_scheduler_config
