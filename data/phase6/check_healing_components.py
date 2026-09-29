import hashlib
import json
from pathlib import Path

source_path = Path('data/phase6/healing_components.py')
source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
report_path = Path('data/phase6/healing_component_checks.json')
if report_path.exists():
    report = json.loads(report_path.read_text(encoding='utf-8'))
    if report['source_sha256'] != source_sha or report['status'] != 'passed':
        raise RuntimeError('Component checks differ; preserve prior results.')
    print('Reusing saved component checks; no Torch import or gradient computation.')
else:
    import copy
    import torch
    from torch import nn
    from types import SimpleNamespace

    namespace = {}
    exec(compile(source_path.read_text(encoding='utf-8'), str(source_path), 'exec'), namespace)
    Wrapped = namespace['FunctionalCheckpointBlock']

    class Cache:
        def store(self, k, v):
            self.k, self.v = k, v
        def get(self):
            return self.k, self.v

    class ToyBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Linear(4, 4, bias=False, dtype=torch.float64)
            self.attn = SimpleNamespace(processor=None)
        def forward(self, hidden_states, modulation, rotary_emb=None,
                    layer_cache=None, kv_cache_mode=None, cache_write_slice=None, **kwargs):
            projected = self.projection(hidden_states).tanh()
            if kv_cache_mode == 'extract':
                layer_cache.store(projected[:, cache_write_slice].clone(),
                                  (projected[:, cache_write_slice] * 0.7).clone())
            elif kv_cache_mode == 'cached':
                k, v = layer_cache.get()
                projected = projected + k.mean(1, keepdim=True) + v.mean(1, keepdim=True)
            return projected + modulation

    torch.manual_seed(20260929)
    direct = [ToyBlock(), ToyBlock()]
    wrapped = [Wrapped(copy.deepcopy(block), Cache) for block in direct]
    early = torch.randn(1, 5, 4, dtype=torch.float64)
    middle = torch.randn(1, 3, 4, dtype=torch.float64)
    modulation = torch.randn(1, 1, 4, dtype=torch.float64)

    def run(blocks):
        caches = [Cache(), Cache()]
        early_input = early.clone().requires_grad_()
        middle_input = middle.clone().requires_grad_()
        h = early_input
        for block, cache in zip(blocks, caches):
            h = block(h, modulation, layer_cache=cache, kv_cache_mode='extract',
                      cache_write_slice=slice(0, 2))
        # Like a middle-stage training sample, only the cached output has loss.
        prefix_objects = [cache.get() for cache in caches]
        h = middle_input
        for block, cache in zip(blocks, caches):
            h = block(h, modulation, layer_cache=cache, kv_cache_mode='cached')
        prediction = h.detach().clone()
        h.square().mean().backward()
        if any(cache.get()[0] is not saved[0] or cache.get()[1] is not saved[1]
               for cache, saved in zip(caches, prefix_objects)):
            raise RuntimeError('Backward recomputation mutated the shared prefix cache.')
        gradients = [p.grad.clone() for block in blocks for p in block.parameters()]
        return prediction, gradients, early_input.grad, middle_input.grad

    plain, checkpointed = run(direct), run(wrapped)
    maximum = 0.0
    for left, right in zip([plain[0], *plain[1], plain[2], plain[3]],
                           [checkpointed[0], *checkpointed[1], checkpointed[2], checkpointed[3]]):
        torch.testing.assert_close(left, right, atol=1e-12, rtol=1e-12)
        maximum = max(maximum, float((left-right).abs().max()))
    if not checkpointed[2].abs().sum() > 0:
        raise RuntimeError('Cached-stage gradients did not reach prefix extraction.')

    # Zero-initialized adapter must preserve the base linear's values exactly.
    base = nn.Linear(4, 4, bias=False, dtype=torch.float32)
    adapter = namespace['MasterLoRALinear'](base, rank=2, alpha=2)
    value = torch.randn(2, 3, 4)
    if not torch.equal(adapter(value), base(value)):
        raise RuntimeError('Zero adapter changed the baseline.')
    report = {'version': 'phase6-functional-cache-check-v1', 'status': 'passed',
              'source_sha256': source_sha, 'maximum_absolute_difference': maximum,
              'prefix_input_gradient_norm': float(checkpointed[2].norm()),
              'shared_cache_unchanged_after_backward': True,
              'zero_adapter_exact_baseline': True,
              'scope': 'two-block CPU float64 cache/gradient equivalence and zero-adapter identity; real-model GPU check still required'}
    with report_path.open('x', encoding='utf-8') as handle:
        handle.write(json.dumps(report, indent=2) + '\n')
print('Functional-cache checks:', report['status'])
print('Maximum direct/checkpoint difference:', report['maximum_absolute_difference'])
print('Gradient norm reaching prefix extraction:', report['prefix_input_gradient_norm'])
print('Shared cache preserved:', report['shared_cache_unchanged_after_backward'])
print('Zero adapter preserves baseline:', report['zero_adapter_exact_baseline'])
