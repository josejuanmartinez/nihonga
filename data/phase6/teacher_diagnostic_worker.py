def diagnose_teacher(request, key):
    """One independent diagnostic; preserve all outputs, including failed controls."""
    import hashlib
    import importlib.metadata
    import inspect
    import json
    import math
    from pathlib import Path
    import torch
    from diffusers import QwenImage21Pipeline
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache

    directory = Path('/phase3/phase6/teacher_diagnostic') / key
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / 'summary.json'
    if report_path.exists():
        return json.loads(report_path.read_text())
    if (directory / 'started.json').exists():
        raise RuntimeError('Incomplete diagnostic exists; inspect rather than repeat calls.')
    (directory / 'started.json').write_text(json.dumps({'request_key': key}))
    (directory / 'request.json').write_text(json.dumps({'request_key': key, 'request': request}, indent=2))
    results_volume.commit()
    report = {'request_key': key, 'status': 'failed', 'teacher_calls': 0,
              'student_calls': 0, 'optimizer_updates': 0, 'velocity_metrics': [], 'pair_metrics': []}
    tensors = {'request_key': key, 'velocities': [], 'sampled_pairs': [], 'buffers': {}}
    hooks = []

    def digest(path):
        h = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                h.update(chunk)
        return h.hexdigest()

    def gpu(value):
        if isinstance(value, torch.Tensor):
            return value.to('cuda')
        if isinstance(value, dict):
            return {k: gpu(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return type(value)(gpu(v) for v in value)
        return value

    def metrics(pred, target):
        x, y = pred.float(), target.float()
        if not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise RuntimeError('Nonfinite diagnostic tensor.')
        delta = x - y
        return {'relative_l2_error': float(delta.norm() / y.norm().clamp_min(1e-12)),
                'max_abs_error': float(delta.abs().max()), 'rmse': float(delta.square().mean().sqrt()),
                'exact_equal': bool(torch.equal(pred, target))}

    try:
        teacher = request['teacher']
        record = request['record']
        path = Path('/phase3/phase5/capture') / record['request_key'] / 'targets.pt'
        if digest(path) != record['tensor_sha256']:
            raise RuntimeError('Diagnostic target differs.')
        bundle = torch.load(path, weights_only=True, map_location='cpu')
        if bundle['request_key'] != record['request_key'] or bundle['row']['split'] != 'validation':
            raise RuntimeError('Diagnostic target provenance differs.')
        report['gpu'] = torch.cuda.get_device_name(0)
        if 'H100' not in report['gpu']:
            raise RuntimeError('GPU differs from capture runtime.')
        report['packages'] = {p: importlib.metadata.version(p) for p in ('torch', 'diffusers', 'transformers', 'accelerate', 'pillow')}
        report['torch_cuda'] = torch.version.cuda
        report['math_settings'] = {
            'float32_matmul_precision': torch.get_float32_matmul_precision(),
            'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
            'bf16_reduced_precision_reduction': torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'flash_sdp': torch.backends.cuda.flash_sdp_enabled(),
            'math_sdp': torch.backends.cuda.math_sdp_enabled(),
            'mem_efficient_sdp': torch.backends.cuda.mem_efficient_sdp_enabled(),
        }
        pipe = QwenImage21Pipeline.from_pretrained(teacher['model_repo'], revision=teacher['model_revision'],
            dtype=torch.bfloat16, local_files_only=True).to('cuda')
        model = pipe.transformer.eval()
        for component in (model, pipe.text_encoder, pipe.vae):
            component.requires_grad_(False)
        transformer_path = Path(inspect.getfile(type(model)))
        if digest(transformer_path) != request['transformer_source_sha256'] or len(model.transformer_blocks) != 32:
            raise RuntimeError('Model source or depth differs.')
        config_path = (Path('/root/.cache/huggingface/hub') / ('models--' + teacher['model_repo'].replace('/', '--'))
                       / 'snapshots' / teacher['model_revision'] / 'transformer/config.json')
        if digest(config_path) != teacher['config_sha256']:
            raise RuntimeError('Model configuration differs.')
        report['attention_processors'] = sorted({type(b.attn.processor).__name__ for b in model.transformer_blocks})
        report['attention_backends'] = sorted({str(getattr(b.attn.processor, '_attention_backend', None)) for b in model.transformer_blocks})
        report['buffers'] = []
        for name, buffer in model.named_buffers():
            report['buffers'].append({'name': name, 'dtype': str(buffer.dtype), 'shape': list(buffer.shape),
                'finite': bool(torch.isfinite(buffer).all()), 'device': str(buffer.device)})
            if buffer.numel() <= 4096:
                tensors['buffers'][name] = buffer.detach().cpu().clone()
        temporal = model.time_text_embed.time_proj
        reference = torch.exp(-math.log(10000) * torch.arange(128, dtype=torch.float32) / 128)
        tensors['buffers']['canonical_temporal_freqs_fp32'] = reference
        report['temporal_frequency_comparison'] = metrics(temporal.freqs.detach().cpu(), reference.to(temporal.freqs.dtype))
        report['input_metadata'] = []
        common = gpu(bundle['shared_transformer_kwargs'])
        state = {}
        pair_by_key = {(p['stage'], p['layer']): p for p in bundle['pairs']}

        def hook_for(layer, role):
            def hook(module, args, kwargs, output=None):
                stage = state['stage']
                saved = pair_by_key[(stage['stage'], layer)]
                full = kwargs['hidden_states'] if role == 'input' else output
                sample = full.index_select(1, saved['token_indices'].to(full.device)).detach().cpu().contiguous()
                tensors['sampled_pairs'].append({'run': state['run'], 'stage': stage['stage'],
                    'layer': layer, 'role': role, 'value': sample})
                report['pair_metrics'].append({'run': state['run'], 'stage': stage['stage'], 'layer': layer,
                    'role': role, **metrics(sample, saved['input' if role == 'input' else 'target'])})
            return hook

        for layer in (2, 3, 4, 5):
            hooks.append(model.transformer_blocks[layer].register_forward_pre_hook(hook_for(layer, 'input'), with_kwargs=True))
            hooks.append(model.transformer_blocks[layer].register_forward_hook(hook_for(layer, 'output'), with_kwargs=True))
        for run in ('fresh_1', 'fresh_2'):
            cache = QwenImage21KVCache(32)
            with torch.inference_mode(), model.cache_context('cond'):
                for stage in bundle['stages']:
                    state.update(run=run, stage=stage)
                    kwargs = {**common, **gpu(stage['kwargs']), 'kv_cache': cache}
                    if run == 'fresh_1':
                        report['input_metadata'].append({'stage': stage['stage'], 'tensor_fields': {
                            k: {'shape': list(v.shape), 'dtype': str(v.dtype), 'stride': list(v.stride()),
                                'contiguous': v.is_contiguous()} for k, v in kwargs.items() if isinstance(v, torch.Tensor)},
                            'cache_mode': kwargs['kv_cache_mode']})
                    pred = model(**kwargs)[0][:, -stage['target_image_tokens']:].detach().cpu().contiguous()
                    report['teacher_calls'] += 1
                    tensors['velocities'].append({'run': run, 'stage': stage['stage'], 'velocity': pred})
                    report['velocity_metrics'].append({'run': run, 'stage': stage['stage'],
                        **metrics(pred, stage['teacher_velocity'])})
        report['repeatability'] = []
        for stage in bundle['stages']:
            values = [v['velocity'] for v in tensors['velocities'] if v['stage'] == stage['stage']]
            report['repeatability'].append({'stage': stage['stage'], **metrics(values[1], values[0])})
        report['status'] = 'diagnostic_complete'
        report['control_passed'] = all(v['relative_l2_error'] <= 0.001 for v in report['velocity_metrics'])
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        for hook in hooks:
            hook.remove()
        # Preserve measured tensors even if the diagnostic itself fails.
        tensor_path = directory / 'diagnostic_tensors.pt'
        temporary = tensor_path.with_suffix('.tmp')
        torch.save(tensors, temporary)
        temporary.replace(tensor_path)
        pipeline_source = Path(inspect.getfile(QwenImage21Pipeline)).read_text()
        source_path = directory / 'pipeline_qwenimage21_runtime.py'
        source_path.write_text(pipeline_source)
        report['artifacts'] = [{'name': p.name, 'sha256': digest(p), 'bytes': p.stat().st_size}
                               for p in (tensor_path, source_path)]
        report_path.write_text(json.dumps(report, indent=2) + '\n')
        results_volume.commit()
    return report
