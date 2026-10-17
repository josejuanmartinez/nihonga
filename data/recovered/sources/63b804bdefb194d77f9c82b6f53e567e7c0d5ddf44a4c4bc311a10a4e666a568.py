def verify_segment_attention(request, key):
    """Verify unchanged cached targets using the original capture environment."""
    import hashlib
    import importlib.metadata
    import inspect
    import json
    import os
    import sys
    from collections import Counter
    from pathlib import Path
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from diffusers import QwenImage21Pipeline
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache

    directory = Path('/phase3/phase6/segment_attention') / key
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / 'summary.json'
    if report_path.exists():
        return json.loads(report_path.read_text())
    if (directory / 'started.json').exists():
        raise RuntimeError('Incomplete verification exists; inspect before repeating forwards.')
    (directory / 'started.json').write_text(json.dumps({'request_key': key}))
    (directory / 'request.json').write_text(json.dumps({'request': request, 'request_key': key}, indent=2))
    results_volume.commit()
    report = {'request_key': key, 'status': 'failed', 'teacher_calls': 0, 'student_calls': 0,
              'optimizer_updates': 0, 'controls': [], 'artifacts': [],
              'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG')}

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
        if isinstance(value, (list, tuple)):
            return type(value)(gpu(v) for v in value)
        return value

    try:
        if report['cublas_workspace_config'] is not None:
            raise RuntimeError('The original capture environment did not set CUBLAS_WORKSPACE_CONFIG.')
        teacher = request['teacher']
        report['gpu'] = torch.cuda.get_device_name(0)
        if 'H100' not in report['gpu']:
            raise RuntimeError('GPU differs.')
        report['packages'] = {p: importlib.metadata.version(p) for p in ('torch', 'diffusers', 'transformers', 'accelerate', 'pillow')}
        pipe = QwenImage21Pipeline.from_pretrained(teacher['model_repo'], revision=teacher['model_revision'],
            dtype=torch.bfloat16, local_files_only=True).to('cuda')
        model = pipe.transformer.eval()
        for component in (model, pipe.text_encoder, pipe.vae):
            component.requires_grad_(False)
        if digest(Path(inspect.getfile(type(model)))) != request['transformer_source_sha256'] or len(model.transformer_blocks) != 32:
            raise RuntimeError('Model implementation differs.')
        config_path = (Path('/root/.cache/huggingface/hub') / ('models--' + teacher['model_repo'].replace('/', '--'))
                       / 'snapshots' / teacher['model_revision'] / 'transformer/config.json')
        if digest(config_path) != teacher['config_sha256']:
            raise RuntimeError('Model configuration differs.')
        module = sys.modules[type(model).__module__]
        original_dispatch = module.dispatch_attention_fn
        routes = Counter()
        def routed_attention(query, key_tensor, value, **kwargs):
            masked = kwargs.get('attn_mask') is not None
            route = 'original_masked' if masked else 'flash_unmasked'
            routes[(route, query.shape[1], key_tensor.shape[1])] += 1
            if masked:
                return original_dispatch(query, key_tensor, value, **kwargs)
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
                return original_dispatch(query, key_tensor, value, **kwargs)
        module.dispatch_attention_fn = routed_attention
        for record in request['records']:
            path = Path('/phase3/phase5/capture') / record['request_key'] / 'targets.pt'
            if digest(path) != record['tensor_sha256']:
                raise RuntimeError('Capture bundle differs.')
            bundle = torch.load(path, weights_only=True, map_location='cpu')
            if bundle['request_key'] != record['request_key'] or bundle['row']['source_id'] != record['source_id']:
                raise RuntimeError('Bundle provenance differs.')
            cache = QwenImage21KVCache(32)
            outputs, controls = [], []
            with torch.inference_mode(), model.cache_context('cond'):
                for stage in bundle['stages']:
                    kwargs = {**gpu(bundle['shared_transformer_kwargs']), **gpu(stage['kwargs']), 'kv_cache': cache}
                    pred = model(**kwargs)[0][:, -stage['target_image_tokens']:].detach().cpu().contiguous()
                    report['teacher_calls'] += 1
                    target = stage['teacher_velocity']
                    if not torch.isfinite(pred).all():
                        raise RuntimeError('Nonfinite prediction.')
                    delta = pred.float() - target.float()
                    relative = float(delta.norm() / target.float().norm().clamp_min(1e-12))
                    entry = {'source_id': record['source_id'], 'split': bundle['row']['split'], 'stage': stage['stage'],
                             'relative_l2_error': relative, 'max_abs_error': float(delta.abs().max()),
                             'exact_equal': bool(torch.equal(pred, target)), 'passed': relative <= 0.001}
                    controls.append(entry)
                    outputs.append({'stage': stage['stage'], 'velocity': pred})
                    if relative > 0.001:
                        break
            name = record['capture_id'] + '.pt'
            output_path = directory / name
            temporary = output_path.with_suffix('.tmp')
            torch.save({'request_key': key, 'source_id': record['source_id'], 'outputs': outputs, 'controls': controls}, temporary)
            temporary.replace(output_path)
            report['controls'].extend(controls)
            report['artifacts'].append({'name': name, 'sha256': digest(output_path), 'bytes': output_path.stat().st_size})
            (directory / 'progress.json').write_text(json.dumps(report, indent=2))
            results_volume.commit()
            if not all(c['passed'] for c in controls):
                report['status'] = 'control_mismatch'
                break
        else:
            report['status'] = 'passed'
        report['max_relative_l2_error'] = max(c['relative_l2_error'] for c in report['controls'])
        report['exact_equal_controls'] = sum(c['exact_equal'] for c in report['controls'])
        report['completed_prompts'] = len(report['artifacts'])
        report['attention_routes'] = [{'route': k[0], 'query_tokens': k[1], 'key_tokens': k[2], 'calls': count} for k, count in routes.items()]
        module.dispatch_attention_fn = original_dispatch
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    results_volume.commit()
    return report
