def probe_attention(request, key):
    import contextlib
    import hashlib
    import inspect
    import json
    from pathlib import Path
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from diffusers import QwenImage21Pipeline
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache

    directory = Path('/phase3/phase6/attention_probe') / key
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / 'summary.json'
    if report_path.exists():
        return json.loads(report_path.read_text())
    if (directory / 'started.json').exists():
        raise RuntimeError('Incomplete probe exists; inspect before repeating calls.')
    (directory / 'started.json').write_text(json.dumps({'request_key': key}))
    (directory / 'request.json').write_text(json.dumps({'request': request, 'request_key': key}, indent=2))
    results_volume.commit()
    report = {'request_key': key, 'status': 'failed', 'teacher_calls': 0, 'student_calls': 0,
              'optimizer_updates': 0, 'variants': [], 'artifacts': []}

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
        teacher, record = request['teacher'], request['record']
        report['gpu'] = torch.cuda.get_device_name(0)
        if 'H100' not in report['gpu']:
            raise RuntimeError('GPU differs.')
        path = Path('/phase3/phase5/capture') / record['request_key'] / 'targets.pt'
        if digest(path) != record['tensor_sha256']:
            raise RuntimeError('Target bundle differs.')
        bundle = torch.load(path, weights_only=True, map_location='cpu')
        pipe = QwenImage21Pipeline.from_pretrained(teacher['model_repo'], revision=teacher['model_revision'],
            dtype=torch.bfloat16, local_files_only=True).to('cuda')
        model = pipe.transformer.eval()
        for component in (model, pipe.text_encoder, pipe.vae):
            component.requires_grad_(False)
        if digest(Path(inspect.getfile(type(model)))) != request['transformer_source_sha256']:
            raise RuntimeError('Model source differs.')
        backends = {'flash': SDPBackend.FLASH_ATTENTION, 'efficient': SDPBackend.EFFICIENT_ATTENTION,
                    'cudnn': SDPBackend.CUDNN_ATTENTION, 'math': SDPBackend.MATH}
        original_reduction = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        for name in request['variants']:
            variant = {'name': name, 'controls': [], 'status': 'failed'}
            outputs = []
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (False if name == 'default_full_reduction' else original_reduction)
            context = sdpa_kernel([backends[name]]) if name in backends else contextlib.nullcontext()
            try:
                cache = QwenImage21KVCache(32)
                with torch.inference_mode(), model.cache_context('cond'), context:
                    for stage in bundle['stages']:
                        kwargs = {**gpu(bundle['shared_transformer_kwargs']), **gpu(stage['kwargs']), 'kv_cache': cache}
                        pred = model(**kwargs)[0][:, -stage['target_image_tokens']:].detach().cpu().contiguous()
                        report['teacher_calls'] += 1
                        delta = pred.float() - stage['teacher_velocity'].float()
                        relative = float(delta.norm() / stage['teacher_velocity'].float().norm().clamp_min(1e-12))
                        if not torch.isfinite(pred).all():
                            raise RuntimeError('Nonfinite prediction.')
                        variant['controls'].append({'stage': stage['stage'], 'relative_l2_error': relative,
                            'max_abs_error': float(delta.abs().max()), 'exact_equal': bool(torch.equal(pred, stage['teacher_velocity']))})
                        outputs.append({'stage': stage['stage'], 'velocity': pred})
                variant['status'] = 'complete'
                variant['control_passed'] = all(c['relative_l2_error'] <= 0.001 for c in variant['controls'])
            except Exception as exc:
                variant['error'] = f'{type(exc).__name__}: {exc}'
            output_path = directory / (name + '.pt')
            torch.save({'request_key': key, 'variant': name, 'outputs': outputs}, output_path)
            report['artifacts'].append({'name': output_path.name, 'sha256': digest(output_path), 'bytes': output_path.stat().st_size})
            report['variants'].append(variant)
            (directory / 'progress.json').write_text(json.dumps(report, indent=2))
            results_volume.commit()
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = original_reduction
        report['status'] = 'probe_complete'
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    results_volume.commit()
    return report
