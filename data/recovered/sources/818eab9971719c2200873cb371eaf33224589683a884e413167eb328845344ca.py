def audit_model_state(request, key):
    import hashlib
    import inspect
    import json
    import os
    import platform
    from pathlib import Path
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from diffusers import QwenImage21Pipeline
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache

    directory = Path('/phase3/phase6/state_audit') / key
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / 'summary.json'
    if report_path.exists():
        return json.loads(report_path.read_text())
    if (directory / 'started.json').exists():
        raise RuntimeError('Incomplete state audit; preserve instead of repeating.')
    (directory / 'started.json').write_text(json.dumps({'request_key': key}))
    (directory / 'request.json').write_text(json.dumps({'request': request, 'request_key': key}, indent=2))
    results_volume.commit()
    report = {'request_key': key, 'status': 'failed', 'teacher_calls': 0, 'student_calls': 0,
              'optimizer_updates': 0, 'artifacts': [], 'state': {}, 'controls': [],
              'process': {'pid': os.getpid(), 'modal_task_id': os.environ.get('MODAL_TASK_ID')}}

    def digest(path):
        h = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                h.update(chunk)
        return h.hexdigest()

    def tensor_digest(value):
        cpu = value.detach().cpu().contiguous()
        return hashlib.sha256(memoryview(cpu.view(torch.uint8).numpy())).hexdigest()

    def gpu(value):
        if isinstance(value, torch.Tensor):
            return value.to('cuda')
        if isinstance(value, dict):
            return {k: gpu(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(gpu(v) for v in value)
        return value

    def save(name, data):
        path = directory / name
        torch.save(data, path)
        report['artifacts'].append({'name': name, 'sha256': digest(path), 'bytes': path.stat().st_size})

    try:
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
        report['gpu'] = torch.cuda.get_device_name(0)
        report['cpu'] = {'machine': platform.machine(), 'threads': torch.get_num_threads(),
                         'cpuinfo': Path('/proc/cpuinfo').read_text().split('\n\n')[0]}
        teacher = request['teacher']
        pipe = QwenImage21Pipeline.from_pretrained(teacher['model_repo'], revision=teacher['model_revision'],
            dtype=torch.bfloat16, local_files_only=True).to('cuda')
        model = pipe.transformer.eval()
        for component in (model, pipe.text_encoder, pipe.vae):
            component.requires_grad_(False)
        if digest(Path(inspect.getfile(type(model)))) != request['transformer_source_sha256']:
            raise RuntimeError('Source differs.')
        constants = {}
        for kind, values in (('parameter', model.named_parameters()), ('buffer', model.named_buffers())):
            for name, value in values:
                report['state'][name] = {'kind': kind, 'dtype': str(value.dtype), 'shape': list(value.shape),
                    'stride': list(value.stride()), 'sha256': tensor_digest(value)}
                if kind == 'buffer':
                    constants[name] = value.detach().cpu().clone()
        for axis, value in enumerate(model.pos_embed.freqs):
            name = f'pos_embed.freqs.{axis}'
            report['state'][name] = {'kind': 'unregistered_rope_table', 'dtype': str(value.dtype),
                'shape': list(value.shape), 'stride': list(value.stride()), 'sha256': tensor_digest(value)}
            constants[name] = value.detach().cpu().clone()
        report['state_sha256'] = hashlib.sha256(json.dumps(report['state'], sort_keys=True).encode()).hexdigest()
        save('constants.pt', {'request_key': key, 'constants': constants})
        if request.get('previous_state'):
            previous = request['previous_state']
            report['state_differences'] = {name: {'previous': previous.get(name), 'current': value}
                for name, value in report['state'].items() if previous.get(name) != value}
        record = request['record']
        path = Path('/phase3/phase5/capture') / record['request_key'] / 'targets.pt'
        if digest(path) != record['tensor_sha256']:
            raise RuntimeError('Input bundle differs.')
        bundle = torch.load(path, weights_only=True, map_location='cpu')
        cache = QwenImage21KVCache(32)
        outputs = []
        with torch.inference_mode(), model.cache_context('cond'), sdpa_kernel([SDPBackend.MATH]):
            for stage in bundle['stages']:
                kwargs = gpu({**bundle['shared_transformer_kwargs'], **stage['kwargs']})
                kwargs['kv_cache'] = cache
                pred = model(**kwargs)[0][:, -stage['target_image_tokens']:].detach().cpu().contiguous()
                report['teacher_calls'] += 1
                target = stage['teacher_velocity']
                d = pred.float() - target.float()
                report['controls'].append({'stage': stage['stage'], 'relative_l2_error': float(d.norm()/target.float().norm().clamp_min(1e-12)),
                    'velocity_sha256': tensor_digest(pred)})
                outputs.append({'stage': stage['stage'], 'velocity': pred})
        save('velocities.pt', {'request_key': key, 'outputs': outputs})
        report['status'] = 'audit_complete'
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    results_volume.commit()
    return report
