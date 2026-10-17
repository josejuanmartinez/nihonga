def generate_pairs(request, key):
    """Save paired teacher and pretrained-bridge images for fixed validation rows."""
    import hashlib
    import inspect
    import json
    import time
    from pathlib import Path

    import torch
    from diffusers import QwenImage21Pipeline
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache

    directory = Path('/phase3/phase6/paired_images') / key
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / 'summary.json'
    if report_path.exists():
        return json.loads(report_path.read_text())
    request_path = directory / 'request.json'
    if request_path.exists():
        if json.loads(request_path.read_text()) != {'request_key': key, 'request': request}:
            raise RuntimeError('Existing image request differs.')
    else:
        request_path.write_text(json.dumps({'request_key': key, 'request': request}, indent=2))
        results_volume.commit()

    def digest(path):
        h = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                h.update(chunk)
        return h.hexdigest()

    report = {'request_key': key, 'status': 'running', 'rows': [], 'artifacts': [],
              'calls': {'teacher': 0, 'student': 0}}
    try:
        teacher = request['teacher']
        report['gpu'] = torch.cuda.get_device_name(0)
        if 'H100' not in report['gpu']:
            raise RuntimeError('Expected H100.')
        pipe = QwenImage21Pipeline.from_pretrained(
            teacher['model_repo'], revision=teacher['model_revision'],
            dtype=torch.bfloat16, local_files_only=True).to('cuda')
        pipe.set_progress_bar_config(disable=True)
        model = pipe.transformer.eval()
        for component in (model, pipe.text_encoder, pipe.vae):
            component.requires_grad_(False)
        if digest(Path(inspect.getfile(type(model)))) != request['transformer_source_sha256']:
            raise RuntimeError('Transformer source differs.')
        config_path = (Path('/root/.cache/huggingface/hub') /
                       ('models--' + teacher['model_repo'].replace('/', '--')) /
                       'snapshots' / teacher['model_revision'] / 'transformer/config.json')
        if digest(config_path) != teacher['config_sha256'] or len(model.transformer_blocks) != 32:
            raise RuntimeError('Teacher configuration differs.')
        if type(pipe.scheduler).__name__ != teacher['scheduler_class']:
            raise RuntimeError('Scheduler class differs.')

        def generate(row, variant):
            path = directory / (row['source_id'] + '-' + variant + '.png')
            if path.exists():
                return {'name': path.name, 'sha256': digest(path), 'bytes': path.stat().st_size,
                        'seconds': None, 'reused': True}
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                image = pipe(prompt=row['prompt'], width=row['width'], height=row['height'],
                    num_inference_steps=40, true_cfg_scale=teacher['true_cfg_scale'],
                    generator=torch.Generator('cpu').manual_seed(row['seed']),
                    use_kv_cache=True, output_type='pil').images[0]
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            temporary = path.with_suffix('.tmp')
            image.save(temporary, format='PNG')
            temporary.replace(path)
            results_volume.commit()
            report['calls'][variant] += 1
            return {'name': path.name, 'sha256': digest(path), 'bytes': path.stat().st_size,
                    'seconds': elapsed, 'reused': False}

        outputs = {'teacher': {}, 'student': {}}
        for row in request['rows']:
            outputs['teacher'][row['source_id']] = generate(row, 'teacher')
        selected = request['selected_candidate']
        checkpoint_path = (Path('/phase3/phase5/fitting') / request['fit_key'] /
                           Path(selected['checkpoint_path']).name)
        if digest(checkpoint_path) != selected['checkpoint_sha256']:
            raise RuntimeError('Selected bridge checkpoint differs.')
        checkpoint = torch.load(checkpoint_path, weights_only=True, map_location='cpu')
        if checkpoint['request_key'] != request['fit_key'] or checkpoint['layer'] != 5:
            raise RuntimeError('Bridge checkpoint provenance differs.')
        namespace = {}
        exec(compile(request['wrapper_source'], 'bridge_block_definition.py', 'exec'), namespace)
        exec(compile(request['components_source'], 'healing_components.py', 'exec'), namespace)
        bridge = namespace['MasterBridge'](4096, 256)
        bridge.load_state_dict(checkpoint['state_dict'])
        removed, _ = namespace['insert_bridge'](model, 5, bridge, QwenImage21KVCache)
        del removed
        if len(model.transformer_blocks) != 32 or not hasattr(model.transformer_blocks[5], 'bridge'):
            raise RuntimeError('Student bridge insertion failed.')
        for row in request['rows']:
            outputs['student'][row['source_id']] = generate(row, 'student')
        for row in request['rows']:
            source_id = row['source_id']
            report['rows'].append({**row, 'teacher': outputs['teacher'][source_id],
                                   'student': outputs['student'][source_id]})
            report['artifacts'].extend([outputs['teacher'][source_id], outputs['student'][source_id]])
        report['status'] = 'passed'
    except Exception as exc:
        report['status'] = 'failed'
        report['error'] = type(exc).__name__ + ': ' + str(exc)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True) + '\n')
    results_volume.commit()
    return report
