def generate_bridge_images(request, key):
    """Matched 40-step images: teacher, B (block-5 pretrained bridge) and C (B plus the learned block-13 bridge).
    Same prompt, CPU noise seed, size, scheduler and 40 steps for every model. Qualitative screen only."""
    import gc
    import hashlib
    import inspect
    import json
    import os
    import time
    from pathlib import Path

    import torch
    from diffusers import QwenImage21Pipeline
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache, QwenImage21KVLayerCache

    directory = Path('/phase3/phase7/bridge_images') / key
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / 'summary.json'
    if report_path.exists():
        return json.loads(report_path.read_text())
    if (directory / 'started.json').exists():
        raise RuntimeError('Incomplete image attempt exists; inspect before repeating calls.')
    (directory / 'started.json').write_text(json.dumps({'request_key': key}))
    (directory / 'request.json').write_text(json.dumps({'request_key': key, 'request': request}, indent=2))
    results_volume.commit()

    def digest(path):
        h = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                h.update(chunk)
        return h.hexdigest()

    def table_hash(value):
        return hashlib.sha256(memoryview(value.detach().cpu().contiguous().view(torch.uint8).numpy())).hexdigest()

    variants = ['teacher', 'B', 'C']
    report = {'request_key': key, 'status': 'running', 'rows': [], 'artifacts': [],
              'calls': {v: 0 for v in variants}, 'optimizer_updates': 0}
    try:
        if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
            raise RuntimeError('Deterministic workspace environment differs.')
        teacher = request['teacher']
        report['gpu'] = torch.cuda.get_device_name(0)
        if 'H100' not in report['gpu']:
            raise RuntimeError('Expected H100.')
        pipe = QwenImage21Pipeline.from_pretrained(teacher['model_repo'], revision=teacher['model_revision'],
                                                   dtype=torch.bfloat16, local_files_only=True).to('cuda')
        pipe.set_progress_bar_config(disable=True)
        model = pipe.transformer.eval()
        for component in (model, pipe.text_encoder, pipe.vae):
            component.requires_grad_(False)
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
        if digest(Path(inspect.getfile(type(model)))) != request['transformer_source_sha256']:
            raise RuntimeError('Transformer source differs.')
        config_path = (Path('/root/.cache/huggingface/hub') / ('models--' + teacher['model_repo'].replace('/', '--')) /
                       'snapshots' / teacher['model_revision'] / 'transformer/config.json')
        if digest(config_path) != teacher['config_sha256'] or len(model.transformer_blocks) != 32:
            raise RuntimeError('Teacher configuration differs.')
        # Derived position/time tables: use the audited values on every host (as in steps 7-11).
        derived = {f'pos_embed.freqs.{axis}': value for axis, value in enumerate(model.pos_embed.freqs)}
        derived['time_text_embed.time_proj.freqs'] = model.time_text_embed.time_proj.freqs
        expected = request['expected_derived_tables']
        mismatched = sorted(n for n, v in derived.items() if table_hash(v) != expected[n])
        if mismatched:
            canonical = torch.load(Path('/phase3/phase7/canonical_derived_tables') / (request['canonical_tables_key'] + '.pt'),
                                   weights_only=True, map_location='cpu')
            if any(table_hash(canonical[n]) != expected[n] for n in expected):
                raise RuntimeError('Canonical derived tables differ from the audited hashes.')
            original = model.pos_embed.freqs
            replaced = [canonical[f'pos_embed.freqs.{a}'].to(device=v.device, dtype=v.dtype) for a, v in enumerate(original)]
            model.pos_embed.freqs = tuple(replaced) if isinstance(original, tuple) else replaced
            with torch.no_grad():
                model.time_text_embed.time_proj.freqs.copy_(canonical['time_text_embed.time_proj.freqs'])
            if any(table_hash(v) != expected[n] for n, v in
                   list({f'pos_embed.freqs.{a}': v for a, v in enumerate(model.pos_embed.freqs)}.items()) +
                   [('time_text_embed.time_proj.freqs', model.time_text_embed.time_proj.freqs)]):
                raise RuntimeError('Derived tables still differ after loading the audited copy.')
            report['canonical_tables_applied'] = mismatched

        def generate(row, variant):
            path = directory / (row['source_id'] + '-' + variant + '.png')
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
            (directory / 'progress.json').write_text(json.dumps(report, indent=2))
            results_volume.commit()
            return {'name': path.name, 'sha256': digest(path), 'bytes': path.stat().st_size, 'seconds': elapsed}

        namespace = {}
        exec(compile(request['core_source'], 'bridge_definition.py', 'exec'), namespace)
        exec(compile(request['wrapper_source'], 'bridge_block_definition.py', 'exec'), namespace)

        def add_bridge(slot, spec):
            path = Path(spec['remote_path'])
            if digest(path) != spec['sha256']:
                raise RuntimeError('Bridge checkpoint differs: ' + spec['remote_path'])
            checkpoint = torch.load(path, weights_only=True, map_location='cpu')
            if checkpoint['layer'] != slot or checkpoint['step'] != spec['step'] or checkpoint['request_key'] != spec['request_key']:
                raise RuntimeError('Bridge provenance differs.')
            bridge = namespace['ResidualBottleneckBridge'](4096, spec['rank'])
            bridge.load_state_dict(checkpoint['state_dict'])
            bridge.requires_grad_(False)
            namespace['insert_bridge'](model, slot, bridge, QwenImage21KVCache)
            model.eval()

        outputs = {v: {} for v in variants}
        for row in request['rows']:
            outputs['teacher'][row['source_id']] = generate(row, 'teacher')
        add_bridge(5, request['bridges']['5'])
        for row in request['rows']:
            outputs['B'][row['source_id']] = generate(row, 'B')
        add_bridge(13, request['bridges']['13'])
        report['surviving_blocks'] = sum(not hasattr(b, 'bridge') for b in model.transformer_blocks)
        if report['surviving_blocks'] != 30:
            raise RuntimeError('Candidate C assembly differs.')
        for row in request['rows']:
            outputs['C'][row['source_id']] = generate(row, 'C')
        for row in request['rows']:
            report['rows'].append({**row, **{v: outputs[v][row['source_id']] for v in variants}})
            report['artifacts'].extend(outputs[v][row['source_id']] for v in variants)
        report['status'] = 'passed'
    except Exception as exc:
        report['status'] = 'failed'
        report['error'] = type(exc).__name__ + ': ' + str(exc)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True) + '\n')
    results_volume.commit()
    return report
