def replay_students(request, key):
    """Generate or verify a separately versioned deterministic teacher reference."""
    import hashlib
    import importlib.metadata
    import inspect
    import json
    import os
    import socket
    import uuid
    from pathlib import Path
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from diffusers import QwenImage21Pipeline
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache

    directory = Path('/phase3/phase6/verified_student_replay') / key
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / 'summary.json'
    if report_path.exists():
        return json.loads(report_path.read_text())
    if (directory / 'started.json').exists():
        raise RuntimeError('Incomplete reference attempt exists; inspect before repeating calls.')
    (directory / 'started.json').write_text(json.dumps({'request_key': key}))
    (directory / 'request.json').write_text(json.dumps({'request_key': key, 'request': request}, indent=2))
    results_volume.commit()
    report = {'request_key': key, 'status': 'failed', 'mode': request['mode'], 'teacher_calls': 0,
              'student_calls': 0, 'optimizer_updates': 0, 'controls': [], 'artifacts': [], 'records': [],
              'process': {'pid': os.getpid(), 'host': socket.gethostname(), 'nonce': uuid.uuid4().hex,
                          'modal_task_id': os.environ.get('MODAL_TASK_ID')}}

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

    def compare(x, y):
        a, b = x.float(), y.float()
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise RuntimeError('Nonfinite reference tensor.')
        d = a - b
        return {'relative_l2_error': float(d.norm()/b.norm().clamp_min(1e-12)),
                'max_abs_error': float(d.abs().max()), 'exact_equal': bool(torch.equal(x, y))}

    try:
        if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
            raise RuntimeError('Deterministic workspace environment differs.')
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
        teacher = request['teacher']
        report['gpu'] = torch.cuda.get_device_name(0)
        if 'H100' not in report['gpu']:
            raise RuntimeError('GPU differs.')
        report['packages'] = {p: importlib.metadata.version(p) for p in ('torch', 'diffusers', 'transformers', 'accelerate', 'pillow')}
        report['torch_cuda'] = torch.version.cuda
        report['cudnn_version'] = torch.backends.cudnn.version()
        report['settings'] = request['arithmetic_contract']
        reference_report = None
        if request['mode'] == 'verify':
            reference_dir = Path('/phase3/phase6/deterministic_reference') / request['reference_key']
            reference_report_path = reference_dir / 'summary.json'
            if hashlib.sha256(json.dumps(json.loads(reference_report_path.read_text()), sort_keys=True).encode()).hexdigest() != request['reference_report_semantic_sha256']:
                raise RuntimeError('Generation report differs.')
            reference_report = json.loads(reference_report_path.read_text())
            if reference_report['status'] != 'reference_generated':
                raise RuntimeError('Reference generation did not complete.')
            if reference_report['settings'] != request['arithmetic_contract']:
                raise RuntimeError('Arithmetic contract differs from generation.')
            report['reference_process'] = reference_report['process']
            if reference_report['process']['nonce'] == report['process']['nonce']:
                raise RuntimeError('Independent process requirement failed.')
            if reference_report['packages'] != report['packages'] or reference_report['torch_cuda'] != report['torch_cuda'] or reference_report['cudnn_version'] != report['cudnn_version']:
                raise RuntimeError('Numerical runtime differs across generation and verification.')
        pipe = QwenImage21Pipeline.from_pretrained(teacher['model_repo'], revision=teacher['model_revision'],
            dtype=torch.bfloat16, local_files_only=True).to('cuda')
        model = pipe.transformer.eval()
        for component in (model, pipe.text_encoder, pipe.vae):
            component.requires_grad_(False)
        if digest(Path(inspect.getfile(type(model)))) != request['transformer_source_sha256']:
            raise RuntimeError('Model source differs.')
        config_path = (Path('/root/.cache/huggingface/hub') / ('models--' + teacher['model_repo'].replace('/', '--'))
                       / 'snapshots' / teacher['model_revision'] / 'transformer/config.json')
        if digest(config_path) != teacher['config_sha256'] or len(model.transformer_blocks) != 32:
            raise RuntimeError('Teacher configuration differs.')
        state = {}
        for kind, values in (('parameter', model.named_parameters()), ('buffer', model.named_buffers())):
            for name, value in values:
                cpu = value.detach().cpu().contiguous()
                state[name] = {'kind': kind, 'dtype': str(value.dtype), 'shape': list(value.shape),
                    'stride': list(value.stride()), 'sha256': hashlib.sha256(memoryview(cpu.view(torch.uint8).numpy())).hexdigest()}
        for axis, value in enumerate(model.pos_embed.freqs):
            name = f'pos_embed.freqs.{axis}'
            cpu = value.detach().cpu().contiguous()
            state[name] = {'kind': 'unregistered_rope_table', 'dtype': str(value.dtype), 'shape': list(value.shape),
                'stride': list(value.stride()), 'sha256': hashlib.sha256(memoryview(cpu.view(torch.uint8).numpy())).hexdigest()}
        if state != request['expected_state']:
            raise RuntimeError('Actual loaded teacher tensor state differs from audited reference.')
        report['state_sha256'] = hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()
        torch.cuda.synchronize()
        argument_order = ('hidden_states', 'timestep', 'encoder_hidden_states', 'encoder_hidden_states_mask',
                          'img_shapes', 'img_mask', 'attention_kwargs', 'kv_cache_mode', 'return_dict')
        for record in request['records']:
            path = Path('/phase3/phase5/capture') / record['request_key'] / 'targets.pt'
            if digest(path) != record['tensor_sha256']:
                raise RuntimeError('Legacy input bundle differs.')
            bundle = torch.load(path, weights_only=True, map_location='cpu')
            outputs, controls = [], []
            cache = QwenImage21KVCache(32)
            with torch.inference_mode(), model.cache_context('cond'), sdpa_kernel([SDPBackend.MATH]):
                for stage in bundle['stages']:
                    full = {**bundle['shared_transformer_kwargs'], **stage['kwargs']}
                    kwargs = gpu({field: full[field] for field in argument_order})
                    kwargs['kv_cache'] = cache
                    pred = model(**kwargs)[0][:, -stage['target_image_tokens']:].detach().cpu().contiguous()
                    report['teacher_calls'] += 1
                    legacy = compare(pred, stage['teacher_velocity'])
                    entry = {'source_id': record['source_id'], 'split': bundle['row']['split'],
                             'stage': stage['stage'], 'legacy_difference': legacy}
                    outputs.append({'stage': stage['stage'], 'step_index': stage['step_index'],
                                    'scheduler_sigma': stage['scheduler_sigma'], 'velocity': pred})
                    controls.append(entry)
            filename = record['capture_id'] + '.pt'
            artifact = next(a for a in reference_report['artifacts'] if a['name'] == filename)
            if digest(reference_dir / filename) != artifact['sha256']:
                raise RuntimeError('Reference output checksum differs.')
            reference_bundle = torch.load(reference_dir / filename, weights_only=True, map_location='cpu')
            if reference_bundle['source_id'] != record['source_id'] or reference_bundle['input_sha256'] != record['tensor_sha256']:
                raise RuntimeError('Reference input provenance differs.')
            reference_outputs = {v['stage']: v['velocity'] for v in reference_bundle['outputs']}
            for entry, measured in zip(controls, outputs):
                entry['reference_difference'] = compare(measured['velocity'], reference_outputs[entry['stage']])
                entry['passed'] = entry['reference_difference']['relative_l2_error'] <= 0.001
            name = record['capture_id'] + '.pt'
            output_path = directory / name
            temporary = output_path.with_suffix('.tmp')
            torch.save({'request_key': key, 'source_id': record['source_id'], 'input_sha256': record['tensor_sha256'],
                        'outputs': outputs, 'controls': controls}, temporary)
            temporary.replace(output_path)
            report['controls'].extend(controls)
            report['artifacts'].append({'name': name, 'sha256': digest(output_path), 'bytes': output_path.stat().st_size})
            (directory / 'progress.json').write_text(json.dumps(report, indent=2))
            results_volume.commit()
            if request['mode'] == 'verify' and not all(c['passed'] for c in controls):
                raise RuntimeError('Independent-process reference replay exceeds tolerance.')
        if report['teacher_calls'] != 90:
            raise RuntimeError('Reference attempt did not cover all 90 saved stage inputs.')
        report['legacy_max_relative_l2'] = max(c['legacy_difference']['relative_l2_error'] for c in report['controls'])
        report['status'] = 'reference_generated' if request['mode'] == 'generate' else 'reference_verified'
        if request['mode'] == 'verify':
            report['max_reference_relative_l2'] = max(c['reference_difference']['relative_l2_error'] for c in report['controls'])
            report['exact_equal_controls'] = sum(c['reference_difference']['exact_equal'] for c in report['controls'])

        # Complete every intact-teacher gate before modifying a block.
        if not all(c['passed'] for c in report['controls']):
            raise RuntimeError('Teacher controls did not pass.')
        report['status'] = 'teacher_verified'
        (directory / 'teacher_control_report.json').write_text(json.dumps(report, indent=2))
        results_volume.commit()
        print('All 90 intact-teacher controls passed; beginning independent student replay.', flush=True)
        namespace = {}
        exec(compile(request['core_source'], 'bridge_definition.py', 'exec'), namespace)
        exec(compile(request['wrapper_source'], 'bridge_block_definition.py', 'exec'), namespace)
        Bridge, insert = namespace['ResidualBottleneckBridge'], namespace['insert_bridge']
        weights = {}
        for candidate in request['candidates']:
            path = Path('/phase3/phase5/fitting') / request['fit_key'] / Path(candidate['checkpoint_path']).name
            if digest(path) != candidate['checkpoint_sha256']:
                raise RuntimeError('Selected bridge checkpoint differs.')
            checkpoint = torch.load(path, weights_only=True, map_location='cpu')
            if checkpoint['request_key'] != request['fit_key'] or checkpoint['layer'] != candidate['removed_teacher_slots'][0] or checkpoint['step'] != candidate['selected_update']:
                raise RuntimeError('Bridge provenance differs.')
            weights[candidate['candidate_id']] = checkpoint['state_dict']

        def metrics(pred, target):
            a, b = pred.float(), target.float()
            if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                raise RuntimeError('Nonfinite student velocity.')
            d = a - b
            return {'relative_l2_error': float(d.norm()/b.norm().clamp_min(1e-12)),
                    'mse': float(d.square().mean()), 'rmse': float(d.square().mean().sqrt()),
                    'max_abs_error': float(d.abs().max()),
                    'cosine': float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0))}

        for record in request['validation']:
            path = Path('/phase3/phase5/capture') / record['request_key'] / 'targets.pt'
            if digest(path) != record['tensor_sha256']:
                raise RuntimeError('Student input bundle differs.')
            bundle = torch.load(path, weights_only=True, map_location='cpu')
            if bundle['row']['split'] != 'validation' or bundle['row']['source_id'] != record['source_id']:
                raise RuntimeError('Student input split or provenance differs.')
            outputs = []
            for candidate in request['candidates']:
                layer = candidate['removed_teacher_slots'][0]
                original = model.transformer_blocks[layer]
                try:
                    for variant in ('identity', 'trained'):
                        model.transformer_blocks[layer] = original
                        bridge = Bridge(4096, 256)
                        bridge.load_state_dict(weights[candidate['candidate_id']])
                        if variant == 'identity':
                            with torch.no_grad():
                                bridge.up.weight.zero_()
                        bridge.requires_grad_(False)
                        insert(model, layer, bridge, QwenImage21KVCache)
                        cache = QwenImage21KVCache(32)
                        with torch.inference_mode(), model.cache_context('cond'), sdpa_kernel([SDPBackend.MATH]):
                            for stage in bundle['stages']:
                                full = {**bundle['shared_transformer_kwargs'], **stage['kwargs']}
                                kwargs = gpu({field: full[field] for field in argument_order})
                                kwargs['kv_cache'] = cache
                                pred = model(**kwargs)[0][:, -stage['target_image_tokens']:].detach().cpu().contiguous()
                                report['student_calls'] += 1
                                outputs.append({'source_id': record['source_id'], 'dimension': record['dimension'],
                                    'candidate_id': candidate['candidate_id'], 'variant': variant, 'stage': stage['stage'],
                                    'step_index': stage['step_index'], 'scheduler_sigma': stage['scheduler_sigma'], 'velocity': pred})
                finally:
                    model.transformer_blocks[layer] = original
            filename = record['capture_id'] + '.pt'
            artifact = next(a for a in reference_report['artifacts'] if a['name'] == filename)
            if digest(reference_dir / filename) != artifact['sha256']:
                raise RuntimeError('Student reference differs.')
            reference_bundle = torch.load(reference_dir / filename, weights_only=True, map_location='cpu')
            if reference_bundle['source_id'] != record['source_id'] or reference_bundle['input_sha256'] != record['tensor_sha256']:
                raise RuntimeError('Student reference provenance differs.')
            targets = {o['stage']: o['velocity'] for o in reference_bundle['outputs']}
            for output in outputs:
                entry = {k: v for k, v in output.items() if k != 'velocity'}
                entry.update(metrics(output['velocity'], targets[output['stage']]))
                output.update({k: v for k, v in entry.items() if k not in output})
                report['records'].append(entry)
            output_path = directory / ('students-' + filename)
            temporary = output_path.with_suffix('.tmp')
            torch.save({'request_key': key, 'input_sha256': record['tensor_sha256'], 'source_id': record['source_id'],
                        'outputs': outputs}, temporary)
            temporary.replace(output_path)
            report['artifacts'].append({'name': output_path.name, 'sha256': digest(output_path), 'bytes': output_path.stat().st_size})
            (directory / 'progress.json').write_text(json.dumps(report, indent=2))
            results_volume.commit()
            print('Saved student prompt', record['source_id'], '| forwards', report['student_calls'], flush=True)
        if report['student_calls'] != 240 or len(report['records']) != 240:
            raise RuntimeError('Student replay count differs.')
        aggregates = []
        for candidate in request['candidates']:
            for stage in ('early', 'middle', 'late', 'all'):
                rows = [r for r in report['records'] if r['candidate_id'] == candidate['candidate_id'] and (stage == 'all' or r['stage'] == stage)]
                baseline = [r for r in rows if r['variant'] == 'identity']
                trained = [r for r in rows if r['variant'] == 'trained']
                expected = 30 if stage == 'all' else 10
                if len(baseline) != expected or len(trained) != expected:
                    raise RuntimeError('Aggregate counts differ.')
                by_key = {(r['source_id'], r['stage']): r for r in baseline}
                identity_mean = sum(r['relative_l2_error'] for r in baseline) / expected
                trained_mean = sum(r['relative_l2_error'] for r in trained) / expected
                aggregates.append({'candidate_id': candidate['candidate_id'], 'stage': stage, 'comparisons': expected,
                    'identity_mean_relative_l2': identity_mean, 'trained_mean_relative_l2': trained_mean,
                    'relative_error_reduction': 1 - trained_mean / identity_mean,
                    'trained_max_relative_l2': max(r['relative_l2_error'] for r in trained),
                    'improved_comparisons': sum(r['relative_l2_error'] < by_key[(r['source_id'], r['stage'])]['relative_l2_error'] for r in trained)})
        report.update(status='passed', aggregates=aggregates,
            scope='fixed-input validation only; four independent one-block replacements; existing Phase 5 weights; no images, trajectories, training or healing')
    except Exception as exc:
        report['status'] = 'failed'
        report['error'] = f'{type(exc).__name__}: {exc}'
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    results_volume.commit()
    return report
