def capture_hidden(request, key):
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

    directory = Path('/phase3/phase6/aligned_hidden_capture') / key
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
              'student_calls': 0, 'optimizer_updates': 0, 'controls': [], 'artifacts': [], 'hidden_differences': [],
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
            outputs, controls, pairs = [], [], []
            current = {}
            old_pairs = {(p['stage'], p['layer']): p for p in bundle['pairs']}
            hooks = []
            def before_block(layer):
                def hook(module, args, kwargs):
                    old = old_pairs[(current['stage'], layer)]
                    h = kwargs['hidden_states']
                    indices = old['token_indices'].to(h.device)
                    if list(h.shape) != old['full_hidden_shape']:
                        raise RuntimeError('Hidden token layout differs.')
                    if not torch.equal(kwargs['target_token_mask'][indices].detach().cpu(), old['token_roles'].bool()):
                        raise RuntimeError('Hidden token roles differ.')
                    current[('input', layer)] = h.index_select(1, indices).detach().cpu().contiguous()
                return hook
            def after_block(layer):
                def hook(module, args, kwargs, output):
                    old = old_pairs[(current['stage'], layer)]
                    target = output.index_select(1, old['token_indices'].to(output.device)).detach().cpu().contiguous()
                    source = current.pop(('input', layer))
                    if not torch.isfinite(source).all() or not torch.isfinite(target).all():
                        raise RuntimeError('Nonfinite captured hidden states.')
                    pairs.append({**{k: v for k, v in old.items() if k not in ('input', 'target')}, 'input': source, 'target': target})
                return hook
            for layer in request['candidate_layers']:
                hooks.extend([model.transformer_blocks[layer].register_forward_pre_hook(before_block(layer), with_kwargs=True),
                              model.transformer_blocks[layer].register_forward_hook(after_block(layer), with_kwargs=True)])
            cache = QwenImage21KVCache(32)
            with torch.inference_mode(), model.cache_context('cond'), sdpa_kernel([SDPBackend.MATH]):
                for stage in bundle['stages']:
                    current['stage'] = stage['stage']
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
            for hook in hooks:
                hook.remove()
            if len(pairs) != 12 or any(isinstance(k, tuple) for k in current):
                raise RuntimeError('Hidden capture is incomplete.')
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
            by_stage = {v['stage']: v['velocity'] for v in outputs}
            stages = [{**stage, 'teacher_velocity': by_stage[stage['stage']]} for stage in bundle['stages']]
            torch.save({'version': 'phase6-contract-aligned-hidden-v1', 'request_key': key,
                        'source_id': record['source_id'], 'input_sha256': record['tensor_sha256'],
                        'row': bundle['row'], 'shared_transformer_kwargs': bundle['shared_transformer_kwargs'],
                        'stages': stages, 'pairs': pairs, 'outputs': outputs, 'controls': controls}, temporary)
            temporary.replace(output_path)
            for pair in pairs:
                old = old_pairs[(pair['stage'], pair['layer'])]
                for role, label in ((0, 'prefix'), (1, 'image')):
                    mask = pair['token_roles'] == role
                    if not mask.any():
                        continue
                    report['hidden_differences'].append({'source_id': record['source_id'], 'split': bundle['row']['split'],
                        'layer': pair['layer'], 'stage': pair['stage'], 'role': label,
                        'input_difference': compare(pair['input'][:, mask], old['input'][:, mask]),
                        'target_difference': compare(pair['target'][:, mask], old['target'][:, mask])})
            report['controls'].extend(controls)
            report['artifacts'].append({'name': name, 'sha256': digest(output_path), 'bytes': output_path.stat().st_size})
            (directory / 'progress.json').write_text(json.dumps(report, indent=2))
            results_volume.commit()
            print('Saved aligned hidden targets:', record['source_id'], '| teacher forwards', report['teacher_calls'], flush=True)
            if request['mode'] == 'verify' and not all(c['passed'] for c in controls):
                raise RuntimeError('Independent-process reference replay exceeds tolerance.')
        if report['teacher_calls'] != 90:
            raise RuntimeError('Reference attempt did not cover all 90 saved stage inputs.')
        report['legacy_max_relative_l2'] = max(c['legacy_difference']['relative_l2_error'] for c in report['controls'])
        report['status'] = 'reference_generated' if request['mode'] == 'generate' else 'reference_verified'
        if request['mode'] == 'verify':
            report['max_reference_relative_l2'] = max(c['reference_difference']['relative_l2_error'] for c in report['controls'])
            report['exact_equal_controls'] = sum(c['reference_difference']['exact_equal'] for c in report['controls'])

        # Evaluate saved bridges only after all teacher forwards and controls.
        import torch.nn.functional as F
        namespace = {}
        exec(compile(request['core_source'], 'bridge_definition.py', 'exec'), namespace)
        Bridge = namespace['ResidualBottleneckBridge']
        bridges = {}
        for candidate in request['candidates']:
            path = Path('/phase3/phase5/fitting') / request['fit_key'] / Path(candidate['checkpoint_path']).name
            if digest(path) != candidate['checkpoint_sha256']:
                raise RuntimeError('Selected checkpoint differs.')
            checkpoint = torch.load(path, weights_only=True, map_location='cpu')
            if checkpoint['request_key'] != request['fit_key'] or checkpoint['layer'] != candidate['removed_teacher_slots'][0] or checkpoint['step'] != candidate['selected_update']:
                raise RuntimeError('Selected checkpoint provenance differs.')
            bridge = Bridge(4096, 256).to('cuda').eval().requires_grad_(False)
            bridge.load_state_dict(checkpoint['state_dict'])
            bridges[checkpoint['layer']] = bridge
        measured = []
        for record in request['records']:
            bundle = torch.load(directory / (record['capture_id'] + '.pt'), weights_only=True, map_location='cpu')
            for pair in bundle['pairs']:
                for role, group in ((0, 'prompt_tokens'), (1, pair['stage'] + '_image')):
                    selected = pair['token_roles'] == role
                    if not selected.any():
                        continue
                    x, y = pair['input'][:, selected].to('cuda').float(), pair['target'][:, selected].to('cuda').float()
                    for variant in ('identity', 'trained'):
                        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                            pred = x if variant == 'identity' else bridges[pair['layer']](x)
                        pred, y = pred.float(), y.float()
                        directional = (F.normalize(pred, dim=-1, eps=1e-12)-F.normalize(y, dim=-1, eps=1e-12)).square().sum(-1).mean()
                        relative_squared = ((pred-y).square().sum(-1)/y.square().sum(-1).clamp_min(1e-12)).mean()
                        measured.append({'source_id': record['source_id'], 'split': bundle['row']['split'], 'layer': pair['layer'],
                            'group': group, 'variant': variant, 'directional_loss': float(directional),
                            'relative_squared_loss': float(relative_squared),
                            'objective': float(directional + request['magnitude_weight'] * relative_squared),
                            'relative_l2_error': float((pred-y).norm()/y.norm().clamp_min(1e-12))})
        aggregates = []
        groups = ('prompt_tokens', 'early_image', 'middle_image', 'late_image')
        for layer in request['candidate_layers']:
            for split, count in (('train', 20), ('validation', 10)):
                variants = {}
                for variant in ('identity', 'trained'):
                    group_scores = []
                    for group in groups:
                        rows = [r for r in measured if r['layer']==layer and r['split']==split and r['variant']==variant and r['group']==group]
                        if len(rows) != count:
                            raise RuntimeError('Hidden fitting metric counts differ.')
                        group_scores.append({'group': group, 'prompts': count,
                            'objective': sum(r['objective'] for r in rows)/count,
                            'mean_relative_l2': sum(r['relative_l2_error'] for r in rows)/count})
                    variants[variant] = {'balanced_objective': sum(r['objective'] for r in group_scores)/4, 'groups': group_scores}
                aggregates.append({'layer': layer, 'split': split, **variants,
                    'relative_objective_reduction': 1-variants['trained']['balanced_objective']/variants['identity']['balanced_objective']})
        report['bridge_hidden_metrics'] = measured
        report['bridge_hidden_aggregates'] = aggregates
        report['status'] = 'passed'
        report['scope'] = 'aligned teacher pairs for blocks 2-5 on saved pilot inputs; frozen-bridge diagnostic only; no optimization or full trajectories'
    except Exception as exc:
        report['status'] = 'failed'
        report['error'] = f'{type(exc).__name__}: {exc}'
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    results_volume.commit()
    return report
