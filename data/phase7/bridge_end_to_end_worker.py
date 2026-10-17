def train_bridge_end_to_end(request, key):
    """Train a bridge for one block inside frozen B against the final velocity (bridge weights only).

    Verifies the teacher contract and exact B replay, computes live teacher targets for TRAIN inputs, then trains
    each bridge variant end to end through the frozen downstream blocks and evaluates the whole student.
    """
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

    directory = Path('/phase3/phase7/bridge_end_to_end') / key
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / 'summary.json'
    if report_path.exists():
        return json.loads(report_path.read_text())
    if (directory / 'started.json').exists():
        raise RuntimeError('Incomplete healing attempt exists; inspect before repeating calls.')
    (directory / 'started.json').write_text(json.dumps({'request_key': key}))
    (directory / 'request.json').write_text(json.dumps({'request_key': key, 'request': request}, indent=2))
    results_volume.commit()
    report = {'request_key': key, 'status': 'failed', 'mode': request['mode'], 'teacher_calls': 0,
              'student_calls': 0, 'optimizer_updates': 0, 'controls': [], 'artifacts': [],
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
        if request['mode'] != 'verify':
            raise RuntimeError('Healing requires the verified teacher reference.')
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
        expected = request['expected_state']
        derived_names = sorted(n for n in expected if n.startswith('pos_embed.freqs.') or n == 'time_text_embed.time_proj.freqs')

        def derived_tensors():
            values = {f'pos_embed.freqs.{axis}': value for axis, value in enumerate(model.pos_embed.freqs)}
            values['time_text_embed.time_proj.freqs'] = model.time_text_embed.time_proj.freqs
            return values

        def table_hash(value):
            return hashlib.sha256(memoryview(value.detach().cpu().contiguous().view(torch.uint8).numpy())).hexdigest()

        canonical_path = Path('/phase3/phase7/canonical_derived_tables') / (request['canonical_tables_key'] + '.pt')
        import time
        mismatched = [n for n in derived_names if state.get(n) != expected.get(n)]
        if not mismatched:
            if not canonical_path.exists():
                canonical_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = canonical_path.with_suffix('.' + report['process']['nonce'] + '.tmp')
                torch.save({n: v.detach().cpu().contiguous() for n, v in derived_tensors().items()}, temporary)
                temporary.replace(canonical_path)
                results_volume.commit()
                report['canonical_tables_saved'] = True
        else:
            deadline = time.time() + 900
            while not canonical_path.exists():
                if time.time() > deadline:
                    raise RuntimeError('Derived tables differ on this host and no canonical copy became available.')
                time.sleep(20)
                results_volume.reload()
            saved = torch.load(canonical_path, weights_only=True, map_location='cpu')
            if sorted(saved) != derived_names or any(table_hash(saved[n]) != expected[n]['sha256'] for n in derived_names):
                raise RuntimeError('Canonical derived tables differ from the audited hashes.')
            original = model.pos_embed.freqs
            replaced = [saved[f'pos_embed.freqs.{axis}'].to(device=value.device, dtype=value.dtype)
                        for axis, value in enumerate(original)]
            model.pos_embed.freqs = tuple(replaced) if isinstance(original, tuple) else replaced
            with torch.no_grad():
                model.time_text_embed.time_proj.freqs.copy_(saved['time_text_embed.time_proj.freqs'])
            for n, value in derived_tensors().items():
                state[n] = {**state[n], 'sha256': table_hash(value)}
            report['canonical_tables_applied'] = mismatched
        if state != request['expected_state']:
            expected = request['expected_state']
            different_names = sorted(name for name in expected.keys() & state.keys()
                                     if state[name] != expected[name])
            report['state_mismatch'] = {
                'actual_count': len(state), 'expected_count': len(expected),
                'missing_names': sorted(set(expected) - set(state))[:12],
                'extra_names': sorted(set(state) - set(expected))[:12],
                'different': [{'name': name, 'actual': state[name], 'expected': expected[name]}
                              for name in different_names][:12]}
            allowed_derived_tables = {'pos_embed.freqs.1', 'pos_embed.freqs.2', 'time_text_embed.time_proj.freqs'}
            if set(state) != set(expected) or not set(different_names).issubset(allowed_derived_tables):
                raise RuntimeError('Loaded teacher learned state differs from audited reference.')
            report['derived_table_variation_requires_output_replay'] = different_names
        report['state_sha256'] = hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()
        torch.cuda.synchronize()
        argument_order = ('hidden_states', 'timestep', 'encoder_hidden_states', 'encoder_hidden_states_mask',
                          'img_shapes', 'img_mask', 'attention_kwargs', 'kv_cache_mode', 'return_dict')
        # Every intact-teacher control must pass before any block is touched.
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
            output_path = directory / filename
            temporary = output_path.with_suffix('.tmp')
            torch.save({'request_key': key, 'source_id': record['source_id'], 'input_sha256': record['tensor_sha256'],
                        'outputs': outputs, 'controls': controls}, temporary)
            temporary.replace(output_path)
            report['controls'].extend(controls)
            report['artifacts'].append({'name': filename, 'sha256': digest(output_path), 'bytes': output_path.stat().st_size})
            (directory / 'progress.json').write_text(json.dumps(report, indent=2))
            results_volume.commit()
            if not all(c['passed'] for c in controls):
                raise RuntimeError('Independent-process reference replay exceeds tolerance.')
        if report['teacher_calls'] != 90:
            raise RuntimeError('Reference attempt did not cover all 90 saved stage inputs.')
        report['legacy_max_relative_l2'] = max(c['legacy_difference']['relative_l2_error'] for c in report['controls'])
        report['max_reference_relative_l2'] = max(c['reference_difference']['relative_l2_error'] for c in report['controls'])
        report['exact_equal_controls'] = sum(c['reference_difference']['exact_equal'] for c in report['controls'])
        if not all(c['passed'] for c in report['controls']):
            raise RuntimeError('Teacher controls did not pass.')
        report['status'] = 'teacher_verified'
        (directory / 'teacher_control_report.json').write_text(json.dumps(report, indent=2))
        results_volume.commit()
        print('All 90 intact-teacher controls passed; assembling frozen 31-block candidate B.', flush=True)

        import gc
        import time
        import statistics
        from types import SimpleNamespace
        
        # Inputs are independently identified; 10 gate and 50 wider development prompts.
        panels = {'gate10': [], 'wider50': []}
        for panel, records in (('gate10', [r for r in request['records'] if r['capture_id'].startswith('p5-validation-')]),
                               ('wider50', request['expanded_validation']['records'])):
            for record in records:
                path = Path('/phase3/phase5/capture') / record['request_key'] / 'targets.pt'
                if digest(path) != record['tensor_sha256']:
                    raise RuntimeError('Fixed input hash differs')
                item = torch.load(path, weights_only=True, map_location='cpu')
                if item['row']['source_id'] != record['source_id'] or item['row']['split'] != 'validation':
                    raise RuntimeError('Validation input identity differs')
                panels[panel].append(item)
        if [len(panels[p]) for p in ('gate10', 'wider50')] != [10, 50]:
            raise RuntimeError('Saved validation panel sizes differ')

        def forward(item, stage, cache):
            full = {**item['shared_transformer_kwargs'], **stage['kwargs']}
            kwargs = gpu({field: full[field] for field in argument_order})
            kwargs['kv_cache'] = cache
            return model(**kwargs)[0][:, -stage['target_image_tokens']:].detach().cpu().contiguous()

        def predictions():
            rows = []
            with torch.inference_mode(), model.cache_context('cond'), sdpa_kernel([SDPBackend.MATH]):
                for panel, items in panels.items():
                    for item in items:
                        cache = QwenImage21KVCache(32)
                        for stage in item['stages']:
                            pred = forward(item, stage, cache)
                            if not torch.isfinite(pred).all() or pred.float().norm() == 0:
                                raise RuntimeError('Invalid reference velocity')
                            rows.append(dict(panel=panel, source_id=item['row']['source_id'],
                                             dimension=item['row']['dimension'],
                                             subdimension=item['row'].get('subdimension'),
                                             stage=stage['stage'], velocity=pred))
            return rows

        def save(name, value):
            path = directory / name
            temporary = path.with_suffix('.tmp')
            torch.save(value, temporary)
            temporary.replace(path)
            artifact = dict(name=name, sha256=digest(path), bytes=path.stat().st_size)
            report['artifacts'].append(artifact)
            return artifact

        teacher_predictions = predictions()
        save('teacher_predictions.pt', teacher_predictions)
        report['teacher_calls'] += len(teacher_predictions)
        # ---- Live teacher targets for TRAIN inputs (stored legacy targets differ by up to ~2.6%). ----
        def load_verified(path, sha256, size=None):
            if (size is not None and path.stat().st_size != size) or digest(path) != sha256:
                raise RuntimeError('Input tensor differs: ' + str(path))
            return torch.load(path, weights_only=True, map_location='cpu')

        train_full, early_pool = [], []
        for record in request['records']:
            if not record['capture_id'].startswith('p5-train-'):
                continue
            item = load_verified(Path('/phase3/phase5/capture') / record['request_key'] / 'targets.pt', record['tensor_sha256'])
            if item['row']['split'] != 'train' or item['row']['source_id'] != record['source_id']:
                raise RuntimeError('Training record identity differs.')
            train_full.append(item)
        fresh = request['fresh_aligned_train']
        for artifact in fresh['artifacts']:
            item = load_verified(Path(fresh['remote_directory']) / artifact['name'], artifact['sha256'], artifact['bytes'])
            if item['source_id'] != artifact['source_id'] or item['row']['split'] != 'train' or len(item['stages']) != 3:
                raise RuntimeError('Fresh aligned training provenance differs.')
            train_full.append(item)
        for example in request['early_pool']['examples']:
            item = load_verified(Path(example['remote_path']), example['sha256'], example['bytes'])
            if (item['capture_id'] != example['capture_id'] or item['row']['split'] != 'train'
                    or [s['stage'] for s in item['stages']] != ['early']):
                raise RuntimeError('Early pool example provenance differs.')
            early_pool.append(item)
        validation_ids = {r['source_id'] for r in request['records'] if r['capture_id'].startswith('p5-validation-')}
        validation_ids |= {r['source_id'] for r in request['expanded_validation']['records']}
        train_ids = {i['row']['source_id'] for i in train_full} | {i['row']['source_id'] for i in early_pool}
        if len(train_full) != 60 or train_ids & validation_ids:
            raise RuntimeError('Training set size or split differs.')
        for item in train_full + early_pool:
            cache = QwenImage21KVCache(32)
            item['live_targets'] = []
            for stage in item['stages']:
                with torch.inference_mode(), model.cache_context('cond'), sdpa_kernel([SDPBackend.MATH]):
                    item['live_targets'].append(forward(item, stage, cache).to(torch.bfloat16))
                report['teacher_calls'] += 1
        report['train_counts'] = {'full_prompts': len(train_full), 'early_pool': len(early_pool)}
        print('Live teacher targets for', report['train_counts'], flush=True)

        # Rebuild the Phase 6 selected student exactly as its verified replay did:
        # 31 teacher blocks plus the frozen pretrained rank-256 bridge at slot 5.
        namespace = {}
        exec(compile(request['core_source'], 'bridge_definition.py', 'exec'), namespace)
        exec(compile(request['wrapper_source'], 'bridge_block_definition.py', 'exec'), namespace)
        candidate = request['selected_candidate']
        path = Path('/phase3/phase5/fitting') / request['fit_key'] / Path(candidate['checkpoint_path']).name
        if digest(path) != candidate['checkpoint_sha256']:
            raise RuntimeError('Selected bridge checkpoint differs.')
        checkpoint = torch.load(path, weights_only=True, map_location='cpu')
        if (checkpoint['request_key'] != request['fit_key'] or checkpoint['layer'] != 5
                or checkpoint['step'] != candidate['selected_update']):
            raise RuntimeError('Bridge provenance differs.')
        bridge = namespace['ResidualBottleneckBridge'](4096, 256)
        bridge.load_state_dict(checkpoint['state_dict'])
        bridge.requires_grad_(False)
        namespace['insert_bridge'](model, 5, bridge, QwenImage21KVCache)
        del checkpoint
        model.requires_grad_(False).eval()
        gc.collect()
        torch.cuda.empty_cache()
        baseline = predictions()
        report['student_calls'] += len(baseline)
        save('candidate_predictions.pt', baseline)
        # The 10 gate prompts must reproduce the saved Phase 6 trained-bridge velocities.
        replay = request['selected_replay']
        replay_dir = Path('/phase3/phase6/verified_student_replay') / replay['request_key']
        reference = {}
        for record in request['records']:
            if not record['capture_id'].startswith('p5-validation-'):
                continue
            name = 'students-' + record['capture_id'] + '.pt'
            if digest(replay_dir / name) != replay['artifacts'][name]:
                raise RuntimeError('Phase 6 student replay checksum differs.')
            saved = torch.load(replay_dir / name, weights_only=True, map_location='cpu')
            for output in saved['outputs']:
                if output['candidate_id'] == candidate['candidate_id'] and output['variant'] == 'trained':
                    reference[(output['source_id'], output['stage'])] = output['velocity']
        gate_rows = [r for r in baseline if r['panel'] == 'gate10']
        if len(reference) != 30 or len(gate_rows) != 30:
            raise RuntimeError('Phase 6 replay coverage differs.')
        reproduction = [compare(r['velocity'], reference[(r['source_id'], r['stage'])]) for r in gate_rows]
        report['candidate_replay_max_relative_l2'] = max(r['relative_l2_error'] for r in reproduction)
        if report['candidate_replay_max_relative_l2'] > 0.001:
            raise RuntimeError('Selected student failed Phase 6 output replay')
        del reference

        class IdentityBlock(torch.nn.Module):
            def __init__(self, original):
                super().__init__()
                self.attn = SimpleNamespace(processor=original.attn.processor)

            def forward(self, hidden_states, *args, **kwargs):
                return hidden_states

        def metrics(rows):
            output = []
            for measured, base, teacher_row in zip(rows, baseline, teacher_predictions):
                identity = ('panel', 'source_id', 'stage')
                if any(measured[k] != base[k] or base[k] != teacher_row[k] for k in identity):
                    raise RuntimeError('Prediction alignment differs')
                teacher_error = compare(measured['velocity'], teacher_row['velocity'])['relative_l2_error']
                baseline_error = compare(base['velocity'], teacher_row['velocity'])['relative_l2_error']
                output.append({k: measured[k] for k in measured if k != 'velocity'} |
                              dict(baseline_displacement=compare(measured['velocity'], base['velocity'])['relative_l2_error'],
                                   teacher_relative_l2=teacher_error, baseline_teacher_relative_l2=baseline_error,
                                   teacher_error_change_percentage_points=100 * (teacher_error - baseline_error)))
            return output

        def aggregate(rows):
            fields = ('baseline_displacement', 'teacher_relative_l2', 'baseline_teacher_relative_l2',
                      'teacher_error_change_percentage_points')
            result = dict(comparisons=len(rows))
            for field in fields:
                values = sorted(r[field] for r in rows)
                def percentile(fraction):
                    position = fraction * (len(values) - 1)
                    low = int(position)
                    return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (position - low)
                result[field] = dict(mean=statistics.mean(values), maximum=max(values),
                                     p10=percentile(.1), p50=percentile(.5), p90=percentile(.9))
            result['worst_regressions'] = sorted(rows, key=lambda r:r['teacher_error_change_percentage_points'], reverse=True)[:10]
            return result

        report['baseline'] = aggregate(metrics(baseline))

        # ---- End-to-end bridge training: only the bridge at the slot learns, from the final velocity error. ----
        import random
        from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVLayerCache
        slot = request['slot']
        fit = request['fit']
        conditioned = {}
        exec(compile(request['conditioned_bridge_source'], 'conditioned_bridge.py', 'exec'), conditioned)
        components = {}
        exec(compile(request['components_source'], 'healing_components.py', 'exec'), components)
        warm_spec = request['warm_start']
        warm_path = Path(warm_spec['remote_path'])
        if digest(warm_path) != warm_spec['sha256']:
            raise RuntimeError('Warm-start checkpoint differs.')
        warm = torch.load(warm_path, weights_only=True, map_location='cpu')
        if warm['layer'] != slot or warm['variant']['name'] != warm_spec['name']:
            raise RuntimeError('Warm-start provenance differs.')
        original_blocks = list(model.transformer_blocks)
        for index in range(slot + 1, len(original_blocks)):
            model.transformer_blocks[index] = components['FunctionalCheckpointBlock'](original_blocks[index], QwenImage21KVLayerCache)
        model.requires_grad_(False)
        model.gradient_checkpointing = False
        gate_teacher = [r for r in teacher_predictions if r['panel'] == 'gate10']

        def gate_error():
            errors = []
            with torch.inference_mode(), model.cache_context('cond'), sdpa_kernel([SDPBackend.MATH]):
                position = 0
                for item in panels['gate10']:
                    cache = QwenImage21KVCache(32)
                    for stage in item['stages']:
                        reference = gate_teacher[position]
                        if reference['source_id'] != item['row']['source_id'] or reference['stage'] != stage['stage']:
                            raise RuntimeError('Gate alignment differs.')
                        errors.append(compare(forward(item, stage, cache), reference['velocity'])['relative_l2_error'])
                        position += 1
            return sum(errors) / len(errors)

        def train_loss(item):
            cache = QwenImage21KVCache(32)
            losses = []
            with model.cache_context('cond'), sdpa_kernel([SDPBackend.MATH]):
                for stage, target in zip(item['stages'], item['live_targets']):
                    full = {**item['shared_transformer_kwargs'], **stage['kwargs']}
                    kwargs = gpu({field: full[field] for field in argument_order})
                    kwargs['kv_cache'] = cache
                    prediction = model(**kwargs)[0][:, -stage['target_image_tokens']:].float()
                    target = target.to('cuda').float()
                    losses.append((prediction - target).square().mean() / target.square().mean().clamp_min(1e-12))
                    report['student_calls'] += 1
            loss = torch.stack(losses).mean()
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss.')
            return loss

        report['variants'] = {}
        for variant in request['variants']:
            if variant['attention']:
                bridge = conditioned['AttentionConditionedBridge'](4096, 1024, variant['heads'], 128)
            else:
                bridge = conditioned['ConditionedBridge'](4096, 1024, True)
            missing, unexpected = bridge.load_state_dict(warm['state_dict'], strict=False)
            if unexpected or (missing and not variant['attention']):
                raise RuntimeError(f'Warm start mismatch: missing {missing}, unexpected {unexpected}')
            torch.manual_seed(fit['init_seed'])
            if variant['attention']:
                for projection in (bridge.to_q, bridge.to_k, bridge.to_v):
                    torch.nn.init.normal_(projection.weight, std=projection.in_features ** -0.5)
            bridge = bridge.float()
            conditioned['insert_conditioned_bridge'](model, slot, bridge, keep_dtype=True)
            bridge = model.transformer_blocks[slot].bridge
            bridge.requires_grad_(True)
            optimizer = torch.optim.AdamW(bridge.parameters(), lr=fit['learning_rate'], weight_decay=0.0)
            rng = random.Random(fit['sampling_seed'])
            curve, best = [], None
            started = time.perf_counter()
            for step in range(fit['updates'] + 1):
                if step % fit['eval_every'] == 0:
                    bridge.eval()
                    score = gate_error()
                    curve.append(dict(step=step, gate10=score, seconds=time.perf_counter() - started))
                    if best is None or score < best[0]:
                        best = (score, step, {k: v.detach().cpu().clone() for k, v in bridge.state_dict().items()})
                    print(f'{variant["name"]} step {step}: gate10 {score:.6f} ({time.perf_counter() - started:.0f}s)', flush=True)
                    (directory / 'progress.json').write_text(json.dumps(report | {'current_curve': curve}, indent=2))
                    results_volume.commit()
                if step == fit['updates']:
                    break
                for group in optimizer.param_groups:
                    group['lr'] = fit['learning_rate'] * min(1.0, (step + 1) / fit['warmup'])
                optimizer.zero_grad(set_to_none=True)
                # Backward recomputes checkpointed blocks: keep the same attention kernel and cache context as forward.
                with model.cache_context('cond'), sdpa_kernel([SDPBackend.MATH]):
                    (train_loss(rng.choice(train_full)) * (1 - fit['early_weight'])).backward()
                    (train_loss(rng.choice(early_pool)) * fit['early_weight']).backward()
                torch.nn.utils.clip_grad_norm_(bridge.parameters(), fit['gradient_clip'])
                optimizer.step()
                report['optimizer_updates'] += 1
            bridge.load_state_dict(best[2])
            bridge.requires_grad_(False)
            bridge.eval()
            checkpoint_path = directory / f'bridge_block_{slot:02d}_{variant["name"]}.pt'
            temporary = checkpoint_path.with_suffix('.tmp')
            torch.save({'request_key': key, 'layer': slot, 'variant': variant, 'step': best[1], 'state_dict': best[2]}, temporary)
            temporary.replace(checkpoint_path)
            artifact = dict(name=checkpoint_path.name, sha256=digest(checkpoint_path), bytes=checkpoint_path.stat().st_size)
            report['artifacts'].append(artifact)
            rows = predictions()
            report['student_calls'] += len(rows)
            records = metrics(rows)
            report['variants'][variant['name']] = dict(
                variant=variant, checkpoint=artifact, parameters=sum(v.numel() for v in best[2].values()),
                fit=dict(curve=curve, selected_step=best[1], selected_gate10=best[0]),
                overall=aggregate(records),
                panels={p: aggregate([r for r in records if r['panel'] == p]) for p in panels},
                stages={s: aggregate([r for r in records if r['stage'] == s]) for s in ('early', 'middle', 'late')},
                records=records)
            (directory / 'progress.json').write_text(json.dumps(report, indent=2))
            results_volume.commit()
            print(f'{variant["name"]}: teacher-relative {report["variants"][variant["name"]]["overall"]["teacher_relative_l2"]["mean"]:.6f}', flush=True)
            del optimizer
            gc.collect()
            torch.cuda.empty_cache()
        report['status'] = 'passed'
        report['scope'] = 'End-to-end bridge training (bridge weights only) and fixed-input velocity evaluation; images are separate.'
    except Exception as exc:
        import traceback
        report['status'] = 'failed'
        report['error'] = repr(exc)
        report['traceback'] = traceback.format_exc()
    (directory / 'summary.json').write_text(json.dumps(report, indent=2))
    results_volume.commit()
    return report
