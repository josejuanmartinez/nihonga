def heal_student(request, key):
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

    directory = Path('/phase3/phase6/healing_pilot') / key
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

        # This body is appended inside the audited teacher worker's try block.
        import gc
        import math
        import time
        from torch.nn import functional as F
        from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVLayerCache

        report.update(status='healing_preflight', validation_calls=0, training_calls=0,
                      smoke_calls=0, evaluations=[], history=[], baseline_reproduction=[])
        del pipe, component
        gc.collect()
        torch.cuda.empty_cache()
        torch.manual_seed(request['seed'])
        torch.cuda.manual_seed_all(request['seed'])

        namespace = {}
        exec(compile(request['wrapper_source'], 'bridge_block_definition.py', 'exec'), namespace)
        exec(compile(request['components_source'], 'healing_components.py', 'exec'), namespace)
        selected = request['selected_candidate']
        checkpoint_path = (Path('/phase3/phase5/fitting') / request['fit_key'] /
                           Path(selected['checkpoint_path']).name)
        if digest(checkpoint_path) != selected['checkpoint_sha256']:
            raise RuntimeError('Selected bridge checkpoint differs.')
        checkpoint_value = torch.load(checkpoint_path, weights_only=True, map_location='cpu')
        if (checkpoint_value['request_key'] != request['fit_key'] or checkpoint_value['layer'] != 5
                or checkpoint_value['step'] != selected['selected_update']):
            raise RuntimeError('Selected bridge checkpoint provenance differs.')
        trainable, adapters = namespace['attach_adapters'](
            model, checkpoint_value['state_dict'], namespace['insert_bridge'],
            QwenImage21KVCache, QwenImage21KVLayerCache,
            rank=request['adapter_rank'], alpha=request['adapter_alpha'])
        report['trainable_parameters'] = sum(p.numel() for p in trainable.values())
        report['trainable_tensor_count'] = len(trainable)
        report['adapter_modules'] = adapters
        report['trainable_dtype'] = sorted({str(p.dtype) for p in trainable.values()})
        if report['trainable_parameters'] != 10223616 or report['trainable_dtype'] != ['torch.float32']:
            raise RuntimeError('Trainable parameter contract differs.')
        if len(adapters) != 124:
            raise RuntimeError('Adapter count differs.')
        optimizer = torch.optim.AdamW(list(trainable.values()), lr=request['learning_rate'],
                                     weight_decay=request['weight_decay'], foreach=False)
        data = {'train': [], 'validation': []}
        index = request['aligned_target_index']
        aligned_directory = Path('/phase3') / index['remote_directory'].lstrip('/')
        for split in ('train', 'validation'):
            for record in index[split]:
                path = aligned_directory / Path(record['tensor_path']).name
                if digest(path) != record['tensor_sha256']:
                    raise RuntimeError('Aligned target tensor differs.')
                item = torch.load(path, weights_only=True, map_location='cpu')
                if (item['request_key'] != index['capture_request_key'] or item['row']['split'] != split
                        or item['source_id'] != record['source_id']
                        or item['input_sha256'] != record['legacy_input_sha256']):
                    raise RuntimeError('Aligned target provenance differs.')
                data[split].append(item)
        if len(data['train']) != 20 or len(data['validation']) != 10:
            raise RuntimeError('Healing split counts differ.')

        schedule = []
        generator = torch.Generator('cpu').manual_seed(request['sampling_seed'])
        for repeat in range(2):
            for position in torch.randperm(60, generator=generator).tolist():
                schedule.append({'prompt_index': position // 3, 'stage_index': position % 3})
        if len(schedule) != request['updates']:
            raise RuntimeError('Schedule length differs.')

        def cpu_tree(value):
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().clone()
            if isinstance(value, dict):
                return {k: cpu_tree(v) for k, v in value.items()}
            if isinstance(value, (tuple, list)):
                return type(value)(cpu_tree(v) for v in value)
            return value

        def artifact(path):
            report['artifacts'].append({'name': path.name, 'sha256': digest(path),
                                       'bytes': path.stat().st_size})

        def save_tensor(name, value):
            path = directory / name
            temporary = path.with_suffix('.tmp')
            torch.save(value, temporary)
            temporary.replace(path)
            artifact(path)
            return path

        def persist():
            (directory / 'progress.json').write_text(json.dumps(report, indent=2))
            results_volume.commit()

        save_tensor('sampling_schedule.pt', {'request_key': key, 'schedule': schedule})
        capture = {'enabled': False, 'pairs': {}, 'states': {}}
        hooks = []
        for slot in (2, 3, 4, 5):
            def hook(module, args, kwargs, output, slot=slot):
                if capture['enabled']:
                    pair = capture['pairs'][slot]
                    if list(output.shape) != pair['full_hidden_shape']:
                        raise RuntimeError('Student hidden layout differs from saved target.')
                    capture['states'][slot] = output.index_select(1, pair['token_indices'].to(output.device))
            hooks.append(model.transformer_blocks[slot].register_forward_hook(hook, with_kwargs=True))

        def forward(item, stage, cache, kind):
            full = {**item['shared_transformer_kwargs'], **stage['kwargs']}
            kwargs = gpu({field: full[field] for field in argument_order})
            kwargs['kv_cache'] = cache
            prediction = model(**kwargs)[0][:, -stage['target_image_tokens']:]
            report[kind] += 1
            report['student_calls'] += 1
            return prediction

        def losses(prediction, stage):
            target = stage['teacher_velocity'].to('cuda').float()
            output_loss = (prediction.float()-target).square().mean() / target.square().mean().clamp_min(1e-12)
            distances = {}
            for slot, state in capture['states'].items():
                pair = capture['pairs'][slot]
                y = pair['target'].to('cuda').float()
                role_losses = []
                for role in (0, 1):
                    selected_role = pair['token_roles'].to('cuda') == role
                    if selected_role.any():
                        x_role, y_role = state[:, selected_role].float(), y[:, selected_role]
                        distance = (F.normalize(x_role, dim=-1, eps=1e-12) -
                                    F.normalize(y_role, dim=-1, eps=1e-12)).square().sum(-1).mean()
                        role_losses.append(distance)
                distances[slot] = torch.stack(role_losses).mean()
            if set(distances) != {2, 3, 4, 5}:
                raise RuntimeError('Missing hidden distillation outputs.')
            hidden_loss = torch.stack([distances[s] for s in (2, 3, 4)]).mean()
            bridge_loss = distances[5]
            total = output_loss + request['hidden_weight']*hidden_loss + request['bridge_weight']*bridge_loss
            if not torch.isfinite(total):
                raise RuntimeError('Nonfinite healing objective.')
            return total, {'output_loss': float(output_loss.detach()),
                           'hidden_loss': float(hidden_loss.detach()),
                           'bridge_loss': float(bridge_loss.detach()),
                           'objective': float(total.detach())}

        def configure_capture(item, stage):
            capture.update(enabled=True, states={},
                           pairs={p['layer']: p for p in item['pairs'] if p['stage'] == stage['stage']})

        def evaluate(step, name, compare_baseline=False):
            rows, outputs = [], []
            model.eval()
            with torch.inference_mode(), model.cache_context('cond'), sdpa_kernel([SDPBackend.MATH]):
                for item in data['validation']:
                    cache = QwenImage21KVCache(32)
                    prompt_outputs = []
                    for stage in item['stages']:
                        configure_capture(item, stage)
                        prediction = forward(item, stage, cache, 'validation_calls')
                        _, parts = losses(prediction, stage)
                        cpu = prediction.detach().cpu().contiguous()
                        row = {'source_id': item['source_id'], 'dimension': item['row']['dimension'],
                               'stage': stage['stage'], **compare(cpu, stage['teacher_velocity']), **parts}
                        rows.append(row)
                        entry = {**row, 'velocity': cpu}
                        outputs.append(entry)
                        prompt_outputs.append(entry)
                        capture.update(enabled=False, states={})
                    if compare_baseline:
                        previous = request['previous_replay_report']
                        filename = 'students-' + item['row']['capture_id'] + '.pt'
                        item_artifact = next(a for a in previous['artifacts'] if a['name'] == filename)
                        previous_path = Path('/phase3/phase6/verified_student_replay') / previous['request_key'] / filename
                        if digest(previous_path) != item_artifact['sha256']:
                            raise RuntimeError('Prior student baseline artifact differs.')
                        prior = torch.load(previous_path, weights_only=True, map_location='cpu')
                        matched = {v['stage']: v['velocity'] for v in prior['outputs']
                                   if v['candidate_id'] == selected['candidate_id'] and v['variant'] == 'trained'}
                        for entry in prompt_outputs:
                            difference = compare(entry['velocity'], matched[entry['stage']])
                            report['baseline_reproduction'].append({'source_id': item['source_id'],
                                'stage': entry['stage'], **difference})
                            if difference['relative_l2_error'] > 0.001:
                                raise RuntimeError('Zero-adapter pilot baseline differs from saved student.')
            aggregate = {'step': step, 'comparisons': len(rows),
                'mean_relative_l2': sum(r['relative_l2_error'] for r in rows)/len(rows),
                'max_relative_l2': max(r['relative_l2_error'] for r in rows),
                'mean_objective': sum(r['objective'] for r in rows)/len(rows), 'records': rows,
                'stages': [{'stage': stage, 'mean_relative_l2': sum(r['relative_l2_error'] for r in rows if r['stage']==stage)/10}
                           for stage in ('early', 'middle', 'late')]}
            path = save_tensor(name, {'request_key': key, 'step': step, 'outputs': outputs})
            aggregate['artifact'] = path.name
            return aggregate, outputs

        def save_checkpoint(step):
            path = save_tensor(f'checkpoint_step_{step:03d}.pt', {
                'request_key': key, 'step': step, 'trainable_state': cpu_tree(trainable),
                'optimizer_state': cpu_tree(optimizer.state_dict()), 'schedule': schedule,
                'rng_cpu': torch.get_rng_state(), 'rng_cuda': torch.cuda.get_rng_state_all(),
                'history': report['history'], 'evaluations': report['evaluations'],
                'selected_candidate': selected, 'components_sha256': request['components_sha256']})
            persist()
            return path.name

        baseline, baseline_outputs = evaluate(0, 'validation_step_000.pt', compare_baseline=True)
        report['evaluations'].append(baseline)
        report['selected_step'] = 0
        report['selected_checkpoint'] = save_checkpoint(0)
        best = baseline['mean_relative_l2']
        print('Baseline validation relative L2:', best, '| all prior-baseline gates passed.', flush=True)

        # Test a cached middle stage, including differentiable prefix extraction.
        item = data['train'][0]
        stage = item['stages'][1]
        with torch.inference_mode(), model.cache_context('cond'), sdpa_kernel([SDPBackend.MATH]):
            capture['enabled'] = False
            cache = QwenImage21KVCache(32)
            forward(item, item['stages'][0], cache, 'smoke_calls')
            inference_prediction = forward(item, stage, cache, 'smoke_calls').detach().cpu().contiguous()
        optimizer.zero_grad(set_to_none=True)
        with torch.enable_grad(), model.cache_context('cond'), sdpa_kernel([SDPBackend.MATH]):
            capture['enabled'] = False
            cache = QwenImage21KVCache(32)
            forward(item, item['stages'][0], cache, 'smoke_calls')
            prefix_objects = {slot: cache.get_layer(slot).get() for slot in range(32) if slot != 5}
            retained_prefix = {}
            for slot in (0, 4, 6, 31):
                for label, tensor in zip(('k', 'v'), prefix_objects[slot]):
                    if not tensor.requires_grad:
                        raise RuntimeError('Prefix cache was detached from extraction.')
                    tensor.retain_grad()
                    retained_prefix[f'{slot}.{label}'] = tensor
            configure_capture(item, stage)
            gradient_prediction = forward(item, stage, cache, 'smoke_calls')
            equivalence = compare(gradient_prediction.detach().cpu(), inference_prediction)
            if equivalence['relative_l2_error'] > 0.001:
                raise RuntimeError('Checkpointed gradient forward differs from inference.')
            smoke_loss, smoke_parts = losses(gradient_prediction, stage)
            smoke_loss.backward()
            capture.update(enabled=False, states={})
        if any(cache.get_layer(slot).get()[0] is not values[0] or
               cache.get_layer(slot).get()[1] is not values[1]
               for slot, values in prefix_objects.items()):
            raise RuntimeError('Real-model backward mutated a shared prefix cache.')
        prefix_gradient_norms = {}
        for label, tensor in retained_prefix.items():
            if tensor.grad is None or not torch.isfinite(tensor.grad).all() or not tensor.grad.abs().sum() > 0:
                raise RuntimeError('Real-model gradients did not reach prefix cache: ' + label)
            prefix_gradient_norms[label] = float(tensor.grad.float().norm())
        gradient_norms = {}
        for name, param in trainable.items():
            if param.grad is None or not torch.isfinite(param.grad).all():
                raise RuntimeError('Missing or nonfinite trainable gradient: ' + name)
            gradient_norms[name] = float(param.grad.norm())
        bridge_gradient = sum(n*n for name, n in gradient_norms.items() if '.bridge.' in name)**0.5
        adapter_gradient = sum(n*n for name, n in gradient_norms.items() if '.bridge.' not in name)**0.5
        if bridge_gradient <= 0 or adapter_gradient <= 0:
            raise RuntimeError('Bridge or adapter gradients are zero.')
        if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
            raise RuntimeError('Frozen base received gradients.')
        report['gradient_smoke'] = {'source_id': item['source_id'], 'stage': 'middle',
            'inference_equivalence': equivalence, 'losses': smoke_parts,
            'bridge_gradient_norm': bridge_gradient, 'adapter_gradient_norm': adapter_gradient,
            'gradient_norms': gradient_norms, 'prefix_gradient_norms': prefix_gradient_norms,
            'shared_cache_unchanged_after_backward': True, 'frozen_base_gradients': 0}
        save_tensor('gradient_smoke.pt', {'request_key': key, 'inference_prediction': inference_prediction,
            'gradient_prediction': gradient_prediction.detach().cpu(),
            'gradients': {name: p.grad.detach().cpu() for name, p in trainable.items()},
            'prefix_gradients': {name: tensor.grad.detach().cpu() for name, tensor in retained_prefix.items()},
            'report': report['gradient_smoke']})
        optimizer.zero_grad(set_to_none=True)
        del cache, gradient_prediction, smoke_loss, retained_prefix, prefix_objects, tensor
        gc.collect()
        torch.cuda.empty_cache()
        persist()
        print('GPU cached-stage gradient preflight passed; starting optimizer updates.', flush=True)

        report['status'] = 'healing_running'
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        for update, sample in enumerate(schedule, start=1):
            item = data['train'][sample['prompt_index']]
            stage = item['stages'][sample['stage_index']]
            optimizer.zero_grad(set_to_none=True)
            with torch.enable_grad(), model.cache_context('cond'), sdpa_kernel([SDPBackend.MATH]):
                cache = QwenImage21KVCache(32)
                capture['enabled'] = False
                if sample['stage_index']:
                    forward(item, item['stages'][0], cache, 'training_calls')
                configure_capture(item, stage)
                prediction = forward(item, stage, cache, 'training_calls')
                loss, parts = losses(prediction, stage)
                loss.backward()
                capture.update(enabled=False, states={})
                norm = torch.nn.utils.clip_grad_norm_(list(trainable.values()), request['gradient_clip'],
                                                    error_if_nonfinite=True)
                optimizer.step()
            report['optimizer_updates'] += 1
            report['history'].append({'update': update, 'source_id': item['source_id'],
                'stage': stage['stage'], 'gradient_norm_before_clip': float(norm), **parts})
            del cache, prediction, loss
            if update % request['checkpoint_every'] == 0:
                evaluation, _ = evaluate(update, f'validation_step_{update:03d}.pt')
                report['evaluations'].append(evaluation)
                checkpoint_name = save_checkpoint(update)
                if evaluation['mean_relative_l2'] < best:
                    best = evaluation['mean_relative_l2']
                    report['selected_step'] = update
                    report['selected_checkpoint'] = checkpoint_name
                persist()
                print('Update', update, '| validation relative L2:', evaluation['mean_relative_l2'],
                      '| selected:', report['selected_step'], flush=True)
            elif update % 5 == 0:
                # Every update's history is durable; optimizer states are checkpointed every 30.
                persist()
                print('Completed update', update, '| objective', parts['objective'], flush=True)
        torch.cuda.synchronize()
        report['training_seconds'] = time.perf_counter()-started
        report['peak_allocated_gib'] = torch.cuda.max_memory_allocated()/2**30
        if report['optimizer_updates'] != 120 or report['training_calls'] != 200:
            raise RuntimeError('Optimizer or forward schedule counts differ.')

        chosen = torch.load(directory / report['selected_checkpoint'], weights_only=True, map_location='cpu')
        with torch.no_grad():
            for name, param in trainable.items():
                param.copy_(chosen['trainable_state'][name].to(param.device))
        selected_evaluation = next(e for e in report['evaluations'] if e['step']==report['selected_step'])
        reload_evaluation, reloaded_outputs = evaluate(report['selected_step'], 'selected_reload_validation.pt')
        saved = torch.load(directory / selected_evaluation['artifact'], weights_only=True, map_location='cpu')
        prior_outputs = {(o['source_id'], o['stage']): o['velocity'] for o in saved['outputs']}
        reload_controls = []
        for output in reloaded_outputs:
            difference = compare(output['velocity'], prior_outputs[(output['source_id'], output['stage'])])
            reload_controls.append({'source_id': output['source_id'], 'stage': output['stage'], **difference})
        if any(c['relative_l2_error'] > 0.001 for c in reload_controls):
            raise RuntimeError('Selected checkpoint reload changes validation outputs.')
        report['selected_reload_controls'] = reload_controls
        report['selected_evaluation'] = reload_evaluation
        baseline_rows = {(r['source_id'], r['stage']): r for r in baseline['records']}
        report['selected_improved_comparisons'] = sum(
            r['relative_l2_error'] < baseline_rows[(r['source_id'], r['stage'])]['relative_l2_error']
            for r in reload_evaluation['records'])
        report['selected_relative_error_reduction'] = 1-reload_evaluation['mean_relative_l2']/baseline['mean_relative_l2']

        # Audit every surviving frozen parameter, mapping wrapper names to the pinned teacher.
        frozen_count = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                continue
            original_name = name.replace('.inner.', '.').replace('.base.', '.')
            expected = request['expected_state'].get(original_name)
            cpu = param.detach().cpu().contiguous()
            actual_sha = hashlib.sha256(memoryview(cpu.view(torch.uint8).numpy())).hexdigest()
            if expected is None or actual_sha != expected['sha256']:
                raise RuntimeError('Frozen base parameter changed: ' + name)
            frozen_count += 1
        expected_count = sum(v['kind']=='parameter' and not name.startswith('transformer_blocks.5.')
                             for name, v in request['expected_state'].items())
        if frozen_count != expected_count:
            raise RuntimeError('Frozen parameter audit count differs.')
        report['frozen_parameter_audit_count'] = frozen_count
        report['frozen_parameters_unchanged'] = True
        report['validation_optimizer_updates'] = 0
        report['status'] = 'passed'
        report['scope'] = '120-update parameter-efficient block-5 healing pilot on 20 TRAIN/10 VALIDATION cached inputs; no full-weight fine-tuning, images, trajectories, independent test or speed benchmark'
        for hook in hooks:
            hook.remove()
    except Exception as exc:
        report['status'] = 'failed'
        report['error'] = f'{type(exc).__name__}: {exc}'
        if 'optimizer' in locals() and 'trainable' in locals():
            try:
                save_tensor('failure_checkpoint.pt', {'request_key': key, 'step': report['optimizer_updates'],
                    'trainable_state': cpu_tree(trainable), 'optimizer_state': cpu_tree(optimizer.state_dict()),
                    'history': report.get('history', []), 'rng_cpu': torch.get_rng_state(),
                    'rng_cuda': torch.cuda.get_rng_state_all()})
            except Exception as snapshot_error:
                report['failure_snapshot_error'] = str(snapshot_error)
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    results_volume.commit()
    return report
