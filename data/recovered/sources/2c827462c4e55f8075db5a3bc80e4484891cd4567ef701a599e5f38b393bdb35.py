def heal_region_student(request, key):
    """Stage-balanced healing of a student whose removed region is request['region_slots'] (step 11: blocks 4-5),
    with a fresh early-moment input from the step-7 pool in every update."""
    # Derived from heal_stage_balanced; its docstring follows.
    """Stage-balanced healing of the 28-block student: every update uses one prompt at early, middle and late noise,
    each stage's objective normalized by its update-0 training value; linear warm-up; one learning rate per request."""
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

    directory = Path('/phase3/phase7/blocks45_healing') / key
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
        print('All 90 intact-teacher controls passed; assembling the 28-block student.', flush=True)

        import gc
        import time
        from torch.nn import functional as F
        from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVLayerCache

        report.update(status='healing_preflight', validation_calls=0, gate_calls=0, training_calls=0,
                      smoke_calls=0, normalizer_calls=0, evaluations=[], history=[], baseline_reproduction=[])
        del pipe, component
        gc.collect()
        torch.cuda.empty_cache()
        torch.manual_seed(request['seed'])
        torch.cuda.manual_seed_all(request['seed'])

        namespace = {}
        exec(compile(request['core_source'], 'bridge_definition.py', 'exec'), namespace)
        exec(compile(request['wrapper_source'], 'bridge_block_definition.py', 'exec'), namespace)
        exec(compile(request['components_source'], 'healing_components.py', 'exec'), namespace)
        region = request['region_bridge']
        region_path = Path('/phase3/phase7/region_bridge_fit') / region['fit_key'] / region['checkpoint']
        if digest(region_path) != region['sha256']:
            raise RuntimeError('Region bridge checkpoint differs.')
        region_value = torch.load(region_path, weights_only=True, map_location='cpu')
        if (region_value['request_key'] != region['fit_key'] or region_value['region'] != request['region_slots']
                or region_value['rank'] != region['rank'] or region_value['step'] != region['selected_step']
                or region_value['variant'] != region['variant']):
            raise RuntimeError('Region bridge provenance differs.')
        teacher_parameters = sum(p.numel() for p in model.parameters())
        slots = request['region_slots']
        removed_parameters = sum(p.numel() for slot in slots for p in model.transformer_blocks[slot].parameters())
        bridge = namespace['MasterBridge'](4096, region['rank']).to('cuda')
        bridge.load_state_dict(region_value['state_dict'])
        removed, _ = namespace['insert_bridge'](model, slots[0], bridge, QwenImage21KVCache)
        # insert_bridge casts to the inference dtype; restore the FP32 master copy of the fitted weights.
        bridge.float()
        bridge.load_state_dict(region_value['state_dict'])
        del removed
        for slot in slots[1:]:
            # The region bridge already covers these blocks: an exact pass-through, as in the step-3 replay.
            passthrough = namespace['ResidualBottleneckBridge'](4096, 256)
            with torch.no_grad():
                passthrough.up.weight.zero_()
            removed, _ = namespace['insert_bridge'](model, slot, passthrough, QwenImage21KVCache)
            del removed
        model.requires_grad_(True)
        for slot in slots[1:]:
            model.transformer_blocks[slot].bridge.requires_grad_(False)
        for slot, block in enumerate(model.transformer_blocks):
            if slot not in slots:
                model.transformer_blocks[slot] = namespace['FunctionalCheckpointBlock'](block, QwenImage21KVLayerCache)
        model.eval()
        model.gradient_checkpointing = False
        trainable = {name: param for name, param in model.named_parameters() if param.requires_grad}
        frozen = sorted(name for name, param in model.named_parameters() if not param.requires_grad)
        region_parameters = sum(p.numel() for p in bridge.parameters())
        report['teacher_parameters'] = teacher_parameters
        report['removed_block_parameters'] = removed_parameters
        report['region_bridge_parameters'] = region_parameters
        report['trainable_parameters'] = sum(p.numel() for p in trainable.values())
        report['trainable_tensor_count'] = len(trainable)
        report['frozen_tensors'] = frozen
        report['adapter_modules'] = []
        report['trainable_dtype'] = sorted({str(p.dtype) for p in trainable.values()})
        report['surviving_blocks'] = sum(isinstance(b, namespace['FunctionalCheckpointBlock']) for b in model.transformer_blocks)
        if report['surviving_blocks'] != 32 - len(slots) or region_parameters != 2 * 4096 * region['rank']:
            raise RuntimeError('Student assembly differs.')
        if report['trainable_parameters'] != teacher_parameters - removed_parameters + region_parameters:
            raise RuntimeError('Trainable parameter contract differs.')
        if report['trainable_dtype'] != ['torch.bfloat16', 'torch.float32'] or sum('.bridge.' in n for n in trainable) != 2:
            raise RuntimeError('Trainable tensor contract differs.')
        if frozen != sorted(f'transformer_blocks.{s}.bridge.{t}.weight' for s in slots[1:] for t in ('down', 'up')):
            raise RuntimeError('Only the three pass-through slots may be frozen.')
        from transformers.optimization import Adafactor
        masters = {name: torch.nn.Parameter(param.detach().float().clone()) for name, param in trainable.items()}
        optimizer = Adafactor(list(masters.values()), lr=request['learning_rate'],
                              scale_parameter=False, relative_step=False, warmup_init=False,
                              weight_decay=request['weight_decay'])
        report['master_parameters'] = sum(p.numel() for p in masters.values())

        def clear_gradients():
            for param in trainable.values():
                param.grad = None
            optimizer.zero_grad(set_to_none=True)

        def update_model():
            with torch.no_grad():
                for name, param in trainable.items():
                    grad = param.grad
                    masters[name].grad = None if grad is None else grad.detach().float()
                    param.grad = None
                optimizer.step()
                for name, param in trainable.items():
                    param.copy_(masters[name].to(param.dtype))
                optimizer.zero_grad(set_to_none=True)

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
            raise RuntimeError('Original healing split counts differ.')
        fresh = request['fresh_aligned_train']
        fresh_directory = Path(fresh['remote_directory'])
        for artifact in fresh['artifacts']:
            path = fresh_directory / artifact['name']
            if path.stat().st_size != artifact['bytes'] or digest(path) != artifact['sha256']:
                raise RuntimeError('Fresh aligned training target differs.')
            item = torch.load(path, weights_only=True, map_location='cpu')
            if (item['request_key'] != fresh['request_key'] or item['source_id'] != artifact['source_id']
                    or item['row']['split'] != 'train' or len(item['stages']) != 3 or len(item['pairs']) != 12):
                raise RuntimeError('Fresh aligned training provenance differs.')
            data['train'].append(item)
        if len(data['train']) != 60 or len({item['source_id'] for item in data['train']}) != 60:
            raise RuntimeError('Training prompt count differs.')
        expanded = request['expanded_validation']
        data['expanded_validation'] = []
        for record in expanded['records']:
            path = Path('/phase3/phase5/capture') / record['request_key'] / 'targets.pt'
            if digest(path) != record['tensor_sha256']:
                raise RuntimeError('Expanded validation teacher input differs.')
            item = torch.load(path, weights_only=True, map_location='cpu')
            if item['row']['source_id'] != record['source_id'] or item['row']['split'] != 'validation':
                raise RuntimeError('Expanded validation provenance differs.')
            data['expanded_validation'].append(item)
        if len(data['expanded_validation']) != 50 or len({i['row']['source_id'] for i in data['expanded_validation']}) != 50:
            raise RuntimeError('Expanded validation count differs.')
        train_ids = {item['source_id'] for item in data['train']}
        if train_ids & {i['source_id'] for i in data['validation']} or train_ids & {i['row']['source_id'] for i in data['expanded_validation']}:
            raise RuntimeError('A validation prompt appears in the training set.')

        # Early-moment pool from step 7: verify every tensor before use.
        data['early_pool'] = []
        for example in request['early_pool']['examples']:
            path = Path(example['remote_path'])
            if path.stat().st_size != example['bytes'] or digest(path) != example['sha256']:
                raise RuntimeError('Early pool example differs: ' + example['capture_id'])
            item = torch.load(path, weights_only=True, map_location='cpu')
            if (item['capture_id'] != example['capture_id'] or item['row']['split'] != 'train'
                    or [s['stage'] for s in item['stages']] != ['early'] or [p['layer'] for p in item['pairs']] != [5]):
                raise RuntimeError('Early pool example provenance differs: ' + example['capture_id'])
            data['early_pool'].append(item)
        if len(data['early_pool']) != len(request['early_pool']['examples']):
            raise RuntimeError('Early pool count differs.')
        # Middle and late: passes over the 60 full TRAIN prompts. Early: drawn without replacement from the 60 original
        # early inputs plus the step-7 pool, so no early input repeats within this run.
        generator = torch.Generator('cpu').manual_seed(request['sampling_seed'])
        prompts = [i for _ in range(request['passes']) for i in torch.randperm(len(data['train']), generator=generator).tolist()]
        early_sources = [('train', i) for i in range(len(data['train']))] + [('pool', j) for j in range(len(data['early_pool']))]
        order = torch.randperm(len(early_sources), generator=generator).tolist()[:request['updates']]
        schedule = [{'prompt_index': p, 'early_source': early_sources[o]} for p, o in zip(prompts, order)]
        if len(schedule) != request['updates'] or len({tuple(s['early_source']) for s in schedule}) != request['updates']:
            raise RuntimeError('Schedule length or early-input uniqueness differs.')
        report['early_sources_used'] = {'original': sum(s['early_source'][0] == 'train' for s in schedule),
                                        'pool': sum(s['early_source'][0] == 'pool' for s in schedule)}
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
            report['artifacts'].append({'name': path.name, 'sha256': digest(path), 'bytes': path.stat().st_size})

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
        # Region supervision: the student state after slot 5 (= the region bridge output, since 3-5 pass through)
        # is compared with the teacher's saved state after original block 5 on the same sampled tokens.
        capture = {'enabled': False, 'pair': None, 'state': None}

        def hook(module, args, kwargs, output):
            if capture['enabled']:
                pair = capture['pair']
                if list(output.shape) != pair['full_hidden_shape']:
                    raise RuntimeError('Student hidden layout differs from saved target.')
                capture['state'] = output.index_select(1, pair['token_indices'].to(output.device))
        hook_handle = model.transformer_blocks[5].register_forward_hook(hook, with_kwargs=True)

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
            if capture['state'] is None:
                raise RuntimeError('Missing region hidden state.')
            pair = capture['pair']
            y = pair['target'].to('cuda').float()
            role_losses = []
            for role in (0, 1):
                selected_role = pair['token_roles'].to('cuda') == role
                if selected_role.any():
                    x_role, y_role = capture['state'][:, selected_role].float(), y[:, selected_role]
                    distance = (F.normalize(x_role, dim=-1, eps=1e-12) - F.normalize(y_role, dim=-1, eps=1e-12)).square().sum(-1).mean()
                    role_losses.append(distance)
            bridge_loss = torch.stack(role_losses).mean()
            total = output_loss + request['bridge_weight']*bridge_loss
            if not torch.isfinite(total):
                raise RuntimeError('Nonfinite healing objective.')
            return total, {'output_loss': float(output_loss.detach()), 'bridge_loss': float(bridge_loss.detach()),
                           'objective': float(total.detach())}

        def configure_capture(item, stage):
            pairs = [p for p in item['pairs'] if p['stage'] == stage['stage'] and p['layer'] == 5]
            if len(pairs) != 1:
                raise RuntimeError('Expected exactly one block-5 target for this stage.')
            capture.update(enabled=True, state=None, pair=pairs[0])

        def run_set(items, kind, id_of):
            rows, outputs = [], []
            with torch.inference_mode(), model.cache_context('cond'), sdpa_kernel([SDPBackend.MATH]):
                for item in items:
                    cache = QwenImage21KVCache(32)
                    for stage in item['stages']:
                        prediction = forward(item, stage, cache, kind)
                        cpu = prediction.detach().cpu().contiguous()
                        row = {'source_id': id_of(item), 'dimension': item['row']['dimension'], 'stage': stage['stage'],
                               **compare(cpu, stage['teacher_velocity'])}
                        rows.append(row)
                        outputs.append({**row, 'velocity': cpu})
            return rows, outputs

        def aggregate_rows(rows, per_stage):
            return {'comparisons': len(rows),
                    'mean_relative_l2': sum(r['relative_l2_error'] for r in rows)/len(rows),
                    'max_relative_l2': max(r['relative_l2_error'] for r in rows),
                    'stages': [{'stage': s, 'mean_relative_l2': sum(r['relative_l2_error'] for r in rows if r['stage'] == s)/per_stage}
                               for s in ('early', 'middle', 'late')],
                    'records': rows}

        def evaluate(step, name):
            model.eval()
            capture.update(enabled=False, state=None, pair=None)
            gate_rows, gate_outputs = run_set(data['validation'], 'gate_calls', lambda i: i['source_id'])
            monitor_rows, monitor_outputs = run_set(data['expanded_validation'], 'validation_calls', lambda i: i['row']['source_id'])
            if len(gate_rows) != 30 or len(monitor_rows) != 150:
                raise RuntimeError('Evaluation comparison counts differ.')
            gate, monitor = aggregate_rows(gate_rows, 10), aggregate_rows(monitor_rows, 50)
            references = request['gate_references']
            one, skip = references['one_block_bridge'], references[request['gate_skip_key']]
            gate['points_above_one_block'] = 100*(gate['mean_relative_l2'] - one)
            gate['skip_penalty_removed'] = (skip - gate['mean_relative_l2'])/(skip - one)
            gate['passed'] = (gate['mean_relative_l2'] - one) <= request['gate_rule']['max_points_above_one_block']/100 and \
                gate['skip_penalty_removed'] >= request['gate_rule']['min_skip_penalty_removed']
            aggregate = {'step': step, 'gate': gate, 'monitor': monitor,
                         'mean_relative_l2': monitor['mean_relative_l2'], 'gate_mean_relative_l2': gate['mean_relative_l2']}
            path = save_tensor(name, {'request_key': key, 'step': step, 'gate_outputs': gate_outputs, 'monitor_outputs': monitor_outputs})
            aggregate['artifact'] = path.name
            return aggregate, gate_outputs, monitor_outputs

        def save_checkpoint(step):
            path = save_tensor(f'checkpoint_step_{step:03d}.pt', {
                'request_key': key, 'step': step, 'master_state': cpu_tree(masters),
                'optimizer_state': cpu_tree(optimizer.state_dict()), 'schedule': schedule,
                'rng_cpu': torch.get_rng_state(), 'rng_cuda': torch.cuda.get_rng_state_all(),
                'history': report['history'], 'evaluations': report['evaluations'],
                'region_bridge': region, 'components_sha256': request['components_sha256']})
            persist()
            return str(path)

        # Update 0 must reproduce the step-3 replay of this exact student before any weight changes.
        baseline, baseline_gate_outputs, _ = evaluate(0, 'validation_step_000.pt')
        replay = request['region_replay']
        replay_directory = Path(replay['remote_directory'])
        saved_rows = {}
        for item in replay['validation_artifacts']:
            path = replay_directory / item['name']
            if digest(path) != item['sha256']:
                raise RuntimeError('Step-3 replay student outputs differ.')
            saved = torch.load(path, weights_only=True, map_location='cpu')
            for output in saved['outputs']:
                if output['variant'] == replay['variant']:
                    saved_rows[(output['source_id'], output['stage'])] = output['velocity']
        if len(saved_rows) != 30:
            raise RuntimeError('Step-3 replay rows for the region student are incomplete.')
        for output in baseline_gate_outputs:
            difference = compare(output['velocity'], saved_rows[(output['source_id'], output['stage'])])
            report['baseline_reproduction'].append({'source_id': output['source_id'], 'stage': output['stage'], **difference})
            if difference['relative_l2_error'] > 0.001:
                raise RuntimeError('Update-0 student differs from the step-3 replay of the same student.')
        baseline['gate']['replay_reproduction_difference'] = baseline['gate']['mean_relative_l2'] - replay['gate_mean_relative_l2']
        if abs(baseline['gate']['replay_reproduction_difference']) > 0.001:
            raise RuntimeError('Update-0 gate mean differs from the step-3 replay mean.')
        report['evaluations'].append(baseline)
        persist()
        print('Update 0 reproduces the step-3 replay | gate', baseline['gate_mean_relative_l2'],
              '| monitor', baseline['mean_relative_l2'], flush=True)

        # Cached middle stage with differentiable prefix extraction, as in Phase 6.
        item = data['train'][0]
        stage = item['stages'][1]
        with torch.inference_mode(), model.cache_context('cond'), sdpa_kernel([SDPBackend.MATH]):
            capture['enabled'] = False
            cache = QwenImage21KVCache(32)
            forward(item, item['stages'][0], cache, 'smoke_calls')
            inference_prediction = forward(item, stage, cache, 'smoke_calls').detach().cpu().contiguous()
        clear_gradients()
        with torch.enable_grad(), model.cache_context('cond'), sdpa_kernel([SDPBackend.MATH]):
            capture['enabled'] = False
            cache = QwenImage21KVCache(32)
            forward(item, item['stages'][0], cache, 'smoke_calls')
            prefix_objects = {slot: cache.get_layer(slot).get() for slot in range(32) if slot not in slots}
            retained_prefix = {}
            for slot in (0, 1, 6, 31):
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
            capture.update(enabled=False, state=None, pair=None)
        if any(cache.get_layer(slot).get()[0] is not values[0] or cache.get_layer(slot).get()[1] is not values[1]
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
        base_gradient = sum(n*n for name, n in gradient_norms.items() if '.bridge.' not in name)**0.5
        if bridge_gradient <= 0 or base_gradient <= 0:
            raise RuntimeError('Bridge or surviving-block gradients are zero.')
        report['gradient_smoke'] = {'source_id': item['source_id'], 'stage': 'middle',
            'inference_equivalence': equivalence, 'losses': smoke_parts,
            'bridge_gradient_norm': bridge_gradient, 'surviving_gradient_norm': base_gradient,
            'gradient_norms': gradient_norms, 'prefix_gradient_norms': prefix_gradient_norms,
            'shared_cache_unchanged_after_backward': True}
        sampled_grads = {name: p.grad.detach().cpu() for name, p in trainable.items()
                         if '.bridge.' in name or name in request['gradient_sample_names']}
        save_tensor('gradient_smoke.pt', {'request_key': key, 'inference_prediction': inference_prediction,
            'gradient_prediction': gradient_prediction.detach().cpu(), 'gradients': sampled_grads,
            'prefix_gradients': {name: tensor.grad.detach().cpu() for name, tensor in retained_prefix.items()},
            'report': report['gradient_smoke']})
        clear_gradients()
        del cache, gradient_prediction, smoke_loss, retained_prefix, prefix_objects, tensor
        gc.collect()
        torch.cuda.empty_cache()
        persist()
        print('GPU cached-stage gradient preflight passed; starting optimizer updates.', flush=True)

        # Stage normalizers: mean update-0 objective per noise stage over all TRAIN prompts (no gradients).
        sums = {'early': 0.0, 'middle': 0.0, 'late': 0.0}
        with torch.inference_mode(), model.cache_context('cond'), sdpa_kernel([SDPBackend.MATH]):
            for item in data['train']:
                cache = QwenImage21KVCache(32)
                for stage in item['stages']:
                    configure_capture(item, stage)
                    prediction = forward(item, stage, cache, 'normalizer_calls')
                    _, parts = losses(prediction, stage)
                    sums[stage['stage']] += parts['objective']
                    capture.update(enabled=False, state=None, pair=None)
        stage_means = {s: v / len(data['train']) for s, v in sums.items()}
        overall = sum(stage_means.values()) / 3
        weights = {s: overall / stage_means[s] for s in stage_means}
        report['stage_update0_objective'] = stage_means
        report['stage_weights'] = weights
        persist()
        print('Stage normalizers:', stage_means, '| weights:', weights, flush=True)

        report['status'] = 'healing_running'
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        for update, sample in enumerate(schedule, start=1):
            item = data['train'][sample['prompt_index']]
            kind, index = sample['early_source']
            early_item = data['train'][index] if kind == 'train' else data['early_pool'][index]
            learning_rate = request['learning_rate'] * min(1.0, update / request['warmup_updates'])
            for group in optimizer.param_groups:
                group['lr'] = learning_rate
            clear_gradients()
            stage_parts = {}
            for stage_index, stage in enumerate(item['stages']):
                source = early_item if stage_index == 0 else item
                if stage_index == 0:
                    stage = early_item['stages'][0]
                    if stage['stage'] != 'early':
                        raise RuntimeError('Early source is not an early-moment input.')
                # Separate graph per stage (as in step 4) keeps peak memory unchanged; gradients accumulate.
                with torch.enable_grad(), model.cache_context('cond'), sdpa_kernel([SDPBackend.MATH]):
                    cache = QwenImage21KVCache(32)
                    capture.update(enabled=False, state=None, pair=None)
                    if stage_index:
                        forward(item, item['stages'][0], cache, 'training_calls')
                    configure_capture(source, stage)
                    prediction = forward(source, stage, cache, 'training_calls')
                    loss, parts = losses(prediction, stage)
                    (weights[stage['stage']] * loss / 3).backward()
                    capture.update(enabled=False, state=None, pair=None)
                stage_parts[stage['stage']] = parts
                del cache, prediction, loss
            if set(stage_parts) != {'early', 'middle', 'late'}:
                raise RuntimeError('An update did not cover all three stages.')
            norm = torch.nn.utils.clip_grad_norm_(list(trainable.values()), request['gradient_clip'], error_if_nonfinite=True)
            update_model()
            report['optimizer_updates'] += 1
            report['history'].append({'update': update, 'source_id': item['source_id'], 'early_source': list(sample['early_source']),
                                      'early_capture_id': early_item.get('capture_id', early_item['source_id']), 'learning_rate': learning_rate,
                                      'gradient_norm_before_clip': float(norm), 'stages': stage_parts,
                                      'balanced_objective': sum(weights[s] * p['objective'] for s, p in stage_parts.items()) / 3})
            if update % request['evaluate_every'] == 0:
                evaluation, _, _ = evaluate(update, f'validation_step_{update:03d}.pt')
                report['evaluations'].append(evaluation)
                evaluation['checkpoint'] = save_checkpoint(update)
                persist()
                print('Update', update, '| gate', evaluation['gate_mean_relative_l2'], '| monitor',
                      evaluation['mean_relative_l2'], flush=True)
            elif update % 5 == 0:
                persist()
                print('Completed update', update, '| balanced objective', report['history'][-1]['balanced_objective'], flush=True)
        torch.cuda.synchronize()
        report['training_seconds'] = time.perf_counter()-started
        report['peak_allocated_gib'] = torch.cuda.max_memory_allocated()/2**30
        if report['optimizer_updates'] != request['updates'] or len(report['history']) != request['updates']:
            raise RuntimeError('Optimizer schedule count differs.')
        final = report['evaluations'][-1]

        def stage_of(evaluation):
            return {s['stage']: s['mean_relative_l2'] for s in evaluation['monitor']['stages']}
        report['final_monitor_stages'] = stage_of(final)
        report['early_not_worse'] = stage_of(final)['early'] <= stage_of(baseline)['early']
        baseline_rows = {(r['source_id'], r['stage']): r for r in baseline['monitor']['records']}
        report['final_improved_comparisons'] = {s: sum(
            r['relative_l2_error'] < baseline_rows[(r['source_id'], r['stage'])]['relative_l2_error']
            for r in final['monitor']['records'] if r['stage'] == s) for s in ('early', 'middle', 'late')}
        report['final_gate_passed'] = final['gate']['passed']
        baseline['checkpoint'] = None  # update 0 = teacher weights + the saved region bridge
        eligible = [e for e in report['evaluations'] if stage_of(e)['early'] <= stage_of(baseline)['early']]
        report['eligible_steps'] = [e['step'] for e in eligible]
        chosen = min(eligible, key=lambda e: e['mean_relative_l2'])
        report['selected_step'], report['selected_checkpoint'] = chosen['step'], chosen['checkpoint']
        report['gate_passed_steps'] = [e['step'] for e in report['evaluations'] if e['gate']['passed']]
        if any(not p.requires_grad for n, p in model.named_parameters() if n not in frozen):
            raise RuntimeError('A surviving student parameter was left frozen.')
        report['surviving_parameters_trainable'] = True
        report['validation_optimizer_updates'] = 0
        report['status'] = 'passed'
        report['scope'] = request['scope']
        hook_handle.remove()
    except Exception as exc:
        report['status'] = 'failed'
        report['error'] = f'{type(exc).__name__}: {exc}'
        if 'optimizer' in locals() and 'masters' in locals():
            try:
                save_tensor('failure_checkpoint.pt', {'request_key': key, 'step': report['optimizer_updates'],
                    'master_state': cpu_tree(masters), 'optimizer_state': cpu_tree(optimizer.state_dict()),
                    'history': report.get('history', []), 'rng_cpu': torch.get_rng_state(),
                    'rng_cuda': torch.cuda.get_rng_state_all()})
            except Exception as snapshot_error:
                report['failure_snapshot_error'] = str(snapshot_error)
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    results_volume.commit()
    return report
