def capture_one(expected, request_key, pipeline=None):
    """Capture one immutable teacher trajectory; results_volume is supplied by the coordinator."""
    import hashlib
    import importlib.metadata
    import inspect
    import json
    import math
    import time
    from datetime import datetime, timezone
    from pathlib import Path
    import torch
    import torch.nn.functional as F
    from diffusers import QwenImage21Pipeline
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache

    directory = Path('/phase3/phase5/capture') / request_key
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / 'summary.json'
    tensor_path = directory / 'targets.pt'
    if report_path.exists():
        return json.loads(report_path.read_text())
    started_path = directory / 'started.json'
    if started_path.exists() or tensor_path.exists():
        raise RuntimeError('Earlier capture is incomplete; preserve and inspect, never recapture automatically.')
    started_path.write_text(json.dumps({'request_key': request_key, 'started_utc': datetime.now(timezone.utc).isoformat()}))
    (directory / 'request.json').write_text(json.dumps({'request': expected, 'request_key': request_key}, indent=2) + '\n')
    results_volume.commit()
    config, row = expected['capture_request'], expected['row']
    report = {'request_key': request_key, 'capture_manifest_key': expected['capture_manifest_key'],
              'row': row, 'status': 'failed', 'recorded_utc': datetime.now(timezone.utc).isoformat()}
    hooks = []

    def cpu(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().contiguous()
        if isinstance(value, dict):
            return {k: cpu(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(cpu(v) for v in value)
        return value

    def gpu(value):
        if isinstance(value, torch.Tensor):
            return value.to('cuda')
        if isinstance(value, dict):
            return {k: gpu(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(gpu(v) for v in value)
        return value

    def hash_file(path):
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(chunk)
        return digest.hexdigest()

    def difference(x, y):
        x, y = x.float(), y.float()
        delta = x - y
        return {'relative_l2_error': float(delta.norm() / y.norm().clamp_min(1e-12)),
                'max_abs_error': float(delta.abs().max()),
                'rmse': float(delta.square().mean().sqrt())}

    try:
        if hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest() != expected['capture_manifest_key']:
            raise RuntimeError('Capture contract key differs.')
        report['gpu'] = torch.cuda.get_device_name(0)
        if 'H100' not in report['gpu']:
            raise RuntimeError('Allocated GPU differs from the saved runtime.')
        report['packages'] = {p: importlib.metadata.version(p) for p in ('torch', 'diffusers', 'transformers', 'accelerate', 'pillow')}
        pipe = pipeline if pipeline is not None else QwenImage21Pipeline.from_pretrained(config['model_repo'], revision=config['model_revision'],
            dtype=getattr(torch, config['dtype']), local_files_only=True).to('cuda')
        pipe.set_progress_bar_config(disable=True)
        model = pipe.transformer.eval()
        for component in (pipe.transformer, pipe.text_encoder, pipe.vae):
            component.requires_grad_(False)
        if hashlib.sha256(Path(inspect.getfile(type(model))).read_bytes()).hexdigest() != config['transformer_source_sha256']:
            raise RuntimeError('Teacher source differs from Phase 3.')
        if (len(model.transformer_blocks) != config['num_teacher_blocks'] or not model.config.causal_condition
                or type(pipe.scheduler).__name__ != config['scheduler_class']
                or json.loads(json.dumps(dict(pipe.scheduler.config), default=str)) != config['scheduler_config']):
            raise RuntimeError('Teacher architecture or scheduler changed.')
        snapshot_config = (Path('/root/.cache/huggingface/hub') / ('models--' + config['model_repo'].replace('/', '--'))
                           / 'snapshots' / config['model_revision'] / 'transformer/config.json')
        if hash_file(snapshot_config) != config['config_sha256']:
            raise RuntimeError('Pinned configuration checksum differs.')
        stage_by_step = {item['step_index']: item['name'] for item in config['stages']}
        state = {'calls': 0, 'selected': False, 'stages': [], 'pairs': [], 'token_selections': {}, 'inputs': {}}

        def before_transformer(module, args, kwargs):
            step = state['calls']
            state['calls'] += 1
            state['selected'] = step in stage_by_step
            state['step'] = step
            if state['selected']:
                mode = 'extract' if step == 0 else 'cached'
                if kwargs['kv_cache_mode'] != mode:
                    raise RuntimeError('Unexpected teacher cache mode.')
                state['stages'].append({'stage': stage_by_step[step], 'step_index': step,
                    'scheduler_timestep': float(pipe.scheduler.timesteps[step]),
                    'scheduler_sigma': float(pipe.scheduler.sigmas[step]),
                    'transformer_timestep': cpu(kwargs['timestep']).tolist(), 'cache_mode': mode,
                    'target_image_tokens': math.prod(kwargs['img_shapes'][0][-1]),
                    'kwargs': cpu({k: v for k, v in kwargs.items() if k != 'kv_cache'})})

        def after_transformer(module, args, kwargs, output):
            if state['selected']:
                current = state['stages'][-1]
                prediction = output[0] if isinstance(output, tuple) else output.sample
                current['teacher_velocity'] = cpu(prediction[:, -current['target_image_tokens']:])

        def choose_indices(available, cap, role, step):
            seed_text = f"{config['capture']['token_selection_seed_salt']}:{row['capture_id']}:{step}:{role}"
            seed = int(hashlib.sha256(seed_text.encode()).hexdigest()[:16], 16) % (2**63 - 1)
            permutation = torch.randperm(len(available), generator=torch.Generator('cpu').manual_seed(seed))
            return available[permutation[:min(cap, len(available))]].sort().values

        def block_before(layer):
            def hook(module, args, kwargs):
                if not state['selected']:
                    return
                step = state['step']
                h = kwargs['hidden_states']
                if h.shape[0] != 1 or h.shape[-1] != config['hidden_dim'] or h.dtype != torch.bfloat16:
                    raise RuntimeError('Unexpected teacher hidden tensor shape/dtype.')
                mask = cpu(kwargs['target_token_mask']).bool()
                if step not in state['token_selections']:
                    image_available = mask.nonzero(as_tuple=True)[0]
                    if len(image_available) != state['stages'][-1]['target_image_tokens']:
                        raise RuntimeError('Image token count differs from target metadata.')
                    prefix_available = (~mask).nonzero(as_tuple=True)[0]
                    prefix_count = len(prefix_available)
                    if step == 0:
                        state['prefix_count'] = prefix_count
                    elif prefix_count != 0:
                        raise RuntimeError('Cached forward unexpectedly contains prefix queries.')
                    key_valid = kwargs.get('key_valid')
                    if key_valid is not None:
                        valid = cpu(key_valid)[0].bool()
                        prefix_available = prefix_available[valid[prefix_available]]
                    sampled_image = choose_indices(image_available, config['capture']['image_token_cap_per_stage'], 'image', step)
                    sampled_prefix = choose_indices(prefix_available, config['capture']['valid_prefix_token_cap_at_step_zero'], 'prefix', step)
                    indices = torch.cat([sampled_prefix, sampled_image])
                    roles = torch.cat([torch.zeros(len(sampled_prefix), dtype=torch.int64), torch.ones(len(sampled_image), dtype=torch.int64)])
                    state['token_selections'][step] = {'indices': indices, 'roles': roles,
                        'valid_mask': torch.ones(len(indices), dtype=torch.bool),
                        'joint_indices': indices if step == 0 else indices + state['prefix_count'],
                        'prefix_available': len(prefix_available), 'prefix_total': prefix_count,
                        'image_available': len(image_available), 'full_shape': list(h.shape)}
                selection = state['token_selections'][step]
                if list(h.shape) != selection['full_shape']:
                    raise RuntimeError('Candidate layers have different token layouts.')
                assert torch.equal(mask[selection['indices']], selection['roles'].bool())
                state['inputs'][(step, layer)] = cpu(h.index_select(1, selection['indices'].to(h.device)))
            return hook

        def block_after(layer):
            def hook(module, args, kwargs, output):
                if not state['selected']:
                    return
                step = state['step']
                selection = state['token_selections'][step]
                target = cpu(output.index_select(1, selection['indices'].to(output.device)))
                source = state['inputs'].pop((step, layer))
                if source.shape != target.shape or not torch.isfinite(source).all() or not torch.isfinite(target).all():
                    raise RuntimeError('Paired teacher states have invalid shape or values.')
                state['pairs'].append({'layer': layer, 'stage': stage_by_step[step], 'step_index': step,
                    'input': source, 'target': target, 'token_indices': selection['indices'],
                    'joint_token_indices': selection['joint_indices'], 'token_roles': selection['roles'],
                    'valid_token_mask': selection['valid_mask'], 'full_hidden_shape': selection['full_shape'],
                    'prefix_token_count': selection['prefix_total'], 'valid_prefix_token_count': selection['prefix_available'],
                    'image_token_count': selection['image_available']})
            return hook

        hooks = [model.register_forward_pre_hook(before_transformer, with_kwargs=True),
                 model.register_forward_hook(after_transformer, with_kwargs=True)]
        for layer in config['candidate_layers']:
            hooks.extend([model.transformer_blocks[layer].register_forward_pre_hook(block_before(layer), with_kwargs=True),
                          model.transformer_blocks[layer].register_forward_hook(block_after(layer), with_kwargs=True)])
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            terminal_latent = pipe(prompt=row['prompt'], width=row['width'], height=row['height'],
                num_inference_steps=40, true_cfg_scale=config['true_cfg_scale'],
                generator=torch.Generator(config['noise_device']).manual_seed(row['seed']),
                use_kv_cache=True, output_type='latent').images
        torch.cuda.synchronize()
        report['capture_seconds'] = time.perf_counter() - started
        report['peak_allocated_gib'] = torch.cuda.max_memory_allocated() / 2**30
        for hook in hooks:
            hook.remove()
        hooks = []
        if state['calls'] != 40 or len(state['pairs']) != 12 or len(state['stages']) != 3 or state['inputs']:
            raise RuntimeError('Capture did not produce all expected paired records.')
        # Replay three saved teacher inputs with a freshly rebuilt teacher prefix cache.
        # These controls run only during first capture, and their outputs are persisted.
        replay_cache = QwenImage21KVCache(config['num_teacher_blocks'])
        controls, replay_outputs = [], []
        with torch.inference_mode(), model.cache_context('cond'):
            for stage in state['stages']:
                kwargs = gpu(stage['kwargs'])
                kwargs['kv_cache'] = replay_cache
                result = model(**kwargs)[0][:, -stage['target_image_tokens']:]
                measured = difference(result, gpu(stage['teacher_velocity']))
                if not math.isfinite(measured['relative_l2_error']) or measured['relative_l2_error'] > expected['replay_relative_l2_tolerance']:
                    raise RuntimeError('Intact teacher replay exceeds the saved control tolerance.')
                controls.append({'stage': stage['stage'], 'step_index': stage['step_index'], **measured})
                replay_outputs.append(cpu(result))
        metrics = []
        for pair in state['pairs']:
            for role, role_name in ((0, 'prefix_text'), (1, 'target_image')):
                selected = pair['token_roles'] == role
                if not selected.any():
                    continue
                source, target = pair['input'][:, selected].float(), pair['target'][:, selected].float()
                cosine = F.cosine_similarity(source, target, dim=-1, eps=1e-12)
                normalized_mse = (F.normalize(source, dim=-1, eps=1e-12) - F.normalize(target, dim=-1, eps=1e-12)).square().mean()
                values = difference(source, target)
                metrics.append({'layer': pair['layer'], 'stage': pair['stage'], 'role': role_name,
                    'sampled_tokens': int(selected.sum()), 'identity_relative_l2_error': values['relative_l2_error'],
                    'identity_normalized_mse': float(normalized_mse), 'identity_mean_token_cosine': float(cosine.mean()),
                    'teacher_target_rms': float(target.square().mean().sqrt())})
        # Prompt embeddings and masks are shared; stage kwargs retain exact replay latents and call metadata.
        common_fields = ('encoder_hidden_states', 'encoder_hidden_states_mask', 'img_mask', 'img_shapes', 'attention_kwargs', 'return_dict')
        common = {name: state['stages'][0]['kwargs'].get(name) for name in common_fields if name in state['stages'][0]['kwargs']}
        for stage in state['stages']:
            for name, value in common.items():
                saved_value = stage['kwargs'].pop(name)
                if isinstance(value, torch.Tensor):
                    assert torch.equal(value, saved_value)
                else:
                    assert value == saved_value
        bundle = {'version': expected['version'], 'request_key': request_key,
                  'capture_manifest_key': expected['capture_manifest_key'], 'row': row,
                  'role_mapping': {'0': 'prefix_text', '1': 'target_image'},
                  'shared_transformer_kwargs': common, 'stages': state['stages'], 'pairs': state['pairs'],
                  'initial_packed_latent': state['stages'][0]['kwargs']['hidden_states'],
                  'terminal_latent': cpu(terminal_latent), 'replay_velocities': replay_outputs,
                  'replay_controls': controls}
        temporary = tensor_path.with_suffix('.tmp')
        torch.save(bundle, temporary)
        temporary.replace(tensor_path)
        report.update(status='passed', tensor_sha256=hash_file(tensor_path), tensor_bytes=tensor_path.stat().st_size,
            teacher_trajectory_calls=40, teacher_replay_calls=3, captured_block_pair_records=12,
            selected_token_pairs=sum(pair['input'].shape[1] for pair in state['pairs']),
            stages=[{k: v for k, v in stage.items() if k not in ('kwargs', 'teacher_velocity')} for stage in state['stages']],
            identity_metrics=metrics, replay_controls=controls,
            selected_shapes=[{'stage': pair['stage'], 'layer': pair['layer'], 'shape': list(pair['input'].shape),
                              'prefix_tokens': int((pair['token_roles'] == 0).sum()),
                              'image_tokens': int((pair['token_roles'] == 1).sum())} for pair in state['pairs']],
            optimizer_updates=0, teacher_frozen=True, images_decoded=0)
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
    finally:
        for hook in hooks:
            hook.remove()
    temporary = report_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(report_path)
    results_volume.commit()
    return report


def capture_many(expected, batch_key, tasks):
    """Load the pinned teacher once and durably complete one prompt before yielding it."""
    import hashlib
    import json
    from pathlib import Path
    import torch
    from diffusers import QwenImage21Pipeline

    if hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest() != batch_key:
        raise RuntimeError('Batch request checksum differs.')
    config = expected['capture_request']
    directory = Path('/phase3/phase5/capture_batches') / batch_key
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {'request': expected, 'request_key': batch_key}
    path = directory / 'request.json'
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise RuntimeError('Saved batch request differs.')
    else:
        path.write_text(json.dumps(manifest, indent=2) + '\n')
        results_volume.commit()
    pending = []
    for task in tasks:
        root = Path('/phase3/phase5/capture') / task['request_key']
        report_path = root / 'summary.json'
        if report_path.exists():
            report = json.loads(report_path.read_text())
            yield report
            if report['status'] != 'passed':
                return
        elif (root / 'started.json').exists() or (root / 'targets.pt').exists():
            raise RuntimeError('A prompt started without a complete report; inspect instead of recapturing.')
        else:
            pending.append(task)
    if not pending:
        return
    # Reuse one pipeline while each prompt retains independent deterministic noise and a fresh prefix cache.
    pipe = QwenImage21Pipeline.from_pretrained(config['model_repo'], revision=config['model_revision'],
        dtype=getattr(torch, config['dtype']), local_files_only=True).to('cuda')
    for task in pending:
        report = capture_one(task['request'], task['request_key'], pipeline=pipe)
        yield report
        if report['status'] != 'passed':
            return
