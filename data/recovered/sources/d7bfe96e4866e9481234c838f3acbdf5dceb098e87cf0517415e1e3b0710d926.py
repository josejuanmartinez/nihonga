def replay_students(request, key):
    """Replay immutable validation inputs; results_volume supplied by coordinator."""
    import hashlib
    import inspect
    import json
    import math
    import time
    from pathlib import Path
    import torch
    from diffusers import QwenImage21Pipeline
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache

    directory = Path('/phase3/phase6/replay') / key
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / 'summary.json'
    if report_path.exists():
        return json.loads(report_path.read_text())
    if (directory / 'started.json').exists():
        raise RuntimeError('Incomplete replay exists; inspect instead of repeating forwards.')
    (directory / 'started.json').write_text(json.dumps({'request_key': key}))
    (directory / 'request.json').write_text(json.dumps({'request': request, 'request_key': key}, indent=2))
    results_volume.commit()
    report = {'request_key': key, 'status': 'failed', 'teacher_calls': 0, 'student_calls': 0,
              'optimizer_updates': 0, 'records': [], 'controls': []}

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

    def metrics(pred, target):
        x, y = pred.float(), target.float()
        if not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise RuntimeError('Nonfinite velocity.')
        delta = x - y
        return {'relative_l2_error': float(delta.norm() / y.norm().clamp_min(1e-12)),
                'mse': float(delta.square().mean()), 'rmse': float(delta.square().mean().sqrt()),
                'max_abs_error': float(delta.abs().max()),
                'cosine': float(torch.nn.functional.cosine_similarity(x.flatten(), y.flatten(), dim=0))}

    try:
        teacher = request['teacher']
        report['gpu'] = torch.cuda.get_device_name(0)
        if 'H100' not in report['gpu']:
            raise RuntimeError('Runtime GPU differs.')
        pipe = QwenImage21Pipeline.from_pretrained(teacher['model_repo'], revision=teacher['model_revision'],
            dtype=torch.bfloat16, local_files_only=True).to('cuda')
        model = pipe.transformer.eval()
        for component in (model, pipe.text_encoder, pipe.vae):
            component.requires_grad_(False)
        if digest(Path(inspect.getfile(type(model)))) != request['transformer_source_sha256']:
            raise RuntimeError('Pinned transformer source differs.')
        if len(model.transformer_blocks) != 32 or not model.config.causal_condition:
            raise RuntimeError('Teacher architecture differs.')
        config_path = (Path('/root/.cache/huggingface/hub') / ('models--' + teacher['model_repo'].replace('/', '--'))
                       / 'snapshots' / teacher['model_revision'] / 'transformer/config.json')
        if digest(config_path) != teacher['config_sha256']:
            raise RuntimeError('Teacher configuration differs.')
        if type(pipe.scheduler).__name__ != teacher['scheduler_class'] or json.loads(json.dumps(dict(pipe.scheduler.config), default=str)) != teacher['scheduler_config']:
            raise RuntimeError('Scheduler differs.')
        namespace = {}
        exec(compile(request['core_source'], 'bridge_definition.py', 'exec'), namespace)
        exec(compile(request['wrapper_source'], 'bridge_block_definition.py', 'exec'), namespace)
        Bridge, insert = namespace['ResidualBottleneckBridge'], namespace['insert_bridge']
        weights = {}
        for candidate in request['candidates']:
            path = Path('/phase3/phase5/fitting') / request['fit_key'] / Path(candidate['checkpoint_path']).name
            if digest(path) != candidate['checkpoint_sha256']:
                raise RuntimeError('Remote selected checkpoint differs.')
            checkpoint = torch.load(path, weights_only=True, map_location='cpu')
            if checkpoint['request_key'] != request['fit_key'] or checkpoint['layer'] != candidate['removed_teacher_slots'][0] or checkpoint['step'] != candidate['selected_update']:
                raise RuntimeError('Checkpoint provenance differs.')
            weights[candidate['candidate_id']] = checkpoint['state_dict']
        outputs = []
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        for record in request['validation']:
            path = Path('/phase3/phase5/capture') / record['request_key'] / 'targets.pt'
            if digest(path) != record['tensor_sha256']:
                raise RuntimeError('Remote teacher bundle differs.')
            bundle = torch.load(path, weights_only=True, map_location='cpu')
            if bundle['request_key'] != record['request_key'] or bundle['row']['source_id'] != record['source_id'] or bundle['row']['split'] != 'validation':
                raise RuntimeError('Target provenance or split differs.')
            stages = bundle['stages']
            if [s['step_index'] for s in stages] != [0, 20, 39]:
                raise RuntimeError('Saved stage order differs.')
            common = gpu(bundle['shared_transformer_kwargs'])

            def replay(candidate_id, variant):
                cache = QwenImage21KVCache(32)
                with torch.inference_mode(), model.cache_context('cond'):
                    for stage in stages:
                        kwargs = {**common, **gpu(stage['kwargs']), 'kv_cache': cache}
                        pred = model(**kwargs)[0][:, -stage['target_image_tokens']:]
                        values = metrics(pred, gpu(stage['teacher_velocity']))
                        entry = {'source_id': record['source_id'], 'dimension': record['dimension'],
                                 'candidate_id': candidate_id, 'variant': variant, 'stage': stage['stage'],
                                 'step_index': stage['step_index'], 'scheduler_sigma': stage['scheduler_sigma'], **values}
                        outputs.append({**entry, 'velocity': pred.detach().cpu().contiguous()})
                        if variant == 'teacher_control':
                            report['teacher_calls'] += 1
                            report['controls'].append(entry)
                            if values['relative_l2_error'] > 0.001:
                                raise RuntimeError('Intact teacher control exceeds 0.001 relative L2.')
                        else:
                            report['student_calls'] += 1
                            report['records'].append(entry)

            replay('teacher', 'teacher_control')
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
                        replay(candidate['candidate_id'], variant)
                finally:
                    model.transformer_blocks[layer] = original
            del bundle, common
        if report['teacher_calls'] != 30 or report['student_calls'] != 240:
            raise RuntimeError('Unexpected forward counts.')
        torch.cuda.synchronize()
        report['replay_seconds'] = time.perf_counter() - started
        report['peak_allocated_gib'] = torch.cuda.max_memory_allocated() / 2**30
        aggregates = []
        for candidate in request['candidates']:
            for stage in ('early', 'middle', 'late'):
                selected = [r for r in report['records'] if r['candidate_id'] == candidate['candidate_id'] and r['stage'] == stage]
                baseline = [r for r in selected if r['variant'] == 'identity']
                trained = [r for r in selected if r['variant'] == 'trained']
                if len(baseline) != 10 or len(trained) != 10:
                    raise RuntimeError('Aggregation prompt counts differ.')
                by_id = {r['source_id']: r for r in baseline}
                aggregates.append({'candidate_id': candidate['candidate_id'], 'stage': stage, 'prompts': 10,
                    'identity_mean_relative_l2': sum(r['relative_l2_error'] for r in baseline) / 10,
                    'trained_mean_relative_l2': sum(r['relative_l2_error'] for r in trained) / 10,
                    'trained_max_relative_l2': max(r['relative_l2_error'] for r in trained),
                    'improved_prompts': sum(r['relative_l2_error'] < by_id[r['source_id']]['relative_l2_error'] for r in trained)})
        tensor_path = directory / 'velocities.pt'
        temporary = tensor_path.with_suffix('.tmp')
        torch.save({'request_key': key, 'outputs': outputs}, temporary)
        temporary.replace(tensor_path)
        report.update(status='passed', aggregates=aggregates,
                      artifacts=[{'name': tensor_path.name, 'sha256': digest(tensor_path), 'bytes': tensor_path.stat().st_size}],
                      scope='fixed teacher-input BF16 inference replay only; no student trajectories, images, gradients or healing')
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    results_volume.commit()
    return report
