def fit_bridges(expected, request_key, initial_payloads):
    """Fit only adapters to cached TRAIN states; results_volume is supplied by coordinator."""
    import hashlib
    import importlib.metadata
    import io
    import json
    import math
    import time
    from datetime import datetime, timezone
    from pathlib import Path
    import torch
    import torch.nn.functional as F

    directory = Path('/phase3/phase5/fitting') / request_key
    directory.mkdir(parents=True, exist_ok=True)
    summary_path = directory / 'summary.json'
    if summary_path.exists():
        return json.loads(summary_path.read_text())
    if (directory / 'started.json').exists():
        raise RuntimeError('Earlier fitting attempt is incomplete; preserve checkpoints and inspect before resuming.')
    (directory / 'started.json').write_text(json.dumps({'request_key': request_key, 'utc': datetime.now(timezone.utc).isoformat()}))
    (directory / 'request.json').write_text(json.dumps({'request': expected, 'request_key': request_key}, indent=2) + '\n')
    results_volume.commit()
    report = {'request_key': request_key, 'status': 'failed', 'recorded_utc': datetime.now(timezone.utc).isoformat(),
              'teacher_calls': 0, 'validation_optimizer_updates': 0}

    def digest(path):
        h = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                h.update(chunk)
        return h.hexdigest()

    def save_tensor(path, value):
        temporary = path.with_suffix('.tmp')
        torch.save(value, temporary)
        temporary.replace(path)

    def save_json(path, value):
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(value, indent=2) + '\n')
        temporary.replace(path)

    try:
        if hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest() != request_key:
            raise RuntimeError('Fitting request checksum differs.')
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError('Allocated GPU lacks BF16 support.')
        report['gpu'] = torch.cuda.get_device_name(0)
        if 'L4' not in report['gpu']:
            raise RuntimeError('Allocated GPU differs from requested L4.')
        report['torch'] = importlib.metadata.version('torch')
        torch.manual_seed(expected['seed'])
        torch.cuda.manual_seed_all(expected['seed'])
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        namespace = {}
        exec(compile(expected['core_source'], 'bridge_definition.py', 'exec'), namespace)
        Bridge = namespace['ResidualBottleneckBridge']
        groups = expected['groups']
        layers = expected['candidate_layers']
        data = {split: {layer: {group: [] for group in groups} for layer in layers} for split in ('train', 'validation')}
        # Read the existing remote captures once. No teacher model is loaded or executed.
        for split in ('train', 'validation'):
            for record in expected['target_index'][split]:
                path = Path('/phase3/phase5/capture') / record['request_key'] / 'targets.pt'
                if digest(path) != record['tensor_sha256']:
                    raise RuntimeError('Remote teacher target checksum differs.')
                bundle = torch.load(path, weights_only=True, map_location='cpu')
                if bundle['row']['split'] != split or bundle['request_key'] != record['request_key']:
                    raise RuntimeError('Cached target split or provenance differs.')
                for pair in bundle['pairs']:
                    for role, name in ((0, 'prompt_tokens'), (1, pair['stage'] + '_image')):
                        selected = pair['token_roles'] == role
                        if not selected.any():
                            continue
                        if name not in groups:
                            raise RuntimeError('Unexpected token role or stage.')
                        x = pair['input'][0, selected].clone()
                        y = pair['target'][0, selected].clone()
                        data[split][pair['layer']][name].append({'source_id': bundle['row']['source_id'], 'x': x, 'y': y})
                del bundle
        for split, count in (('train', 20), ('validation', 10)):
            for layer in layers:
                for group in groups:
                    records = data[split][layer][group]
                    if len(records) != count or len(set(record['source_id'] for record in records)) != count:
                        raise RuntimeError('Prompt coverage is incomplete or duplicated within a loss group.')
        # Common sample schedule: prompt uniformly, then token uniformly within that prompt.
        # Layer captures share indices, so identical offsets select identical examples for every bridge.
        lengths, offsets = {}, {}
        for group in groups:
            sizes = torch.tensor([len(record['x']) for record in data['train'][layers[0]][group]], dtype=torch.int64)
            lengths[group] = sizes
            offsets[group] = torch.cat([torch.zeros(1, dtype=torch.int64), sizes.cumsum(0)[:-1]])
            for layer in layers:
                assert [record['source_id'] for record in data['train'][layer][group]] == [record['source_id'] for record in data['train'][layers[0]][group]]
                assert [len(record['x']) for record in data['train'][layer][group]] == sizes.tolist()
        generator = torch.Generator('cpu').manual_seed(expected['sampling_seed'])
        sampling = torch.empty(expected['updates_per_bridge'], len(groups), expected['tokens_per_group'], dtype=torch.int64)
        for step in range(expected['updates_per_bridge']):
            for g, group in enumerate(groups):
                prompts = torch.randint(20, (expected['tokens_per_group'],), generator=generator)
                token = (torch.rand(expected['tokens_per_group'], generator=generator) * lengths[group][prompts]).long()
                sampling[step, g] = offsets[group][prompts] + token
        save_tensor(directory / 'sampling_schedule.pt', {'request_key': request_key, 'groups': groups, 'indices': sampling})
        results_volume.commit()

        def token_losses(prediction, target):
            prediction, target = prediction.float(), target.float()
            directional = (F.normalize(prediction, dim=-1, eps=1e-12) - F.normalize(target, dim=-1, eps=1e-12)).square().sum(-1)
            relative_squared = (prediction - target).square().sum(-1) / target.square().sum(-1).clamp_min(1e-12)
            return directional, relative_squared

        def evaluate(bridge, split, layer):
            bridge.eval()
            measured = []
            with torch.no_grad():
                for group in groups:
                    prompt_metrics = []
                    for record in data[split][layer][group]:
                        x, y = record['x'].to('cuda').float(), record['y'].to('cuda').float()
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            prediction = bridge(x)
                        directional, relative_squared = token_losses(prediction, y)
                        error = prediction - y
                        prompt_metrics.append({'source_id': record['source_id'], 'directional_loss': float(directional.mean()),
                            'relative_squared_loss': float(relative_squared.mean()),
                            'relative_l2_error': float(error.norm() / y.norm().clamp_min(1e-12)),
                            'mean_token_cosine': float(F.cosine_similarity(prediction, y, dim=-1).mean())})
                    def mean(field):
                        return sum(record[field] for record in prompt_metrics) / len(prompt_metrics)
                    measured.append({'group': group, 'n_prompts': len(prompt_metrics),
                        'directional_loss': mean('directional_loss'), 'relative_squared_loss': mean('relative_squared_loss'),
                        'relative_l2_error': mean('relative_l2_error'), 'mean_token_cosine': mean('mean_token_cosine'),
                        'max_prompt_relative_l2_error': max(record['relative_l2_error'] for record in prompt_metrics),
                        'per_prompt': prompt_metrics})
            score = sum(group['directional_loss'] + expected['magnitude_weight'] * group['relative_squared_loss'] for group in measured) / len(groups)
            return {'balanced_objective': score, 'groups': measured}

        fitted, artifacts = [], []
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        for layer in layers:
            if hashlib.sha256(initial_payloads[str(layer)]).hexdigest() != expected['initial_checkpoint_sha256'][str(layer)]:
                raise RuntimeError('Initial checkpoint transfer checksum differs.')
            initial = torch.load(io.BytesIO(initial_payloads[str(layer)]), map_location='cpu', weights_only=True)
            if initial['layer'] != layer or initial['request_key'] != expected['prototype_request_key'] or initial['trained']:
                raise RuntimeError('Initial bridge provenance differs.')
            bridge = Bridge(4096, 256).to('cuda')
            bridge.load_state_dict(initial['state_dict'], strict=True)
            if sum(p.numel() for p in bridge.parameters()) != 2097152:
                raise RuntimeError('Bridge architecture changed.')
            optimizer = torch.optim.AdamW(bridge.parameters(), lr=expected['learning_rate'], weight_decay=expected['weight_decay'])
            baseline_train, baseline_validation = evaluate(bridge, 'train', layer), evaluate(bridge, 'validation', layer)
            evaluations = [{'step': 0, 'train': baseline_train, 'validation': baseline_validation}]
            best_step, best_score = 0, baseline_validation['balanced_objective']
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in bridge.state_dict().items()}
            history = []
            training_groups = {group: {'x': torch.cat([r['x'] for r in data['train'][layer][group]]).to('cuda'),
                                       'y': torch.cat([r['y'] for r in data['train'][layer][group]]).to('cuda')} for group in groups}
            def checkpoint(step):
                path = directory / f'bridge_block_{layer:02d}_step_{step:04d}.pt'
                save_tensor(path, {'request_key': request_key, 'layer': layer, 'step': step, 'trained': step > 0,
                    'state_dict': {name: tensor.detach().cpu().clone() for name, tensor in bridge.state_dict().items()},
                    'optimizer_state_dict': optimizer.state_dict(), 'torch_rng_state': torch.get_rng_state(),
                    'cuda_rng_states': torch.cuda.get_rng_state_all(), 'sampling_schedule_sha256': digest(directory / 'sampling_schedule.pt'),
                    'best_step': best_step, 'best_validation_score': best_score, 'best_state_dict': best_state,
                    'history': history, 'evaluations': evaluations})
                save_json(directory / f'bridge_block_{layer:02d}_progress.json', {'request_key': request_key, 'layer': layer,
                    'step': step, 'checkpoint': path.name, 'checkpoint_sha256': digest(path),
                    'best_step': best_step, 'best_validation_score': best_score})
                results_volume.commit()
            checkpoint(0)
            for step in range(expected['updates_per_bridge']):
                bridge.train()
                optimizer.zero_grad(set_to_none=True)
                xs, ys = [], []
                for g, group in enumerate(groups):
                    selected = sampling[step, g].to('cuda')
                    xs.append(training_groups[group]['x'][selected].float())
                    ys.append(training_groups[group]['y'][selected].float())
                x, y = torch.cat(xs), torch.cat(ys)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    prediction = bridge(x)
                directional, relative_squared = token_losses(prediction, y)
                loss = (directional + expected['magnitude_weight'] * relative_squared).mean()
                if not torch.isfinite(loss):
                    raise RuntimeError('Training produced a nonfinite loss.')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(bridge.parameters(), expected['gradient_clip'], error_if_nonfinite=True)
                optimizer.step()
                history.append({'step': step + 1, 'objective': float(loss.detach()), 'gradient_norm': float(norm)})
                if (step + 1) % expected['checkpoint_every'] == 0:
                    train_metrics, validation_metrics = evaluate(bridge, 'train', layer), evaluate(bridge, 'validation', layer)
                    evaluations.append({'step': step + 1, 'train': train_metrics, 'validation': validation_metrics})
                    score = validation_metrics['balanced_objective']
                    if not math.isfinite(score):
                        raise RuntimeError('Validation objective is nonfinite.')
                    if score < best_score:
                        best_score, best_step = score, step + 1
                        best_state = {name: tensor.detach().cpu().clone() for name, tensor in bridge.state_dict().items()}
                    checkpoint(step + 1)
            best_metrics = next(item for item in evaluations if item['step'] == best_step)
            selected_path = directory / f'bridge_block_{layer:02d}_selected.pt'
            save_tensor(selected_path, {'request_key': request_key, 'layer': layer, 'step': best_step, 'trained': best_step > 0,
                'state_dict': best_state, 'prototype_request_key': expected['prototype_request_key'],
                'selection': 'minimum balanced validation objective at saved checkpoints; no validation gradients'})
            result = {'layer': layer, 'updates': expected['updates_per_bridge'], 'parameters': 2097152,
                'selected_step': best_step, 'selected_validation_score': best_score, 'baseline_train': baseline_train,
                'baseline_validation': baseline_validation, 'selected_train': best_metrics['train'],
                'selected_validation': best_metrics['validation'], 'final_train': evaluations[-1]['train'],
                'final_validation': evaluations[-1]['validation'], 'evaluations': evaluations,
                'selected_checkpoint': selected_path.name, 'selected_checkpoint_sha256': digest(selected_path)}
            save_json(directory / f'bridge_block_{layer:02d}_summary.json', result)
            results_volume.commit()
            fitted.append(result)
            print(f"Block {layer}: {expected['updates_per_bridge']} updates; selected step {best_step}; validation {baseline_validation['balanced_objective']:.6f} -> {best_score:.6f}.", flush=True)
            del bridge, optimizer, training_groups
            torch.cuda.empty_cache()
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.name not in ('summary.json', 'started.json', 'request.json') and path.suffix != '.tmp':
                artifacts.append({'name': path.name, 'bytes': path.stat().st_size, 'sha256': digest(path)})
        report.update(status='passed', results=fitted, artifacts=artifacts,
            total_optimizer_updates=sum(result['updates'] for result in fitted),
            fit_seconds=time.perf_counter() - started, peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            fitting_precision='FP32 parameters/residual/loss; BF16 autocast projections',
            scope='cached local hidden-state fitting; no full student forward or image evaluation')
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
    save_json(summary_path, report)
    results_volume.commit()
    return report
