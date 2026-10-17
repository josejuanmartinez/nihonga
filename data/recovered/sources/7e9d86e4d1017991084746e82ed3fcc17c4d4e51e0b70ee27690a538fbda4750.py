def evaluate_continuation_test(request, key):
    """Compare pretrained, update 60, 120, and 180 on untouched teacher inputs."""
    import gc
    import hashlib
    import inspect
    import json
    import math
    from pathlib import Path

    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from diffusers import QwenImage21Pipeline
    from diffusers.models.transformers.transformer_qwenimage21 import (
        QwenImage21KVCache, QwenImage21KVLayerCache)

    def digest(path):
        h = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
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

    directory = Path('/phase3/phase6/full_student_continuation_test') / key
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / 'summary.json'
    if report_path.exists():
        return json.loads(report_path.read_text())
    request_path = directory / 'request.json'
    manifest = {'request_key': key, 'request': request}
    if request_path.exists():
        if json.loads(request_path.read_text()) != manifest:
            raise RuntimeError('Saved test request differs.')
    else:
        request_path.write_text(json.dumps(manifest, indent=2) + '\n')
        results_volume.commit()
    report = {'request_key': key, 'status': 'running', 'rows': [],
              'student_forwards': 0, 'optimizer_updates': 0, 'artifacts': []}
    try:
        if request['version'] != 'phase6-full-student-continuation-test-v1':
            raise RuntimeError('Test protocol differs.')
        if 'H100' not in torch.cuda.get_device_name(0):
            raise RuntimeError('Expected H100 GPU.')
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
        teacher = request['teacher']
        pipe = QwenImage21Pipeline.from_pretrained(
            teacher['model_repo'], revision=teacher['model_revision'],
            dtype=torch.bfloat16, local_files_only=True).to('cuda')
        model = pipe.transformer.eval()
        if digest(Path(inspect.getfile(type(model)))) != request['transformer_source_sha256']:
            raise RuntimeError('Transformer source differs.')
        config = (Path('/root/.cache/huggingface/hub') /
                  ('models--' + teacher['model_repo'].replace('/', '--')) /
                  'snapshots' / teacher['model_revision'] / 'transformer/config.json')
        if digest(config) != teacher['config_sha256'] or len(model.transformer_blocks) != 32:
            raise RuntimeError('Teacher configuration differs.')
        namespace = {}
        exec(compile(request['wrapper_source'], 'bridge_block_definition.py', 'exec'), namespace)
        exec(compile(request['components_source'], 'healing_components.py', 'exec'), namespace)
        selected = request['selected_candidate']
        bridge_path = (Path('/phase3/phase5/fitting') / request['fit_key'] /
                       Path(selected['checkpoint_path']).name)
        if digest(bridge_path) != selected['checkpoint_sha256']:
            raise RuntimeError('Pretrained bridge checkpoint differs.')
        bridge_state = torch.load(bridge_path, map_location='cpu', weights_only=True)
        if bridge_state['request_key'] != request['fit_key'] or bridge_state['layer'] != 5:
            raise RuntimeError('Bridge checkpoint provenance differs.')
        bridge = namespace['MasterBridge']().to('cuda')
        bridge.load_state_dict(bridge_state['state_dict'])
        removed, _ = namespace['insert_bridge'](model, 5, bridge, QwenImage21KVCache)
        bridge.float()
        bridge.load_state_dict(bridge_state['state_dict'])
        del removed, bridge_state
        for slot, block in enumerate(model.transformer_blocks):
            if slot != 5:
                model.transformer_blocks[slot] = namespace['FunctionalCheckpointBlock'](
                    block, QwenImage21KVLayerCache)
        model.requires_grad_(False)
        parameters = dict(model.named_parameters())
        if sum(p.numel() for p in parameters.values()) != 6899117824:
            raise RuntimeError('Student parameter count differs.')
        fields = ('hidden_states', 'timestep', 'encoder_hidden_states',
                  'encoder_hidden_states_mask', 'img_shapes', 'img_mask',
                  'attention_kwargs', 'kv_cache_mode', 'return_dict')
        bundles = []
        for record in request['records']:
            path = Path('/phase3/phase5/capture') / record['request_key'] / 'targets.pt'
            if path.stat().st_size != record['bytes'] or digest(path) != record['sha256']:
                raise RuntimeError('Untouched teacher capture differs.')
            bundle = torch.load(path, map_location='cpu', weights_only=True)
            if bundle['row']['source_id'] != record['source_id'] or bundle['row']['split'] != 'validation':
                raise RuntimeError('Test row provenance differs.')
            bundles.append(bundle)
        if len(bundles) != 20 or len({b['row']['source_id'] for b in bundles}) != 20:
            raise RuntimeError('Untouched test prompt count differs.')
        predictions = []
        for variant in ('pretrained', 'update60', 'update120', 'update180'):
            if variant != 'pretrained':
                checkpoint = request['checkpoints'][variant]
                path = Path(checkpoint['remote_path'])
                if path.stat().st_size != checkpoint['bytes'] or digest(path) != checkpoint['sha256']:
                    raise RuntimeError('Student checkpoint differs: ' + variant)
                state = torch.load(path, map_location='cpu', weights_only=True)
                if (state['request_key'] != checkpoint['request_key'] or
                        state['step'] != checkpoint['step'] or
                        set(state['master_state']) != set(parameters)):
                    raise RuntimeError('Student checkpoint provenance differs: ' + variant)
                with torch.no_grad():
                    for name, parameter in parameters.items():
                        parameter.copy_(state['master_state'][name].to(
                            device=parameter.device, dtype=parameter.dtype))
                del state
                gc.collect()
            model.eval()
            with torch.inference_mode(), model.cache_context('cond'), sdpa_kernel([SDPBackend.MATH]):
                for bundle in bundles:
                    cache = QwenImage21KVCache(32)
                    for stage in bundle['stages']:
                        full = {**bundle['shared_transformer_kwargs'], **stage['kwargs']}
                        kwargs = gpu({field: full[field] for field in fields})
                        kwargs['kv_cache'] = cache
                        prediction = model(**kwargs)[0][:, -stage['target_image_tokens']:].detach().cpu().contiguous()
                        target = stage['teacher_velocity']
                        error = float((prediction.float() - target.float()).norm() /
                                      target.float().norm().clamp_min(1e-12))
                        if not math.isfinite(error):
                            raise RuntimeError('Nonfinite student velocity error.')
                        row = {'variant': variant, 'source_id': bundle['row']['source_id'],
                               'dimension': bundle['row']['dimension'], 'stage': stage['stage'],
                               'relative_l2_error': error}
                        report['rows'].append(row)
                        predictions.append({**row, 'velocity': prediction})
                        report['student_forwards'] += 1
            print('Test variant', variant, '| mean relative L2',
                  sum(r['relative_l2_error'] for r in report['rows'] if r['variant'] == variant) / 60,
                  flush=True)
        path = directory / 'predictions.pt'
        temporary = path.with_suffix('.tmp')
        torch.save({'request_key': key, 'predictions': predictions}, temporary)
        temporary.replace(path)
        report['artifacts'].append({'name': path.name, 'sha256': digest(path),
                                    'bytes': path.stat().st_size})
        report['means'] = {variant: sum(r['relative_l2_error'] for r in report['rows']
                                        if r['variant'] == variant) / 60
                           for variant in ('pretrained', 'update60', 'update120', 'update180')}
        report['status'] = 'passed'
    except Exception as exc:
        report['status'] = 'failed'
        report['error'] = type(exc).__name__ + ': ' + str(exc)
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    results_volume.commit()
    return report
