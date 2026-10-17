def evaluate_expanded(request, key):
    """Replay a pretrained single-bridge student on durable teacher captures."""
    import hashlib
    import inspect
    import json
    import math
    from pathlib import Path

    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from diffusers import QwenImage21Pipeline
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache

    directory = Path('/phase3/phase6/expanded_validation') / key
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {'request_key': key, 'request': request}
    manifest_path = directory / 'request.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise RuntimeError('Existing evaluation request differs.')
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
        results_volume.commit()
    report_path = directory / 'summary.json'
    if report_path.exists():
        return json.loads(report_path.read_text())

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

    report = {'request_key': key, 'status': 'running', 'records': [],
              'optimizer_updates': 0, 'teacher_capture_count': len(request['new_tasks']),
              'student_forwards': 0}
    progress_path = directory / 'progress.json'
    try:
        if torch.cuda.get_device_name(0).find('H100') < 0:
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
        config_path = (Path('/root/.cache/huggingface/hub') /
                       ('models--' + teacher['model_repo'].replace('/', '--')) /
                       'snapshots' / teacher['model_revision'] / 'transformer/config.json')
        if digest(config_path) != teacher['config_sha256'] or len(model.transformer_blocks) != 32:
            raise RuntimeError('Teacher configuration differs.')
        namespace = {}
        exec(compile(request['wrapper_source'], 'bridge_block_definition.py', 'exec'), namespace)
        exec(compile(request['components_source'], 'healing_components.py', 'exec'), namespace)
        selected = request['selected_candidate']
        bridge_path = (Path('/phase3/phase5/fitting') / request['fit_key'] /
                       Path(selected['checkpoint_path']).name)
        if digest(bridge_path) != selected['checkpoint_sha256']:
            raise RuntimeError('Selected pretrained bridge differs.')
        checkpoint = torch.load(bridge_path, map_location='cpu', weights_only=True)
        if (checkpoint['request_key'] != request['fit_key'] or checkpoint['layer'] != 5 or
                checkpoint['step'] != selected['selected_update']):
            raise RuntimeError('Bridge checkpoint provenance differs.')
        bridge = namespace['MasterBridge']().to('cuda')
        bridge.load_state_dict(checkpoint['state_dict'])
        removed, _ = namespace['insert_bridge'](model, 5, bridge, QwenImage21KVCache)
        bridge.float()
        bridge.load_state_dict(checkpoint['state_dict'])
        del removed, checkpoint
        model.requires_grad_(False)
        fields = ('hidden_states', 'timestep', 'encoder_hidden_states',
                  'encoder_hidden_states_mask', 'img_shapes', 'img_mask',
                  'attention_kwargs', 'kv_cache_mode', 'return_dict')

        for task in request['controls'] + request['new_tasks']:
            capture = task['capture']
            path = Path('/phase3/phase5/capture') / capture['request_key'] / 'targets.pt'
            summary_path = path.with_name('summary.json')
            summary = json.loads(summary_path.read_text())
            if summary['status'] != 'passed' or digest(path) != summary['tensor_sha256']:
                raise RuntimeError('Teacher capture or checksum differs: ' + capture['request_key'])
            bundle = torch.load(path, map_location='cpu', weights_only=True)
            if bundle['request_key'] != capture['request_key'] or bundle['row']['source_id'] != task['source_id']:
                raise RuntimeError('Teacher capture provenance differs.')
            if len(bundle['stages']) != 3 or any(
                    c['relative_l2_error'] > 0.001 for c in bundle['replay_controls']):
                raise RuntimeError('Teacher replay controls failed.')
            cache = QwenImage21KVCache(32)
            with torch.inference_mode(), model.cache_context('cond'), sdpa_kernel([SDPBackend.MATH]):
                for stage in bundle['stages']:
                    full = {**bundle['shared_transformer_kwargs'], **stage['kwargs']}
                    kwargs = gpu({field: full[field] for field in fields})
                    kwargs['kv_cache'] = cache
                    output = model(**kwargs)[0][:, -stage['target_image_tokens']:].float()
                    target = stage['teacher_velocity'].to('cuda').float()
                    difference = output - target
                    relative = float(difference.norm() / target.norm().clamp_min(1e-12))
                    if not math.isfinite(relative):
                        raise RuntimeError('Nonfinite student error.')
                    report['records'].append({
                        'source_id': task['source_id'], 'dimension': task['dimension'],
                        'stage': stage['stage'], 'set': task['set'],
                        'relative_l2_error': relative,
                        'teacher_rms': float(target.square().mean().sqrt()),
                        'student_rms': float(output.square().mean().sqrt()),
                        'difference_rms': float(difference.square().mean().sqrt()),
                    })
                    report['student_forwards'] += 1
            progress_path.write_text(json.dumps(report, indent=2) + '\n')
            results_volume.commit()
            print('Evaluated', task['set'], task['source_id'], flush=True)

        controls = [r for r in report['records'] if r['set'] == 'control']
        fresh = [r for r in report['records'] if r['set'] == 'new_validation']
        if len(controls) != 30 or len(fresh) != 150:
            raise RuntimeError('Evaluation count differs.')
        control_mean = sum(r['relative_l2_error'] for r in controls) / len(controls)
        if abs(control_mean - request['expected_control_mean']) > 0.001:
            raise RuntimeError('Original 10-prompt control mean differs.')
        report['control_mean_relative_l2'] = control_mean
        report['new_mean_relative_l2'] = sum(r['relative_l2_error'] for r in fresh) / len(fresh)
        report['by_stage'] = [
            {'stage': stage, 'mean_relative_l2': sum(r['relative_l2_error'] for r in subset)/len(subset),
             'prompts': len(subset)}
            for stage in ('early', 'middle', 'late')
            for subset in [[r for r in fresh if r['stage'] == stage]]
        ]
        report['by_dimension'] = [
            {'dimension': dimension,
             'mean_relative_l2': sum(r['relative_l2_error'] for r in subset)/len(subset),
             'comparisons': len(subset)}
            for dimension in request['dimensions']
            for subset in [[r for r in fresh if r['dimension'] == dimension]]
        ]
        report['status'] = 'passed'
    except Exception as exc:
        report['status'] = 'failed'
        report['error'] = type(exc).__name__ + ': ' + str(exc)
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    results_volume.commit()
    return report
