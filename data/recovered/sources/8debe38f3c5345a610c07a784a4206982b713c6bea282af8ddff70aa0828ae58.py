def generate_update120_images(request, key):
    """Generate only update-120 images on the already fixed comparison rows."""
    import gc
    import hashlib
    import inspect
    import json
    from pathlib import Path

    import torch
    from diffusers import QwenImage21Pipeline
    from diffusers.models.transformers.transformer_qwenimage21 import (
        QwenImage21KVCache, QwenImage21KVLayerCache)

    def digest(path):
        h = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                h.update(chunk)
        return h.hexdigest()

    directory = Path('/phase3/phase6/full_student_continuation_update120_images') / key
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / 'summary.json'
    if report_path.exists():
        return json.loads(report_path.read_text())
    request_path = directory / 'request.json'
    manifest = {'request_key': key, 'request': request}
    if request_path.exists():
        if json.loads(request_path.read_text()) != manifest:
            raise RuntimeError('Saved update-120 image request differs.')
    else:
        request_path.write_text(json.dumps(manifest, indent=2) + '\n')
        results_volume.commit()
    report = {'request_key': key, 'status': 'running', 'rows': [],
              'artifacts': [], 'calls': 0, 'optimizer_updates': 0}
    try:
        if request['version'] != 'phase6-continuation-update120-images-v1':
            raise RuntimeError('Image protocol differs.')
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
        pipe.set_progress_bar_config(disable=True)
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
            raise RuntimeError('Bridge checkpoint differs.')
        bridge_state = torch.load(bridge_path, weights_only=True, map_location='cpu')
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
        checkpoint = request['checkpoint']
        path = Path(checkpoint['remote_path'])
        if path.stat().st_size != checkpoint['bytes'] or digest(path) != checkpoint['sha256']:
            raise RuntimeError('Update-120 checkpoint differs.')
        state = torch.load(path, weights_only=True, map_location='cpu')
        parameters = dict(model.named_parameters())
        if (state['request_key'] != checkpoint['request_key'] or state['step'] != 120 or
                set(parameters) != set(state['master_state']) or
                sum(p.numel() for p in parameters.values()) != 6899117824):
            raise RuntimeError('Update-120 checkpoint provenance differs.')
        with torch.no_grad():
            for name, parameter in parameters.items():
                parameter.copy_(state['master_state'][name].to(
                    device=parameter.device, dtype=parameter.dtype))
        del state
        gc.collect()
        model.eval()
        for row in request['rows']:
            output = directory / (row['source_id'] + '-update120.png')
            if output.exists():
                raise RuntimeError('Unexpected preexisting update-120 PNG.')
            with torch.inference_mode():
                image = pipe(prompt=row['prompt'], width=row['width'], height=row['height'],
                    num_inference_steps=40, true_cfg_scale=teacher['true_cfg_scale'],
                    generator=torch.Generator('cpu').manual_seed(row['seed']),
                    use_kv_cache=True, output_type='pil').images[0]
            temporary = output.with_suffix('.tmp')
            image.save(temporary, format='PNG')
            temporary.replace(output)
            artifact = {'name': output.name, 'sha256': digest(output),
                        'bytes': output.stat().st_size}
            report['rows'].append({**row, 'update120': artifact})
            report['artifacts'].append(artifact)
            report['calls'] += 1
            results_volume.commit()
            print('Update-120 image', report['calls'], '/', len(request['rows']), flush=True)
        report['status'] = 'passed'
    except Exception as exc:
        report['status'] = 'failed'
        report['error'] = type(exc).__name__ + ': ' + str(exc)
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    results_volume.commit()
    return report
