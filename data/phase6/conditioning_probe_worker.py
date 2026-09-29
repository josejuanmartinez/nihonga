def probe_conditioning(request, key):
    import hashlib
    import inspect
    import json
    import math
    import types
    from pathlib import Path
    import torch
    from diffusers import QwenImage21Pipeline
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache, QwenImage21Rope

    directory = Path('/phase3/phase6/conditioning_probe') / key
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / 'summary.json'
    if report_path.exists():
        return json.loads(report_path.read_text())
    if (directory / 'started.json').exists():
        raise RuntimeError('Incomplete probe exists; inspect before repeating calls.')
    (directory / 'started.json').write_text(json.dumps({'request_key': key}))
    (directory / 'request.json').write_text(json.dumps({'request_key': key, 'request': request}, indent=2))
    results_volume.commit()
    report = {'request_key': key, 'status': 'failed', 'teacher_calls': 0, 'student_calls': 0,
              'optimizer_updates': 0, 'variants': [], 'artifacts': [], 'text_encoder_calls': 0}

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

    def preserve(name, data):
        path = directory / name
        torch.save(data, path)
        report['artifacts'].append({'name': name, 'sha256': digest(path), 'bytes': path.stat().st_size})

    try:
        teacher, record = request['teacher'], request['record']
        report['gpu'] = torch.cuda.get_device_name(0)
        if 'H100' not in report['gpu']:
            raise RuntimeError('GPU differs.')
        path = Path('/phase3/phase5/capture') / record['request_key'] / 'targets.pt'
        if digest(path) != record['tensor_sha256']:
            raise RuntimeError('Target differs.')
        bundle = torch.load(path, weights_only=True, map_location='cpu')
        pipe = QwenImage21Pipeline.from_pretrained(teacher['model_repo'], revision=teacher['model_revision'],
            dtype=torch.bfloat16, local_files_only=True).to('cuda')
        model = pipe.transformer.eval()
        for component in (model, pipe.text_encoder, pipe.vae):
            component.requires_grad_(False)
        if digest(Path(inspect.getfile(type(model)))) != request['transformer_source_sha256']:
            raise RuntimeError('Model source differs.')
        temporal = model.time_text_embed.time_proj
        original_forward = temporal.forward
        original_freqs = temporal.freqs
        original_rope = model.pos_embed.freqs
        report['initial_frequency_dtype'] = str(original_freqs.dtype)
        # Reconstruct the exact argument order used by the pinned pipeline and capture replay.
        argument_order = ('hidden_states', 'timestep', 'encoder_hidden_states', 'encoder_hidden_states_mask',
                          'img_shapes', 'img_mask', 'attention_kwargs', 'kv_cache_mode', 'return_dict')
        for name in request['variants']:
            temporal.freqs = original_freqs
            temporal.forward = original_forward
            model.pos_embed.freqs = original_rope
            if name == 'bf16_frequency_buffer':
                temporal.freqs = original_freqs.to(torch.bfloat16)
            elif name == 'cpu_timestep_embedding':
                def cpu_forward(self, timestep):
                    t = self.time_factor * timestep.detach().cpu().float()
                    args = t[:, None] * original_freqs.detach().cpu()[None]
                    result = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
                    return result.to(device=timestep.device, dtype=timestep.dtype)
                temporal.forward = types.MethodType(cpu_forward, temporal)
            elif name == 'canonical_cpu_rope':
                rope = QwenImage21Rope(model.pos_embed.theta, model.pos_embed.axes_dim)
                model.pos_embed.freqs = rope.freqs
            elif name == 'after_novel_text_warmup':
                with torch.inference_mode():
                    warm = pipe.encode_prompt(prompt=request['warmup_prompt'], device=torch.device('cuda'))
                report['text_encoder_calls'] += 1
                preserve('novel_text_warmup.pt', {'request_key': key, 'prompt': request['warmup_prompt'], 'outputs': [v.detach().cpu() if isinstance(v, torch.Tensor) else v for v in warm]})
            variant = {'name': name, 'status': 'failed', 'controls': []}
            outputs, timestep_embeddings = [], []
            try:
                cache = QwenImage21KVCache(32)
                with torch.inference_mode(), model.cache_context('cond'):
                    for stage in bundle['stages']:
                        full = {**bundle['shared_transformer_kwargs'], **stage['kwargs']}
                        ordered = {field: full[field] for field in argument_order}
                        kwargs = gpu(ordered)
                        kwargs['kv_cache'] = cache
                        timestep_embeddings.append({'stage': stage['stage'], 'value': temporal(kwargs['timestep']).detach().cpu()})
                        pred = model(**kwargs)[0][:, -stage['target_image_tokens']:].detach().cpu().contiguous()
                        report['teacher_calls'] += 1
                        target = stage['teacher_velocity']
                        d = pred.float() - target.float()
                        relative = float(d.norm()/target.float().norm().clamp_min(1e-12))
                        variant['controls'].append({'stage': stage['stage'], 'relative_l2_error': relative,
                            'max_abs_error': float(d.abs().max()), 'exact_equal': bool(torch.equal(pred, target))})
                        outputs.append({'stage': stage['stage'], 'velocity': pred})
                variant['status'] = 'complete'
                variant['control_passed'] = all(c['relative_l2_error'] <= 0.001 for c in variant['controls'])
            except Exception as exc:
                variant['error'] = f'{type(exc).__name__}: {exc}'
            preserve(name + '.pt', {'request_key': key, 'variant': name, 'outputs': outputs, 'timestep_embeddings': timestep_embeddings})
            report['variants'].append(variant)
            (directory / 'progress.json').write_text(json.dumps(report, indent=2))
            results_volume.commit()
        temporal.freqs, temporal.forward = original_freqs, original_forward
        model.pos_embed.freqs = original_rope
        report['status'] = 'probe_complete'
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    results_volume.commit()
    return report
