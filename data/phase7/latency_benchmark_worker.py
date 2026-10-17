def benchmark_candidate(request, key):
    """Matched 40-step images for the Phase 7 candidates. family 'blocks45': teacher, C (bridge only), A and B (healed);
    family 'blocks25': D (28-block healed). Same prompt, CPU noise seed, size, scheduler and 40 steps for every model."""
    import gc
    import hashlib
    import inspect
    import json
    import os
    import time
    from pathlib import Path

    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from diffusers import QwenImage21Pipeline
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache, QwenImage21KVLayerCache

    directory = Path('/phase3/phase7/latency_benchmark') / key
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / 'summary.json'
    if report_path.exists():
        return json.loads(report_path.read_text())
    if (directory / 'started.json').exists():
        raise RuntimeError('Incomplete image attempt exists; inspect before repeating calls.')
    (directory / 'started.json').write_text(json.dumps({'request_key': key}))
    (directory / 'request.json').write_text(json.dumps({'request_key': key, 'request': request}, indent=2))
    results_volume.commit()

    def digest(path):
        h = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                h.update(chunk)
        return h.hexdigest()

    def table_hash(value):
        return hashlib.sha256(memoryview(value.detach().cpu().contiguous().view(torch.uint8).numpy())).hexdigest()

    variants = ['teacher', 'candidate_A']
    report = {'request_key': key, 'status': 'running', 'rows': [], 'artifacts': [],
              'calls': {v: 0 for v in variants}, 'optimizer_updates': 0}
    try:
        if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
            raise RuntimeError('Deterministic workspace environment differs.')
        teacher = request['teacher']
        report['gpu'] = torch.cuda.get_device_name(0)
        if 'H100' not in report['gpu']:
            raise RuntimeError('Expected H100.')
        load_started = time.perf_counter()
        pipe = QwenImage21Pipeline.from_pretrained(teacher['model_repo'], revision=teacher['model_revision'],
                                                   dtype=torch.bfloat16, local_files_only=True).to('cuda')
        torch.cuda.synchronize()
        pipeline_load_seconds = time.perf_counter()-load_started
        pipe.set_progress_bar_config(disable=True)
        model = pipe.transformer.eval()
        for component in (model, pipe.text_encoder, pipe.vae):
            component.requires_grad_(False)
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
        if digest(Path(inspect.getfile(type(model)))) != request['transformer_source_sha256']:
            raise RuntimeError('Transformer source differs.')
        config_path = (Path('/root/.cache/huggingface/hub') / ('models--' + teacher['model_repo'].replace('/', '--')) /
                       'snapshots' / teacher['model_revision'] / 'transformer/config.json')
        if digest(config_path) != teacher['config_sha256'] or len(model.transformer_blocks) != 32:
            raise RuntimeError('Teacher configuration differs.')
        # Derived position/time tables: use the audited values on every host (as in steps 7-11).
        derived = {f'pos_embed.freqs.{axis}': value for axis, value in enumerate(model.pos_embed.freqs)}
        derived['time_text_embed.time_proj.freqs'] = model.time_text_embed.time_proj.freqs
        expected = request['expected_derived_tables']
        mismatched = sorted(n for n, v in derived.items() if table_hash(v) != expected[n])
        if mismatched:
            canonical = torch.load(Path('/phase3/phase7/canonical_derived_tables') / (request['canonical_tables_key'] + '.pt'),
                                   weights_only=True, map_location='cpu')
            if any(table_hash(canonical[n]) != expected[n] for n in expected):
                raise RuntimeError('Canonical derived tables differ from the audited hashes.')
            original = model.pos_embed.freqs
            replaced = [canonical[f'pos_embed.freqs.{a}'].to(device=v.device, dtype=v.dtype) for a, v in enumerate(original)]
            model.pos_embed.freqs = tuple(replaced) if isinstance(original, tuple) else replaced
            with torch.no_grad():
                model.time_text_embed.time_proj.freqs.copy_(canonical['time_text_embed.time_proj.freqs'])
            if any(table_hash(v) != expected[n] for n, v in
                   list({f'pos_embed.freqs.{a}': v for a, v in enumerate(model.pos_embed.freqs)}.items()) +
                   [('time_text_embed.time_proj.freqs', model.time_text_embed.time_proj.freqs)]):
                raise RuntimeError('Derived tables still differ after loading the audited copy.')
            report['canonical_tables_applied'] = mismatched

        import random
        import statistics
        from types import SimpleNamespace
        started_loading = time.perf_counter()
        teacher_state = {n:v.detach().cpu().clone() for n,v in model.state_dict().items()}
        original_blocks = list(model.transformer_blocks)
        namespace = {}
        for field in ('core_source','wrapper_source','components_source'):
            exec(compile(request[field],field+'.py','exec'),namespace)
        checkpoint = request['checkpoints']['A_healed_u60']
        path = Path(checkpoint['remote_path'])
        if path.stat().st_size != checkpoint['bytes'] or digest(path) != checkpoint['sha256']:
            raise RuntimeError('Candidate A checkpoint differs')
        state = torch.load(path,weights_only=True,map_location='cpu')
        if state['request_key']!=checkpoint['request_key'] or state['step']!=60:
            raise RuntimeError('Candidate A provenance differs')
        masters = state['master_state']
        del state
        student = request['students']['blocks45']
        region = student['region_bridge']
        region_path = Path('/phase3/phase7/region_bridge_fit')/region['fit_key']/region['checkpoint']
        if digest(region_path)!=region['sha256']:
            raise RuntimeError('Bridge checkpoint differs')
        region_value = torch.load(region_path,weights_only=True,map_location='cpu')
        report['cpu_snapshot_and_checkpoint_seconds']=time.perf_counter()-started_loading
        report['protocol']=request['benchmark_protocol']
        report['packages']={p:__import__('importlib.metadata',fromlist=['version']).version(p)
                            for p in ('torch','diffusers','transformers','accelerate','pillow')}
        report['torch_cuda']=torch.version.cuda
        report['cold_pipeline_load_seconds']=pipeline_load_seconds

        def restore(variant):
            model.transformer_blocks=torch.nn.ModuleList(original_blocks)
            model.to('cuda')
            model.load_state_dict(teacher_state,strict=True)
            if variant=='candidate_A':
                bridge=namespace['MasterBridge'](4096,512)
                bridge.load_state_dict(region_value['state_dict'])
                removed,_=namespace['insert_bridge'](model,4,bridge,QwenImage21KVCache)
                bridge.float()
                bridge.load_state_dict(region_value['state_dict'])
                del removed
                passthrough=namespace['ResidualBottleneckBridge'](4096,256)
                with torch.no_grad():passthrough.up.weight.zero_()
                removed,_=namespace['insert_bridge'](model,5,passthrough,QwenImage21KVCache)
                del removed
                for slot,block in enumerate(model.transformer_blocks):
                    if slot not in (4,5):
                        model.transformer_blocks[slot]=namespace['FunctionalCheckpointBlock'](block,QwenImage21KVLayerCache)
                frozen={'transformer_blocks.5.bridge.down.weight','transformer_blocks.5.bridge.up.weight'}
                parameters={n:p for n,p in model.named_parameters() if n not in frozen}
                if set(parameters)!=set(masters):raise RuntimeError('Candidate parameter names differ')
                with torch.no_grad():
                    for name,p in parameters.items():p.copy_(masters[name].to(device=p.device,dtype=p.dtype))
                original_blocks[4].cpu()
                original_blocks[5].cpu()
            model.requires_grad_(False).eval()
            model.gradient_checkpointing=False
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        events=[]
        def before(module,args):
            start=torch.cuda.Event(enable_timing=True)
            end=torch.cuda.Event(enable_timing=True)
            start.record()
            events.append((start,end))
        def after(module,args,output):events[-1][1].record()
        pre=model.register_forward_pre_hook(before)
        post=model.register_forward_hook(after)

        def generate(row):
            events.clear()
            torch.cuda.synchronize()
            started=time.perf_counter()
            with torch.inference_mode(), sdpa_kernel([SDPBackend.MATH]):
                image=pipe(prompt=row['prompt'],width=row['width'],height=row['height'],
                           num_inference_steps=40,true_cfg_scale=teacher['true_cfg_scale'],
                           generator=torch.Generator('cpu').manual_seed(row['seed']),
                           use_kv_cache=True,output_type='pil').images[0]
            torch.cuda.synchronize()
            seconds=time.perf_counter()-started
            denoiser=sum(a.elapsed_time(b) for a,b in events)/1000
            return image,seconds,denoiser

        # Fifteen AB and fifteen BA pairs, shuffled before observing timings.
        rng=random.Random(request['benchmark_protocol']['order_seed'])
        orders=[('teacher','candidate_A')]*15+[('candidate_A','teacher')]*15
        rng.shuffle(orders)
        report['pair_orders']=orders
        report['rows']=[]
        for index,(row,order) in enumerate(zip(request['rows'],orders)):
            for position,variant in enumerate(order):
                restore_started=time.perf_counter()
                restore(variant)
                restore_seconds=time.perf_counter()-restore_started
                for _ in range(request['benchmark_protocol']['warmups_per_restore']):
                    warm,_,_=generate(row)
                    del warm
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                image,seconds,denoiser=generate(row)
                record=dict(pair=index,position=position,variant=variant,source_id=row['source_id'],
                            seed=row['seed'],width=row['width'],height=row['height'],seconds=seconds,
                            denoiser_seconds=denoiser,restore_seconds_excluded=restore_seconds,
                            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                            resident_allocated_gib=torch.cuda.memory_allocated()/2**30)
                image_path=directory/f'pair_{index:02d}_{variant}.png'
                image.save(image_path)
                record['image']=dict(name=image_path.name,sha256=digest(image_path),bytes=image_path.stat().st_size)
                del image
                report['rows'].append(record)
                report['calls'][variant]+=1
                (directory/'progress.json').write_text(json.dumps(report,indent=2))
                results_volume.commit()
                print(f'Pair {index+1}/30 {variant}: {seconds:.3f}s, {record["peak_allocated_gib"]:.3f} GiB',flush=True)
        pre.remove()
        post.remove()
        ratios=[]
        for index in range(30):
            pair={r['variant']:r for r in report['rows'] if r['pair']==index}
            ratios.append(pair['teacher']['seconds']/pair['candidate_A']['seconds'])
        bootstrap_rng=random.Random(20261017)
        boot=sorted(statistics.median(bootstrap_rng.choices(ratios,k=len(ratios))) for _ in range(20000))
        report['paired_speedup']=dict(median=statistics.median(ratios),bootstrap_95pct=[boot[499],boot[19499]],
                                      replicates=20000,ratios=ratios)
        report['summary']={}
        for variant in variants:
            rows=[r for r in report['rows'] if r['variant']==variant]
            values=sorted(r['seconds'] for r in rows)
            median=statistics.median(values)
            report['summary'][variant]=dict(pairs=len(rows),median_seconds=median,p90_seconds=values[26]+.1*(values[27]-values[26]),
                throughput_images_per_second=1/median,median_denoiser_seconds=statistics.median(r['denoiser_seconds'] for r in rows),
                peak_allocated_gib=max(r['peak_allocated_gib'] for r in rows),peak_reserved_gib=max(r['peak_reserved_gib'] for r in rows))
        report['status']='passed'
        report['scope']='Matched 40-step teacher vs provisional candidate A, one H100, 30 balanced randomized pairs; no quality promotion.'
    except Exception as exc:
        import traceback
        report['status']='failed'
        report['error']=repr(exc)
        report['traceback']=traceback.format_exc()
    (directory/'summary.json').write_text(json.dumps(report,indent=2))
    results_volume.commit()
    return report
