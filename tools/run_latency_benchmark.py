"""Launch the matched randomized latency/VRAM benchmark; reuse completed remote results."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-account', action='store_true')
    args = parser.parse_args()
    if args.old_account:
        parser.error('Old account is download-only; GPU execution must use the new account.')
    from modal_accounts import set_modal_account
    set_modal_account('current', purpose='compute')
    import modal
    root = Path(__file__).resolve().parents[1]
    recovered = root / 'data/recovered/modal'
    images_key = 'dda8c184e657dfb604c35fb261d83371c108c39fac033a0f57db66f7686c30ad'
    request = json.loads((recovered / 'phase7/candidate_images' / images_key / 'request.json').read_bytes())['request']
    source = (root / 'data/phase7/latency_benchmark_worker.py').read_text(encoding='utf-8')
    request.update(version='phase7-latency-benchmark-v1', worker_source=source, execution_account='new',
                   benchmark_protocol=dict(pairs=30, order_seed=20261017, warmups_per_restore=1,
                     precision='BF16 with original FP32 bridge masters cast per forward',
                     attention='SDPBackend.MATH', steps=40, batch_size=1, kv_cache=True,
                     timing='CUDA-synchronized pipeline wall time, excluding restoration, warmup and image serialization; CUDA events for denoiser',
                     memory='one active variant; unused teacher blocks moved to CPU; reset peaks after warmup; no offload of active pipeline'))
    if len(request['rows']) != 30:
        raise RuntimeError('Benchmark requires 30 saved prompt/seed pairs')
    request_key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    local = root / 'data/phase7/latency_benchmark' / request_key
    local.mkdir(parents=True, exist_ok=True)
    (local / 'request.json').write_text(json.dumps(dict(request_key=request_key, request=request), indent=2)+'\n',encoding='utf-8')
    runtime = request['teacher']['runtime']
    results_volume = modal.Volume.from_name(runtime['results_volume'])
    remote = '/phase7/latency_benchmark/' + request_key
    try:
        report = json.loads(b''.join(results_volume.read_file(remote + '/summary.json')))
        print('Reusing completed remote report; no GPU scheduled.', flush=True)
    except FileNotFoundError:
        try:
            b''.join(results_volume.read_file(remote + '/started.json'))
        except FileNotFoundError:
            pass
        else:
            raise RuntimeError('Run already started; inspect its progress. No duplicate GPU scheduled.')
        app = modal.App('nihonga-phase7-latency-benchmark')
        image = (modal.Image.debian_slim(python_version=runtime['python_version']).apt_install('git')
                 .pip_install(*runtime['packages']).env(request['runtime_environment']))
        weights = modal.Volume.from_name(runtime['weights_volume'])
        namespace = {'results_volume': results_volume}
        exec(compile(source, 'latency_benchmark_worker.py', 'exec'), namespace)
        worker = app.function(image=image, gpu='H100!', serialized=True, timeout=28800, cpu=8,
                              memory=98304, max_containers=1, retries=0,
                              volumes={'/phase3':results_volume,
                                       '/root/.cache/huggingface':weights.with_mount_options(read_only=True)})(namespace['benchmark_candidate'])
        with modal.enable_output(), app.run(detach=True):
            call = worker.spawn(request, request_key)
            (local / 'call.json').write_text(json.dumps(dict(request_key=request_key, function_call_id=call.object_id, account='new'),indent=2)+'\n')
            print('Request:',request_key,'Call:',call.object_id,flush=True)
            report = call.get()
    (local / 'summary.json').write_text(json.dumps(report, indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:report.get(k) for k in ('status','error','paired_speedup','summary')}),flush=True)
    if report.get('status') != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
