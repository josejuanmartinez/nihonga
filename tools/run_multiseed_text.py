"""Render the text prompts with extra seeds for teacher, B and C (best block-13 bridge); qualitative only."""
import argparse
import hashlib
import json
import sys
from pathlib import Path


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--variant', default='cond_r1024_long')
    parser.add_argument('--source', default='block13_bridge_v3', help='Run folder holding the bridge checkpoint.')
    parser.add_argument('--with-base-prompts', action='store_true', help='Also render the 30 base prompts at seed 0; candidate only.')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    recovered = root / 'data/recovered/modal'
    images_key = 'dda8c184e657dfb604c35fb261d83371c108c39fac033a0f57db66f7686c30ad'
    base = json.loads((recovered / 'phase7/candidate_images' / images_key / 'request.json').read_bytes())['request']
    scan_b = json.loads(next((root / 'data/phase7/conditional_importance_b').glob('*/request.json')).read_bytes())['request']
    fit = next(r for r in (json.loads(p.read_bytes()) for p in (root / 'data/phase7' / args.source).glob('*/summary.json')) if r.get('status') == 'passed')
    best = fit['variants'][args.variant]
    remote_dir = {'block13_bridge_v3': 'bridge_variants_v3', 'block13_bridge_e2e': 'bridge_end_to_end', 'block13_bridge_e2e_text': 'bridge_end_to_end_v2', 'block13_bridge_e2e_text8': 'bridge_end_to_end_v3'}[args.source]
    selected = scan_b['selected_candidate']
    text_rows = [r for r in base['rows'] if r['subdimension'] in ('text_rendering', 'posters')]
    rows = [{**r, 'source_id': f"{r['source_id']}-s{k}", 'seed': (r['seed'] + 1000003 * k) % 2**32}
            for r in text_rows for k in (1, 2, 3)]
    if args.with_base_prompts:
        rows += base['rows']
    request = {k: base[k] for k in ('teacher', 'transformer_source_sha256', 'runtime_environment', 'expected_derived_tables',
                                    'canonical_tables_key', 'core_source', 'wrapper_source')}
    request.update(
        version='phase7-multiseed-text-v1', rows=rows,
        worker_source=(root / 'data/phase7/multiseed_text_worker.py').read_text(encoding='utf-8'),
        conditioned_bridge_source=(root / 'data/phase7/conditioned_bridge.py').read_text(encoding='utf-8'),
        execution_account='new',
        block5_bridge=dict(remote_path=f"/phase3/phase5/fitting/{scan_b['fit_key']}/{Path(selected['checkpoint_path']).name}",
                           sha256=selected['checkpoint_sha256'], step=selected['selected_update']),
        c_variant=dict(name=args.variant, remote_path=f"/phase3/phase7/{remote_dir}/{fit['request_key']}/{best['checkpoint']['name']}",
                       sha256=best['checkpoint']['sha256']),
        candidate_only=args.with_base_prompts,
        scope='Extra seeds (3 per prompt) for the 5 text prompts of the Phase 7 image set' + ('; plus the 30 base prompts at seed 0, candidate only' if args.with_base_prompts else '; seed 0 images are reused'))
    from experiment_runtime import pin_reference_runtime
    pin_reference_runtime(request, recovered)
    request_key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    local = root / 'data/phase7/multiseed_text' / request_key
    local.mkdir(parents=True, exist_ok=True)
    (local / 'request.json').write_text(json.dumps(dict(request_key=request_key, request=request), indent=2)+'\n', encoding='utf-8')
    print('Request:', request_key, flush=True)
    if args.dry_run:
        return
    from modal_accounts import set_modal_account
    set_modal_account('current', purpose='compute')
    import modal
    runtime = request['teacher']['runtime']
    results_volume = modal.Volume.from_name(runtime['results_volume'])
    remote = '/phase7/multiseed_text/' + request_key
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
        app = modal.App('nihonga-phase7-multiseed-text')
        image = (modal.Image.debian_slim(python_version=runtime['python_version']).apt_install('git')
                 .pip_install(*request['execution_packages']).env(request['runtime_environment']))
        weights = modal.Volume.from_name(runtime['weights_volume'])
        namespace = {'results_volume': results_volume}
        exec(compile(request['worker_source'], 'multiseed_text_worker.py', 'exec'), namespace)
        worker = app.function(image=image, gpu='H100!', serialized=True, timeout=28800, cpu=8,
                              memory=98304, max_containers=1, retries=0,
                              volumes={'/phase3': results_volume,
                                       '/root/.cache/huggingface': weights.with_mount_options(read_only=True)})(namespace['generate_multiseed_text'])
        with modal.enable_output(), app.run(detach=True):
            call = worker.spawn(request, request_key)
            (local / 'call.json').write_text(json.dumps(dict(request_key=request_key, function_call_id=call.object_id, account='new'), indent=2)+'\n')
            print('Call:', call.object_id, flush=True)
            report = call.get()
    (local / 'summary.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: report.get(k) for k in ('status', 'error', 'calls')}), flush=True)
    if report.get('status') != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
