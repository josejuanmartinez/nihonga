"""Render teacher, B and C (B plus learned block-13 bridge) on the 30 earlier Phase 7 image prompts; qualitative only."""
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
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    recovered = root / 'data/recovered/modal'
    images_key = 'dda8c184e657dfb604c35fb261d83371c108c39fac033a0f57db66f7686c30ad'
    base = json.loads((recovered / 'phase7/candidate_images' / images_key / 'request.json').read_bytes())['request']
    scan_b = json.loads(next((root / 'data/phase7/conditional_importance_b').glob('*/request.json')).read_bytes())['request']
    fit = json.loads(next((root / 'data/phase7/block13_bridge').glob('*/summary.json')).read_bytes())
    if fit.get('status') != 'passed':
        raise RuntimeError('Block-13 bridge run did not complete.')
    selected = scan_b['selected_candidate']
    bridge13 = next(a for a in fit['artifacts'] if a['name'] == 'bridge_block_13_selected.pt')
    request = {k: base[k] for k in ('teacher', 'transformer_source_sha256', 'runtime_environment', 'expected_derived_tables',
                                    'canonical_tables_key', 'core_source', 'wrapper_source', 'rows')}
    request.update(
        version='phase7-bridge-images-v1', worker_source=(root / 'data/phase7/bridge_images_worker.py').read_text(encoding='utf-8'),
        execution_account='new',
        bridges={'5': dict(remote_path=f"/phase3/phase5/fitting/{scan_b['fit_key']}/{Path(selected['checkpoint_path']).name}",
                           sha256=selected['checkpoint_sha256'], step=selected['selected_update'], rank=256,
                           request_key=scan_b['fit_key']),
                 '13': dict(remote_path=f"/phase3/phase7/bridge_candidate/{fit['request_key']}/{bridge13['name']}",
                            sha256=bridge13['sha256'], step=fit['fit']['selected_step'], rank=256, request_key=fit['request_key'])},
        scope='Qualitative matched-seed screen of teacher, B and C on the 30 earlier Phase 7 image prompts; cannot override the failed velocity gates')
    from experiment_runtime import pin_reference_runtime
    pin_reference_runtime(request, recovered)
    request_key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    local = root / 'data/phase7/bridge_images' / request_key
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
    remote = '/phase7/bridge_images/' + request_key
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
        app = modal.App('nihonga-phase7-bridge-images')
        image = (modal.Image.debian_slim(python_version=runtime['python_version']).apt_install('git')
                 .pip_install(*request['execution_packages']).env(request['runtime_environment']))
        weights = modal.Volume.from_name(runtime['weights_volume'])
        namespace = {'results_volume': results_volume}
        exec(compile(request['worker_source'], 'bridge_images_worker.py', 'exec'), namespace)
        worker = app.function(image=image, gpu='H100!', serialized=True, timeout=28800, cpu=8,
                              memory=98304, max_containers=1, retries=0,
                              volumes={'/phase3': results_volume,
                                       '/root/.cache/huggingface': weights.with_mount_options(read_only=True)})(namespace['generate_bridge_images'])
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
