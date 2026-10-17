"""Launch the hash-pinned conditional scan; reuse completed remote results."""
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
    healing_key = '95fcd2cfae35cb7f124b8af2865c30f928cbd681a0d015df5e56e8ae5ba4fd78'
    images_key = 'dda8c184e657dfb604c35fb261d83371c108c39fac033a0f57db66f7686c30ad'
    healing = recovered / 'phase7/blocks45_healing' / healing_key
    request = json.loads((healing / 'request.json').read_bytes())['request']
    summary = json.loads((healing / 'summary.json').read_bytes())
    request['image_request'] = json.loads((recovered / 'phase7/candidate_images' / images_key / 'request.json').read_bytes())['request']
    request['historical_validation_sha256'] = next(a['sha256'] for a in summary['artifacts'] if a['name']=='validation_step_060.pt')
    source = (root / 'data/phase7/conditional_scan_worker.py').read_text(encoding='utf-8')
    request.update(version='phase7-conditional-importance-v1', worker_source=source, execution_account='new',
                   eligible_original_block_ids=[i for i in range(32) if i not in (4, 5)],
                   selection='Screening only: independent frozen bypasses; no pruning or quality promotion',
                   scope='Fixed gate10 and wider50 development prompts, three saved stages; compare A and teacher')
    request_key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    local = root / 'data/phase7/conditional_importance' / request_key
    local.mkdir(parents=True, exist_ok=True)
    (local / 'request.json').write_text(json.dumps(dict(request_key=request_key, request=request), indent=2)+'\n',encoding='utf-8')
    runtime = request['teacher']['runtime']
    results_volume = modal.Volume.from_name(runtime['results_volume'])
    remote = '/phase7/conditional_importance/' + request_key
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
        app = modal.App('nihonga-phase7-conditional-importance')
        image = (modal.Image.debian_slim(python_version=runtime['python_version']).apt_install('git')
                 .pip_install(*runtime['packages']).env(request['runtime_environment']))
        weights = modal.Volume.from_name(runtime['weights_volume'])
        namespace = {'results_volume': results_volume}
        exec(compile(source, 'conditional_scan_worker.py', 'exec'), namespace)
        worker = app.function(image=image, gpu='H100!', serialized=True, timeout=28800, cpu=8,
                              memory=65536, max_containers=1, retries=0,
                              volumes={'/phase3':results_volume,
                                       '/root/.cache/huggingface':weights.with_mount_options(read_only=True)})(namespace['scan_candidate'])
        with modal.enable_output(), app.run(detach=True):
            call = worker.spawn(request, request_key)
            (local / 'call.json').write_text(json.dumps(dict(request_key=request_key, function_call_id=call.object_id, account='new'),indent=2)+'\n')
            print('Request:',request_key,'Call:',call.object_id,flush=True)
            report = call.get()
    (local / 'summary.json').write_text(json.dumps(report, indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:report.get(k) for k in ('status','error','ranking','candidate_A_replay_max_relative_l2')}),flush=True)
    if report.get('status') != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
