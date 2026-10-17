"""Fit several bridge architectures for one block inside candidate B and evaluate each whole student; reuse completed results."""
import argparse
import hashlib
import json
import sys
from pathlib import Path


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--slot', type=int, default=13)
    parser.add_argument('--old-account', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Write the request locally without scheduling GPU work.')
    args = parser.parse_args()
    if args.old_account:
        parser.error('Old account is download-only; GPU execution must use the new account.')
    root = Path(__file__).resolve().parents[1]
    protocol_path = root / f'data/phase7/block{args.slot}_bridge_v2/protocol.json'
    protocol = json.loads(protocol_path.read_bytes())
    scan_b = json.loads(next((root / 'data/phase7/conditional_importance_b').glob('*/request.json')).read_bytes())['request']
    scan_a = json.loads((root / 'data/phase7/conditional_importance/7454da6a5c1dc33af23dd46833e698f9b759d14985ca96340bc3d40cd47fb4f5/request.json').read_bytes())['request']
    if args.slot == 5 or not 0 < args.slot < 31:
        parser.error('Slot must be a surviving interior block.')
    request = {k: v for k, v in scan_b.items() if k not in ('worker_source', 'eligible_original_block_ids', 'selection', 'scope', 'version')}
    source = (root / 'data/phase7/bridge_variants_worker.py').read_text(encoding='utf-8')
    request.update(
        version='phase7-bridge-variants-v1', worker_source=source, slot=args.slot,
        conditioned_bridge_source=(root / 'data/phase7/conditioned_bridge.py').read_text(encoding='utf-8'),
        variants=[dict(name='plain_r1024', rank=1024, conditioned=False), dict(name='cond_r256', rank=256, conditioned=True),
                  dict(name='cond_r1024', rank=1024, conditioned=True)],
        protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        fresh_aligned_train=scan_a['fresh_aligned_train'], early_pool=scan_a['early_pool'],
        fit=dict(init_seed=20261017, capture_seed=20261018, sampling_seed=20261019, learning_rate=1e-3,
                 weight_decay=0.01, gradient_clip=1.0, magnitude_weight=0.1, updates=1000, checkpoint_every=100,
                 tokens_per_group_full=256, tokens_per_group_early_pool=64, tokens_per_group_update=256),
        selection='Minimum validation hidden-state objective at saved checkpoints including identity start',
        scope=f'Bridge variants for block {args.slot} inside frozen B; TRAIN-only fitting; whole-student velocity on scan inputs; no healing, images or latency')
    request_key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    local = protocol_path.parent / request_key
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
    remote = '/phase7/bridge_variants/' + request_key
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
        app = modal.App('nihonga-phase7-bridge-variants')
        image = (modal.Image.debian_slim(python_version=runtime['python_version']).apt_install('git')
                 .pip_install(*request['execution_packages']).env(request['runtime_environment']))
        weights = modal.Volume.from_name(runtime['weights_volume'])
        namespace = {'results_volume': results_volume}
        exec(compile(source, 'bridge_variants_worker.py', 'exec'), namespace)
        worker = app.function(image=image, gpu='H100!', serialized=True, timeout=28800, cpu=8,
                              memory=98304, max_containers=1, retries=0,
                              volumes={'/phase3': results_volume,
                                       '/root/.cache/huggingface': weights.with_mount_options(read_only=True)})(namespace['fit_bridge_variants'])
        with modal.enable_output(), app.run(detach=True):
            call = worker.spawn(request, request_key)
            (local / 'call.json').write_text(json.dumps(dict(request_key=request_key, function_call_id=call.object_id, account='new'), indent=2)+'\n')
            print('Call:', call.object_id, flush=True)
            report = call.get()
    (local / 'summary.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: report.get(k) for k in ('status', 'error', 'candidate_replay_max_relative_l2')}), flush=True)
    for name, v in (report.get('variants') or {}).items():
        print(name, 'step', v['fit']['selected_step'], 'teacher-relative', v['overall']['teacher_relative_l2']['mean'], flush=True)
    if report.get('status') != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
