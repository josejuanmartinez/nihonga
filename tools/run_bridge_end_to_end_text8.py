"""Grow the attention bridge to 8 heads and continue it on 450 text prompts recorded (and saved) in the same job; reuse completed results."""
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
    protocol_path = root / f'data/phase7/block{args.slot}_bridge_e2e_text8/protocol.json'
    protocol = json.loads(protocol_path.read_bytes())
    scan_b = json.loads(next((root / 'data/phase7/conditional_importance_b').glob('*/request.json')).read_bytes())['request']
    scan_a = json.loads((root / 'data/phase7/conditional_importance/7454da6a5c1dc33af23dd46833e698f9b759d14985ca96340bc3d40cd47fb4f5/request.json').read_bytes())['request']
    if args.slot == 5 or not 0 < args.slot < 31:
        parser.error('Slot must be a surviving interior block.')
    request = {k: v for k, v in scan_b.items() if k not in ('worker_source', 'eligible_original_block_ids', 'selection', 'scope', 'version')}
    warm = next(r for r in (json.loads(p.read_bytes()) for p in (root / 'data/phase7/block13_bridge_e2e_text').glob('*/summary.json')) if r.get('status') == 'passed')
    import random
    used = {r['source_id'] for r in scan_a['records']} | {a['source_id'] for a in scan_a['fresh_aligned_train']['artifacts']}
    used |= {'p1-' + e['capture_id'].split('p1-')[1].rsplit('-s', 1)[0] for e in scan_a['early_pool']['examples']}
    used |= {r['source_id'] for r in scan_a['expanded_validation']['records']}
    resolutions = json.loads((root / 'data/phase2/protocol.json').read_bytes())['RESOLUTIONS']
    pilot = [json.loads(line) for line in (root / 'data/phase1/pilot_20k/train.jsonl').read_text(encoding='utf-8').splitlines()]
    pilot = [r for r in pilot if r['split'] == 'train' and r['id'] not in used and r['aspect_ratio'] in resolutions]
    picker = random.Random(20261020)
    text_rows = picker.sample(sorted((r for r in pilot if r['subdimension'] == 'text_rendering'), key=lambda r: r['id']), 150)
    general_rows = picker.sample(sorted((r for r in pilot if r['subdimension'] != 'text_rendering'), key=lambda r: r['id']), 50)
    chosen = {r['id'] for r in text_rows + general_rows}
    extra_text = random.Random(20261022).sample(sorted((r for r in pilot if r['subdimension'] == 'text_rendering' and r['id'] not in chosen),
                                                       key=lambda r: r['id']), 300)
    new_prompts = [dict(source_id=r['id'], prompt=r['prompt'], seed=r['seed'], dimension=r['dimension'], subdimension=r['subdimension'],
                        width=resolutions[r['aspect_ratio']][0], height=resolutions[r['aspect_ratio']][1])
                   for r in text_rows + general_rows + extra_text]
    source = (root / 'data/phase7/bridge_end_to_end_v3_worker.py').read_text(encoding='utf-8')
    request.update(
        version='phase7-bridge-end-to-end-text8-v1', new_prompts=new_prompts, worker_source=source, slot=args.slot,
        protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        conditioned_bridge_source=(root / 'data/phase7/conditioned_bridge.py').read_text(encoding='utf-8'),
        components_source=scan_a['components_source'],
        fresh_aligned_train=scan_a['fresh_aligned_train'], early_pool=scan_a['early_pool'],
        warm_start=dict(name='e2e_attn_text', remote_path=f"/phase3/phase7/bridge_end_to_end_v2/{warm['request_key']}/{warm['variants']['e2e_attn_text']['checkpoint']['name']}",
                        sha256=warm['variants']['e2e_attn_text']['checkpoint']['sha256']),
        variants=[dict(name='e2e_attn8_text', attention=True, heads=8)],
        fit=dict(init_seed=20261017, sampling_seed=20261023, learning_rate=1e-4, warmup=20, gradient_clip=1.0,
                 updates=600, eval_every=100, early_weight=0.3, text_fraction=0.4),
        selection='Lowest gate10 velocity error at evaluations every 50 updates, including the warm start',
        scope=f'End-to-end bridges for block {args.slot} inside frozen B; TRAIN-only; whole-student velocity on scan inputs; no images or latency')
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
    remote = '/phase7/bridge_end_to_end_v3/' + request_key
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
        app = modal.App('nihonga-phase7-bridge-end-to-end-text8')
        image = (modal.Image.debian_slim(python_version=runtime['python_version']).apt_install('git')
                 .pip_install(*request['execution_packages']).env(request['runtime_environment']))
        weights = modal.Volume.from_name(runtime['weights_volume'])
        namespace = {'results_volume': results_volume}
        exec(compile(source, 'bridge_end_to_end_v3_worker.py', 'exec'), namespace)
        worker = app.function(image=image, gpu='H100!', serialized=True, timeout=28800, cpu=8,
                              memory=98304, max_containers=1, retries=0,
                              volumes={'/phase3': results_volume,
                                       '/root/.cache/huggingface': weights.with_mount_options(read_only=True)})(namespace['train_bridge_end_to_end'])
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
