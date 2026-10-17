"""Greedy conditional block selection inside candidate B (identity skips, rerank after each); reuse completed results."""
import argparse
import hashlib
import json
import sys
from pathlib import Path


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-account', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Write the request locally without scheduling GPU work.')
    args = parser.parse_args()
    if args.old_account:
        parser.error('Old account is download-only; GPU execution must use the new account.')
    root = Path(__file__).resolve().parents[1]
    recovered = root / 'data/recovered/modal'
    replay_key = 'ccabf81790b354ca69bcb5d0213c7559e15ba3458bf5d5c17036ecb79b73d12c'
    validation_key = '173d4a2d3ab854f15ae4c710e44b9e7afe122e429742f46d8fb7c72cce1880da'
    scan_a_key = '7454da6a5c1dc33af23dd46833e698f9b759d14985ca96340bc3d40cd47fb4f5'
    replay_dir = recovered / 'phase6/verified_student_replay' / replay_key
    replay = json.loads((replay_dir / 'request.json').read_bytes())['request']
    replay_summary = json.loads((replay_dir / 'summary.json').read_bytes())
    selected = json.loads((recovered / 'phase6/expanded_validation' / validation_key / 'request.json').read_bytes())['request']
    # Scan inputs (fixed panels, teacher contract, runtime) are shared with the candidate A scan.
    scan_a = json.loads((root / 'data/phase7/conditional_importance' / scan_a_key / 'request.json').read_bytes())
    if hashlib.sha256(json.dumps(scan_a['request'], sort_keys=True).encode()).hexdigest() != scan_a_key:
        raise RuntimeError('Candidate A scan request differs from its key.')
    scan_a = scan_a['request']
    candidate = selected['selected_candidate']
    if candidate['candidate_id'] != 'single_bridge_block_05' or selected['fit_key'] != replay['fit_key']:
        raise RuntimeError('Selected Phase 6 candidate differs.')
    if replay['core_source'] != scan_a['core_source'] or replay['wrapper_source'] != scan_a['wrapper_source']:
        raise RuntimeError('Bridge sources differ between Phase 6 replay and the scan contract.')
    shared = ('mode', 'teacher', 'handoff_manifest_sha256', 'replay_contract_sha256', 'expected_state',
              'arithmetic_contract', 'runtime_environment', 'reference_key', 'reference_report_semantic_sha256',
              'records', 'transformer_source_sha256', 'expanded_validation', 'core_source', 'wrapper_source',
              'canonical_tables_key', 'cache_policy', 'gate_rule', 'gate_references',
              'execution_packages', 'execution_reference_runtime')
    request = {k: scan_a[k] for k in shared}
    source = (root / 'data/phase7/greedy_selection_worker.py').read_text(encoding='utf-8')
    request.update(
        version='phase7-greedy-selection-b-v1', greedy_rounds=3, worker_source=source, execution_account='new',
        selected_candidate=candidate, fit_key=selected['fit_key'],
        selected_replay=dict(request_key=replay_key, artifacts={a['name']: a['sha256'] for a in replay_summary['artifacts']
                                                                  if a['name'].startswith('students-')}),
        eligible_original_block_ids=[4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 25, 26],
        selection='Greedy: per round, skip the block with the lowest teacher-relative mean; candidates are blocks under ~12% in the B scan',
        scope='Fixed gate10 and wider50 development prompts, three saved stages; choose three blocks to remove after block 5')
    request_key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    local = root / 'data/phase7/greedy_selection' / request_key
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
    remote = '/phase7/greedy_selection/' + request_key
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
        app = modal.App('nihonga-phase7-greedy-selection')
        image = (modal.Image.debian_slim(python_version=runtime['python_version']).apt_install('git')
                 .pip_install(*request['execution_packages']).env(request['runtime_environment']))
        weights = modal.Volume.from_name(runtime['weights_volume'])
        namespace = {'results_volume': results_volume}
        exec(compile(source, 'greedy_selection_worker.py', 'exec'), namespace)
        worker = app.function(image=image, gpu='H100!', serialized=True, timeout=28800, cpu=8,
                              memory=65536, max_containers=1, retries=0,
                              volumes={'/phase3': results_volume,
                                       '/root/.cache/huggingface': weights.with_mount_options(read_only=True)})(namespace['greedy_select'])
        with modal.enable_output(), app.run(detach=True):
            call = worker.spawn(request, request_key)
            (local / 'call.json').write_text(json.dumps(dict(request_key=request_key, function_call_id=call.object_id, account='new'), indent=2)+'\n')
            print('Call:', call.object_id, flush=True)
            report = call.get()
    (local / 'summary.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: report.get(k) for k in ('status', 'error', 'selected_blocks', 'candidate_replay_max_relative_l2')}), flush=True)
    if report.get('status') != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
