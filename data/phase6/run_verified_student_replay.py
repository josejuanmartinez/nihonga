# Compare independent BF16 students on cached validation inputs, with durable reuse.
import hashlib
import json
from pathlib import Path

root = Path.cwd()
def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

handoff_path = Path('data/phase6/handoff/manifest.json')
handoff_summary = read_json('data/phase6/handoff/summary.json')
if digest(handoff_path) != handoff_summary['manifest_sha256']:
    raise RuntimeError('Handoff manifest changed.')
handoff = read_json(handoff_path)
for name, expected in handoff['artifact_sha256'].items():
    if digest(name) != expected:
        raise RuntimeError('Handoff artifact changed: ' + name)
contract = read_json('data/phase6/replay_contract.json')
if contract['status'] != 'verified_for_saved_pilot' or contract['verified_controls'] != 90:
    raise RuntimeError('A verified replay contract is required.')
worker_path = Path('data/phase6/verified_student_replay_worker.py')
worker_source = worker_path.read_text(encoding='utf-8')
request = {
    'version': 'phase6-verified-student-replay-v2', 'mode': 'verify',
    'replay_contract_sha256': digest('data/phase6/replay_contract.json'),
    'expected_state': contract['expected_model_state'],
    'arithmetic_contract': contract['arithmetic_contract'],
    'runtime_environment': contract['runtime_environment'],
    'reference_key': contract['generation_request_key'],
    'reference_report_semantic_sha256': contract['generation_report_semantic_sha256'],
    'records': handoff['target_index']['train'] + handoff['target_index']['validation'],
    'handoff_manifest_sha256': digest(handoff_path),
    'teacher': handoff['teacher'], 'candidates': handoff['candidates'],
    'validation': handoff['target_index']['validation'],
    'fit_key': handoff['phase5_fitting_request_key'],
    'transformer_source_sha256': handoff['artifact_sha256']['data/phase4/transformer_qwenimage21_pinned.py'],
    'core_source': Path('data/phase4/bridge_definition.py').read_text(encoding='utf-8'),
    'wrapper_source': Path('data/phase4/bridge_block_definition.py').read_text(encoding='utf-8'),
    'worker_source': worker_source,
    'variants': ['identity', 'trained'], 'stages': [0, 20, 39],
    'teacher_control_relative_l2_tolerance': 0.001,
    'precision': 'BF16 pipeline weights and residual addition; FP32 metrics',
    'cache_policy': 'fresh prefix cache for every teacher/candidate/variant/prompt; rebuild at extraction',
    'metrics': 'full target-image velocity relative L2, MSE, RMSE, maximum absolute difference and cosine; unweighted prompt means by stage',
    'selection': 'no winner selected; retain all per-prompt outputs and regressions',
    'scope': 'held-out fixed-input inference compatibility; no optimizer, gradient test, images or student trajectories',
}
key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
run_dir = Path('data/phase6/verified_student_replay')
run_dir.mkdir(exist_ok=True)
manifest_path = run_dir / 'request.json'
manifest = {'request': request, 'request_key': key}
if manifest_path.exists():
    if read_json(manifest_path) != manifest:
        raise RuntimeError('Replay request changed; preserve saved results.')
else:
    with manifest_path.open('x', encoding='utf-8') as handle:
        handle.write(json.dumps(manifest, indent=2) + '\n')
report_path = run_dir / 'summary.json'
remote_dir = f'/phase6/verified_student_replay/{key}'
report = read_json(report_path) if report_path.exists() else None

def validate(report):
    if report['request_key'] != key:
        raise RuntimeError('Replay report provenance differs.')
    if report['status'] != 'passed':
        raise RuntimeError('Replay failed; inspect saved attempt, no automatic retry: ' + report.get('error', 'unknown'))
    if report['max_reference_relative_l2'] > 0.001 or len(report['artifacts']) != 40:
        raise RuntimeError('Teacher controls or artifact counts differ.')
    if report['teacher_calls'] != 90 or report['student_calls'] != 240 or report['optimizer_updates'] != 0:
        raise RuntimeError('Unexpected replay counts.')

complete = False
if report is not None:
    validate(report)
    complete = all((run_dir / a['name']).exists() for a in report['artifacts'])
    for a in report['artifacts']:
        path = run_dir / a['name']
        if path.exists() and digest(path) != a['sha256']:
            raise RuntimeError('Saved replay tensor changed.')
if complete:
    print('Reusing verified replay results; no model imports, network calls or forwards.')
else:
    import modal
    runtime = handoff['teacher']['runtime']
    results_volume = modal.Volume.from_name(runtime['results_volume'])
    try:
        remote_report = json.loads(b''.join(results_volume.read_file(remote_dir + '/summary.json')))
        print('Recovering durable remote replay; no GPU scheduled.')
    except FileNotFoundError:
        if report is not None or any(run_dir.glob('*.pt')):
            raise RuntimeError('Local results are incomplete and no remote report exists; inspect before retrying.')
        app = modal.App('nihonga-phase6-verified-student-replay')
        image = (modal.Image.debian_slim(python_version=runtime['python_version']).apt_install('git')
            .pip_install(*runtime['packages'])
            .env(request['runtime_environment']))
        weights_volume = modal.Volume.from_name(runtime['weights_volume'])
        namespace = {'results_volume': results_volume}
        exec(compile(worker_source, str(worker_path), 'exec'), namespace)
        worker = app.function(image=image, gpu='H100!', serialized=True, timeout=3600,
            max_containers=1, retries=0,
            volumes={'/phase3': results_volume, '/root/.cache/huggingface': weights_volume.with_mount_options(read_only=True)})(namespace['replay_students'])
        with modal.enable_output(), app.run():
            remote_report = worker.remote(request, key)
    if remote_report['request_key'] != key:
        raise RuntimeError('Remote report provenance differs.')
    if report is not None and report != remote_report:
        raise RuntimeError('Local and remote replay reports differ.')
    if not report_path.exists():
        with report_path.open('x', encoding='utf-8') as handle:
            handle.write(json.dumps(remote_report, indent=2) + '\n')
    report = remote_report
    validate(report)
    for artifact in report['artifacts']:
        path = run_dir / artifact['name']
        if path.exists():
            if digest(path) != artifact['sha256']:
                raise RuntimeError('Local velocity artifact differs.')
            continue
        temporary = path.with_suffix('.download')
        with temporary.open('wb') as handle:
            for chunk in results_volume.read_file(remote_dir + '/' + artifact['name']):
                handle.write(chunk)
        if digest(temporary) != artifact['sha256']:
            raise RuntimeError('Downloaded velocities differ.')
        temporary.replace(path)
print('Student replay:', report['status'], '| GPU:', report['gpu'])
print('Teacher control calls:', report['teacher_calls'], '| student calls:', report['student_calls'], '| updates:', report['optimizer_updates'])
for row in report['aggregates']:
    print(row['candidate_id'], '|', row['stage'], '| identity L2:', round(row['identity_mean_relative_l2'], 6),
          '| trained L2:', round(row['trained_mean_relative_l2'], 6), '| improved prompts:', str(row['improved_comparisons']) + '/' + str(row['comparisons']))
print('Saved report and velocity tensors:', run_dir)

print('Teacher exact comparisons:', report['exact_equal_controls'], '| max reference relative L2:', report['max_reference_relative_l2'])
