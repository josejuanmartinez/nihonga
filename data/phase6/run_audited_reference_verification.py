# Independently verify the deterministic reference and preserve its replay contract.
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
worker_path = Path('data/phase6/audited_reference_verification_worker.py')
worker_source = worker_path.read_text(encoding='utf-8')
request = {
    'version': 'phase6-audited-reference-verification-v3', 'mode': 'verify',
    'expected_state': read_json('data/phase6/state_audit_A/summary.json')['state'],
    'handoff_manifest_sha256': digest(handoff_path), 'teacher': handoff['teacher'],
    'records': handoff['target_index']['train'] + handoff['target_index']['validation'],
    'transformer_source_sha256': handoff['artifact_sha256']['data/phase4/transformer_qwenimage21_pinned.py'],
    'worker_source': worker_source,
    'runtime_environment': {'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'},
    'arithmetic_contract': {
        'weights_dtype': 'bfloat16', 'attention_backend': 'SDPBackend.MATH',
        'deterministic_algorithms': True, 'float32_matmul_precision': 'highest',
        'tf32_matmul': False, 'tf32_cudnn': False, 'cudnn_benchmark': False, 'cudnn_deterministic': True,
        'bf16_reduced_precision_matmul_reduction': False, 'fp16_bf16_reduction_math_sdp': False,
        'input_order': 'original pipeline argument order; stage tensors copied inside inference mode',
        'prefix_cache': 'fresh teacher cache per prompt; populate at early; use at middle and late',
    },
    'reference_key': read_json('data/phase6/deterministic_reference_generation/request.json')['request_key'],
    'reference_report_semantic_sha256': hashlib.sha256(json.dumps(read_json('data/phase6/deterministic_reference_generation/summary.json'), sort_keys=True).encode()).hexdigest(),
    'policy': 'new versioned teacher velocities from unchanged saved input samples; preserve all legacy targets and compare against them; no bridge or gradient; verification must use a fresh process and original 0.001 gate',
}
key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
run_dir = Path('data/phase6/audited_reference_verification')
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
remote_dir = f'/phase6/audited_reference_verification/{key}'
report = read_json(report_path) if report_path.exists() else None

def validate(report):
    if report['request_key'] != key or report['status'] != 'reference_verified':
        raise RuntimeError('Deterministic generation incomplete: ' + report.get('error', 'unknown'))
    if report['teacher_calls'] != 90 or report['student_calls'] != 0 or report['optimizer_updates'] != 0 or len(report['artifacts']) != 30:
        raise RuntimeError('Unexpected deterministic reference counts.')

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
        app = modal.App('nihonga-phase6-audited-reference-verification-v3')
        image = (modal.Image.debian_slim(python_version=runtime['python_version']).apt_install('git')
            .pip_install(*runtime['packages'])
            .env(request['runtime_environment']))
        weights_volume = modal.Volume.from_name(runtime['weights_volume'])
        namespace = {'results_volume': results_volume}
        exec(compile(worker_source, str(worker_path), 'exec'), namespace)
        worker = app.function(image=image, gpu='H100!', serialized=True, timeout=1800,
            max_containers=1, retries=0,
            volumes={'/phase3': results_volume, '/root/.cache/huggingface': weights_volume.with_mount_options(read_only=True)})(namespace['deterministic_reference'])
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
print('Deterministic reference:', report['status'], '| GPU:', report['gpu'])
print('Teacher forwards:', report['teacher_calls'], '| student forwards:', report['student_calls'], '| updates:', report['optimizer_updates'])
print('Saved prompts:', len(report['artifacts']), '| maximum difference from legacy:', report['legacy_max_relative_l2'])
print('Generation process:', report['process'])
print('Saved outputs:', run_dir)

generation_dir = Path('data/phase6/deterministic_reference_generation')
generation = read_json(generation_dir / 'summary.json')
first_task = generation['process']['modal_task_id']
second_task = report['process']['modal_task_id']
if not first_task or not second_task or first_task == second_task:
    raise RuntimeError('Independent Modal tasks were not established.')
if report['max_reference_relative_l2'] > 0.001:
    raise RuntimeError('Independent verification exceeds the unchanged gate.')
for artifact in generation['artifacts']:
    if digest(generation_dir / artifact['name']) != artifact['sha256']:
        raise RuntimeError('Local deterministic reference differs.')
contract = {
    'version': 'phase6-audited-replay-contract-v2',
    'expected_model_state': request['expected_state'],
    'execution_policy': 'audit the complete model state before inference; load reference output bundles only after prompt replay',
    'status': 'verified_for_saved_pilot',
    'legacy_cache_reproduced': False,
    'handoff_manifest_sha256': digest(handoff_path),
    'generation_request_key': generation['request_key'],
    'verification_request_key': key,
    'generation_report_sha256': digest(generation_dir / 'summary.json'),
    'generation_report_semantic_sha256': request['reference_report_semantic_sha256'],
    'verification_report_sha256': digest(report_path),
    'teacher': request['teacher'],
    'arithmetic_contract': request['arithmetic_contract'],
    'runtime_environment': request['runtime_environment'],
    'packages': report['packages'], 'torch_cuda': report['torch_cuda'], 'cudnn_version': report['cudnn_version'],
    'teacher_velocity_directory': generation_dir.as_posix(),
    'reference_artifacts': generation['artifacts'],
    'input_records': request['records'],
    'control_relative_l2_tolerance': 0.001,
    'verified_controls': 90, 'max_reference_relative_l2': report['max_reference_relative_l2'],
    'exact_equal_controls': report['exact_equal_controls'],
    'independent_tasks': [first_task, second_task],
    'student_policy': 'teacher/student comparisons must use the same deterministic arithmetic contract and exact saved inputs; discard old prefix caches after bridge insertion',
    'hidden_supervision_policy': 'legacy Phase 5 hidden pairs remain historical bridge-fitting data; surviving-layer hidden targets for healing need matching-contract capture',
    'scope': 'cached 20 TRAIN / 10 VALIDATION pilot and three stages; no student, healing, image quality or speed benchmark established',
    'legacy_policy': 'preserve original inputs, velocities, hidden pairs, trained bridges and failed attempts; legacy mismatch remains unexplained',
}
contract_path = Path('data/phase6/replay_contract.json')
if contract_path.exists():
    if read_json(contract_path) != contract:
        raise RuntimeError('Verified replay contract changed; preserve existing result.')
else:
    with contract_path.open('x', encoding='utf-8') as handle:
        handle.write(json.dumps(contract, indent=2) + '\n')
print('Independent replay controls:', len(report['controls']), '| exactly equal:', report['exact_equal_controls'])
print('Maximum independent replay relative L2:', report['max_reference_relative_l2'])
print('Verified replay contract:', contract_path)
