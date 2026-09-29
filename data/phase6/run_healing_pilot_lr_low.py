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
assessment = read_json('data/phase6/bridge_assessment.json')
for name, expected in assessment['input_sha256'].items():
    if digest(name) != expected:
        raise RuntimeError('Assessment source changed: ' + name)
if digest(assessment['target_index_path']) != assessment['target_index_sha256']:
    raise RuntimeError('Aligned target index changed.')
aligned_index = read_json(assessment['target_index_path'])
for split in ('train', 'validation'):
    for record in aligned_index[split]:
        if digest(record['tensor_path']) != record['tensor_sha256']:
            raise RuntimeError('Aligned target changed.')
components_path = Path('data/phase6/healing_components.py')
component_checks = read_json('data/phase6/healing_component_checks.json')
if component_checks['status'] != 'passed' or component_checks['source_sha256'] != digest(components_path):
    raise RuntimeError('Functional cache component checks are required.')
worker_path = Path('data/phase6/healing_pilot_worker.py')
worker_source = worker_path.read_text(encoding='utf-8')
request = {
    'version': 'phase6-parameter-efficient-healing-pilot-low-lr-v2', 'mode': 'verify',
    'teacher': handoff['teacher'],
    'handoff_manifest_sha256': digest(handoff_path),
    'replay_contract_sha256': digest('data/phase6/replay_contract.json'),
    'assessment_sha256': digest('data/phase6/bridge_assessment.json'),
    'component_checks_sha256': digest('data/phase6/healing_component_checks.json'),
    'expected_state': contract['expected_model_state'],
    'arithmetic_contract': contract['arithmetic_contract'],
    'runtime_environment': contract['runtime_environment'],
    'reference_key': contract['generation_request_key'],
    'reference_report_semantic_sha256': contract['generation_report_semantic_sha256'],
    'records': handoff['target_index']['train'] + handoff['target_index']['validation'],
    'transformer_source_sha256': handoff['artifact_sha256']['data/phase4/transformer_qwenimage21_pinned.py'],
    'selected_candidate': assessment['selected_candidate'],
    'fit_key': handoff['phase5_fitting_request_key'],
    'aligned_target_index': aligned_index,
    'previous_replay_report': read_json('data/phase6/verified_student_replay/summary.json'),
    'wrapper_source': Path('data/phase4/bridge_block_definition.py').read_text(encoding='utf-8'),
    'components_source': components_path.read_text(encoding='utf-8'),
    'components_sha256': digest(components_path), 'worker_source': worker_source,
    'adapter_rank': 8, 'adapter_alpha': 8, 'seed': 20260929, 'sampling_seed': 20260930,
    'updates': 120, 'checkpoint_every': 30, 'learning_rate': 0.00001,
    'weight_decay': 0.01, 'gradient_clip': 1.0,
    'hidden_weight': 0.1, 'bridge_weight': 0.1,
    'optimizer': 'AdamW on FP32 bridge and adapter masters; original surviving weights frozen',
    'sampling': 'two independently shuffled passes over all 20 TRAIN prompts and their three stages; batch one',
    'objective': 'relative velocity MSE + 0.1 mean normalized-vector squared distance at surviving slots 2/3/4 + 0.1 corresponding bridge-slot-5 distance; equal available prefix/image role weights',
    'cache_policy': 'fresh student cache each update/evaluation prompt; middle/late rebuild differentiable prefix from saved early input; functional checkpointing snapshots cached K/V and returns extracted K/V',
    'selection': 'minimum equal-prompt/equal-stage VALIDATION mean velocity relative L2 at step 0/30/60/90/120; preserve step 0 if no improvement',
    'checkpoint_policy': 'save baseline, gradient smoke, schedule, every 30-update optimizer/model/RNG checkpoint and validation outputs, then reload and verify selected checkpoint; save emergency state on handled failure',
    'scope': 'parameter-efficient healing implementation pilot on saved TRAIN/VALIDATION; no full-weight fine-tuning, generated images or independent test',
}

key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
run_dir = Path('data/phase6/healing_pilot_lr_low')
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
remote_dir = f'/phase6/healing_pilot/{key}'
report = read_json(report_path) if report_path.exists() else None

def validate(report):
    if report['request_key'] != key or report['status'] != 'passed':
        raise RuntimeError('Healing pilot failed; saved state preserved, no automatic retry: ' + report.get('error', 'unknown'))
    if report['teacher_calls'] != 90 or report['student_calls'] != 384 or report['optimizer_updates'] != 120:
        raise RuntimeError('Unexpected pilot forward/update counts.')
    if report['max_reference_relative_l2'] > 0.001 or not report['frozen_parameters_unchanged']:
        raise RuntimeError('Teacher control or frozen-weight audit failed.')
    if report['validation_optimizer_updates'] != 0 or len(report['evaluations']) != 5:
        raise RuntimeError('Validation/checkpoint policy differs.')

complete = False
if report is not None:
    if report['request_key'] != key:
        raise RuntimeError('Report provenance differs.')
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
        try:
            json.loads(b''.join(results_volume.read_file(remote_dir + '/started.json')))
        except FileNotFoundError:
            pass
        else:
            raise RuntimeError('This GPU attempt already started; inspect its progress and wait for its durable report. No GPU is rescheduled.')
        app = modal.App('nihonga-phase6-healing-pilot-low-lr')
        image = (modal.Image.debian_slim(python_version=runtime['python_version']).apt_install('git')
            .pip_install(*runtime['packages'])
            .env(request['runtime_environment']))
        weights_volume = modal.Volume.from_name(runtime['weights_volume'])
        namespace = {'results_volume': results_volume}
        exec(compile(worker_source, str(worker_path), 'exec'), namespace)
        worker = app.function(image=image, gpu='H100!', serialized=True, timeout=7200, cpu=8, memory=65536,
            max_containers=1, retries=0,
            volumes={'/phase3': results_volume, '/root/.cache/huggingface': weights_volume.with_mount_options(read_only=True)})(namespace['heal_student'])
        with modal.enable_output(), app.run(detach=True):
            remote_report = worker.remote(request, key)
    if remote_report['request_key'] != key:
        raise RuntimeError('Remote report provenance differs.')
    if report is not None and report != remote_report:
        raise RuntimeError('Local and remote replay reports differ.')
    if not report_path.exists():
        with report_path.open('x', encoding='utf-8') as handle:
            handle.write(json.dumps(remote_report, indent=2) + '\n')
    report = remote_report
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
validate(report)
print('Healing pilot:', report['status'], '| GPU:', report['gpu'])
print('Teacher forwards:', report['teacher_calls'], '| student forwards:', report['student_calls'], '| optimizer updates:', report['optimizer_updates'])
print('Trainable parameters:', report['trainable_parameters'], '| adapter modules:', len(report['adapter_modules']))
print('Gradient preflight:', report['gradient_smoke']['inference_equivalence'])
for evaluation in report['evaluations']:
    print('Step', evaluation['step'], '| validation mean relative L2:', evaluation['mean_relative_l2'], '| maximum:', evaluation['max_relative_l2'])
print('Selected step:', report['selected_step'], '| error reduction:', report['selected_relative_error_reduction'])
print('Improved comparisons:', report['selected_improved_comparisons'], '/30')
print('Frozen parameters checked:', report['frozen_parameter_audit_count'])
print('Selected checkpoint:', run_dir / report['selected_checkpoint'])
print('Saved pilot artifacts:', run_dir)
