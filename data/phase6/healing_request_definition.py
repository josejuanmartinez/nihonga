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
    'version': 'phase6-parameter-efficient-healing-pilot-v1', 'mode': 'verify',
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
    'updates': 120, 'checkpoint_every': 30, 'learning_rate': 0.0001,
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
