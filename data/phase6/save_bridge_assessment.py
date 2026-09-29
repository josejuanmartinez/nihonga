# Save the bridge decision and index the versioned, matching-contract targets.
import hashlib
import json
from pathlib import Path

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def save_once(path, value):
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise RuntimeError('Saved decision artifact differs: ' + str(path))
    else:
        with path.open('x', encoding='utf-8') as handle:
            handle.write(json.dumps(value, indent=2) + '\n')

student_path = Path('data/phase6/verified_student_replay/summary.json')
hidden_path = Path('data/phase6/aligned_hidden_capture/summary.json')
contract_path = Path('data/phase6/replay_contract.json')
inputs = {p.as_posix(): digest(p) for p in (student_path, hidden_path, contract_path)}
decision_path = Path('data/phase6/bridge_assessment.json')
index_path = Path('data/phase6/aligned_hidden_capture/target_index.json')
if decision_path.exists():
    decision = read_json(decision_path)
    if decision['input_sha256'] != inputs or digest(index_path) != decision['target_index_sha256']:
        raise RuntimeError('Decision inputs or target index changed.')
    print('Reusing the saved bridge decision; no metrics or selection recalculated.')
else:
    student, hidden = read_json(student_path), read_json(hidden_path)
    if student['status'] != 'passed' or hidden['status'] != 'passed':
        raise RuntimeError('Both evaluations must pass before the decision.')
    if student['max_reference_relative_l2'] > 0.001 or hidden['max_reference_relative_l2'] > 0.001:
        raise RuntimeError('Teacher control gate failed.')
    if student['optimizer_updates'] or hidden['optimizer_updates']:
        raise RuntimeError('Unexpected training during assessment.')
    handoff = read_json('data/phase6/handoff/manifest.json')
    capture_request = read_json('data/phase6/aligned_hidden_capture/request.json')
    artifacts = {a['name']: a for a in hidden['artifacts']}
    target_index = {'version': 'phase6-aligned-target-index-v1', 'capture_request_key': hidden['request_key'],
        'teacher_replay_contract_sha256': inputs[contract_path.as_posix()],
        'remote_directory': '/phase6/aligned_hidden_capture/' + hidden['request_key'], 'train': [], 'validation': []}
    for record in capture_request['request']['records']:
        name = record['capture_id'] + '.pt'
        artifact = artifacts[name]
        target_index['validation' if 'validation' in record['capture_id'] else 'train'].append({
            'capture_id': record['capture_id'], 'source_id': record['source_id'], 'dimension': record['dimension'],
            'request_key': hidden['request_key'], 'tensor_path': (hidden_path.parent / name).as_posix(),
            'tensor_sha256': artifact['sha256'], 'legacy_input_path': record['tensor_path'],
            'legacy_input_sha256': record['tensor_sha256'], 'records': 12})
    if len(target_index['train']) != 20 or len(target_index['validation']) != 10:
        raise RuntimeError('Aligned target index split counts differ.')
    save_once(index_path, target_index)
    global_rows = [r for r in student['aggregates'] if r['stage'] == 'all']
    fresh_rows = {r['layer']: r for r in hidden['bridge_hidden_aggregates'] if r['split'] == 'validation'}
    candidates = []
    for row in global_rows:
        recipe = next(c for c in handoff['candidates'] if c['candidate_id'] == row['candidate_id'])
        layer = recipe['removed_teacher_slots'][0]
        old = read_json('data/phase5/bridge_fitting_pilot/bridge_block_' + str(layer).zfill(2) + '_summary.json')
        fresh = fresh_rows[layer]
        candidates.append({'candidate_id': row['candidate_id'], 'layer': layer,
            'mean_velocity_relative_l2': row['trained_mean_relative_l2'],
            'velocity_error_reduction': row['relative_error_reduction'],
            'improved_velocity_comparisons': row['improved_comparisons'],
            'total_velocity_comparisons': row['comparisons'],
            'historical_validation_objective': old['selected_validation']['balanced_objective'],
            'fresh_validation_objective': fresh['trained']['balanced_objective'],
            'fresh_objective_relative_change': fresh['trained']['balanced_objective']/old['selected_validation']['balanced_objective'] - 1,
            'fresh_objective_reduction_from_identity': fresh['relative_objective_reduction']})
    supported = [r for r in candidates if r['velocity_error_reduction'] > 0 and r['fresh_objective_reduction_from_identity'] > 0]
    if not supported:
        raise RuntimeError('No existing bridge improves both local fit and complete velocity; assess refitting.')
    chosen = min(supported, key=lambda r: r['mean_velocity_relative_l2'])
    recipe = next(c for c in handoff['candidates'] if c['candidate_id'] == chosen['candidate_id'])
    decision = {'version': 'phase6-existing-bridge-assessment-v1', 'input_sha256': inputs,
        'target_index_path': index_path.as_posix(), 'target_index_sha256': digest(index_path),
        'decision': 'retain existing pretrained candidate for a healing pilot; no Phase 5 rerun justified solely by the replay change',
        'selected_candidate': recipe, 'candidate_evidence': candidates,
        'selection_policy': 'lowest equal-prompt/equal-stage mean validation velocity relative L2 among existing candidates improving both velocity and fresh local fit versus identity; no claim of statistical superiority',
        'original_targets_preserved': True, 'optimizer_updates': 0,
        'limitations': 'same small validation split used for Phase 5 checkpoint selection; no independent test, generated images, student trajectories, healing or speed benchmark',
        'teacher_contract': contract_path.as_posix()}
    save_once(decision_path, decision)
index = read_json(index_path)
for split in ('train', 'validation'):
    for record in index[split]:
        if digest(record['tensor_path']) != record['tensor_sha256']:
            raise RuntimeError('Aligned target artifact changed.')
if digest(decision['selected_candidate']['checkpoint_path']) != decision['selected_candidate']['checkpoint_sha256']:
    raise RuntimeError('Selected pretrained checkpoint changed.')
print('Decision:', decision['decision'])
print('Candidate:', decision['selected_candidate']['candidate_id'])
for row in decision['candidate_evidence']:
    print('Block', row['layer'], '| velocity relative L2:', row['mean_velocity_relative_l2'],
          '| historical hidden objective:', row['historical_validation_objective'],
          '| fresh hidden objective:', row['fresh_validation_objective'],
          '| objective relative change:', row['fresh_objective_relative_change'])
print('Aligned target index:', index_path, '| TRAIN:', len(index['train']), '| VALIDATION:', len(index['validation']))
print('Saved decision:', decision_path)
