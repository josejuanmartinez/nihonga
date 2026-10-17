"""Verify selected continuation tensors against hashes recovered from original requests."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEY = '95fcd2cfae35cb7f124b8af2865c30f928cbd681a0d015df5e56e8ae5ba4fd78'
ARCHIVE = ROOT / '.archive/nihonga-phase3-artifacts'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    run = ROOT / 'data/recovered/modal/phase7/blocks45_healing' / KEY
    request = json.loads((run / 'request.json').read_bytes())['request']
    summary = json.loads((run / 'summary.json').read_bytes())
    expected = {f"phase5/capture/{r['request_key']}/targets.pt":r['tensor_sha256']
                for r in request['records']+request['expanded_validation']['records']}
    for split in ('train','validation'):
        for r in request['aligned_target_index'][split]:
            expected[request['aligned_target_index']['remote_directory'].strip('/')+'/'+Path(r['tensor_path']).name] = r['tensor_sha256']
    for artifact in summary['artifacts']:
        if artifact['name'] in ('checkpoint_step_060.pt','validation_step_060.pt'):
            expected[f"phase7/blocks45_healing/{KEY}/{artifact['name']}"] = artifact['sha256']
    region = request['region_bridge']
    expected[f"phase7/region_bridge_fit/{region['fit_key']}/{region['checkpoint']}"] = region['sha256']
    rows = []
    for name, checksum in expected.items():
        path = ARCHIVE / name
        actual = digest(path) if path.is_file() else None
        rows.append(dict(path=name, expected_sha256=checksum, actual_sha256=actual,
                         status='missing' if actual is None else 'verified' if actual==checksum else 'FAILED'))
    counts = {s:sum(r['status']==s for r in rows) for s in ('verified','missing','FAILED')}
    report = dict(counts=counts, artifacts=rows, local_root='.archive/nihonga-phase3-artifacts',
                  status='verified' if counts['verified']==len(rows) else 'incomplete')
    (ROOT / 'data/recovery/essential-tensor-verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(counts)
    if counts['FAILED']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
