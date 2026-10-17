"""Back up completed new-account experiments locally and verify artifact hashes."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

from modal_accounts import set_modal_account


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    set_modal_account('current', purpose='download')
    import modal
    root = Path(__file__).resolve().parents[1]
    volume = modal.Volume.from_name('nihonga-phase3-artifacts')
    for experiment in ('conditional_importance', 'latency_benchmark', 'blind_quality'):
        for call_path in (root / 'data/phase7' / experiment).glob('*/call.json'):
            call = json.loads(call_path.read_bytes())
            if call.get('account') != 'new':
                raise RuntimeError('Only new-account calls can be archived here')
            remote = f"/phase7/{experiment}/{call['request_key']}"
            try:
                raw = b''.join(volume.read_file(remote + '/summary.json'))
            except FileNotFoundError:
                print(experiment, 'not complete yet', flush=True)
                continue
            summary = json.loads(raw)
            if summary['request_key'] != call['request_key']:
                raise RuntimeError('Summary request identity differs')
            (call_path.parent / 'summary.json').write_bytes(raw)
            if summary['status'] != 'passed':
                print(experiment, 'failed; summary preserved', flush=True)
                continue
            artifacts = {a['name']: a for a in summary['artifacts']}
            for row in summary.get('rows', []):
                if 'image' in row:
                    artifacts[row['image']['name']] = row['image']
            destination = root / '.archive/new-experiments' / experiment / call['request_key']
            destination.mkdir(parents=True, exist_ok=True)
            (destination / 'summary.json').write_bytes(raw)

            def download(item):
                name, expected = item
                target = destination / name
                if not target.resolve().is_relative_to(destination.resolve()):
                    raise ValueError('Unsafe artifact path')
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    temporary = target.with_suffix(target.suffix + '.partial')
                    with temporary.open('wb') as stream:
                        volume._read_file_into_fileobj(remote + '/' + name, stream, concurrency=2)
                    if (temporary.stat().st_size != expected['bytes']
                            or digest(temporary) != expected['sha256']):
                        raise RuntimeError('Downloaded artifact hash differs: ' + name)
                    temporary.replace(target)
                elif target.stat().st_size != expected['bytes'] or digest(target) != expected['sha256']:
                    raise RuntimeError('Existing local artifact hash differs: ' + name)
                return name, dict(size=expected['bytes'], sha256=expected['sha256'])

            with ThreadPoolExecutor(max_workers=3) as executor:
                verified = dict(executor.map(download, artifacts.items()))
            report = dict(status='verified', account='new', request_key=call['request_key'],
                          summary_sha256=hashlib.sha256(raw).hexdigest(), files=verified,
                          artifact_count=len(verified), bytes=sum(a['size'] for a in verified.values()))
            (destination / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
            (call_path.parent / 'local-archive-verification.json').write_text(json.dumps(report, indent=2) + '\n')
            print(experiment, len(verified), 'artifacts verified locally', flush=True)


if __name__ == '__main__':
    main()
