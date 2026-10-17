"""Preserve a published snapshot and audit its source hashes without model calls."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    destination = root / 'data/published/2026-10-04'
    destination.mkdir(parents=True, exist_ok=True)
    source = args.site / 'results.json'
    target = destination / 'results.json'
    if target.exists() and sha256(target) != sha256(source):
        raise RuntimeError('Snapshot differs; use a separately versioned destination.')
    shutil.copyfile(source, target)
    published = json.loads(target.read_text(encoding='utf-8'))
    records = []
    for name, expected in published['source_sha256'].items():
        path = root / name
        if not path.resolve().is_relative_to(root):
            raise ValueError('Source path escapes repository')
        actual = sha256(path) if path.is_file() else None
        records.append(dict(path=name, expected_sha256=expected, actual_sha256=actual,
                            status='missing' if actual is None else
                            'match' if actual == expected else 'different'))
    report = {
        'status': 'recovered_available_sources_with_gaps' if any(r['status']=='missing' for r in records) else 'all_published_sources_verified',
        'snapshot_sha256': sha256(target),
        'snapshot_updated': published['updated'],
        'source_repository': 'https://github.com/Neopolita/niwaki-page',
        'source_commit': '57ce7e4',
        'counts': {status: sum(r['status'] == status for r in records)
                   for status in ('match', 'missing', 'different')},
        'artifacts': records,
        'limitations': ['Only hash-identical source bytes count as recovered; missing artifacts remain explicit.',
                        'This reconciliation performs no model computation or promotion.'],
    }
    (destination / 'reconciliation.json').write_text(
        json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report['counts']))


if __name__ == '__main__':
    main()
