"""Download a Modal volume inventory with atomic writes and a local hash ledger.

Credentials are supplied by the caller's environment; no credentials are saved.
Existing ledger-verified files are reused. Large files require --max-file-gib.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path

import modal


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--volume', default='nihonga-phase3-artifacts')
    parser.add_argument('--destination', type=Path, default=Path('.archive'))
    parser.add_argument('--max-file-gib', type=float, default=0.02)
    parser.add_argument('--prefix', default='')
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--metadata-only', action='store_true')
    parser.add_argument('--ledger', default='download-ledger.json')
    parser.add_argument('--paths', type=Path, help='JSON array of exact volume-relative paths')
    parser.add_argument('--old-account', action='store_true', help='Read the old archive account; never schedules computation')
    args = parser.parse_args()
    from modal_accounts import set_modal_account
    set_modal_account('old' if args.old_account else 'current', purpose='download')
    inventory_path = Path('data/recovery') / (args.volume + '-inventory.json')
    inventory = json.loads(inventory_path.read_text())
    destination = args.destination.resolve() / args.volume
    destination.mkdir(parents=True, exist_ok=True)
    ledger_path = destination / args.ledger
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    volume = modal.Volume.from_name(args.volume)
    entries = [e for e in inventory if e['type'] == 1 and
               e['size'] <= args.max_file_gib * 2**30 and e['path'].startswith(args.prefix)]
    if args.metadata_only:
        entries = [e for e in entries if e['path'].endswith(('.json', '.jsonl', '.py', '.ipynb', '.txt'))]
    if args.paths:
        selected = set(json.loads(args.paths.read_bytes()))
        entries = [e for e in entries if e['path'] in selected]
        if selected != {e['path'] for e in entries}:
            raise ValueError('Selected paths absent from inventory or above size cap')
    entries.sort(key=lambda e: (not e['path'].endswith('.json'), e['size']))

    def fetch(entry):
        name = entry['path']
        target = destination / name
        if not target.resolve().is_relative_to(destination):
            raise ValueError('Archive path escapes destination')
        previous = ledger.get(name)
        if target.exists() and previous and target.stat().st_size == entry['size']:
            if digest(target) == previous['sha256']:
                return name, previous
            raise RuntimeError('Previously downloaded artifact changed: ' + name)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + '.partial')
        with temporary.open('wb') as stream:
            # Bound SDK block concurrency rather than multiplying it by CPU count
            # for each file in this thread pool (SDK 1.6.2).
            volume._read_file_into_fileobj(name, stream, concurrency=2)
        if temporary.stat().st_size != entry['size']:
            raise RuntimeError('Size mismatch: ' + name)
        checksum = digest(temporary)
        temporary.replace(target)
        return name, dict(size=entry['size'], sha256=checksum)

    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch, e): e['path'] for e in entries}
        for index, future in enumerate(as_completed(futures), 1):
            try:
                name, record = future.result()
                ledger[name] = record
            except Exception as exc:
                failures.append(dict(path=futures[future], error=str(exc)))
            if index % 25 == 0 or index == len(entries):
                temporary = ledger_path.with_suffix('.tmp')
                temporary.write_text(json.dumps(ledger, indent=2) + '\n')
                temporary.replace(ledger_path)
                print(f'{index}/{len(entries)} processed; {len(failures)} failures', flush=True)
    if failures:
        print(json.dumps(failures, indent=2))
        raise SystemExit(1)


if __name__ == '__main__':
    main()
