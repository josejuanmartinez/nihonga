"""Recover hash-identical published sources and embedded workers from a local archive."""
import hashlib
import json
from pathlib import Path
import shutil
import argparse

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / '.archive/nihonga-phase3-artifacts'


def digest(value):
    return hashlib.sha256(value).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site', type=Path)
    args = parser.parse_args()
    published = json.loads((ROOT / 'data/published/2026-10-04/results.json').read_bytes())
    wanted = {}
    for name, checksum in published['source_sha256'].items():
        if not (ROOT / name).exists():
            wanted.setdefault(checksum, []).append(name)
    provenance = []

    def offer(content, origin):
        variants = {content}
        if b'\x00' not in content:
            unix = content.replace(b'\r\n', b'\n')
            for ending in (unix.rstrip(b'\n'), unix.rstrip(b'\n') + b'\n'):
                variants.update((ending, ending.replace(b'\n', b'\r\n')))
        for variant in variants:
            for name in wanted.get(digest(variant), []):
                target = ROOT / name
                if not target.resolve().is_relative_to(ROOT):
                    raise ValueError(name)
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    target.write_bytes(variant)
                    provenance.append(dict(path=name, sha256=digest(variant), recovered_from=origin))

    def walk(value, origin):
        if isinstance(value, dict):
            for key, item in value.items():
                if key.endswith('_source') and isinstance(item, str) and '\n' in item:
                    content = item.encode('utf-8')
                    target = ROOT / 'data/recovered/sources' / (digest(content) + '.py')
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(content)
                    offer(content, origin + ':' + key)
                elif isinstance(item, (dict, list)):
                    walk(item, origin + ':' + key)
        if isinstance(value, (dict, list)):
            for ascii_only in (True, False):
                offer(json.dumps(value, indent=2, ensure_ascii=ascii_only).encode(), origin)

    # The publisher retained some full protocol/review dictionaries inline.
    # Recover them only when their serialized bytes match an original source hash.
    walk(published, 'published-snapshot:embedded-record')
    if args.site:
        for source in (args.site / 'images').rglob('*'):
            if source.is_file():
                offer(source.read_bytes(), 'published-site/images/' + source.name)
    for source in ARCHIVE.rglob('*'):
        if source.suffix in ('.png', '.jpg', '.jpeg', '.webp'):
            offer(source.read_bytes(), str(source.relative_to(ARCHIVE)))
        if source.suffix not in ('.json', '.jsonl', '.py', '.ipynb', '.txt'):
            continue
        relative = source.relative_to(ARCHIVE)
        if relative.name.endswith('ledger.json'):
            continue
        content = source.read_bytes()
        target = ROOT / 'data/recovered/modal' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        offer(content, str(relative))
        if source.suffix == '.json':
            value = json.loads(content)
            walk(value, str(relative))
    target = ROOT / 'data/recovery/restored-sources.json'
    previous = json.loads(target.read_bytes()) if target.exists() else []
    by_path = {item['path']: item for item in previous + provenance}
    target.write_text(json.dumps(list(by_path.values()), indent=2) + '\n')
    print(f'Restored {len(provenance)} hash-identical published source files.')


if __name__ == '__main__':
    main()
