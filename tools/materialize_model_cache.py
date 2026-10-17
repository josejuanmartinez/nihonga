"""Rebuild the pinned model's cache aliases locally, using verified downloaded blobs."""
import hashlib
import json
import os
from pathlib import Path


def main():
    root=Path(__file__).resolve().parents[1]
    base=root/'.archive/nihonga-hf-cache'
    repo=Path('hub/models--Qwen--Qwen-Image-2.1')
    revision='790c92633540aa0cb11d9abf19eb46d861714758'
    tree=json.loads((base/repo/'trees'/f'{revision}.json').read_bytes())
    inventory=json.loads((root/'data/recovery/nihonga-hf-cache-inventory.json').read_bytes())
    present={entry['path'] for entry in inventory if entry['type']==1}
    aliases=[]
    blobs=[]
    for filename,spec in tree['files'].items():
        if 'xet_hash' in spec:
            canonical=Path('hub/blobs')/spec['xet_hash'][:2]/spec['xet_hash']
            if canonical.as_posix() not in present:continue
            repo_blob=repo/'blobs'/spec['lfs_sha256']
            checksum=spec['lfs_sha256']
            aliases.append(dict(path=repo_blob.as_posix(),target=canonical.as_posix()))
        else:
            canonical=repo/'blobs'/spec['blob_id']
            if canonical.as_posix() not in present:continue
            repo_blob=canonical
            checksum=None
        blobs.append(dict(path=canonical.as_posix(),size=spec['size'],expected_sha256=checksum,
                          expected_git_blob_sha1=None if checksum else spec['blob_id']))
        aliases.append(dict(path=(repo/'snapshots'/revision/filename).as_posix(),target=canonical.as_posix()))
    missing=[]
    checksums={}
    for entry in blobs:
        source=base/entry['path']
        if not source.exists():missing.append(entry['path']);continue
        if source.stat().st_size!=entry['size']:raise RuntimeError('Model blob size mismatch')
        sha256=hashlib.sha256()
        git_hash=hashlib.sha1(f"blob {entry['size']}\0".encode())
        with source.open('rb') as stream:
            for chunk in iter(lambda:stream.read(8*1024*1024),b''):
                sha256.update(chunk);git_hash.update(chunk)
        actual=sha256.hexdigest()
        if entry['expected_sha256'] and actual!=entry['expected_sha256']:
            raise RuntimeError('Model LFS SHA-256 mismatch')
        if entry['expected_git_blob_sha1'] and git_hash.hexdigest()!=entry['expected_git_blob_sha1']:
            raise RuntimeError('Model Git blob SHA-1 mismatch')
        entry['verified_sha256']=actual
        checksums[entry['path']]=actual
    for alias in aliases:
        source=base/alias['target'];target=base/alias['path']
        if alias['target'] not in checksums:continue
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():os.link(source,target)
    report=dict(status='verified' if not missing else 'incomplete',revision=revision,blobs=blobs,
                aliases=aliases,missing=missing,local_cache='.archive/nihonga-hf-cache')
    (root/'data/recovery/model-cache-verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'{len(blobs)-len(missing)}/{len(blobs)} model blobs verified; {len(missing)} still downloading.')


if __name__=='__main__':main()
