"""Verify migrated files and rebuild cache symlinks in the NEW account, CPU only."""
import hashlib
import json
from pathlib import Path
import sys

from modal_accounts import set_modal_account


def verify_migration(manifest,cache_layout,key):
    import hashlib
    import json
    import os
    from pathlib import Path
    roots={'nihonga-phase3-artifacts':Path('/research'),'nihonga-hf-cache':Path('/hf')}
    report=dict(status='verifying',account='new',gpu_requested=False,files_verified=0,bytes_verified=0,failures=[])
    for volume,files in manifest.items():
        base=roots[volume]
        for name,expected in files.items():
            path=base/name
            if not path.resolve().is_relative_to(base):raise ValueError('Unsafe archive path')
            if not path.is_file():
                report['failures'].append(dict(volume=volume,path=name,error='missing'));continue
            h=hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
            if path.stat().st_size!=expected['size'] or h.hexdigest()!=expected['sha256']:
                report['failures'].append(dict(volume=volume,path=name,error='size or SHA-256 mismatch'));continue
            report['files_verified']+=1
            report['bytes_verified']+=expected['size']
            if report['files_verified']%250==0:print('Verified',report['files_verified'],'files',flush=True)
    if report['failures']:raise RuntimeError(json.dumps(report))
    base=roots['nihonga-hf-cache']
    for alias in cache_layout['aliases']:
        target=base/alias['path'];source=base/alias['target']
        if not source.is_file() or not source.resolve().is_relative_to(base):
            raise RuntimeError('Cache canonical blob missing')
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():target.symlink_to(os.path.relpath(source,target.parent))
        if target.resolve()!=source.resolve():raise RuntimeError('Cache alias resolves incorrectly')
    cache_volume.commit()
    report.update(status='verified',cache_aliases=len(cache_layout['aliases']),
                  continuation_payload_independent_of_old_account=True)
    directory=Path('/research/migration')/key
    directory.mkdir(parents=True,exist_ok=True)
    (directory/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (directory/'verification.json').write_text(json.dumps(report,indent=2))
    results_volume.commit()
    return report


def main():
    sys.stdout.reconfigure(encoding='utf-8');sys.stderr.reconfigure(encoding='utf-8')
    set_modal_account('current',purpose='compute')
    import modal
    root=Path(__file__).resolve().parents[1]
    tensor_report=json.loads((root/'data/recovery/essential-tensor-verification.json').read_bytes())
    layout=json.loads((root/'data/recovery/model-cache-verification.json').read_bytes())
    if tensor_report['status']!='verified' or layout['status']!='verified':
        raise RuntimeError('Local tensor/cache verification is incomplete')
    manifest={}
    for name in ('nihonga-phase3-artifacts','nihonga-hf-cache'):
        files={}
        for path in (root/'.archive'/name).glob('*ledger.json'):
            files.update(json.loads(path.read_bytes()))
        uploaded=json.loads((root/'data/recovery'/('new-account-'+name+'-upload.json')).read_bytes())['files']
        if any(uploaded.get(path)!=record for path,record in files.items()):
            raise RuntimeError('Upload is incomplete for '+name)
        manifest[name]=files
    for artifact in tensor_report['artifacts']:
        record=manifest['nihonga-phase3-artifacts'].get(artifact['path'])
        if record is None or record['sha256']!=artifact['expected_sha256']:
            raise RuntimeError('Required tensor omitted from destination manifest: '+artifact['path'])
    for blob in layout['blobs']:
        record=manifest['nihonga-hf-cache'].get(blob['path'])
        if record is None or record['sha256']!=blob['verified_sha256']:
            raise RuntimeError('Required model blob omitted from destination manifest: '+blob['path'])
    key=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    results=modal.Volume.from_name('nihonga-phase3-artifacts')
    cache=modal.Volume.from_name('nihonga-hf-cache')
    globals()['results_volume']=results
    globals()['cache_volume']=cache
    app=modal.App('nihonga-new-account-migration-verification')
    function=app.function(image=modal.Image.debian_slim(python_version='3.13'),serialized=True,
        cpu=4,memory=4096,timeout=7200,volumes={'/research':results,'/hf':cache})(verify_migration)
    with modal.enable_output(),app.run(detach=True):
        report=function.remote(manifest,layout,key)
    report['manifest_sha256']=key
    (root/'data/recovery/new-account-verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
