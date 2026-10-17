"""Upload locally verified archive files to the NEW Modal account only."""
import argparse
import hashlib
import json
from pathlib import Path

from modal_accounts import set_modal_account


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--volume',default='nihonga-phase3-artifacts')
    parser.add_argument('--batch-files',type=int,default=100)
    args=parser.parse_args()
    set_modal_account('current',purpose='upload')
    import modal
    root=Path(__file__).resolve().parents[1]
    archive=root/'.archive'/args.volume
    entries={}
    for path in archive.glob('*ledger.json'):
        for name,record in json.loads(path.read_bytes()).items():
            if name in entries and entries[name]!=record:raise RuntimeError('Conflicting local ledger entries')
            entries[name]=record
    if not entries:raise RuntimeError('No verified local download ledgers')
    progress_path=root/'data/recovery'/('new-account-'+args.volume+'-upload.json')
    progress=json.loads(progress_path.read_bytes()) if progress_path.exists() else dict(account='new',volume=args.volume,files={})
    volume=modal.Volume.from_name(args.volume)
    pending=[(name,record) for name,record in entries.items() if progress['files'].get(name)!=record]
    pending.sort(key=lambda item:item[1]['size'])
    print('NEW account upload:',args.volume,len(pending),'files',round(sum(r['size'] for _,r in pending)/2**30,3),'GiB',flush=True)
    for start in range(0,len(pending),args.batch_files):
        batch=pending[start:start+args.batch_files]
        for name,record in batch:
            path=archive/name
            if not path.resolve().is_relative_to(archive.resolve()):raise ValueError(name)
            if path.stat().st_size!=record['size'] or digest(path)!=record['sha256']:
                raise RuntimeError('Local file changed before upload: '+name)
        with volume.batch_upload(force=True) as upload:
            for name,_ in batch:upload.put_file(archive/name,'/'+name)
        progress['files'].update(batch)
        temporary=progress_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(progress,indent=2)+'\n')
        temporary.replace(progress_path)
        print(f'Uploaded {min(start+len(batch),len(pending))}/{len(pending)} files to NEW account',flush=True)
    print('All currently complete local files uploaded.',flush=True)


if __name__=='__main__':main()
