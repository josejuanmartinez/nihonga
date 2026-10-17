"""Finish verified local-to-new-account transfers as archive downloads complete.

Does not access old credentials and never schedules GPU work.
"""
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    root=Path(__file__).resolve().parents[1]
    def run(tool,*args):
        print('Running',tool,*args,flush=True)
        subprocess.run([sys.executable,'-u',str(root/'tools'/tool),*args],cwd=root,check=True)
    def uploaded(volume):
        path=root/'data/recovery'/('new-account-'+volume+'-upload.json')
        return json.loads(path.read_bytes())['files'] if path.exists() else {}
    candidate='phase7/blocks45_healing/95fcd2cfae35cb7f124b8af2865c30f928cbd681a0d015df5e56e8ae5ba4fd78/checkpoint_step_060.pt'
    previous=None
    while True:
        cache_report=json.loads((root/'data/recovery/model-cache-verification.json').read_bytes())
        if any((root/'.archive/nihonga-hf-cache'/name).exists() for name in cache_report['missing']):
            run('materialize_model_cache.py')
            cache_report=json.loads((root/'data/recovery/model-cache-verification.json').read_bytes())
        cache_uploaded=uploaded('nihonga-hf-cache')
        if any(blob.get('verified_sha256') and cache_uploaded.get(blob['path'],{}).get('sha256')!=blob['verified_sha256']
               for blob in cache_report['blobs']):
            run('upload_modal_archive.py','--volume','nihonga-hf-cache')
            cache_uploaded=uploaded('nihonga-hf-cache')
        tensor_report=json.loads((root/'data/recovery/essential-tensor-verification.json').read_bytes())
        if tensor_report['status']!='verified' and (root/'.archive/nihonga-phase3-artifacts'/candidate).is_file():
            run('verify_recovered_tensors.py')
            tensor_report=json.loads((root/'data/recovery/essential-tensor-verification.json').read_bytes())
        candidate_ready=tensor_report['status']=='verified'
        candidate_ledger=root/'.archive/nihonga-phase3-artifacts/candidate-A-ledger.json'
        if candidate_ready and candidate_ledger.exists() and candidate not in uploaded('nihonga-phase3-artifacts'):
            run('upload_modal_archive.py')
        status=(len(cache_report['missing']),candidate_ready,candidate in uploaded('nihonga-phase3-artifacts'))
        if status!=previous:
            print('Migration state:',dict(model_blobs_pending=status[0],candidate_verified_locally=status[1],candidate_uploaded=status[2]),flush=True)
            previous=status
        if cache_report['status']=='verified' and candidate_ready and status[2]:
            # Capture final download-ledger metadata as well as all verified canonical blobs.
            run('upload_modal_archive.py','--volume','nihonga-hf-cache')
            run('verify_new_account.py')
            print('NEW ACCOUNT FILE VERIFICATION COMPLETE. Offline replay and experiments can run solely in the new account.',flush=True)
            return
        time.sleep(15)


if __name__=='__main__':main()
