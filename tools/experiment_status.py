"""Fetch durable experiment progress without scheduling model work."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-account',action='store_true')
    args=parser.parse_args()
    if args.old_account:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER,'Environment') as key:
            for name in ('MODAL_TOKEN_ID','MODAL_TOKEN_SECRET'):
                os.environ[name]=winreg.QueryValueEx(key,name+'_OLD')[0]
    import modal
    root=Path(__file__).resolve().parents[1]
    volume=modal.Volume.from_name('nihonga-phase3-artifacts')
    for experiment in ('conditional_importance','latency_benchmark'):
        for call_path in (root/'data/phase7'/experiment).glob('*/call.json'):
            call=json.loads(call_path.read_bytes())
            for filename in ('summary.json','progress.json','started.json'):
                try:
                    raw=b''.join(volume.read_file(f"/phase7/{experiment}/{call['request_key']}/{filename}"))
                except FileNotFoundError:
                    continue
                report=json.loads(raw)
                if report['request_key']!=call['request_key']:
                    raise RuntimeError('Unexpected report identity')
                (call_path.parent/filename).write_bytes(raw)
                status=report.get('status')
                if filename!='summary.json' and not report.get('error'):
                    status='in_progress'
                print(json.dumps(dict(experiment=experiment,file=filename,status=status,
                    teacher_controls=len(report.get('controls',[])),blocks=len(report.get('blocks',[])),
                    timed_images=len(report.get('rows',[])),error=report.get('error'))))
                break
            else:
                print(experiment+': waiting for container start')


if __name__=='__main__':main()
