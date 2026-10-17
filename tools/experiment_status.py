"""Fetch durable experiment progress without scheduling model work."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-account',action='store_true')
    args=parser.parse_args()
    from modal_accounts import set_modal_account
    set_modal_account('old' if args.old_account else 'current', purpose='download')
    import modal
    root=Path(__file__).resolve().parents[1]
    volume=modal.Volume.from_name('nihonga-phase3-artifacts')
    # Local folder -> remote experiment folder; candidate B's scan shares the remote scan folder.
    for local,experiment in (('conditional_importance','conditional_importance'),('conditional_importance_b','conditional_importance'),
                             ('latency_benchmark','latency_benchmark'),('blind_quality','blind_quality'),
                             ('block13_bridge','bridge_candidate')):
        for call_path in (root/'data/phase7'/local).glob('*/call.json'):
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
                print(json.dumps(dict(experiment=local,file=filename,status=status,
                    teacher_controls=len(report.get('controls',[])),blocks=len(report.get('blocks',[])),
                    timed_images=len(report.get('rows',[])),calls=report.get('calls'),error=report.get('error'))))
                break
            else:
                print(local+': waiting for container start')


if __name__=='__main__':main()
