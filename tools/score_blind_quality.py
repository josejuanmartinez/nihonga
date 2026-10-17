"""Unblind committed, locked visual scores and apply the preregistered gates."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scores',type=Path,default=Path('data/phase7/blind_quality_v1/scores.json'))
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    source=args.scores.resolve()
    if not source.is_relative_to(root):raise ValueError('Scores must be in the repository')
    relative=source.relative_to(root).as_posix()
    raw=source.read_bytes()
    committed=subprocess.run(['git','show','HEAD:'+relative],cwd=root,capture_output=True,check=True).stdout
    if raw!=committed:raise RuntimeError('Commit the complete blinded scores before revealing identities')
    public=root/'data/phase7/blind_quality_v1'
    protocol=json.loads((public/'protocol.json').read_bytes())
    prompts=json.loads((public/'prompts.json').read_bytes())
    scores=json.loads(raw)
    expected={(f'Q{i+1:03d}',label) for i in range(len(prompts)) for label in ('L','R')}
    identities=[(r['question_id'],r['label']) for r in scores]
    if len(scores)!=318 or len(set(identities))!=318 or set(identities)!=expected:
        raise RuntimeError('Require exactly 318 unique blinded image scores')
    if any(r['score'] not in ('pass','fail','uncertain') or not r.get('note') for r in scores):
        raise RuntimeError('Every score requires a valid status and an observation')
    for row in scores:
        index=int(row['question_id'][1:])-1
        if prompts[index]['subdimension']=='text_rendering' and not isinstance(row.get('requested_phrase_absent'),bool):
            raise RuntimeError('Text reviews must record whether any requested phrase is entirely absent')
    lock=dict(scores_sha256=hashlib.sha256(raw).hexdigest(),images=len(scores),
              commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip())
    lock_path=public/'score_lock.json'
    if lock_path.exists():
        if json.loads(lock_path.read_bytes())['scores_sha256']!=lock['scores_sha256']:
            raise RuntimeError('Scores changed after locking; do not revise after unblinding')
    else:lock_path.write_text(json.dumps(lock,indent=2)+'\n')
    key_path=root/'.archive/blind_quality_v1/private_labels.json'
    commitment=json.loads((public/'blind_key_commitment.json').read_bytes())['sha256']
    if hashlib.sha256(key_path.read_bytes()).hexdigest()!=commitment:
        raise RuntimeError('Blind key commitment differs')
    labels=json.loads(key_path.read_bytes())
    by_id={(r['question_id'],r['label']):r for r in scores}
    pairs=[]
    for index,prompt in enumerate(prompts):
        qid=f'Q{index+1:03d}'
        models={labels[prompt['source_id']][label]:by_id[(qid,label)] for label in ('L','R')}
        a=models['A_healed_u60'];teacher=models['teacher']
        pairs.append(dict(question_id=qid,source_id=prompt['source_id'],group=prompt['subdimension'],
                          teacher=teacher['score'],candidate_A=a['score'],
                          clear_pass_delta=int(a['score']=='pass')-int(teacher['score']=='pass'),
                          missing_phrase_loss=prompt['subdimension']=='text_rendering' and
                            teacher['score']=='pass' and a['score']!='pass' and a.get('requested_phrase_absent',False)))
    adherence=[r for r in pairs if r['group']!='text_rendering']
    text=[r for r in pairs if r['group']=='text_rendering']
    if len(adherence)!=129 or len(text)!=30:raise RuntimeError('Panel counts differ')
    rng=random.Random(protocol['bootstrap_seed'])
    deltas=[r['clear_pass_delta'] for r in adherence]
    bootstrap=sorted(sum(rng.choices(deltas,k=129))/129 for _ in range(protocol['bootstrap_replicates']))
    ci=[bootstrap[499],bootstrap[19499]]
    counts={panel:{model:dict(Counter(r[model] for r in rows)) for model in ('teacher','candidate_A')}
            for panel,rows in (('adherence',adherence),('text',text))}
    adherence_pass=ci[0]>=-.05
    text_pass=(sum(r['candidate_A']=='pass' for r in text)>=sum(r['teacher']=='pass' for r in text)-1
               and not any(r['missing_phrase_loss'] for r in text))
    benchmark=None
    for path in (root/'data/phase7/latency_benchmark').glob('*/summary.json'):
        report=json.loads(path.read_bytes())
        if report.get('status')=='passed':benchmark=report
    speed_pass=None if benchmark is None else benchmark['paired_speedup']['bootstrap_95pct'][0]>1
    report=dict(status='scored',protocol_sha256=hashlib.sha256((public/'protocol.json').read_bytes()).hexdigest(),
                scores_sha256=lock['scores_sha256'],blind_key_sha256=commitment,counts=counts,
                adherence_gain_fraction=sum(deltas)/129,paired_bootstrap_95pct=ci,
                adherence_noninferiority_passed=adherence_pass,text_gate_passed=text_pass,
                performance_gate_passed=speed_pass,research_replacement_gate_passed=adherence_pass and text_pass and speed_pass is True,
                historical_velocity_gate_passed=False,production_promoted=False,pairs=pairs,
                limitations=['One model-blind AI reviewer; one saved seed per prompt.',
                             'No independent external image-quality benchmark; no production promotion.'])
    (public/'review_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    shutil.copyfile(key_path,public/'revealed_labels.json')
    print(json.dumps({k:v for k,v in report.items() if k not in ('pairs',)},indent=2))


if __name__=='__main__':main()
