"""Fetch available quality pairs and build model-blind review sheets.

The per-prompt randomized identity key stays in ignored .archive until scores
are locked. Sheets contain only prompt IDs, questions and neutral image labels.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import textwrap


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-account',action='store_true')
    args=parser.parse_args()
    from modal_accounts import set_modal_account
    set_modal_account('old' if args.old_account else 'current', purpose='download')
    import modal
    from PIL import Image,ImageDraw,ImageFont
    root=Path(__file__).resolve().parents[1]
    public=root/'data/phase7/blind_quality_v1'
    private=root/'.archive/blind_quality_v1'
    private.mkdir(parents=True,exist_ok=True)
    prompts=json.loads((public/'prompts.json').read_bytes())
    key_path=private/'private_labels.json'
    if not key_path.exists():
        labels={}
        for row in prompts:
            variants=['teacher','A_healed_u60']
            random.SystemRandom().shuffle(variants)
            labels[row['source_id']]=dict(zip(('L','R'),variants))
        key_path.write_text(json.dumps(labels,indent=2)+'\n')
    labels=json.loads(key_path.read_bytes())
    (public/'blind_key_commitment.json').write_text(json.dumps(dict(
        sha256=hashlib.sha256(key_path.read_bytes()).hexdigest(),
        policy='Reveal this identity key only after all visual scores are locked.'),indent=2)+'\n')
    volume=modal.Volume.from_name('nihonga-phase3-artifacts')
    calls=list((root/'data/phase7/blind_quality').glob('*/call.json'))
    if len(calls)!=1:raise RuntimeError('Expected exactly one registered quality run')
    request_key=json.loads(calls[0].read_bytes())['request_key']
    remote='/phase7/blind_quality/'+request_key
    for filename in ('summary.json','progress.json'):
        try:report=json.loads(b''.join(volume.read_file(remote+'/'+filename)))
        except FileNotFoundError:continue
        break
    else:raise RuntimeError('Image job has not reported progress yet')
    available=min(report['calls'].get('teacher',0),report['calls'].get('A_healed_u60',0))
    font_path='C:/Windows/Fonts/arial.ttf'
    for candidate in ('C:/Windows/Fonts/msyh.ttc','C:/Windows/Fonts/simsun.ttc'):
        if Path(candidate).exists():font_path=candidate;break
    font=ImageFont.truetype(font_path,19)
    title_font=ImageFont.truetype(font_path,24)
    sheet_dir=public/'review_sheets'
    sheet_dir.mkdir(exist_ok=True)
    manifest=[]
    for index,row in enumerate(prompts[:available]):
        paths={}
        for label,variant in labels[row['source_id']].items():
            name=row['source_id']+'-'+variant+'.png'
            path=private/name
            if not path.exists():
                temporary=path.with_suffix('.partial')
                with temporary.open('wb') as stream:volume._read_file_into_fileobj(remote+'/'+name,stream,concurrency=2)
                temporary.replace(path)
            if report.get('status')=='passed':
                artifact=next(a for a in report['artifacts'] if a['name']==name)
                if hashlib.sha256(path.read_bytes()).hexdigest()!=artifact['sha256']:
                    raise RuntimeError('Quality image hash mismatch')
            paths[label]=path
        sheet=Image.new('RGB',(2048,1280),'white')
        draw=ImageDraw.Draw(sheet)
        draw.text((12,6),f"Q{index+1:03d} | {row['source_id']} | {row['subdimension']}",font=title_font,fill='black')
        prompt_lines=textwrap.wrap(row['prompt'],width=150)
        for line_number,line in enumerate(prompt_lines[:6]):draw.text((12,43+line_number*25),line,font=font,fill='black')
        draw.text((12,200),row['question'],font=font,fill='black')
        for x,label in ((0,'L'),(1024,'R')):
            draw.text((x+12,235),label,font=title_font,fill='black')
            with Image.open(paths[label]) as image:sheet.paste(image.convert('RGB'),(x,256))
        target=sheet_dir/f'Q{index+1:03d}.jpg'
        if not target.exists():sheet.save(target,quality=95,subsampling=0)
        manifest.append(dict(question_id=f'Q{index+1:03d}',source_id=row['source_id'],
                             prompt=row['prompt'],question=row['question'],group=row['subdimension'],
                             sheet=str(target.relative_to(root)).replace('\\','/'),
                             sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    (public/'review_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    contact_dir=public/'review_contacts'
    contact_dir.mkdir(exist_ok=True)
    for offset in range(0,len(manifest)-2,3):
        target=contact_dir/f'panel_{offset//3+1:02d}.jpg'
        if target.exists():continue
        canvas=Image.new('RGB',(1024,1680),'white')
        draw=ImageDraw.Draw(canvas)
        for position,entry in enumerate(manifest[offset:offset+3]):
            y=position*560
            draw.text((10,y+5),entry['question_id']+'  L | R  '+entry['group'],font=title_font,fill='black')
            with Image.open(root/entry['sheet']) as image:
                crop=image.crop((0,256,2048,1280)).resize((1024,512),Image.Resampling.LANCZOS)
                canvas.paste(crop,(0,y+40))
        canvas.save(target,quality=95,subsampling=0)
    print(f'{available}/{len(prompts)} blinded pairs available; {len(manifest)} sheets ready.')


if __name__=='__main__':main()
