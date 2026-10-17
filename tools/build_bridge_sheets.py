"""Download the teacher/B/C images and build labeled comparison sheets (no GPU work)."""
import hashlib
import json
import textwrap
from pathlib import Path


def main():
    from modal_accounts import set_modal_account
    set_modal_account('current', purpose='download')
    import modal
    from PIL import Image, ImageDraw, ImageFont
    root = Path(__file__).resolve().parents[1]
    run = next((root / 'data/phase7/bridge_images').glob('*/summary.json'))
    report = json.loads(run.read_bytes())
    if report.get('status') != 'passed':
        raise RuntimeError('Image run did not pass.')
    key = report['request_key']
    store = root / '.archive/bridge_images' / key
    store.mkdir(parents=True, exist_ok=True)
    volume = modal.Volume.from_name('nihonga-phase3-artifacts')
    for artifact in report['artifacts']:
        path = store / artifact['name']
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != artifact['sha256']:
            path.write_bytes(b''.join(volume.read_file(f'/phase7/bridge_images/{key}/{artifact["name"]}')))
        if hashlib.sha256(path.read_bytes()).hexdigest() != artifact['sha256']:
            raise RuntimeError('Image hash mismatch: ' + artifact['name'])
    # Determinism check against the earlier Phase 7 teacher renders of the same prompts and seeds.
    earlier = json.loads(next((root / 'data/recovered/modal/phase7/candidate_images').glob('*/summary_blocks45.json')).read_bytes())
    earlier_hashes = {a['name']: a['sha256'] for a in earlier.get('artifacts', [])}
    matches = []
    for row in report['rows']:
        old = next((h for n, h in earlier_hashes.items() if n.startswith(row['source_id']) and 'teacher' in n), None)
        matches.append(None if old is None else old == row['teacher']['sha256'])
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 20)
    big = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 26)
    sheets = run.parent / 'sheets'
    sheets.mkdir(exist_ok=True)
    tile = 640
    for index, row in enumerate(report['rows']):
        canvas = Image.new('RGB', (3 * tile, tile + 150), 'white')
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 6), f"P{index + 1:02d} | {row['source_id']} | {row.get('subdimension', '')}", font=big, fill='black')
        for line, text in enumerate(textwrap.wrap(row['prompt'], 170)[:3]):
            draw.text((10, 42 + 24 * line), text, font=font, fill='black')
        for column, variant in enumerate(('teacher', 'B', 'C')):
            label = {'teacher': 'Teacher (32 blocks)', 'B': 'B: block 5 bridged (accepted)', 'C': 'C: blocks 5 + 13 bridged'}[variant]
            draw.text((column * tile + 10, 118), label, font=big, fill='black')
            with Image.open(store / row[variant]['name']) as image:
                canvas.paste(image.convert('RGB').resize((tile, tile), Image.Resampling.LANCZOS), (column * tile, 150))
        canvas.save(sheets / f'P{index + 1:02d}.jpg', quality=90)
    print(json.dumps(dict(sheets=len(report['rows']), teacher_matches_earlier=sum(m is True for m in matches),
                          teacher_differs=sum(m is False for m in matches), teacher_unmatched=sum(m is None for m in matches))))


if __name__ == '__main__':
    main()
