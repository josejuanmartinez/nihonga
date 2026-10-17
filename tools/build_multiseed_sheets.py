"""Build per-prompt seed grids (rows: teacher, B, C; columns: 4 seeds) for the multi-seed text check (no GPU work)."""
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
    volume = modal.Volume.from_name('nihonga-phase3-artifacts')
    run_dir = next((root / 'data/phase7/multiseed_text').glob('*/request.json')).parent
    request = json.loads((run_dir / 'request.json').read_bytes())
    key, c_name = request['request_key'], request['request']['c_variant']['name']
    remote = f'/phase7/multiseed_text/{key}'
    summary = json.loads(b''.join(volume.read_file(remote + '/summary.json')))
    if summary.get('status') != 'passed':
        raise RuntimeError('Multi-seed run did not pass.')
    (run_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    expected = {a['name']: a['sha256'] for a in summary['artifacts']}
    store = root / '.archive/multiseed_text' / key
    store.mkdir(parents=True, exist_ok=True)
    for name, sha in expected.items():
        path = store / name
        if not path.exists():
            path.write_bytes(b''.join(volume.read_file(f'{remote}/{name}')))
        if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            raise RuntimeError('Image hash mismatch: ' + name)
    # Seed 0: teacher and B from the first image run, C from the variant image run.
    first = json.loads(next((root / 'data/phase7/bridge_images').glob('*/summary.json')).read_bytes())
    first_store = root / '.archive/bridge_images' / first['request_key']
    variant_key = next(json.loads(p.read_bytes())['request_key'] for p in (root / 'data/phase7/bridge_variant_images').glob('*/request.json')
                       if any(v['name'] == c_name for v in json.loads(p.read_bytes())['request']['variants']))
    variant_store = root / '.archive/bridge_variant_images' / variant_key
    face = next((f for f in ('C:/Windows/Fonts/msyh.ttc', 'C:/Windows/Fonts/simsun.ttc') if Path(f).exists()), 'C:/Windows/Fonts/arial.ttf')
    font, big = ImageFont.truetype(face, 20), ImageFont.truetype(face, 24)
    sheets = run_dir / 'sheets'
    sheets.mkdir(exist_ok=True)
    tile, label_width = 420, 230
    base_rows = [r for r in first['rows'] if r['subdimension'] in ('text_rendering', 'posters')]
    for row in base_rows:
        sid = row['source_id']
        canvas = Image.new('RGB', (label_width + 4 * tile, 110 + 3 * tile), 'white')
        draw = ImageDraw.Draw(canvas)
        for line, text in enumerate(textwrap.wrap(row['prompt'], 200 if row['prompt'].isascii() else 100)[:2]):
            draw.text((10, 8 + 26 * line), text, font=font, fill='black')
        for column in range(4):
            draw.text((label_width + column * tile + 10, 76), f'seed {column}', font=big, fill='black')
        models = [('teacher', 'Teacher'), ('B', 'B'), (c_name, 'C (best bridge)')]
        for line, (model, label) in enumerate(models):
            y = 110 + line * tile
            draw.text((10, y + tile // 2 - 12), label, font=big, fill='black')
            for column in range(4):
                if column == 0:
                    path = (first_store / row[model]['name']) if model in ('teacher', 'B') else variant_store / f'{sid}-{model}.png'
                else:
                    path = store / f'{sid}-s{column}-{model}.png'
                with Image.open(path) as image:
                    canvas.paste(image.convert('RGB').resize((tile, tile), Image.Resampling.LANCZOS), (label_width + column * tile, y))
        canvas.save(sheets / f'{sid}.jpg', quality=88)
    print(json.dumps(dict(sheets=len(base_rows), folder=str(sheets))))


if __name__ == '__main__':
    main()
