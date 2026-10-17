"""Build teacher | B | first C | variant comparison sheets from whatever images exist so far (no GPU work)."""
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
    first = json.loads(next((root / 'data/phase7/bridge_images').glob('*/summary.json')).read_bytes())
    first_store = root / '.archive/bridge_images' / first['request_key']
    run_dir = next((root / 'data/phase7/bridge_variant_images').glob('*/request.json')).parent
    request = json.loads((run_dir / 'request.json').read_bytes())
    key, variants = request['request_key'], [v['name'] for v in request['request']['variants']]
    store = root / '.archive/bridge_variant_images' / key
    store.mkdir(parents=True, exist_ok=True)
    volume = modal.Volume.from_name('nihonga-phase3-artifacts')
    remote = f'/phase7/bridge_variant_images/{key}'
    summary = None
    try:
        summary = json.loads(b''.join(volume.read_file(remote + '/summary.json')))
        (run_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    except FileNotFoundError:
        pass
    expected = {a['name']: a['sha256'] for a in summary['artifacts']} if summary else {}
    face = next((f for f in ('C:/Windows/Fonts/msyh.ttc', 'C:/Windows/Fonts/simsun.ttc') if Path(f).exists()), 'C:/Windows/Fonts/arial.ttf')
    font, big = ImageFont.truetype(face, 20), ImageFont.truetype(face, 24)
    labels = {'teacher': 'Teacher', 'B': 'B (block 13 kept)', 'C': 'First C (plain r256)',
              'plain_r1024': 'plain r1024', 'cond_r256': 'cond r256', 'cond_r1024': 'cond r1024'}
    sheets = run_dir / 'sheets'
    sheets.mkdir(exist_ok=True)
    tile, columns = 480, ['teacher', 'B', 'C'] + variants
    ready = 0
    for index, row in enumerate(first['rows']):
        images = {v: first_store / row[v]['name'] for v in ('teacher', 'B', 'C')}
        for variant in variants:
            name = f"{row['source_id']}-{variant}.png"
            path = store / name
            if not path.exists():
                try:
                    data = b''.join(volume.read_file(f'{remote}/{name}'))
                except FileNotFoundError:
                    continue
                path.write_bytes(data)
            if name in expected and hashlib.sha256(path.read_bytes()).hexdigest() != expected[name]:
                raise RuntimeError('Image hash mismatch: ' + name)
            images[variant] = path
        canvas = Image.new('RGB', (len(columns) * tile, tile + 140), 'white')
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 6), f"P{index + 1:02d} | {row.get('subdimension', '')}", font=big, fill='black')
        for line, text in enumerate(textwrap.wrap(row['prompt'], 250 if row['prompt'].isascii() else 120)[:2]):
            draw.text((10, 40 + 24 * line), text, font=font, fill='black')
        for column, variant in enumerate(columns):
            draw.text((column * tile + 8, 105), labels[variant], font=big, fill='black')
            if variant in images:
                with Image.open(images[variant]) as image:
                    canvas.paste(image.convert('RGB').resize((tile, tile), Image.Resampling.LANCZOS), (column * tile, 140))
            else:
                draw.text((column * tile + 150, 140 + tile // 2), 'rendering...', font=big, fill='gray')
        canvas.save(sheets / f'P{index + 1:02d}.jpg', quality=88)
        ready += all(v in images for v in variants)
    print(json.dumps(dict(sheets=len(first['rows']), complete=ready, finished=summary is not None)))


if __name__ == '__main__':
    main()
