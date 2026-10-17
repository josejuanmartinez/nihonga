"""Crop the 'The Rusty Anchor' sign into two site figures: seed 0 across models, and B vs the attention bridge over 4 seeds."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT.parent / 'niwaki-page/nihonga/images'
SID = 'p1-2fa54a3c4eb8'
FIRST = ROOT / '.archive/bridge_images/d4530312b0f69c14bb9bbc8ad15d2514f3e8cb009caa860eed3b1702ff22ad7b'
LONG = ROOT / '.archive/bridge_variant_images/aa4f5189b1d06f81ec32eb4e01b400f790f0b1bbe545e923ae2f38ff40f2c286'
SEEDS_REF = ROOT / '.archive/multiseed_text/af4a34ba1a0c739121706991b89e79ec92bff3debaf1b1ddbf5b6188c182f514'
ATTN = ROOT / '.archive/multiseed_text/8611910a2e5f829c8a5f663372457de6ab2481528914e1f3ea7a49023b9a055b'


def crop(path, top=0.0, bottom=0.36, width=440):
    with Image.open(path) as image:
        w, h = image.size
        part = image.convert('RGB').crop((0, int(top * h), w, int(bottom * h)))
    return part.resize((width, int(part.height * width / part.width)), Image.Resampling.LANCZOS)


def strip(cells, labels, columns, label_height=58, gap=8):
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 20)
    cw, ch = cells[0].size
    rows = (len(cells) + columns - 1) // columns
    canvas = Image.new('RGB', (columns * cw + (columns - 1) * gap, rows * (ch + label_height)), 'white')
    draw = ImageDraw.Draw(canvas)
    for index, (cell, label) in enumerate(zip(cells, labels)):
        x, y = (index % columns) * (cw + gap), (index // columns) * (ch + label_height)
        canvas.paste(cell, (x, y))
        for line, text in enumerate(label.split('\n')):
            draw.text((x + 6, y + ch + 4 + 24 * line), text, font=font, fill='black')
    return canvas


def main():
    seed0 = [FIRST / f'{SID}-teacher.png', FIRST / f'{SID}-B.png', FIRST / f'{SID}-C.png',
             LONG / f'{SID}-cond_r1024_long.png', ATTN / f'{SID}-e2e_attn.png']
    labels = ['Teacher\n"Anchor" correct', 'B (block 13 kept)\n"Anchor" correct', 'First bridge\n"Achor"',
              'Conditioned bridge\n"Achor"', 'Attention bridge\n"Anchor" correct']
    strip([crop(p) for p in seed0], labels, 5).save(SITE / 'phase7-anchor-seed0.jpg', quality=88)
    b = [FIRST / f'{SID}-B.png'] + [SEEDS_REF / f'{SID}-s{k}-B.png' for k in (1, 2, 3)]
    a = [ATTN / f'{SID}-e2e_attn.png'] + [ATTN / f'{SID}-s{k}-e2e_attn.png' for k in (1, 2, 3)]
    b_labels = ['B, seed 0: correct', 'B, seed 1: correct', 'B, seed 2: correct', 'B, seed 3: "Anchar"']
    a_labels = ['Attention, seed 0: correct', 'Attention, seed 1: "Achar"', 'Attention, seed 2: "Ancher"', 'Attention, seed 3: garbled']
    strip([crop(p, width=400) for p in b + a], b_labels + a_labels, 4, label_height=34).save(SITE / 'phase7-anchor-seeds.jpg', quality=88)
    print('ok')


if __name__ == '__main__':
    main()
