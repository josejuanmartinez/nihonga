"""Redraw the Phase 6 giraffe decision example with the human-corrected scores (no GPU)."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / '.archive/nihonga-phase3-artifacts/phase6/alignment_qa_images/adb98bb2d4a656eb05a9f8761611a6a4c3dc4a2e1eefca7de2c6b90ba3455360'
TARGET = ROOT.parent / 'niwaki-page/nihonga/images/phase6-align-example_01_p1-0c99a4e9e487-corrected.jpg'


def tag(draw, x, y, label, colour, font):
    width = draw.textlength(label, font=font) + 16
    draw.rounded_rectangle((x, y, x + width, y + 22), radius=4, fill=colour)
    draw.text((x + 8, y + 3), label, font=font, fill='white')


def main():
    regular = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 15)
    small = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 13)
    bold = ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf', 17)
    tags = ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf', 13)
    canvas = Image.new('RGB', (1100, 600), '#F9F6F1')
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 12), '005. p1-0c99a4e9e487 | spatial_relation', font=regular, fill='#222222')
    draw.text((20, 32), 'A giraffe partially hidden behind a coffee mug, stylized 3D', font=regular, fill='#222222')
    draw.text((20, 52), 'Are all named objects visible in the requested relative positions?', font=small, fill='#444444')
    draw.text((20, 70), 'Corrected on human review: the pretrained bridge hides the giraffe\'s neck behind the mug; update 120 draws the', font=small, fill='#2F5A86')
    draw.text((20, 86), 'neck coming out of the mug, as if the giraffe were inside it. The original blind score had these reversed.', font=small, fill='#2F5A86')
    for x, label, image, verdict, colour, blind in (
            (20, 'Pretrained bridge (selected)', 'p1-0c99a4e9e487-pretrained.png', 'PASS', '#2E7D32', 'blind score: FAIL'),
            (560, 'Update 120 (healed)', 'p1-0c99a4e9e487-update120.png', 'FAIL', '#B3261E', 'blind score: PASS')):
        draw.text((x, 112), label, font=bold, fill='#111111')
        tag(draw, x + 330, 112, verdict, colour, tags)
        draw.text((x + 330, 138), blind, font=small, fill='#777777')
        with Image.open(SOURCE / image) as source:
            picture = source.convert('RGB')
        picture.thumbnail((360, 420))
        canvas.paste(picture, (x + 90, 160))
    canvas.save(TARGET, quality=90)
    print(TARGET)


if __name__ == '__main__':
    main()
