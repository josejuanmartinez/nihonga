"""Draw the Nihonga v1 architecture (30 blocks + 2 bridges, fused adapter, 10-step schedule) as an SVG in the site's style."""
import sys
from pathlib import Path

from build_bridge_explainer_svgs import BLUE, GREEN, GREY, HEAD, INK, PAPER, SAND, arrow, box, text

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1].parent / 'niwaki-page/nihonga/v1-architecture.svg'


def main():
    W, H, top, x, w = 1100, 1100, 150, 40, 400
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img" aria-labelledby="t d">',
         '<title id="t">Nihonga v1 architecture</title>',
         '<desc id="d">Qwen-Image-2.1 transformer with 32 slots: blocks 5 and 20 are replaced by small attention bridges, and a '
         'rank-32 adapter is fused into the attention of the other 30 blocks. Sampling uses 10 points of the stock 40-step schedule: '
         'three small steps, then seven large ones.</desc>', HEAD,
         f'<rect width="{W}" height="{H}" fill="{PAPER}"/>', '<g font-family="Georgia,serif">',
         text(W / 2, 44, 'Nihonga v1: 30 blocks, 2 bridges, 10 steps', 26, style='font-style="italic"'),
         text(W / 2, 76, 'The Qwen-Image-2.1 transformer, with the text encoder and VAE unchanged', 15, fill=GREY),
         box(x, top - 44, w, 30, PAPER), text(x + w / 2, top - 24, 'prompt tokens + image tokens', 14),
         arrow(f'M{x + w / 2} {top - 14} L{x + w / 2} {top - 2}')]
    for slot in range(32):
        y = top + slot * 27
        if slot in (5, 20):
            s += [box(x, y, w, 23, SAND), text(x + w / 2, y + 16, f'slot {slot} · bridge (attention + MLP) · ~25 M', 13, style='font-weight="bold"')]
        else:
            s += [box(x, y, w, 23, BLUE), text(x + w / 2, y + 16, f'block {slot} · 218 M · adapter fused into q, k, v, out', 12)]
    end = top + 32 * 27
    s += [arrow(f'M{x + w / 2} {end + 2} L{x + w / 2} {end + 16}'), box(x, end + 18, w, 30, PAPER),
          text(x + w / 2, end + 38, 'velocity → next latent (Euler step)', 14)]
    # Right: the bridge, the adapter, the schedule.
    rx, rw = 500, 560
    y2 = top - 44
    s += [box(rx, y2, rw, 200, '#FFFFFF'), text(rx + rw / 2, y2 + 28, 'The fused adapter', 18, style='font-weight="bold"')]
    lines = ['Rank 32 on q, k, v and out of the 30 blocks, trained so', 'that one step lands where five steps of the pruned',
             'model’s own 40-step path land; letter tokens (found', 'by OCR) weigh ten times in the loss.', '',
             'Merged once: W′ = W + B·A. No extra weights or time.']
    s += [text(rx + 22, y2 + 58 + i * 22, line, 14, anchor='start') for i, line in enumerate(lines)]
    y3 = y2 + 230
    s += [box(rx, y3, rw, 330, '#FFFFFF'), text(rx + rw / 2, y3 + 28, 'The 10-step schedule', 18, style='font-weight="bold"'),
          text(rx + rw / 2, y3 + 54, 'points of the stock 40-step schedule (noise → image)', 13, fill=GREY)]
    sx, sw, sy = rx + 30, rw - 60, y3 + 120
    s.append(f'<line x1="{sx}" y1="{sy}" x2="{sx + sw}" y2="{sy}" stroke="{INK}" stroke-width="1.2"/>')
    for t in range(41):
        px = sx + sw * t / 40
        s.append(f'<line x1="{px:.1f}" y1="{sy - 4}" x2="{px:.1f}" y2="{sy + 4}" stroke="{GREY}" stroke-width="0.8"/>')
    points = [0, 1, 3, 5, 10, 15, 20, 25, 30, 35, 40]
    for a, b in zip(points[:-1], points[1:]):
        pa, pb = sx + sw * a / 40, sx + sw * b / 40
        color = SAND if b <= 5 else GREEN
        s.append(f'<path d="M{pa:.1f} {sy - 6} Q{(pa + pb) / 2:.1f} {sy - 30 - (b - a) * 2} {pb:.1f} {sy - 6}" stroke="{INK}" '
                 f'stroke-width="1.3" fill="none"/>')
        s.append(f'<circle cx="{pa:.1f}" cy="{sy}" r="5" fill="{color}" stroke="{INK}" stroke-width="1"/>')
    s.append(f'<circle cx="{sx + sw:.1f}" cy="{sy}" r="5" fill="{INK}"/>')
    s += [text(sx, sy + 26, 'noise', 12, anchor='start', fill=GREY), text(sx + sw, sy + 26, 'image', 12, anchor='end', fill=GREY)]
    notes = ['Three small steps first (points 0, 1, 3, 5): the layout is', 'decided at the start, and one big jump there merged',
             'figures, dropped small objects and broke patterns.', 'Then seven large jumps (5, 10, …, 35, end), each five',
             'steps of the original schedule.', '', 'No classifier-free guidance; 10 transformer calls per image.']
    s += [text(rx + 22, y3 + 178 + i * 21, line, 14, anchor='start') for i, line in enumerate(notes)]
    s.append('</g></svg>')
    OUT.write_text('\n'.join(s), encoding='utf-8')
    print('wrote', OUT)


if __name__ == '__main__':
    main()
