"""One Nihonga v1 diagram: the 32-slot transformer, the bridge in detail (from bridge-architecture.svg), the Lightning LoRA and
the 10-step schedule."""
import re
import sys
from pathlib import Path

from build_bridge_explainer_svgs import BLUE, GREEN, GREY, HEAD, INK, PAPER, SAND, arrow, box, text

BRIDGE = Path(sys.argv[1])
OUT = Path(sys.argv[2])


def main():
    W, H = 1120, 1640
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img" aria-labelledby="t d">',
         '<title id="t">Nihonga v1 architecture</title>',
         '<desc id="d">The Qwen-Image-2.1 transformer with 32 slots: blocks 5 and 20 are replaced by bridges, shown in detail '
         '(an 8-head attention branch with its own prompt cache, then a conditioned MLP); the other 30 blocks carry a fused rank-32 '
         'Lightning LoRA. Sampling uses 10 points of the 40-step schedule: three small steps, then seven large ones.</desc>', HEAD,
         f'<rect width="{W}" height="{H}" fill="{PAPER}"/>', '<g font-family="Georgia,serif">',
         text(W / 2, 44, 'Nihonga v1: 30 blocks, 2 bridges, a fused Lightning LoRA, 10 steps', 25, style='font-style="italic"'),
         text(W / 2, 74, 'The Qwen-Image-2.1 transformer; text encoder and VAE unchanged', 15, fill=GREY)]

    # ---- Left: the 32 slots ----
    x, w, top, step = 40, 400, 160, 27
    s += [box(x, top - 46, w, 30, PAPER), text(x + w / 2, top - 26, 'prompt tokens + image tokens', 14),
          arrow(f'M{x + w / 2} {top - 16} L{x + w / 2} {top - 2}')]
    for slot in range(32):
        y = top + slot * step
        if slot in (5, 20):
            s += [box(x, y, w, 23, SAND), text(x + w / 2, y + 16, f'slot {slot} · bridge · ~25 M', 13, style='font-weight="bold"')]
        else:
            s += [box(x, y, w, 23, BLUE), text(x + w / 2, y + 16, f'block {slot} · 218 M · Lightning LoRA fused in', 12)]
    end = top + 32 * step
    s += [arrow(f'M{x + w / 2} {end + 2} L{x + w / 2} {end + 16}'), box(x, end + 18, w, 30, PAPER),
          text(x + w / 2, end + 38, 'velocity → next latent', 14)]

    # ---- Right: the bridge in detail ----
    bx, by, bscale = 520, 96, 1.0
    inner = BRIDGE.read_text(encoding='utf-8')
    vb = re.search(r'viewBox="([^"]+)"', inner).group(1).split()
    bw, bh = float(vb[2]), float(vb[3])
    body = re.sub(r'^.*?<svg[^>]*>', '', inner, count=1, flags=re.S)
    body = re.sub(r'</svg>\s*$', '', body)
    body = re.sub(r'<title[^>]*>.*?</title>|<desc[^>]*>.*?</desc>|<defs>.*?</defs>', '', body, flags=re.S)
    body = re.sub(r'<rect width="[\d.]+" height="[\d.]+" fill="[^"]+"/>', '', body, count=1)
    s.append(f'<rect x="{bx - 8}" y="{by - 8}" width="{bw + 16}" height="{bh + 16}" fill="none" stroke="{SAND}" stroke-width="3" rx="4"/>')
    s.append(f'<g transform="translate({bx},{by}) scale({bscale})">{body}</g>')
    for slot in (5, 20):
        y = top + slot * step + 11
        s.append(f'<path d="M{x + w} {y} C{x + w + 40} {y}, {bx - 50} {by + 40}, {bx - 8} {by + 40}" stroke="{INK}" '
                 f'stroke-width="1.4" stroke-dasharray="5 4" fill="none" marker-end="url(#arrow)"/>')

    # ---- Bottom: Lightning LoRA and schedule ----
    y2 = by + bh + 40
    s += [box(40, y2, 500, 380, '#FFFFFF'), text(290, y2 + 32, 'The Lightning LoRA (fused)', 19, style='font-weight="bold"')]
    lines = ['A LoRA learns a thin correction beside a frozen', 'matrix: W·x + B·(A·x), with B 4,096 × 32 and A 32 × 4,096.',
             'Ours: q, k, v and out of the 30 blocks, 31.5 M weights.', '',
             'Step distillation: one LoRA step is trained to land', 'where five steps of the pruned model’s own 40-step',
             'path land, on 1,000 prompts; letter tokens, found by', 'OCR, count ten times in the loss.', '',
             'Fused after training: W′ = W + B·A.', 'No extra weights, files or time.']
    s += [text(62, y2 + 66 + i * 26, line, 15, anchor='start') for i, line in enumerate(lines)]
    sx0 = 580
    s += [box(sx0, y2, 500, 380, '#FFFFFF'), text(sx0 + 250, y2 + 32, 'The 10-step schedule', 19, style='font-weight="bold"'),
          text(sx0 + 250, y2 + 56, 'points of the stock 40-step schedule', 13, fill=GREY)]
    sx, sw, sy = sx0 + 30, 440, y2 + 130
    s.append(f'<line x1="{sx}" y1="{sy}" x2="{sx + sw}" y2="{sy}" stroke="{INK}" stroke-width="1.2"/>')
    for t in range(41):
        px = sx + sw * t / 40
        s.append(f'<line x1="{px:.1f}" y1="{sy - 4}" x2="{px:.1f}" y2="{sy + 4}" stroke="{GREY}" stroke-width="0.8"/>')
    points = [0, 1, 3, 5, 10, 15, 20, 25, 30, 35, 40]
    for a, b in zip(points[:-1], points[1:]):
        pa, pb = sx + sw * a / 40, sx + sw * b / 40
        s.append(f'<path d="M{pa:.1f} {sy - 6} Q{(pa + pb) / 2:.1f} {sy - 28 - (b - a) * 2} {pb:.1f} {sy - 6}" stroke="{INK}" stroke-width="1.3" fill="none"/>')
        s.append(f'<circle cx="{pa:.1f}" cy="{sy}" r="5" fill="{SAND if b <= 5 else GREEN}" stroke="{INK}" stroke-width="1"/>')
    s.append(f'<circle cx="{sx + sw:.1f}" cy="{sy}" r="5" fill="{INK}"/>')
    s += [text(sx, sy + 26, 'noise', 12, anchor='start', fill=GREY), text(sx + sw, sy + 26, 'image', 12, anchor='end', fill=GREY)]
    notes = ['Three small steps first (0, 1, 3, 5): the layout', 'is decided there, and one big jump merged',
             'figures and dropped small objects.', 'Then seven big LoRA steps (5, 10, …, 35, end).', '',
             'No classifier-free guidance:', '10 transformer calls per image instead of 40.']
    s += [text(sx0 + 22, y2 + 196 + i * 25, line, 15, anchor='start') for i, line in enumerate(notes)]
    s.append('</g></svg>')
    OUT.write_text('\n'.join(s), encoding='utf-8')
    print('wrote', OUT, 'bridge', bw, bh, 'bottom', y2 + 380)


if __name__ == '__main__':
    main()
