"""Draw the attention-bridge architecture and the RoPE explainer as SVGs in the site's style (no GPU)."""
import math
from pathlib import Path

SITE = Path(__file__).resolve().parents[1].parent / 'niwaki-page/nihonga'
INK, PAPER, GREEN, BLUE, SAND, GREY = '#231F1A', '#F7EEE3', '#A7C2A0', '#C3D3DB', '#E8D5B5', '#5E564B'
HEAD = (f'<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" '
        f'orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10" fill="{INK}"/></marker></defs>')


def text(x, y, s, size=16, anchor='middle', style='', fill=INK):
    return f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" fill="{fill}" {style}>{s}</text>'


def box(x, y, w, h, fill=GREEN):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="2" fill="{fill}" stroke="{INK}" stroke-width="1.2"/>'


def arrow(points):
    return f'<path d="{points}" stroke="{INK}" stroke-width="1.3" fill="none" marker-end="url(#arrow)"/>'


def bridge13():
    W, H = 560, 1060
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img" aria-labelledby="t d">',
         '<title id="t">Attention bridge for block 13</title>',
         '<desc id="d">Inputs: hidden states, the shared timestep modulation, token positions and a prefix key/value cache. '
         'An attention branch (8 heads of 128) updates image tokens, then a conditioned MLP branch updates all tokens. '
         'About 25.2 million weights, against 218 million in the original block.</desc>', HEAD,
         f'<rect width="{W}" height="{H}" fill="{PAPER}"/>', '<g font-family="Georgia,serif">',
         text(W / 2, 36, 'Attention bridge for block 13', 23, style='font-style="italic"')]
    # inputs
    labels = [('hidden h', '4,096 per token', BLUE), ('timestep', 'scale · gate', SAND), ('positions', 'RoPE', SAND), ('cache', 'prompt K, V', SAND)]
    for i, (a, b, c) in enumerate(labels):
        x = 20 + i * 132
        s += [box(x, 56, 122, 46, c), text(x + 61, 76, a, 15), text(x + 61, 94, b, 12)]
    # attention branch frame
    s.append(f'<rect x="30" y="122" width="500" height="430" rx="4" fill="none" stroke="{GREY}" stroke-dasharray="5 4"/>')
    s.append(text(44, 142, 'ATTENTION BRANCH · image tokens', 13, 'start', fill=GREY))
    s += [arrow('M81 102 V160'), box(60, 160, 440, 40), text(280, 186, 'LayerNorm(h) × (1 + scale_attn)', 16)]
    s += [arrow('M150 200 V228'), arrow('M280 200 V228'), arrow('M410 200 V228')]
    for x, name in ((90, 'Q'), (220, 'K'), (350, 'V')):
        s += [box(x, 228, 120, 46), text(x + 60, 250, f'{name} · 4,096 → 1,024', 13), text(x + 60, 266, '8 heads × 128', 12)]
    s += [arrow('M150 274 V300'), arrow('M280 274 V300'), box(90, 300, 250, 44, SAND),
          text(215, 320, 'RMSNorm, then RoPE', 15), text(215, 336, 'on Q and K, per 128-d head', 12)]
    s += [arrow('M347 102 V118'), f'<path d="M347 118 H520 V322 H346" stroke="{INK}" stroke-width="1" fill="none" stroke-dasharray="3 3"/>',
          arrow('M480 102 V112 H530 V400 H471'), arrow('M215 344 V372'), arrow('M410 274 V372'),
          box(90, 372, 380, 58), text(280, 394, 'softmax(Q·Kᵀ / √128) · V', 16),
          text(280, 414, 'queries: image tokens · keys: prompt + image', 12)]
    s += [text(522, 360, 'step 0: store', 11, 'end', fill=GREY), text(522, 374, 'later: reuse', 11, 'end', fill=GREY),
          arrow('M280 430 V458'), box(90, 458, 380, 46), text(280, 480, 'out · 1,024 → 4,096', 15),
          text(280, 496, '× tanh(gate_attn) · starts at zero', 12)]
    s += [arrow('M280 504 V524'), f'<circle cx="280" cy="538" r="14" fill="{PAPER}" stroke="{INK}" stroke-width="1.2"/>',
          text(280, 544, '+', 20), arrow('M44 180 H38 V538 H265'), text(150, 530, 'h kept on the residual path', 11, fill=GREY)]
    # MLP branch
    s += [f'<rect x="30" y="572" width="500" height="372" rx="4" fill="none" stroke="{GREY}" stroke-dasharray="5 4"/>',
          text(44, 592, 'CONDITIONED MLP BRANCH · all tokens', 13, 'start', fill=GREY),
          arrow('M280 552 V610'), box(60, 610, 440, 40), text(280, 636, "LayerNorm(h′) × (1 + scale_mlp)", 16),
          arrow('M280 650 V676'), box(130, 676, 300, 40), text(280, 701, 'down · 4,096 → 1,024', 15),
          arrow('M280 716 V740'), box(200, 740, 160, 36), text(280, 763, 'GELU', 15),
          arrow('M280 776 V800'), box(130, 800, 300, 40), text(280, 825, 'up · 1,024 → 4,096', 15),
          arrow('M280 840 V864'), box(130, 864, 300, 36), text(280, 887, '× tanh(gate_mlp)', 15),
          arrow('M280 900 V912'), f'<circle cx="280" cy="926" r="14" fill="{PAPER}" stroke="{INK}" stroke-width="1.2"/>',
          text(280, 932, '+', 20), arrow('M60 630 H44 V926 H265')]
    s += [arrow('M280 940 V966'), box(110, 966, 340, 42, BLUE), text(280, 992, 'h_out · 4,096 per token', 16),
          text(280, 1036, '≈ 25.2M weights · 8 heads   (original block 13: 218M · 32 heads)', 14, fill=GREY), '</g></svg>']
    (SITE / 'bridge13-architecture.svg').write_text('\n'.join(s) + '\n', encoding='utf-8')


def vec(cx, cy, length, degrees, width=2.2, color=INK):
    r = math.radians(degrees)
    x, y = cx + length * math.cos(r), cy - length * math.sin(r)
    return (f'<line x1="{cx}" y1="{cy}" x2="{x:.1f}" y2="{y:.1f}" stroke="{color}" stroke-width="{width}" marker-end="url(#arrow)"/>'
            f'<circle cx="{cx}" cy="{cy}" r="3" fill="{color}"/>')


def rope():
    W, H = 760, 1090
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img" aria-labelledby="t d">',
         '<title id="t">How RoPE encodes position</title>',
         '<desc id="d">One pair of numbers from a query is drawn as an arrow. RoPE rotates it by an angle equal to position times a fixed '
         'frequency. Two tokens with the same content at different positions get different arrows. The attention score depends only '
         'on the difference in angle, that is, on the distance between the tokens. Different pairs rotate at different speeds, and '
         'in images some pairs follow the row and others the column.</desc>', HEAD,
         f'<rect width="{W}" height="{H}" fill="{PAPER}"/>', '<g font-family="Georgia,serif">',
         text(W / 2, 36, 'RoPE: position as a rotation', 23, style='font-style="italic"')]
    # panel 1
    s += [text(30, 76, '1 · Same content, different place', 17, 'start'),
          text(30, 98, 'Two patches show the same letter, so their query q is the same. One pair of its numbers, drawn as an arrow:', 13, 'start', fill=GREY)]
    for i, (col, deg) in enumerate(((0, 0), (1, 30), (2, 60), (3, 90))):
        cx = 110 + i * 170
        s += [f'<circle cx="{cx}" cy="185" r="58" fill="none" stroke="{GREY}" stroke-width=".6" stroke-dasharray="2 3"/>',
              vec(cx, 185, 52, deg), text(cx, 268, f'column {col}', 14), text(cx, 286, f'rotated {deg}°', 13, fill=GREY)]
    s.append(text(W / 2, 314, 'angle = position × frequency (here 30° per column). The content (length) is unchanged; the direction says where.', 13, fill=GREY))
    # panel 2
    s += [text(30, 360, '2 · The score only sees the distance', 17, 'start'),
          text(30, 382, 'Queries and keys are both rotated. Their dot product depends on the angle between them: θ·(m − n).', 13, 'start', fill=GREY)]
    for i, (m, n) in enumerate(((2, 5), (10, 13))):
        cx = 190 + i * 380
        qa, ka = (m * 30) % 360, (n * 30) % 360
        s += [vec(cx, 470, 60, qa, color='#3E6B4A'), vec(cx, 470, 60, ka, color='#7A5230'),
              text(cx, 548, f'q at {m}, k at {n}', 14), text(cx, 566, f'difference {n - m} positions → same score', 13, fill=GREY)]
    s += [text(110, 420, 'q', 14, fill='#3E6B4A'), text(132, 420, 'k', 14, fill='#7A5230')]
    # panel 3
    s += [text(30, 612, '3 · Each pair turns at its own fixed speed', 17, 'start'),
          text(30, 634, 'freq = 1 / 10000^(2i / d): nothing is learned. Fast pairs see neighbours, slow pairs see the coarse layout.', 13, 'start', fill=GREY)]
    for row, (label, step) in enumerate((('fast pair', 57.3), ('medium pair', 15), ('slow pair', 3))):
        y = 676 + row * 52
        s.append(text(30, y + 5, label, 14, 'start'))
        for p in range(9):
            s.append(vec(170 + p * 62, y, 20, 90 - p * step, width=1.6))
    s += [text(170 + 4 * 62, 822, 'positions 0 → 8', 13, fill=GREY)]
    # panel 4
    s += [text(30, 858, '4 · In images the pairs split by axis', 17, 'start'),
          text(30, 880, 'Each 128-d head = 64 pairs: 8 follow the frame/segment, 28 the row, 28 the column (axes 16 / 56 / 56).', 13, 'start', fill=GREY),
          text(30, 898, 'Image tokens get a (row, column) on their grid, centred at zero; prompt tokens advance one step per word.', 13, 'start', fill=GREY)]
    for r in range(3):
        for c in range(4):
            x, y = 60 + c * 72, 914 + r * 42
            s += [box(x, y, 66, 36, BLUE if (r, c) == (1, 1) else SAND if (r, c) == (1, 2) else GREEN), text(x + 33, y + 23, f'{r - 1}, {c - 1}', 13)]
    s += [text(385, 936, 'Row pairs rotate with the row,', 13, 'start'),
          text(385, 954, 'column pairs with the column.', 13, 'start'),
          text(385, 980, 'So a head can learn "the patch just to', 13, 'start', fill=GREY),
          text(385, 998, 'my right" (blue → sand: same row,', 13, 'start', fill=GREY),
          text(385, 1016, 'column + 1), e.g. the next letter.', 13, 'start', fill=GREY),
          text(W / 2, 1072, 'RoPE is applied to Q and K only, after RMSNorm; V carries content and is not rotated.', 13, fill=GREY),
          '</g></svg>']
    (SITE / 'rope-explainer.svg').write_text('\n'.join(s) + '\n', encoding='utf-8')


if __name__ == '__main__':
    bridge13()
    rope()
    print('ok')
