"""Draw the Nihonga Lightning LoRA explainer: what a LoRA is, how ours was trained, and how it was fused (site style)."""
import sys
from pathlib import Path

from build_bridge_explainer_svgs import BLUE, GREEN, GREY, HEAD, INK, PAPER, SAND, arrow, box, text

OUT = Path(sys.argv[1])


def matrix(x, y, w, h, fill, label, sub, sub2=None):
    out = [box(x, y, w, h, fill), text(x + w / 2, y + h / 2 - (6 if sub else -6), label, 17, style='font-weight="bold"')]
    if sub:
        out.append(text(x + w / 2, y + h / 2 + 14, sub, 12))
    if sub2:
        out.append(text(x + w / 2, y + h + 18, sub2, 12, fill=GREY))
    return out


def main():
    W, H = 1100, 1240
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img" aria-labelledby="t d">',
         '<title id="t">The Nihonga Lightning LoRA</title>',
         '<desc id="d">1. A LoRA leaves a weight matrix W frozen and learns a thin correction B times A of rank 32. 2. The Lightning LoRA '
         'is trained so that one 10-step-sampler step lands where five steps of the model\'s own 40-step path land, with letter tokens '
         'weighted ten times. 3. After training it is fused: W prime equals W plus B times A, the same shape and cost as before.</desc>',
         HEAD, f'<rect width="{W}" height="{H}" fill="{PAPER}"/>', '<g font-family="Georgia,serif">',
         text(W / 2, 44, 'The Lightning LoRA: what it is, how it learned, how it was fused', 25, style='font-style="italic"')]

    # ---- 1. What a LoRA is ----
    y0 = 80
    s += [box(30, y0, W - 60, 330, '#FFFFFF'), text(56, y0 + 32, '1 · What a LoRA is', 19, anchor='start', style='font-weight="bold"'),
          text(56, y0 + 58, 'Instead of changing a big weight matrix, learn a thin correction next to it. Here: every attention projection', 14, anchor='start'),
          text(56, y0 + 78, '(q, k, v, out) of the 30 blocks, 120 matrices of 4,096 × 4,096, each with a rank-32 pair B·A.', 14, anchor='start')]
    s += matrix(80, y0 + 110, 170, 170, GREEN, 'W', '4,096 × 4,096', 'frozen · 16.8 M weights')
    s += [text(285, y0 + 200, '+', 34)]
    s += matrix(330, y0 + 170, 70, 110, SAND, 'B', '4,096 × 32')
    s += [text(420, y0 + 230, '·', 34)]
    s += matrix(445, y0 + 195, 170, 40, SAND, 'A', '32 × 4,096')
    s += [text(472, y0 + 300, 'trained · 262 K weights per matrix (1.6%)', 12, fill=GREY)]
    s += [text(660, y0 + 160, 'y = W·x + B·(A·x)', 20, anchor='start'),
          text(660, y0 + 192, 'A squeezes the input to 32 numbers,', 14, anchor='start'),
          text(660, y0 + 212, 'B expands them back: a low-rank nudge.', 14, anchor='start'),
          text(660, y0 + 244, 'B starts at zero, so training starts from', 14, anchor='start'),
          text(660, y0 + 264, 'the unchanged model.', 14, anchor='start'),
          text(660, y0 + 296, '120 pairs: 31.5 M trainable weights.', 14, anchor='start', style='font-weight="bold"')]

    # ---- 2. How it learned ----
    y1 = y0 + 360
    s += [box(30, y1, W - 60, 430, '#FFFFFF'), text(56, y1 + 32, '2 · How it learned to be fast (step distillation)', 19, anchor='start', style='font-weight="bold"'),
          text(56, y1 + 58, 'The pruned model was run in its normal 40 steps on 1,000 prompts (500 with text), saving the image after every step.', 14, anchor='start'),
          text(56, y1 + 78, 'The LoRA is trained so that one big step lands where five small steps of that recorded path land.', 14, anchor='start')]
    lx, lw, ly = 90, 920, y1 + 190
    s.append(f'<line x1="{lx}" y1="{ly}" x2="{lx + lw}" y2="{ly}" stroke="{INK}" stroke-width="1.2"/>')
    for t in range(41):
        px = lx + lw * t / 40
        s.append(f'<line x1="{px:.1f}" y1="{ly - 5}" x2="{px:.1f}" y2="{ly + 5}" stroke="{GREY}" stroke-width="0.8"/>')
    # Small teacher steps (one segment highlighted) and the student jump over them.
    a, b = 10, 15
    pa, pb = lx + lw * a / 40, lx + lw * b / 40
    for t in range(a, b):
        p1, p2 = lx + lw * t / 40, lx + lw * (t + 1) / 40
        s.append(f'<path d="M{p1:.1f} {ly + 8} Q{(p1 + p2) / 2:.1f} {ly + 26} {p2:.1f} {ly + 8}" stroke="{GREY}" stroke-width="1.2" fill="none"/>')
    s.append(f'<path d="M{pa:.1f} {ly - 8} Q{(pa + pb) / 2:.1f} {ly - 70} {pb:.1f} {ly - 8}" stroke="{INK}" stroke-width="2" fill="none" marker-end="url(#arrow)"/>')
    s += [text((pa + pb) / 2, ly - 78, 'one LoRA step', 14, style='font-weight="bold"'),
          text((pa + pb) / 2, ly + 46, 'five recorded steps of the pruned model', 13, fill=GREY),
          text(lx, ly + 30, 'noise', 12, anchor='start', fill=GREY), text(lx + lw, ly + 30, 'image', 12, anchor='end', fill=GREY)]
    notes = ['Loss: the gap between where the LoRA step lands and where the recorded path lands, over every token of the latent.',
             'Letters are a few percent of the tokens, so an OCR detector marked the tokens that hold text in each training image;',
             'those count ten times in the loss. 3,000 updates on one H100, learning rate decaying to zero.',
             '',
             'At use, the first jump out of noise is split into three small steps (0, 1, 3, 5): one big first jump merged figures',
             'and dropped small objects. Then seven big LoRA steps (5, 10, …, 35, end): 10 steps in all, instead of 40.']
    s += [text(56, y1 + 290 + i * 21, line, 14, anchor='start') for i, line in enumerate(notes)]

    # ---- 3. How it was fused ----
    y2 = y1 + 460
    s += [box(30, y2, W - 60, 330, '#FFFFFF'), text(56, y2 + 32, '3 · How it was fused', 19, anchor='start', style='font-weight="bold"'),
          text(56, y2 + 58, 'After training, each pair is multiplied out once and added to its matrix. The LoRA disappears into the weights.', 14, anchor='start')]
    s += matrix(80, y2 + 100, 150, 150, GREEN, 'W', '4,096 × 4,096')
    s += [text(262, y2 + 180, '+', 34)]
    s += matrix(300, y2 + 100, 150, 150, SAND, 'B·A', '4,096 × 4,096', 'rank 32, computed once')
    s += [arrow(f'M{470} {y2 + 175} L{560} {y2 + 175}'), text(515, y2 + 162, 'merge', 13, fill=GREY)]
    s += matrix(580, y2 + 100, 150, 150, BLUE, 'W′', '4,096 × 4,096')
    s += [text(760, y2 + 140, 'W′ = W + B·A', 19, anchor='start'),
          text(760, y2 + 172, 'Same shape and size as W:', 14, anchor='start'),
          text(760, y2 + 192, 'no extra weights, no extra time,', 14, anchor='start'),
          text(760, y2 + 212, 'one ordinary checkpoint.', 14, anchor='start'),
          text(760, y2 + 244, 'Fused in FP32, stored in BF16.', 14, anchor='start', fill=GREY)]
    s.append('</g></svg>')
    OUT.write_text('\n'.join(s), encoding='utf-8')
    print('wrote', OUT)


if __name__ == '__main__':
    main()
