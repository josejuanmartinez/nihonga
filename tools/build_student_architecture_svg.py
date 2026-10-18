"""Draw the 30-block student before healing and after healing with the LoRA fused, as an SVG in the site's style (no GPU)."""
from build_bridge_explainer_svgs import BLUE, GREEN, GREY, HEAD, INK, PAPER, SAND, SITE, arrow, box, text

BRIDGES = {5: ('bridge 5', 'step 200'), 20: ('bridge 20', 'step 600')}
BLOCK_M, BRIDGE_M = 218.1, 25.2
LORA_M = 30 * 4 * 16 * (4096 + 4096) / 1e6   # rank 16 on to_q, to_k, to_v, to_out of the 30 real blocks


def column(x, top, w, after):
    s = [box(x, top - 40, w, 28, PAPER), text(x + w / 2, top - 21, 'image + prompt tokens', 14)]
    for slot in range(32):
        y = top + slot * 28
        if slot in BRIDGES:
            name, step = BRIDGES[slot]
            s.append(box(x, y, w, 24, '#D9A86C' if after else SAND))
            label = f'{name}′  ·  attention bridge, fine-tuned  ·  {BRIDGE_M}M' if after else f'{name}  ·  attention bridge ({step})  ·  {BRIDGE_M}M'
            s.append(text(x + w / 2, y + 17, label, 13, style='font-weight="bold"'))
        else:
            s.append(box(x, y, w, 24, BLUE if after else GREEN))
            label = f'block {slot}  ·  W + B·A fused  ·  {BLOCK_M}M' if after else f'block {slot}  ·  original weights  ·  {BLOCK_M}M'
            s.append(text(x + w / 2, y + 17, label, 13))
    end = top + 32 * 28
    s += [arrow(f'M{x + w / 2} {top - 12} L{x + w / 2} {top - 2}'), box(x, end + 6, w, 28, PAPER),
          text(x + w / 2, end + 25, 'velocity (one denoising step)', 14)]
    return s


def student30():
    W, H, top = 1200, 1270, 170
    left, right, cw = 30, 790, 380
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img" aria-labelledby="t d">',
         '<title id="t">The 30-block student before and after healing</title>',
         '<desc id="d">Left: 32 slots, 30 original blocks plus attention bridges in slots 5 and 20. Middle: healing trains a rank-16 LoRA '
         'on the attention projections of the 30 blocks together with both bridges, then merges each LoRA into its weight matrix. '
         'Right: the same 32 slots with fused blocks and fine-tuned bridges, with no extra weights.</desc>', HEAD,
         f'<rect width="{W}" height="{H}" fill="{PAPER}"/>', '<g font-family="Georgia,serif">',
         text(W / 2, 40, 'The 30-block student: before healing, and after healing with the LoRA fused', 24, style='font-style="italic"'),
         text(W / 2, 70, 'Blocks 5 and 20 are replaced by attention bridges; the other 30 blocks keep their original architecture.', 15, fill=GREY),
         text(left + cw / 2, 115, 'Before healing', 20, style='font-weight="bold"'),
         text(right + cw / 2, 115, 'After healing, LoRA fused', 20, style='font-weight="bold"')]
    s += column(left, top, cw, False) + column(right, top, cw, True)

    # Middle: what healing does
    mx, mw = 440, 320
    s += [box(mx, top - 40, mw, 330, '#FFFFFF'),
          text(mx + mw / 2, top - 14, 'Healing (option B)', 18, style='font-weight="bold"')]
    lines = ['Trained together, end to end against', 'the teacher velocity (40% text prompts):', '',
             f'• LoRA rank 16 on q, k, v, out of each', f'  of the 30 blocks: {LORA_M:.1f}M weights, lr 2e-5', '',
             f'• both bridges: {2 * BRIDGE_M:.1f}M weights, lr 1e-5', '',
             '• original block weights stay frozen', '', 'Every checkpoint saved; each judged by', 'velocity and by the spelling grids.']
    for i, line in enumerate(lines):
        s.append(text(mx + 18, top + 16 + i * 21, line, 14, anchor='start'))
    s.append(arrow(f'M{mx - 20} {top + 120} L{mx - 4} {top + 120}'))
    s.append(arrow(f'M{mx + mw + 4} {top + 120} L{right - 10} {top + 120}'))

    # Middle: the LoRA on one projection, then fused
    y0 = top + 330
    s += [box(mx, y0, mw, 520, '#FFFFFF'), text(mx + mw / 2, y0 + 26, 'One attention projection (e.g. q)', 16, style='font-weight="bold"'),
          text(mx + mw / 2, y0 + 56, 'during healing', 15, style='font-style="italic"'),
          box(mx + 30, y0 + 80, 110, 50, GREEN), text(mx + 85, y0 + 102, 'W', 16, style='font-weight="bold"'),
          text(mx + 85, y0 + 121, '4096 × 4096', 12), text(mx + 85, y0 + 146, 'frozen', 12, fill=GREY),
          box(mx + 180, y0 + 70, 110, 30, SAND), text(mx + 235, y0 + 90, 'A  16 × 4096', 13),
          box(mx + 180, y0 + 112, 110, 30, SAND), text(mx + 235, y0 + 132, 'B  4096 × 16', 13),
          arrow(f'M{mx + 235} {y0 + 100} L{mx + 235} {y0 + 110}'),
          text(mx + 235, y0 + 162, 'trained (B starts at 0)', 12, fill=GREY),
          text(mx + mw / 2, y0 + 200, 'y = W·x + B·(A·x)', 17),
          text(mx + mw / 2, y0 + 224, '(α / rank = 16 / 16 = 1)', 12, fill=GREY),
          arrow(f'M{mx + mw / 2} {y0 + 240} L{mx + mw / 2} {y0 + 290}'),
          text(mx + mw / 2 + 10, y0 + 270, 'merge after training', 13, anchor='start', fill=GREY),
          text(mx + mw / 2, y0 + 320, 'after fusing', 15, style='font-style="italic"'),
          box(mx + 85, y0 + 336, 150, 56, BLUE), text(mx + 160, y0 + 360, 'W′ = W + B·A', 16, style='font-weight="bold"'),
          text(mx + 160, y0 + 380, '4096 × 4096', 12),
          text(mx + mw / 2, y0 + 420, 'y = W′·x', 17),
          text(mx + mw / 2, y0 + 452, 'Same shape and cost as before:', 13),
          text(mx + mw / 2, y0 + 472, 'the LoRA adds no weights and', 13),
          text(mx + mw / 2, y0 + 492, 'no time at inference.', 13)]

    # Footer: counts
    fy = top + 32 * 28 + 70
    removed = 2 * BLOCK_M - 2 * BRIDGE_M
    s += [box(30, fy, W - 60, 120, '#FFFFFF'),
          text(50, fy + 30, f'Teacher: 32 blocks × {BLOCK_M}M.  Student: 30 blocks + 2 bridges = {removed:.1f}M fewer weights '
               f'(about 5.6%), about 1.05× faster per step.', 15, anchor='start'),
          text(50, fy + 58, f'During healing only: + {LORA_M:.1f}M LoRA weights (trainable, with the {2 * BRIDGE_M:.1f}M of the bridges). '
               'After fusing: exactly the size of the left column.', 15, anchor='start'),
          text(50, fy + 86, 'Later, removals three and four are chosen inside the healed student, and the speed-distillation (lightning) '
               'LoRA,', 15, anchor='start'),
          text(50, fy + 108, 'trained last for about 5× fewer steps, is fused the same way before quantization.', 15, anchor='start'),
          '</g></svg>']
    return '\n'.join(s), W, H


if __name__ == '__main__':
    svg, _, _ = student30()
    (SITE / 'student30-architecture.svg').write_text(svg, encoding='utf-8')
    print('wrote', SITE / 'student30-architecture.svg')
