"""Write Phase 7 conditional scan and latency results into the Nihonga site (no GPU or network calls).

Edits the published page and the matching narrative in the site's build script between
`<!-- p7-scan -->` / `<!-- p7-status -->` markers, and adds the measured values to results.json.
"""
import argparse
import datetime
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCAN_A = 'data/phase7/conditional_importance/7454da6a5c1dc33af23dd46833e698f9b759d14985ca96340bc3d40cd47fb4f5/summary.json'
LATENCY_A = 'data/phase7/latency_benchmark/29f50a4db5f63a6c73135438800ab2d5413e32878bc3fa61284aebbb5c165f4c/summary.json'
SCAN_B_DIR = 'data/phase7/conditional_importance_b'
QUALITY_A = 'data/phase7/blind_quality_v1/review_summary.json'
BRIDGE13_DIR = 'data/phase7/block13_bridge'
GATE = 6.52


def load(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def pct(v, d=2):
    return f'{100 * v:.{d}f}%'


def scan_rows(report):
    rows = []
    for b in report['blocks']:
        rows.append(dict(block=b['original_block_id'], mean=b['overall']['teacher_relative_l2']['mean'],
                         p90=b['overall']['teacher_relative_l2']['p90'],
                         **{s: b['stages'][s]['teacher_relative_l2']['mean'] for s in ('early', 'middle', 'late')}))
    return sorted(rows, key=lambda r: r['mean'])


def panel_baseline(report):
    records = report['blocks'][0]['records']
    return {p: sum(r['baseline_teacher_relative_l2'] for r in records if r['panel'] == p) /
               sum(1 for r in records if r['panel'] == p) for p in ('gate10', 'wider50')}


def scan_b():
    for path in sorted((ROOT / SCAN_B_DIR).glob('*/summary.json')):
        report = json.loads(path.read_text(encoding='utf-8'))
        if report.get('status') == 'passed':
            return report
    return None


def bridge13():
    protocol = load(BRIDGE13_DIR + '/protocol.json')
    for path in sorted((ROOT / BRIDGE13_DIR).glob('*/summary.json')):
        report = json.loads(path.read_text(encoding='utf-8'))
        if report.get('status') == 'passed':
            return protocol, report
    return protocol, None


def review_parts():
    path = ROOT / 'data/phase7/bridge_images/review.json'
    if not path.exists():
        return []
    r = json.loads(path.read_text(encoding='utf-8'))
    c = r['c_vs_b']
    return [f'<p class="summary"><b>Qualitative image check.</b> Because the teacher is not a perfect reference, we also rendered the teacher, B and C on the 30 earlier Phase 7 prompts with matched seeds (the regenerated teacher images matched the earlier renders byte for byte) and judged each image against its prompt. C was worse than B on <b>{c["worse"]}</b> prompts, similar on {c["same"]} and better on {c["better"]}. The failures appear at unchanged composition: hair smearing into the jaw, circuit-board traces dissolving, an incoherent house cross-section, misspelled signs (&ldquo;Happy Birtliday Emma&rdquo;, &ldquo;The Rusty Achor&rdquo;) where B is exact, and a missing alarm clock. B stays close to the teacher.</p>',
            '<p class="note">Unblinded review by an AI visual reviewer, one seed per prompt; a screen, not a benchmark.</p>']


def anchor_parts():
    site = ROOT.parent / 'niwaki-page/nihonga/images'
    if not (site / 'phase7-anchor-seed0.jpg').exists():
        return []
    return ['<h3 id="phase7-anchor">The attention bridge spells &ldquo;The Rusty Anchor&rdquo; on one seed</h3>',
            '<p class="summary">On the original seed, every earlier bridge wrote &ldquo;The Rusty <b>Achor</b>&rdquo;. The attention bridge is the first to write the sign correctly, like the teacher and B. Across four seeds, however, it is right once (B: three times), so attention moves spelling in the right direction without restoring it.</p>',
            '<div class="decision-examples"><figure class="comparison"><a href="images/phase7-anchor-seed0.jpg"><img loading="lazy" src="images/phase7-anchor-seed0.jpg" alt="The Rusty Anchor sign, seed 0: teacher, B and attention bridge correct; first and conditioned bridges write Achor"></a>'
            '<figcaption><b>Seed 0, same prompt and noise for every model</b> | teacher, B and the attention bridge write &ldquo;Anchor&rdquo;; both per-token bridges write &ldquo;Achor&rdquo;</figcaption></figure>'
            '<figure class="comparison"><a href="images/phase7-anchor-seeds.jpg"><img loading="lazy" src="images/phase7-anchor-seeds.jpg" alt="B versus the attention bridge on four seeds: B correct three times, the attention bridge once"></a>'
            '<figcaption><b>Four seeds, B (top) and the attention bridge (bottom)</b> | B 3/4 correct; attention bridge 1/4 (&ldquo;Achar&rdquo;, &ldquo;Ancher&rdquo;, garbled)</figcaption></figure></div>',
            table('Exactly spelled signs out of four seeds, by bridge',
                  ['Text', 'Teacher', 'B', 'Per-token (cond r1024)', 'Attention', '+ 150 text prompts', '8 heads + 450 text prompts'],
                  [['The Rusty Anchor', '4/4', '3/4', '0/4', '1/4', '1/4', '1/4'],
                   ['Happy Birthday Emma', '4/4', '3/4', '1/4', '1/4', '2/4', '2/4'],
                   ['Moonlight Motel', '3/4', '3/4', '1/4', '2/4', '2/4', '3/4'],
                   ['<b>All three</b>', '<b>11/12</b>', '<b>9/12</b>', '<b>2/12</b>', '<b>4/12</b>', '<b>5/12</b>', '<b>6/12</b>']]),
            '<div class="verdict"><p><b>Spelling recovers step by step, then plateaus below B.</b> More text data and a larger attention branch kept improving velocity (6.11% overall) but brought exact signs only from 5/12 to 6/12, against 9/12 for B. The rule registered before this last run said that 6/12 or fewer means a bridge alone does not carry spelling. <b>Strategy change:</b> choose and bridge the four blocks with this recipe, then heal the whole 28-block student on text-rich data, instead of perfecting each bridge on its own.</p></div>']


def bridge5_explainer():
    rows = [['Hidden state of every token (4,096 numbers)', 'yes', 'yes'],
            ['Other tokens (attention)', 'no: each token is transformed alone', 'yes: 8 heads'],
            ['Timestep (how noisy the image is)', 'no: same correction at every step', 'yes: scale and gate'],
            ['Token positions (RoPE)', 'no', 'yes'],
            ['Prompt key/value cache', 'no', 'yes: its own slot'],
            ['Weights', '2.1M', '25.2M (original block: 218M)']]
    return ('<h3 id="bridge-inputs">What this bridge sees, and what it ignores</h3>'
            '<p class="summary">The rank-256 bridge receives exactly one thing: each token&rsquo;s hidden state. It never looks at '
            'other tokens, does not know which denoising step it is in, and has no notion of where a token sits in the image. '
            'That is enough for block 5, whose update is small, but not for a block like 13. The attention bridge designed in '
            '<a href="#phase7-inside-bridge">Phase 7</a> adds each missing input.</p>'
            + table('What each bridge receives', ['Input', 'Rank-256 bridge (block 5)', 'Attention bridge (block 13, Phase 7)'], rows))


def inside_bridge_parts():
    return ['<h3 id="phase7-inside-bridge">Inside the attention bridge</h3>',
            '<p class="summary">The best bridge for block 13 is a miniature block: modulated attention followed by a modulated MLP, '
            'the same structure as the original, with a quarter of the attention heads and a much narrower MLP. Its attention '
            'output starts at zero, so a fresh attention branch leaves the conditioned MLP bridge unchanged.</p>',
            '<div class="charts bridge-diagrams">'
            '<figure class="chart"><figcaption><b>The attention bridge for block 13</b></figcaption><a href="bridge13-architecture.svg">'
            '<img src="bridge13-architecture.svg" width="560" height="1060" alt="Inputs: hidden states, timestep modulation, positions and a prefix cache. '
            'An attention branch with 8 heads updates image tokens; a conditioned MLP branch updates all tokens. About 25.2 million weights."></a></figure>'
            '<figure class="chart"><figcaption><b>RoPE: position as a rotation</b></figcaption><a href="rope-explainer.svg">'
            '<img src="rope-explainer.svg" width="760" height="1090" alt="One pair of query numbers drawn as an arrow, rotated by position times a fixed '
            'frequency; the attention score depends only on the distance between tokens; pairs turn at different speeds; in images some pairs follow '
            'the row and others the column."></a></figure></div>',
            '<p class="summary"><b>Timestep modulation: scale and gate.</b> At every step the model turns the timestep into one vector shared by '
            'all blocks: <code>[scale1, gate1, scale2, gate2]</code>, 4 &times; 4,096 numbers. A real block multiplies its normalised input by '
            '<code>(1 + scale)</code>, which turns each channel up or down for this noise level, and its output by <code>tanh(gate)</code>, which '
            'decides how much of the update to add. The bridge learns per-channel mixes of these same values (about 25,000 weights) and starts '
            'exactly like a real block: <code>scale_attn = scale1</code>, <code>gate_attn = gate1</code>, <code>scale_mlp = scale2</code>, '
            '<code>gate_mlp = gate2</code>. Prompt tokens use the row for a clean image (t = 0), as in the model.</p>',
            '<p class="summary"><b>Heads: 4,096 &rarr; 1,024 = 8 &times; 128.</b> Each token&rsquo;s query, key and value are projected to 1,024 '
            'numbers and split into 8 heads of 128. Each head compares tokens in its own 128-number space, so different heads can attend to '
            'different things; their outputs are joined and projected back to 4,096. The original block has 32 such heads. The bridge keeps '
            '128 per head, because the model&rsquo;s position code is built for that size, and saves cost with fewer heads.</p>',
            '<p class="summary"><b>RMSNorm on Q and K.</b> Before comparing tokens, each 128-number query and key is rescaled to a standard '
            'length, <code>x / &radic;mean(x&sup2;)</code>, times learned weights. An attention score is a dot product; without this, a few large '
            'vectors can make scores explode and the softmax lock onto one token. The original blocks do the same.</p>',
            '<p class="summary"><b>RoPE: position as a rotation.</b> On its own, attention does not know where a token is: shuffle the tokens and '
            'the scores do not change. RoPE pairs up each head&rsquo;s 128 numbers into 64 two-dimensional pairs and rotates every pair by '
            '<code>angle = position &times; frequency</code>. Two patches showing the same letter at different places get the same arrows pointing '
            'in different directions. When a rotated query meets a rotated key, their score depends only on the difference of the angles, '
            'that is, on <b>how far apart</b> the two tokens are, so a head can learn &ldquo;look at the patch just to my right&rdquo; once and use '
            'it anywhere in the image. Nothing about the rotation is learned: <code>frequency = 1 / 10000^(2i/d)</code> per pair, fast for some '
            'pairs (neighbours) and slow for others (coarse layout), like the hands of a clock. In Qwen-Image 2.1 the 64 pairs are split by axis: '
            '8 follow the image or text segment, 28 the row and 28 the column; image tokens get a (row, column) on their grid, centred at zero, '
            'and prompt tokens advance one step per word. The angles are precomputed once as a table. RoPE is applied to Q and K only, after '
            'RMSNorm; V, the content being passed along, is not rotated. This is why the attention bridge can help spelling: to write '
            '&ldquo;Anchor&rdquo;, the patch for &ldquo;c&rdquo; must know that &ldquo;n&rdquo; is just left of it and &ldquo;h&rdquo; just right.</p>',
            '<p class="summary"><b>Why rotations rather than classic position embeddings?</b> The classic approach adds a learned (or fixed '
            'sinusoidal) vector for each absolute position to every token once, at the input. RoPE differs in four ways that matter here. '
            '(1) <b>Relative by construction</b>: the score depends exactly on the offset between two tokens, so a pattern learned in one corner '
            'works everywhere; with absolute vectors the model has to infer offsets indirectly. (2) <b>Applied in every attention layer</b>, '
            'directly to queries and keys, so position information cannot fade as it travels through 32 blocks, and it never mixes into the '
            'content passed in V. (3) <b>Any grid size</b>: the formula works for any number of rows and columns, which suits a model that renders '
            '1024&times;1024, 1376&times;768 or 896&times;1184 images; a learned table only covers the sizes it was trained on. (4) <b>No extra '
            'weights</b>, and two-dimensional by simply splitting the pairs between rows and columns. RoPE is not perfect: positions far beyond '
            'the trained range still behave worse, but it has become the standard choice in modern diffusion transformers.</p>',
            table('The two bridges and the block they replace',
                  ['', 'Rank-256 bridge (block 5)', 'Attention bridge (block 13)', 'Original block 13'],
                  [['Looks at other tokens', 'no', 'yes, 8 heads &times; 128', 'yes, 32 heads &times; 128'],
                   ['Knows the timestep', 'no', 'yes (scale, gate)', 'yes (scale, gate)'],
                   ['Token positions', 'no', 'yes (RoPE)', 'yes (RoPE)'],
                   ['Prompt key/value cache', 'no', 'yes, own slot', 'yes'],
                   ['MLP', '4,096 &rarr; 256 &rarr; 4,096, GELU', '4,096 &rarr; 1,024 &rarr; 4,096, GELU, conditioned', '4,096 &rarr; 12,288 &rarr; 4,096, SwiGLU'],
                   ['Weights', '2.1M', '25.2M', '218.1M']])]


def healing_parts():
    rows = [['step 0 (C, no healing)', '6.11%', '6.08%', '6/12'], ['step 50', '6.11%', '6.06%', '5-6/12'],
            ['step 100', '6.06%', '6.00%', '6/12'], ['<b>step 150</b>', '<b>6.04%</b>', '<b>5.98%</b>', '<b>~7/12</b>'],
            ['step 200', '6.19%', '6.13%', '6-7/12']]
    return ['<h3 id="phase7-healing">Healing the whole student does not bring spelling back</h3>',
            '<p class="summary">If one bridge cannot carry block 13, perhaps the rest of the network can adapt around it. We healed the '
            'two-bridge student (blocks 5 and 13 removed) with LoRA adapters (rank 16) on the attention of all 30 remaining blocks, '
            'the bridges trainable too, on the saved recordings (450 text prompts). A first run with a higher learning rate made the '
            'velocity worse at every evaluation and kept only its best-velocity checkpoint, so its spelling could not be checked; it '
            'was stopped. The second run saved <b>every checkpoint</b> and judged each by both tests: velocity on 180 comparisons '
            '(quantitative) and exact spelling of three signs over four seeds (qualitative).</p>',
            table('Gentle LoRA healing: velocity and spelling at every checkpoint (teacher 11/12, B 9/12 exact signs)',
                  ['Checkpoint', 'Overall velocity', 'Wider (50)', 'Exact signs'], rows),
            '<div class="verdict"><p><b>Healing moves velocity slightly and spelling hardly at all.</b> The same seeds keep failing the '
            'same way at every step (&ldquo;Achar&rdquo;, &ldquo;Acher&rdquo;; &ldquo;Cnna&rdquo;, &ldquo;Enna&rdquo;). By the rule '
            'registered beforehand (8/12 or more), healing does not recover spelling. Removing block 13 takes something the model '
            'needs for spelling. The next step changes <b>which</b> blocks are removed: a text-aware scan measures what each block '
            'costs on text prompts, not on average velocity.</p></div>']


def block13_alone_parts():
    rows = [['The Rusty Anchor', '4', '3', '1', '2', '1'], ['Happy Birthday Emma', '4', '3', '2', '2', '2'],
            ['Moonlight Motel', '3', '3', '3', '~1', '~1-2'],
            ['<b>Total / 12</b>', '<b>11</b>', '<b>9</b>', '<b>6</b>', '<b>~5</b>', '<b>~4-5</b>']]
    return ['<h3 id="phase7-block13-alone">Is it block 13, or block 13 on top of a weak block-5 bridge?</h3>',
            '<p class="summary">B already spells a little worse than the teacher, and its block-5 bridge is the simplest one. Errors could '
            'compound. To separate the two, block 13 was removed from the <b>intact teacher</b> (block 5 kept): once as a plain skip and once '
            'with the best block-13 bridge.</p>',
            table('Exactly spelled signs out of four seeds', ['Text', 'Teacher', 'B', 'C (5 + 13)', 'Teacher &minus; 13 (skip)', 'Teacher &minus; 13 (bridge)'], rows),
            '<div class="verdict"><p><b>Block 13 breaks spelling on its own.</b> Without it, even the intact teacher drops to about 5/12, '
            'with the same errors (&ldquo;Achor&rdquo;, &ldquo;Enna&rdquo;). The block-5 bridge costs much less (11 &rarr; 9). Which blocks '
            'are removed matters more than how they are replaced, so the next cut is chosen by measuring what each block costs on text '
            'prompts.</p></div>']


def text_selection_parts():
    rows = [['The Rusty Anchor', '4', '3', '2', '3', '<b>4</b>', '3', '2', '1'],
            ['Happy Birthday Emma', '4', '3', '2', '3-4', '<b>4</b>', '3', '3', '4'],
            ['Moonlight Motel', '3', '3', '~1', '~2', '~2-3', '~1', '~2-3', '~2'],
            ['<b>Total / 12</b>', '<b>11</b>', '<b>9</b>', '<b>~5</b>', '<b>~8</b>', '<b>~10-11</b>', '<b>~7</b>', '<b>~7-8</b>', '<b>~7</b>']]
    return ['<h3 id="phase7-text-selection">Choosing the next block by spelling</h3>',
            '<p class="summary">A scan skipped each block of B and measured velocity on 80 saved text prompts. It filters overall damage but '
            'cannot see spelling: block 13, which breaks it, sits mid-table, because letters are a tiny fraction of the pixels. So the five '
            'blocks with the lowest velocity cost and not next to block 5 (17, 25, 20, 18, 16) were each skipped inside B and rendered on the '
            'text prompts with four seeds. Block 25 ranked second only after the first images had been launched; it was added afterwards, '
            'as recorded in a dated addendum.</p>',
            table('Exactly spelled signs out of four seeds, with one block skipped inside B (no bridge)',
                  ['Text', 'Teacher', 'B', 'Teacher &minus; 13', 'B &minus; 17', 'B &minus; 20', 'B &minus; 18', 'B &minus; 16', 'B &minus; 25'], rows),
            '<div class="verdict"><p><b>Block 20 keeps spelling.</b> Skipped with no bridge at all, the student writes both long signs correctly '
            'on every seed, at the level of B or better; block 17 also holds up well. Blocks 20 and 13 cost about the same by velocity, yet one '
            'keeps spelling and the other breaks it. Block 25 shows a systematic error of its own (&ldquo;Anchar&rdquo;).</p></div>']


def combination_parts():
    rows = [['5 (B)', '31', '5.40%', '10.0 / 2.6 / 3.6', '9/12'],
            ['<b>5, 20</b>', '30', '10.46%', '12.3 / 5.4 / <b>13.6</b>', '<b>~11/12</b>'],
            ['5, 20, 17', '29', '12.64%', '15.2 / 6.6 / 16.1', '~6/12'],
            ['5, 20, 17, 25', '28', '15.17%', '17.1 / 8.2 / 20.1', '~5/12'],
            ['5, 20, 17, 16', '28', '15.63%', '20.3 / 7.7 / 18.9', '~0/12']]
    return ['<h3 id="phase7-combinations">Removing the chosen blocks together</h3>',
            '<p class="summary">Blocks 20 and 17 each keep spelling when skipped alone. Removals can compound, so combinations were skipped '
            'together inside B, with no new bridges, and judged by velocity on 180 comparisons and by the four-seed text grid.</p>',
            table('Skipped-block combinations inside B: velocity and spelling', ['Removed', 'Blocks', 'Overall velocity', 'Early / middle / late', 'Exact signs'], rows),
            '<div class="verdict"><p><b>Removing 5 and 20 keeps spelling at teacher level, with no bridge for 20.</b> Its velocity error doubles, '
            'almost entirely at the last denoising moment: block 20 refines detail after the letters have formed. The two tests disagree, so '
            'general images decide next. Removals compound: 17 keeps spelling alone but not on top of 20, and four blocks skipped without '
            'bridges break text, completely when 16 sits next to 17.</p></div>',
            '<p class="summary"><b>General images with 5 and 20 removed.</b> On the 30 standard prompts it is worse than B on 9, similar on 20 '
            'and better on 1. Fine detail and texture are intact (lace, circuit board, skin, hair). The losses are mid-scale structure: a '
            'checkerboard breaks apart, simple solids lose their geometry, an alarm clock disappears, crowded scenes and figures lose coherence, '
            'and one letter is malformed (so about 10/12 exact signs). The first block-13 bridge was worse on 15 of these prompts and the best '
            'per-token one on 8: removing 20 with no bridge already matches the best block-13 bridge on general quality while keeping '
            'spelling. A bridge for block 20 is the next step.</p>']


def block20_bridge_parts():
    rows = [['no bridge for 20', '10.46%', '~10/12', '9'], ['bridge, step 200', '7.92%', '<b>~11/12</b>', '7'],
            ['<b>bridge, step 600</b>', '<b>7.76%</b>', '~10/12', '<b>5</b>']]
    return ['<h3 id="phase7-block20-bridge">A bridge for block 20</h3>',
            '<p class="summary">Block 20 got the recipe that worked best for block 13 (timestep-conditioned MLP plus an 8-head attention '
            'branch, trained end to end on 450 text and general recordings), starting from &ldquo;skip block 20&rdquo;. Every 100 updates it '
            'was saved and measured; two checkpoints were rendered on the 30 standard prompts and the four-seed text grid.</p>',
            table('Blocks 5 and 20 removed: velocity, spelling and general quality', ['Block 20', 'Overall velocity', 'Exact signs', 'Worse than B (of 30)'], rows),
            '<div class="verdict"><p><b>Step 600 is chosen.</b> It keeps spelling above B and restores most of the structure lost without a '
            'bridge: simple solids, the four friends, the cyclist, the melting clocks. A missing alarm clock and some clutter remain. The '
            'block-5 bridge was then rebuilt with the same recipe, with this block-20 bridge frozen in place.</p></div>'] + block5_bridge_parts()


def block5_bridge_parts():
    rows = [['old rank-256 bridge (2.1M weights)', '<b>7.76%</b>', '9'], ['new bridge, step 100', '8.09%', '9'],
            ['<b>new bridge, step 200</b>', '8.09%', '<b>10</b>'], ['new bridge, step 300', '8.06%', '8'],
            ['new bridge, step 400', '8.11%', '9'], ['new bridge, step 500', '8.20%', '6'], ['new bridge, step 600', '7.93%', '7']]
    return ['<h3 id="phase7-block5-bridge">A new bridge for block 5</h3>',
            '<p class="summary">Block 5 had kept its first bridge, a plain rank-256 bottleneck fitted per token in Phase 5. It was rebuilt with '
            'exactly the architecture and training of the block-20 bridge (timestep-conditioned rank-1024 MLP plus an 8-head attention branch '
            'with its own prefix cache, about 25.2 million weights, trained end to end against the teacher velocity for 600 updates), with the '
            'block-20 bridge (step 600) frozen. Every 100 updates the bridge was saved, measured on 180 velocity comparisons, and rendered on '
            'the four-seed text grid; the best spelling checkpoint was also rendered on the 25 general prompts.</p>',
            table('Blocks 5 and 20 removed, block-20 bridge fixed: which block-5 bridge', ['Block 5', 'Overall velocity', 'Exact signs (of 11 legible)'], rows),
            '<div class="decision-examples"><figure class="comparison"><a href="images/phase7-block5-anchor.jpg"><img loading="lazy" src="images/phase7-block5-anchor.jpg" '
            'alt="The Rusty Anchor, four seeds: teacher, B, old block-5 bridge (seed 0 Anchar) and new block-5 bridge at step 200 (all correct)"></a>'
            '<figcaption><b>The Rusty Anchor, four seeds</b> | the old block-5 bridge writes &ldquo;Anchar&rdquo; in seed 0; the new bridge at step 200 '
            'spells all four headlines but adds a garbled second plaque in seed 3</figcaption></figure>'
            '<figure class="comparison"><a href="images/phase7-block5-poster.jpg"><img loading="lazy" src="images/phase7-block5-poster.jpg" '
            'alt="GOLDEN RIVER poster, four seeds: the new bridge writes the tagline Live in Berlin legibly in seed 0"></a>'
            '<figcaption><b>Poster, four seeds</b> | the new bridge writes the tagline &ldquo;Live in Berlin&rdquo; in seed 0, which the old '
            'bridge garbles</figcaption></figure></div>',
            '<div class="verdict"><p><b>Decision: the new block-5 bridge at step 200 replaces the old one.</b> It spells one more sign '
            '(10 of 11 against 9; the seed-2 motel plaque is cut off by the frame in every row) and writes the poster tagline, while on the 25 '
            'general prompts it is indistinguishable from the old bridge. Its velocity is 0.3 points worse, but velocity cannot see letters: '
            'the checkpoint with the best velocity (step 600) is among the worst at spelling. The extra 23 million weights are about 0.3% of '
            'the model. One sign over four seeds is close to noise, so the next text grids will show whether the gain holds.</p></div>',
            '<p class="summary"><b>The 30-block student</b> is now B plus both attention bridges (block 5 at step 200, block 20 at step 600). '
            'Next it is healed: a rank-16 LoRA on the attention of all blocks, trained jointly with both bridges at a low learning rate, '
            'with every checkpoint saved and judged by velocity and by spelling. Removals three and four are then chosen by a new text-aware '
            'scan inside the healed student.</p>']


def variant_review_parts():
    path = ROOT / 'data/phase7/bridge_variant_images/review.json'
    if not path.exists():
        return []
    r = json.loads(path.read_text(encoding='utf-8'))
    c, f = r['cond_r1024_vs_B'], r['first_C_vs_B_earlier_review']
    return [f'<p class="summary"><b>Images.</b> On the same 30 prompts and seeds, the conditioned rank-1024 bridge was worse than B on <b>{c["worse"]}</b> prompts, similar on {c["same"]} and better on {c["better"]}, against {f["worse"]} worse for the first bridge. It fixes the hair smearing, spells &ldquo;Happy Birthday Emma&rdquo; correctly, keeps the alarm clock and one clean pair of glasses, and stops duplicating the pot. It does not recover fine local detail (lace and circuit board, the peacock eye, the star), the bent spoon or the melting clocks, and &ldquo;The Rusty Anchor&rdquo; is still misspelled.</p>',
            '<p class="note">Unblinded AI visual review, two reviewers, one seed per prompt; reviewers were not fully consistent, so column-by-column comparisons within a sheet are the reliable signal.</p>'] + multiseed_parts()


def multiseed_parts():
    path = ROOT / 'data/phase7/multiseed_text/review.json'
    if not path.exists():
        return []
    r = json.loads(path.read_text(encoding='utf-8'))
    rows = [[html.escape(text), f"{c['teacher']}/4", f"{c['B']}/4", f"<b>{c['C']}/4</b>"] for text, c in r['counts_correct_of_4'].items()]
    return ['<h3 id="phase7-multiseed-text">Is the misspelling systematic? Four seeds per text prompt</h3>',
            '<p class="summary">One seed per prompt cannot separate a real loss from bad luck, so we rendered every text prompt with three more seeds for the teacher, B and the best bridge (conditioned rank 1024, 3,000 updates) and counted exact spellings.</p>',
            table('Exactly spelled signs out of four seeds', ['Text', 'Teacher', 'B', 'C (best bridge)'], rows),
            '<div class="verdict"><p><b>The loss is systematic.</b> Without block 13 the model loses the spelling of longer, less common Latin words (&ldquo;The Rusty Achor&rdquo;, &ldquo;Happy Bithday Enna&rdquo;, &ldquo;MonLght Motel&rdquo;), while short or common phrases and Chinese survive. The earlier correct &ldquo;Emma&rdquo; was seed luck. A bridge that sees neither the prompt nor neighbouring letters cannot carry this; the next candidate adds a small attention layer.</p></div>']


def variants_parts():
    paths = sorted((ROOT / 'data/phase7/block13_bridge_v2').glob('*/summary.json'))
    if not paths:
        return []
    r = json.loads(paths[0].read_text(encoding='utf-8'))
    first = bridge13()[1]
    names = {'plain_r1024': 'Plain, rank 1024', 'cond_r256': 'Timestep-conditioned, rank 256', 'cond_r1024': 'Timestep-conditioned, rank 1024',
             'cond_r1024_long': 'Conditioned rank 1024, 3,000 updates', 'cond_local_r1024': 'Conditioned rank 1024 + 3×3 local mixing, 3,000 updates',
             'e2e_cond': 'Same, trained on the final output (no gain; best at its start)', 'e2e_attn': '+ attention branch, trained on the final output',
             'e2e_attn_text': '+ 150 recorded text prompts, 600 more updates',
             'e2e_attn8_text': '<b>8 attention heads, 450 recorded text prompts</b>'}

    def row(label, params, c):
        o = c['overall']['teacher_relative_l2']['mean']
        g10, w50 = (c['panels'][p]['teacher_relative_l2']['mean'] for p in ('gate10', 'wider50'))
        e, m, l = (c['stages'][s]['teacher_relative_l2']['mean'] for s in ('early', 'middle', 'late'))
        return [label, params, f'<b>{pct(o)}</b>', pct(g10), pct(w50), f'{pct(e, 1)} / {pct(m, 1)} / {pct(l, 1)}']
    rows = [row('Plain, rank 256 (first bridge)', '2.1M', first['candidate'])]
    rows += [row(names.get(n, n), f"{v['parameters'] / 1e6:.1f}M", v) for n, v in r['variants'].items()]
    for later in sorted(list((ROOT / 'data/phase7/block13_bridge_v3').glob('*/summary.json')) +
                        list((ROOT / 'data/phase7/block13_bridge_e2e').glob('*/summary.json')) +
                        list((ROOT / 'data/phase7/block13_bridge_e2e_text').glob('*/summary.json')) +
                        list((ROOT / 'data/phase7/block13_bridge_e2e_text8').glob('*/summary.json'))):
        extra = json.loads(later.read_text(encoding='utf-8'))
        if extra.get('status') == 'passed':
            rows += [row(names.get(n, n), f"{v['parameters'] / 1e6:.1f}M", v) for n, v in extra['variants'].items()]
    return ['<h3 id="phase7-bridge-variants">Better bridges for block 13: conditioning helps, attention helps more</h3>',
            '<p class="summary">Every real block receives the model’s shared timestep modulation (scale and gate values that tell it how noisy the image is); the first bridge ignored it, and its errors were worst at early, high-noise steps. We fitted three variants on identical data: a 4× wider plain bridge, and a bridge that mirrors a block’s MLP branch, h + tanh(gate) · MLP(LN(h) · (1 + scale)), with scale and gate learned mixes of the model’s own modulation, at rank 256 and 1024.</p>',
            table('Block-13 bridge variants inside B: velocity error against the teacher (B with block 13 kept: 5.40% overall)',
                  ['Bridge', 'Parameters', 'Overall', 'Check (10)', 'Wider (50)', 'Early / middle / late'], rows),
            '<div class="verdict"><p><b>Capacity alone does nothing; conditioning does.</b> The conditioned rank-1024 bridge recovers about a third of the gap between the first bridge and B on the wider prompts. Training three times longer, or adding 3×3 local mixing over the image-token grid, moves the error by only about 0.05 points: the per-token fitting objective has reached its ceiling. Training the same bridge on the final velocity instead (gradients through the frozen blocks 14-31) did not help either. Adding a small attention branch (4 heads, image tokens attending to the prompt and to each other, with its own prefix cache) and training it on the final velocity gave the first clear step: 6.42% overall and 6.29% on the wider prompts, still improving when training stopped. Continuing it on 200 newly recorded training prompts, 150 of them text, brought it to 6.25% overall and 6.21% on the wider prompts, and cut its single worst comparison from about 26% to 18.5% (B: 17.9%). Growing the attention branch to 8 heads and training on 450 text prompts reached 6.11% overall and 6.08% on the wider prompts, 0.7 points from B.</p></div>'] + inside_bridge_parts() + variant_review_parts() + anchor_parts() + healing_parts() + block13_alone_parts() + text_selection_parts() + combination_parts() + block20_bridge_parts()


def bridge13_section(protocol, report):
    g, base = protocol['velocity_gates'], protocol['base']
    c = report['candidate']
    o = c['overall']['teacher_relative_l2']
    gate10 = c['panels']['gate10']['teacher_relative_l2']['mean']
    wider = c['panels']['wider50']['teacher_relative_l2']['mean']
    stages = {s: c['stages'][s]['teacher_relative_l2']['mean'] for s in ('early', 'middle', 'late')}
    skip = base['skip_block_13_gate10_mean_relative_l2']
    removed = (skip - gate10) / (skip - base['gate10_mean_relative_l2'])
    checks = [('Check prompts (10)', gate10, g['gate10_mean_max']), ('Wider prompts (50)', wider, g['wider50_mean_max'])]
    checks += [(s.capitalize() + ' moment', stages[s], g['stage_mean_max'][s]) for s in stages]
    checks += [('90th percentile', o['p90'], g['overall_p90_max']), ('Worst comparison', o['maximum'], g['overall_maximum_max'])]
    rows = [[name, f'<b>{pct(v)}</b>', pct(limit), 'pass' if v <= limit else '<b>fail</b>'] for name, v, limit in checks]
    rows.append(['Skip penalty removed (check prompts)', f'<b>{100 * removed:.0f}%</b>', '50%', 'pass' if removed >= .5 else '<b>fail</b>'])
    passed = all(v <= limit for _, v, limit in checks) and removed >= .5
    fit = report['fit']
    parts = ['<h3 id="phase7-block13">Learned bridge for block 13: the velocity gates fail</h3>',
             f'<p class="summary">Block 13 ranked first inside the accepted student, so we replaced it with a rank-256 bridge while keeping the block-5 bridge and every other weight frozen. Pass rules were committed before the run. The bridge was fitted to block 13’s own inputs and outputs captured inside the student: 60 training prompts at three moments plus 600 early training examples. The 10 check prompts were used only to choose the checkpoint. The fit worked on its own terms: the hidden-state objective fell from {fit["identity_validation"]:.4f} (plain skip) to {fit["selected_validation"]:.4f}, a {100 * (1 - fit["selected_validation"] / fit["identity_validation"]):.0f}% reduction.</p>',
             table('Candidate C (blocks 5 and 13 bridged, 30 blocks): velocity error against the teacher and the preregistered limits',
                   ['Measure', 'C', 'Limit', 'Result'], rows),
             *review_parts(),
             f'<div class="verdict"><p><b>{"All velocity gates pass." if passed else "Candidate C fails the velocity gates; it is not promoted and no images or timings were produced for it."}</b> Skipping block 13 gives {pct(skip)} on the check prompts; the bridge brings it only to {pct(gate10)}, against {pct(base["gate10_mean_relative_l2"])} for the accepted student. Middle and late moments stay within limits; the early moment does not. A likely reason: the bridge acts on each token separately, so it cannot reproduce the cross-token attention a mid-network block performs, and most of the damage of removing block 13 remains.</p></div>']
    return parts


def table(caption, headers, rows):
    head = ''.join(f'<th scope="col">{html.escape(h)}</th>' for h in headers)
    body = ''.join('<tr>' + ''.join(f'<td>{c}</td>' for c in r) + '</tr>' for r in rows)
    label = html.escape(caption, quote=True)
    return (f'<div class="table-scroll" tabindex="0" role="region" aria-label="{label}"><table><caption>{label}</caption>'
            f'<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>')


def build(a, lat, b, q):
    rows_a = scan_rows(a)
    base_a = a['baseline']['teacher_relative_l2']['mean']
    panels_a = panel_baseline(a)
    best_a = rows_a[0]
    ta, tb = lat['summary']['teacher'], lat['summary']['candidate_A']
    speed = lat['paired_speedup']
    a_parts = ['<h3 id="phase7-a-tested">Candidate A under closer test: rejected</h3>',
               '<p class="summary">The plan was to use candidate A (blocks 4-5 replaced, 30 blocks) as the base for further pruning. Before building on it we measured three things: its image quality in a larger blinded test, its real speed, and which block could go next inside it. The blinded image test decided the question.</p>']
    adh, txt = q['counts']['adherence'], q['counts']['text']
    pairs = q['pairs']
    wins = sum(r['clear_pass_delta'] > 0 for r in pairs if r['group'] != 'text_rendering')
    losses = sum(r['clear_pass_delta'] < 0 for r in pairs if r['group'] != 'text_rendering')
    text_losses = sum(r['clear_pass_delta'] < 0 for r in pairs if r['group'] == 'text_rendering')
    text_wins = sum(r['clear_pass_delta'] > 0 for r in pairs if r['group'] == 'text_rendering')
    ci = q['paired_bootstrap_95pct']
    a_parts.append('<h3 id="phase7-blind">Blinded image test: candidate A loses to the teacher</h3>')
    a_parts.append(f'<p class="summary">Before viewing images we registered <b>159 fresh validation prompts</b> (35 counting, 34 spatial, 30 color, 30 multiple-object, 30 text), one seed each, rendered by the teacher and A at 40 steps. Each image was scored pass, fail or uncertain against one fixed question, with left/right positions randomized and the identity key committed by hash and revealed only after all 318 scores were committed.</p>')
    a_parts.append(table('Blinded 159-prompt panel: clear passes for the teacher and candidate A',
                       ['Panel', 'Teacher pass', 'A pass', 'A better', 'A worse', 'Gate'],
                       [['Adherence (129)', adh['teacher'].get('pass', 0), adh['candidate_A'].get('pass', 0), wins, losses,
                         f"gain {100*q['adherence_gain_fraction']:+.1f} points, 95% CI {100*ci[0]:+.1f} to {100*ci[1]:+.1f}; needs lower bound ≥ -5: <b>{'pass' if q['adherence_noninferiority_passed'] else 'fail'}</b>"],
                        ['Text (30)', txt['teacher'].get('pass', 0), txt['candidate_A'].get('pass', 0), text_wins, text_losses,
                         f"A must be within 1 of the teacher: <b>{'pass' if q['text_gate_passed'] else 'fail'}</b>"]]))
    a_parts.append(f'<div class="verdict"><p><b>Candidate A is rejected as a base for further pruning.</b> It fails both image gates as well as the velocity gate; only the speed gate passes. The earlier impression that A matched the teacher came from a small unblinded screen. Further pruning starts from the accepted one-block student.</p></div>')
    a_parts.append('<p class="note">One model-blind AI reviewer and one seed per prompt. The 10 and 50 validation prompts used for velocity are development data, not an untouched final test.</p>')
    a_parts.append('<h3 id="phase7-latency">Measured speed of candidate A</h3>')
    a_parts.append(f'<p class="summary">{len(speed["ratios"])} balanced, randomized teacher/A pairs on one H100, 1024×1024, 40 steps, BF16, same prompts and seeds, warm models. Median end-to-end time: <b>{ta["median_seconds"]:.2f} s</b> for the teacher and <b>{tb["median_seconds"]:.2f} s</b> for A. Paired speedup <b>{speed["median"]:.3f}×</b> (bootstrap 95% CI {speed["bootstrap_95pct"][0]:.3f}-{speed["bootstrap_95pct"][1]:.3f}). Peak allocated memory: {ta["peak_allocated_gib"]:.2f} GiB vs {tb["peak_allocated_gib"]:.2f} GiB.</p>')
    a_parts.append('<h3 id="phase7-scan-a">Which block next inside A? The importance scan</h3>')
    a_parts.append(f'<p class="summary">A alone measures <b>{pct(base_a)}</b> on all 180 comparisons ({pct(panels_a["wider50"])} on the 50 wider prompts and {pct(panels_a["gate10"])} on the 10 check prompts, reproducing the earlier 7.41% and 7.97%).</p>')
    a_parts.append(table('Candidate A scan: velocity error against the teacher with one extra block bypassed (lower is better); A alone measures ' + pct(base_a),
                       ['Block', 'A + bypass', 'Early', 'Middle', 'Late', 'p90'],
                       [[r['block'], f"<b>{pct(r['mean'])}</b>", pct(r['early'], 1), pct(r['middle'], 1), pct(r['late'], 1), pct(r['p90'], 1)] for r in rows_a]))
    a_parts.append(f'<div class="verdict"><p><b>Inside A, no single bypass stays within the gate.</b> The gentlest, block {best_a["block"]}, moves A from {pct(base_a)} to {pct(best_a["mean"])}. The middle blocks (6-17) cost least; blocks 0 and 31 cannot be skipped. A low bypass score only nominates a block for a learned bridge; it is not a removal.</p></div>')
    parts = a_parts + ['<h3 id="phase7-adaptive">Which block next? Importance measured inside the accepted student</h3>',
             '<p class="summary">The original ranking measured blocks in the intact teacher. After a cut, surviving blocks may compensate or become more redundant, so we measure again inside the smaller student. Each trial <b>bypasses one surviving block</b>, compares the prediction with the teacher on the same <b>60 development prompts at early, middle and late moments</b> (180 comparisons), then restores the block. Bypasses never accumulate and no weight is updated. Before any trial, all 90 intact-teacher controls reproduced exactly and the student reproduced its saved predictions.</p>',
             '<p class="summary">Because A is rejected, its scan describes a model we will not build on. The same scan is repeated inside <b>candidate B</b>: the accepted one-block student, 31 blocks plus the pretrained block-5 bridge. Block 4 is eligible again in B. Everything after this point starts from B.</p>']
    if b is None:
        parts.append('<p class="summary"><b>Candidate B scan: running.</b> Its ranking will be compared with A\'s below once it completes.</p>')
    else:
        rows_b = scan_rows(b)
        base_b = b['baseline']['teacher_relative_l2']['mean']
        rank_a = {r['block']: i + 1 for i, r in enumerate(rows_a)}
        parts.append(f'<p class="summary">Candidate B alone measures <b>{pct(base_b)}</b> against the teacher on these 180 comparisons (A: {pct(base_a)}). Each row adds one bypass to B; the last columns show the same block bypassed inside A.</p>')
        parts.append(table('Candidate B scan: velocity error against the teacher with one extra block bypassed (lower is better), next to the same bypass in candidate A',
                           ['Block', 'B + bypass', 'Early', 'Middle', 'Late', 'Rank in A', 'A + bypass'],
                           [[r['block'], f"<b>{pct(r['mean'])}</b>", pct(r['early'], 1), pct(r['middle'], 1), pct(r['late'], 1),
                             rank_a.get(r['block'], 'n/a (replaced)'),
                             next((pct(x['mean']) for x in rows_a if x['block'] == r['block']), 'n/a')] for r in rows_b]))
        within = [r['block'] for r in rows_b if 100 * r['mean'] <= GATE]
        top_b = [r['block'] for r in rows_b[:5]]
        top_a = [r['block'] for r in rows_a[:5]]
        shared = sorted(set(top_b) & set(top_a))
        parts.append(f'<div class="verdict"><p><b>In B, the gentlest bypass is block {rows_b[0]["block"]}: {pct(base_b)} → {pct(rows_b[0]["mean"])}.</b> '
                     + (f'Bypasses within the {GATE:.2f}% gate without any bridge: blocks {", ".join(map(str, within))}. ' if within else f'No single bypass stays within the {GATE:.2f}% gate; any further removal needs a learned bridge. ')
                     + f'Top five in B: {", ".join(map(str, top_b))}; in A: {", ".join(map(str, top_a))}'
                     + (f' (shared: {", ".join(map(str, shared))}).' if shared else ' (none shared).') + '</p></div>')
    b13_protocol, b13 = bridge13()
    if b13 is not None:
        parts.extend(bridge13_section(b13_protocol, b13))
        parts.extend(variants_parts())
    scan = '\n    '.join(parts)

    if b is None:
        status_text = ('<p class="summary"><b>Research direction:</b> the accepted model is still the one-block student (block 5 replaced by its pretrained bridge). '
                       'Candidate A (blocks 4-5) failed the velocity gate and the blinded image test, so the importance scan is being repeated inside the accepted student before choosing a learned replacement. '
                       f'A\'s measured speedup is {speed["median"]:.3f}×.</p>')
    else:
        next_step = ('A rank-256 learned bridge for block 13 then failed the preregistered velocity gates, so the accepted model is unchanged. '
                     if bridge13()[1] is not None else
                     'The next step is a learned bridge for a shortlisted block, with predeclared velocity, image and latency gates. ')
        status_text = ('<p class="summary"><b>Research direction:</b> the accepted model is still the one-block student. '
                       f'Importance scans inside both candidate A and the accepted student are complete; block {scan_rows(b)[0]["block"]} is the gentlest bypass in the accepted student. '
                       + next_step +
                       f'A\'s measured speedup is {speed["median"]:.3f}×.</p>')
    status = (status_text + '\n    <p class="note">Velocity errors are measured on saved teacher inputs at three denoising moments, not on finished images. '
              'Image results come from one model-blind AI reviewer with one seed per prompt.</p>')
    record = dict(candidate_A=dict(request_key=a['request_key'], baseline_mean_relative_l2=base_a, baseline_by_panel=panels_a,
                                   blocks=rows_a),
                  candidate_B=None if b is None else dict(request_key=b['request_key'], baseline_mean_relative_l2=b['baseline']['teacher_relative_l2']['mean'],
                                                         replay_max_relative_l2=b.get('candidate_replay_max_relative_l2'), blocks=scan_rows(b)),
                  candidate_B_status='running' if b is None else 'passed',
                  comparisons_per_block=a['blocks'][0]['overall']['comparisons'], gate_percent=GATE)
    latency = dict(request_key=lat['request_key'], summary=lat['summary'], paired_speedup={k: v for k, v in speed.items() if k != 'ratios'})
    record['blind_quality_candidate_A'] = {k: v for k, v in q.items() if k != 'pairs'}
    b13_protocol, b13 = bridge13()
    if b13 is not None:
        record['block13_bridge'] = dict(request_key=b13['request_key'], protocol=b13_protocol, fit=b13['fit'],
                                        candidate={k: v for k, v in b13['candidate'].items() if k != 'records'})
    return scan, status, record, latency


def replace_between(text, marker, content):
    start, end = f'<!-- {marker} -->', f'<!-- /{marker} -->'
    i, j = text.index(start) + len(start), text.index(end)
    return text[:i] + '\n    ' + content + '\n    ' + text[j:]


def ensure_markers(text):
    if '<!-- p4-bridge -->' not in text:
        anchor = 'Open either diagram for the standalone SVG.</p>'
        i = text.index(anchor) + len(anchor)
        text = text[:i] + '\n    <!-- p4-bridge --><!-- /p4-bridge -->' + text[i:]
    if '<!-- p7-scan -->' not in text:
        i = text.index('<h3 id="phase7-adaptive">')
        j = text.index('</p>', text.index('<p class="note">The existing 10 and 50 validation prompts', i)) + 4
        text = text[:i] + '<!-- p7-scan --><!-- /p7-scan -->' + text[j:]
    if '<!-- p7-status -->' not in text:
        i = text.index('<p class="summary"><b>Research direction:</b>')
        j = text.index('</p>', text.index('<p class="note">Velocity errors are measured', i)) + 4
        text = text[:i] + '<!-- p7-status --><!-- /p7-status -->' + text[j:]
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site', default=str(ROOT.parent / 'niwaki-page'))
    args = parser.parse_args()
    site = Path(args.site)
    a, lat, b, q = load(SCAN_A), load(LATENCY_A), scan_b(), load(QUALITY_A)
    scan, status, record, latency = build(a, lat, b, q)
    today = datetime.date.today()
    stamp = f'<time datetime="{today.isoformat()}">{today.day} {today:%B %Y}</time>'
    edits = {
        'Candidate A is the provisional baseline for adaptive importance screening, not a confirmed release.':
            'Candidate A is not accepted; the one-block student remains the selected model.',
        'Phase 7 now plans to recalculate importance in candidate A across all 30 surviving blocks, then test a learned replacement and rerank after each accepted removal. This conditional scan has not run. A larger blinded quality comparison, measured latency, a speed-distillation LoRA':
            'Importance has been recalculated inside candidate A and is being recalculated inside the accepted one-block student; a learned replacement follows, with reranking after each accepted removal. Scoring of the blinded 159-prompt comparison, a speed-distillation LoRA',
        'Thirty blocks is the research baseline. Further pruning needs a new ranking':
            'One block removed is still the accepted model. The next cut is ranked inside it',
        'It now plans to recalculate importance in the changed student before choosing another removal.':
            'Importance is now recalculated inside the smaller student before choosing another removal.',
        'a learned replacement follows, with reranking after each accepted removal. Scoring of the blinded 159-prompt comparison, a speed-distillation LoRA':
            'a learned replacement follows, with reranking after each accepted removal. A speed-distillation LoRA',
        'Candidate A is not accepted; the one-block student remains the selected model.</dd>':
            'Candidate A is not accepted: in a blinded 159-prompt test it fell behind the teacher on adherence and text. The one-block student remains the selected model.</dd>',
        'it looks like the teacher and renders all five text prompts correctly.</p>':
            'it looked like the teacher and rendered all five text prompts correctly. The larger <a href="#phase7-blind">blinded 159-prompt test</a> then showed A falling behind the teacher, so A is rejected as a base.</p>',
        'A promising two-block cut. Reassess the remaining network':
            'A two-block cut that fails blinded review. Reassess from one block',
    }
    import re
    for path in (site / 'nihonga/index.html', site / 'tools/build_nihonga.py'):
        text = ensure_markers(path.read_text(encoding='utf-8'))
        for old, new in edits.items():
            text = text.replace(old, new)
        is_template = path.suffix == '.py'
        if '{' in scan + status or '}' in scan + status:
            raise RuntimeError('Generated HTML must not contain braces (the build script embeds it in f-strings).')
        text = replace_between(text, 'p4-bridge', bridge5_explainer())
        text = replace_between(text, 'p7-scan', scan)
        text = replace_between(text, 'p7-status', status)
        text = re.sub(r'(Research notebook \| updated )<time datetime="[0-9-]+">[^<]+</time>', lambda m: m.group(1) + stamp, text)
        if is_template:
            text = re.sub(r"'updated': '[0-9-]+'", f"'updated': '{today.isoformat()}'", text)
        path.write_text(text, encoding='utf-8')
    results_path = site / 'nihonga/results.json'
    results = json.loads(results_path.read_text(encoding='utf-8'))
    results['updated'] = today.isoformat()
    results['scope'] = ('Development pilots. The selected model is the one-block student (block 5 bridged). Phase 7 candidate A (blocks 4-5) failed the velocity gate; '
                        'conditional importance scans and A latency are measured on development prompts. No independent image-quality benchmark yet.')
    results['phase7_conditional_importance'] = record
    results['phase7_latency_candidate_A'] = latency
    results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print('Updated site; candidate B', record['candidate_B_status'])


if __name__ == '__main__':
    main()
