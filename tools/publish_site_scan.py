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
    site = ROOT.parent / 'josejuanmartinez.github.io/research/nihonga/images'
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
            '<p class="summary"><b>The 30-block student</b> is B plus both attention bridges (block 5 at step 200, block 20 at step 600).</p>',
            '<figure class="chart"><figcaption><b>The 30-block student before healing, and after healing with the LoRA fused</b></figcaption>'
            '<a href="student30-architecture.svg"><img loading="lazy" src="student30-architecture.svg" width="1200" height="1270" '
            'alt="Left: 32 slots, 30 original blocks plus attention bridges in slots 5 and 20. Middle: healing trains a rank-16 LoRA on q, k, v '
            'and out of the 30 blocks together with both bridges, then merges each LoRA into its weight matrix, W prime equals W plus B times A. '
            'Right: the same slots with fused blocks and fine-tuned bridges and no extra weights."></a></figure>'] + healing30_parts()



def healing30_parts():
    rows = [['unhealed', '8.09%', '4 / 4 / 2', '<b>10</b>'], ['healed, step 50', '8.01%', '4 / 4 / 1', '9'],
            ['healed, step 100', '7.97%', '4 / 4 / 1', '9'], ['healed, step 150', '7.90%', '2 / 4 / 1', '7'],
            ['healed, step 200', '<b>7.84%</b>', '4 / 3 / 2', '9']]
    return ['<h3 id="phase8-healing">Healing the 30-block student</h3>',
            '<p class="summary">A rank-16 LoRA on q, k, v and out of the 30 original blocks (15.7 million weights, learning rate 2e-5) was '
            'trained together with both bridges (learning rate 1e-5) against the teacher velocity, on 450 text and 50 general recordings plus '
            'the original training data, for 200 updates. Every 50 updates the student was saved, measured on the 180 velocity comparisons, '
            'and rendered on the four-seed text grid.</p>',
            table('Healing the 30-block student: velocity and spelling', ['Checkpoint', 'Overall velocity', 'Anchor / Emma / Motel', 'Exact signs (of 11)'], rows),
            '<div class="verdict"><p><b>Healing improves velocity, not spelling.</b> Velocity falls steadily, from 8.09% to 7.84%, and was '
            'still falling at step 200. Spelling does not follow: every healed checkpoint is one to three signs below the unhealed student, '
            'with no trend, and the poster, banner and handwritten note do not change. As with block 13, a gentle LoRA trained on velocity does '
            'not teach letters back.</p></div>',
            '<h3 id="v1-plan">Version 1: two blocks removed</h3>',
            '<p class="summary"><b>The plan changes.</b> The first release removes two blocks (5 and 20) instead of four. Its base is the '
            '<b>unhealed</b> 30-block student, the best speller; the healed checkpoint at step 200 is kept as a backup. Next, a speed-distillation '
            '(lightning) LoRA takes sampling from 40 steps to 8, about five times faster, since the model already runs without classifier-free '
            'guidance. Every checkpoint is judged by velocity, by the text grids and by general images, and the chosen one is fused into the '
            'weights. Quantization follows, judged the same way. Removing blocks three and four may come in a later version.</p>'] + phase9_parts()



def phase9_parts():
    base = [['teacher, 40 steps', '11', 'reference', 'about 6.9'], ['teacher, 8 steps, no LoRA', '0', '15.1 dB', '1.53'],
            ['teacher, 4 steps, no LoRA', '0', '14.1 dB', '0.86'], ['teacher + LightX2V LoRA, 4 steps', '9', '11.8 dB', '0.87']]
    speed = [['teacher (32 blocks)', '165', '5.0 (real block)', '39.6'], ['version 1, FP32 bridges', '172', '8.8', '38.9'],
             ['<b>version 1, BF16 bridges</b>', '<b>159</b>', '<b>2.2</b>', '38.9']]
    return ['<h3 id="phase9-baseline">Few steps without distillation, and a published lightning LoRA</h3>',
            '<p class="summary">Before training our own speed LoRA, the intact teacher was sampled with the same prompts and seeds as every '
            'earlier grid in 8 and 4 steps without any LoRA, and in 4 steps with the LightX2V Lightning LoRA for Qwen-Image 2.1 (a published '
            'distribution-matching distillation, rank 16 on attention, used here for comparison only; it is not reused).</p>',
            table('Few-step baselines on the teacher', ['Variant', 'Exact signs (of 12)', 'PSNR to the 40-step image', 'Seconds per image'], base),
            '<figure class="comparison"><a href="images/phase9-baseline-anchor.jpg"><img loading="lazy" src="images/phase9-baseline-anchor.jpg" '
            'alt="The Rusty Anchor, four seeds: teacher at 40 steps, smeared at 8 steps, no image at 4 steps, LightX2V clean but Acchor three times"></a>'
            '<figcaption><b>The Rusty Anchor, four seeds</b> | without distillation 8 steps smear and 4 steps give no image; the LightX2V LoRA is '
            'clean and sharp but writes &ldquo;Acchor&rdquo; in three of four seeds</figcaption></figure>',
            '<div class="verdict"><p><b>Distillation is required, and it costs letters even when done well.</b> The published 4-step LoRA spells '
            'Emma and the motel sign but misspells the Rusty Anchor three times, so our own training data is mostly text (450 of 500 prompts). '
            'It also has the cleanest images and the lowest PSNR: it reproduces the teacher&rsquo;s kind of image, not the teacher&rsquo;s image for '
            'the same noise, so distance to the teacher is not a quality score on its own. Spelling and the images remain the judges.</p></div>',
            '<h3 id="phase9-speed">What removing two blocks buys</h3>',
            '<p class="summary">Batch 1 at 1024 by 1024 on one H100, each component timed separately. The transformer is almost all the time: '
            'text encoding takes 0.03 s and VAE decoding 0.13 s per image. The transformer shrinks from 7.12 to 6.73 billion weights (14.2 to 13.6 GB), '
            'but the untouched text encoder holds 17.5 GB, so memory falls by only 0.7 GB.</p>',
            table('Time per denoising step and per slot (milliseconds)', ['Model', 'Transformer step', 'Slot 5 (block or bridge)', 'Peak memory (GB)'], speed),
            '<div class="verdict"><p><b>The bridges had to run in BF16.</b> Kept in FP32 from training, each bridge cost 1.75 times a real block '
            'and made version 1 slower than the teacher. In BF16 a bridge costs less than half a block and version 1 is about 4% faster per step, '
            'as expected from removing 2 of 32 blocks, with nearly identical images (one small sign flips). The large gain is still to come from '
            'step distillation: 40 steps take 6.9 s per image, 8 steps about 1.5 s. Memory will fall with quantization.</p></div>',
            ] + phase10_parts()



def phase10_parts():
    dist = [['8 steps', '38.9%', '35.8%', '35.5%', '37.9%', '39.2%'], ['4 steps', '48.0%', '43.5%', '35.0%', '34.8%', '33.8%'],
            ['8 steps, continued from 1500, letters weighted', '37.9%', '34.6%', '33.3%', '<b>31.6%</b>', '31.9%']]
    spell = [['teacher, 40 steps', '4', '4', '4', '4', '16'], ['version 1, 40 steps', '4', '4', '3', '4', '15'],
             ['LightX2V on the teacher, 4 steps', '1', '4', '4', '4', '13'], ['ours, 8 steps, before weighting (step 1500)', '3', '3', '2', '3', '11'],
             ['<b>ours, 8 steps, letters weighted (+2000)</b>', '2', '<b>4</b>', '<b>3</b>', '3', '<b>12</b>']]
    return ['<h3 id="phase10-lightning">Our own lightning LoRAs</h3>',
            '<p class="summary"><b>Method: trajectory distillation.</b> The intact teacher (32 blocks, 40 steps) was run on 500 training prompts '
            '(450 with text) and 45 held-out prompts, saving the latent after every step: 545 trajectories, 11.9 GB. The student&rsquo;s 8 (or 4) '
            'steps are exact subsets of the teacher&rsquo;s 40-step schedule. Starting from a teacher latent, one student step is trained to land '
            'where 5 (or 10) teacher steps land. A rank-32 LoRA on the attention of the 30 blocks of version 1 learns it (31.5 million weights; '
            'the BF16 bridges stay frozen). Every 500 updates the LoRA is saved and judged by the distance between its final latent and the '
            'teacher&rsquo;s 40-step final latent for the same noise, and by the same 45 images.</p>',
            table('Text distance to the teacher&rsquo;s 40-step result, by update (lower is closer)', ['Run', '0', '500', '1000', '1500', '2000'], dist),
            '<p class="summary">The 8-step LoRA learned the teacher&rsquo;s layouts and most letters within 500 updates and then stopped improving. '
            'The 4-step LoRA learned the layouts but stayed soft and grey: copying the teacher&rsquo;s path averages at 4 steps, which '
            'distribution matching (the published LightX2V LoRA) avoids. The rest of this section is about 8 steps.</p>',
            '<h4>Making the loss see letters: OCR text regions</h4>',
            '<p class="summary">A misspelled sign hardly changes the loss: letters cover a few percent of the latent, and every token counts the '
            'same. So the loss was told where the letters are. Each teacher training image (its trajectory&rsquo;s final latent, decoded by the VAE '
            'exactly as the pipeline does) was read by an off-the-shelf OCR detector (EasyOCR, Chinese and English). Each detected word box was '
            'mapped onto the latent token grid, where one token covers 16 by 16 pixels, and widened by one token. Those tokens then count '
            '<b>10 times</b> in the loss: an error on a letter costs ten times an error on a wall. Only the boxes are used, not what the OCR reads, '
            'so a misread word still marks the right place. Text was found in <b>370 of the 450</b> text images, covering a median 12% of their '
            'tokens (and in 3 of the 50 general images).</p>',
            '<figure class="comparison"><a href="images/phase10-ocr-examples.jpg"><img loading="lazy" src="images/phase10-ocr-examples.jpg" '
            'alt="Six teacher training images with the text tokens outlined in red. Found: a Chinese restaurant sign with a small English line, '
            'a Back in 5 Minutes shop sign, every word of a banner, a Quiet Please sign and a small card in a window. Missed: HOTEL written in sand, '
            'and large 3D Chinese characters on a blue sign."></a><figcaption><b>Where the loss now looks hardest</b> | red squares are the '
            'tokens weighted 10 times; stylized lettering (sand, puffy 3D letters, large carved characters) is sometimes missed and keeps weight 1'
            '</figcaption></figure>',
            '<p class="summary">The best 8-step checkpoint was continued for 2000 updates with these weights, a learning rate decaying to zero and '
            '2 examples per update.</p>',
            table('Spelling, four seeds each (16 items)', ['Model', 'Rusty Anchor', 'Emma', 'Poster tagline', 'Handwritten note', 'Total'], spell),
            '<div class="decision-examples"><figure class="comparison"><a href="images/phase10-8step-emma.jpg"><img loading="lazy" src="images/phase10-8step-emma.jpg" '
            'alt="Happy Birthday Emma, four seeds: the letter-weighted 8-step LoRA spells all four"></a><figcaption><b>Happy Birthday Emma</b> | '
            'with letters weighted, all four seeds are right</figcaption></figure>'
            '<figure class="comparison"><a href="images/phase10-8step-anchor.jpg"><img loading="lazy" src="images/phase10-8step-anchor.jpg" '
            'alt="The Rusty Anchor, four seeds: seeds 1 and 2 right, seeds 0 and 3 still wrong at every checkpoint"></a><figcaption><b>The Rusty '
            'Anchor</b> | seeds 1 and 2 are right; seeds 0 and 3 stay wrong at every checkpoint</figcaption></figure></div>',
            '<div class="verdict"><p><b>Weighting letters helped, modestly.</b> The text distance fell from 37.9% to 31.6&ndash;31.9%, the best of '
            'any run; Emma became right in all four seeds and the poster tagline more reliable, from 11 to 12 of 16 items. It did not fix the '
            'hardest sign, and it stays below version 1 at 40 steps (15 of 16).</p></div>',
            '<h4>The main issue: structure</h4>',
            '<p class="summary">Averages hide it. On the 25 general prompts the 8-step LoRA is as close to the teacher as version 1 at 40 steps '
            '(median PSNR 17.7 against 17.8 dB; 8 steps without the LoRA, 15.0 dB), and weighting letters did not change its general images. But '
            'side by side with version 1 at 40 steps, <b>5 of the 25 images are clearly worse</b>, and textures are slightly softer:</p>',
            '<figure class="comparison"><a href="images/phase10-structure-failures.jpg"><img loading="lazy" src="images/phase10-structure-failures.jpg" '
            'alt="Five prompts, teacher, version 1 at 40 steps and ours at 8 steps: the checkerboard plate breaks up, the white star becomes a blob, '
            'four friends become a crowd, the cyclist turns semi-transparent, the alarm clock disappears"></a><figcaption><b>Five structure '
            'failures at 8 steps</b> | teacher, version 1 at 40 steps, ours at 8 steps</figcaption></figure>',
            '<p class="note">All 25 general prompts side by side: <a href="images/phase10-general-triplets.jpg">teacher | version 1 at 40 steps | '
            'ours at 8 steps</a>.</p>',
            '<div class="verdict"><p><b>Why, and what next.</b> These look like averaging: blobs, ghosts and merged figures. The LoRA is asked to '
            'reach the 32-block teacher&rsquo;s images, which version 1 cannot draw even in 40 steps, so it settles between the two; and only 50 of '
            'its 500 training prompts were general. The next run records <b>version 1&rsquo;s own 40-step trajectories</b> on 500 text and 500 '
            'general prompts and trains a fresh 8-step LoRA on them, with letters weighted as here, so it only has to compress steps; its '
            'ceiling is version 1 at 40 steps.</p></div>'] + phase10_v1_parts()



def phase10_v1_parts():
    dist = [['0 (8 steps, no LoRA)', '38.5%', '38.3%', '38.7%', '41.5%'], ['500', '29.2%', '29.5%', '28.9%', '36.7%'],
            ['1000', '29.5%', '28.4%', '30.4%', '36.7%'], ['1500', '29.2%', '29.1%', '29.3%', '36.3%'],
            ['2000', '<b>27.6%</b>', '<b>26.9%</b>', '28.1%', '34.6%']]
    spell = [['version 1, 40 steps', '4', '4', '3', '4', '15'], ['8 steps, teacher targets (+2000)', '2', '4', '3', '3', '12'],
             ['8 steps, version 1 targets, update 1000', '<b>4</b>', '2', '3', '<b>4</b>', '<b>13</b>'],
             ['8 steps, version 1 targets, update 1500', '3', '3', '<b>4</b>', '3', '<b>13</b>']]
    drift = [['Checkerboard plate', '31.8%', '12.5%', '7.3%', '2.4%'], ['White star', '28.5%', '<b>3.1%</b>', '2.1%', '1.2%'],
             ['Four friends', '24.0%', '6.7%', '2.9%', '1.9%'], ['Cyclist', '39.4%', '10.8%', '5.5%', '1.8%'],
             ['Alarm clock', '39.9%', '12.2%', '3.2%', '1.7%']]
    return ['<h3 id="phase10-v1-targets">Learning from version 1&rsquo;s own images</h3>',
            '<p class="summary">Version 1 recorded its own 40-step trajectories (BF16 bridges, as it ships) on 1000 training prompts, now '
            'balanced: 500 with text and 500 general, drawn from the Phase 1 training split across 38 subjects and checked against the held-out '
            'prompts (1045 trajectories with the 45 held-out ones). OCR found text in 406 of the 500 text images. A fresh 8-step LoRA was trained on '
            'them from zero, with letters weighted 10 times and a decaying learning rate. Its targets are now images version 1 can actually draw, '
            'so its ceiling is version 1 at 40 steps.</p>',
            table('Distance to version 1 at 40 steps (its target) and to the teacher, by update', ['Update', 'All 45', 'Text', 'General', 'Teacher, all'], dist),
            table('Spelling, four seeds each (16 items)', ['Model', 'Rusty Anchor', 'Emma', 'Poster tagline', 'Handwritten note', 'Total'], spell),
            '<figure class="comparison"><a href="images/phase10-v1targets-anchor.jpg"><img loading="lazy" src="images/phase10-v1targets-anchor.jpg" '
            'alt="The Rusty Anchor, four seeds: version 1 at 40 steps, the earlier 8-step LoRA, and the new one at updates 500, 1000 and 1500; the '
            'new one spells all four at update 1000"></a><figcaption><b>The Rusty Anchor</b> | learning from version 1&rsquo;s images, the 8-step LoRA '
            'spells all four seeds at update 1000; with teacher targets it never passed two</figcaption></figure>',
            '<p class="summary"><b>Structure.</b> Against version 1 at 40 steps, 20 of the 25 general images match closely; 5 are clearly worse. '
            'The four friends are now right, but a new case appeared (the laptop lighting a face disappears), so the count stays at 5: the '
            'checkerboard pattern, the star&rsquo;s outline, the semi-transparent cyclist, the missing alarm clock and the missing laptop. They are '
            'not random: precise geometry, a small object the image depends on, or a thin subject. These five did not change between updates 500 '
            'and 2000.</p>',
            '<figure class="comparison"><a href="images/phase10-v1targets-general.jpg"><img loading="lazy" src="images/phase10-v1targets-general.jpg" '
            'alt="The 25 general prompts in pairs: version 1 at 40 steps and the new 8-step LoRA at update 2000"></a><figcaption><b>All 25 general '
            'prompts</b> | version 1 at 40 steps (left) and the new 8-step LoRA at update 2000 (right)</figcaption></figure>',
            '<h4>Where the failures are made: a drift test</h4>',
            '<p class="summary">If the student&rsquo;s first jump lands slightly off, the later steps may never recover. On the five failure prompts '
            'the 8-step LoRA was started not from noise but from version 1&rsquo;s own latent after 5, 10 or 20 of its 40 steps.</p>',
            table('Distance to version 1 at 40 steps, by starting point', ['Prompt', 'From noise', 'From step 5', 'From step 10', 'From step 20'], drift),
            '<figure class="comparison"><a href="images/phase10-drift-test.jpg"><img loading="lazy" src="images/phase10-drift-test.jpg" '
            'alt="Five failure prompts: version 1 at 40 steps, our 8-step LoRA from noise (broken), and from version 1 step 5, 10 and 20 (all right)"></a>'
            '<figcaption><b>The first jump is the whole problem</b> | from version 1&rsquo;s step 5 the checkerboard is regular, the star is a star, '
            'the cyclist is solid and the clock is there</figcaption></figure>',
            '<div class="verdict"><p><b>The structure failures are made entirely in the first jump</b>, from noise to step 5 of 40, where the layout '
            'is decided. The LoRA does the other seven jumps well. That suggests a fix without training: let version 1 itself take the start in a '
            'few small steps, then hand over to the LoRA.</p></div>',
            ] + phase10_ship_parts()



def phase10_ship_parts():
    hybrid = [['8 (1 start step)', '37.7%', '25.8%'], ['9 (2)', '23.1%', '26.2%'], ['10 (3)', '14.5%', '29.1%'], ['12 (5)', '<b>7.7%</b>', '34.7%']]
    spell = [['teacher', '40', '16'], ['version 1', '40', '15'], ['<b>version 1, fused, uneven schedule</b>', '12', '15'],
             ['<b>version 1, fused, uneven schedule</b>', '<b>10</b>', '<b>14</b>'], ['version 1, 8-step LoRA from noise', '8', '13'],
             ['LightX2V on the teacher', '4', '13']]
    return ['<h4>A gentler start</h4>',
            '<p class="summary">Version 1 takes the start from noise to step 5 of 40 in a few small Euler steps of its own schedule, then the '
            'LoRA does the seven big jumps. Two variants: with the LoRA switched off for the start (which needs the LoRA kept separate), and with '
            'it on throughout (what a single model with the LoRA fused into its weights does).</p>',
            table('Median distance to version 1 at 40 steps, all 45 held-out images', ['Model calls (start steps)', 'LoRA off for the start', 'LoRA fused'], hybrid),
            '<figure class="comparison"><a href="images/phase10-hybrid-off-fused.jpg"><img loading="lazy" src="images/phase10-hybrid-off-fused.jpg" '
            'alt="Checkerboard, star, laptop, four friends, cyclist, alarm clock, Rusty Anchor, Emma, handwritten note: version 1 at 40 steps, then '
            '10 and 12 steps with the LoRA off for the start and fused"></a><figcaption><b>Version 1 at 40 steps, then 10 and 12 steps, LoRA off '
            'for the start or fused</b> | off reproduces version 1; fused draws different but clean images</figcaption></figure>',
            '<p class="summary">With the LoRA off for the start, 12 steps reproduce version 1 almost exactly (7.7%). Fused, the distance looks '
            'worse, but the images are not broken: the start drifts to a different composition and every step stays clean, so the star, cyclist, '
            'alarm clock and checkerboard come out sharp. Distance cannot tell a different good image from a broken one. A separate LoRA was '
            'ruled out, so the fused route was scored in full.</p>',
            table('Spelling, four seeds each (16 items)', ['Model', 'Steps', 'Correct'], spell),
            '<div class="decision-examples"><figure class="comparison"><a href="images/phase10-fused10-anchor.jpg"><img loading="lazy" src="images/phase10-fused10-anchor.jpg" '
            'alt="The Rusty Anchor, four seeds: version 1 at 40 steps, fused at 10 and 12 steps"></a><figcaption><b>The Rusty Anchor</b> | '
            'version 1 at 40 steps, fused at 10 and 12 steps</figcaption></figure>'
            '<figure class="comparison"><a href="images/phase10-fused10-general.jpg"><img loading="lazy" src="images/phase10-fused10-general.jpg" '
            'alt="The 25 general prompts as triplets: version 1 at 40 steps, fused at 10 steps, fused at 12 steps"></a><figcaption><b>All 25 '
            'general prompts</b> | version 1 at 40, fused 10, fused 12</figcaption></figure></div>',
            '<div class="verdict"><p><b>Version 1 ships fused, at 10 steps</b>: 4 times faster than 40, 14 of 16 spelling items (version 1 itself '
            '15), and 2 of 25 general images clearly wrong (a laptop missing, five friends instead of four) instead of 5. 12 steps gains one sign '
            'and no structure.</p></div>',
            '<h4>How version 1 is run</h4>',
            '<p class="summary">The LoRA is fused into the weights (W + B&middot;A in 120 attention layers), so the model is a single checkpoint. '
            'The 10-step schedule is points 0, 1, 3, 5, 10, 15, 20, 25, 30 and 35 of the stock 40-step schedule for the image size, then the end. '
            'The stock scheduler cannot produce it on its own, and passing those values naively would shift and stretch them a second time. Two '
            'ways to run it:</p>',
            '<dl class="facts"><div><dt>A &middot; scheduler class</dt><dd>A small scheduler that builds the stock 40-step schedule for the image '
            'size and keeps those ten points; the reference implementation, shipped with the model.</dd></div>'
            '<div><dt>B &middot; plain sigmas</dt><dd>A helper returns the ten values for a width and height; they are passed as '
            '<code>sigmas</code> to the stock pipeline with a neutral scheduler (no shift, no terminal stretch). Works in diffusers as is, and in '
            'ComfyUI and other interfaces through a custom-sigmas input.</dd></div></dl>',
            '<dl class="facts"><div><dt>shipping</dt><dd>Both routes ship as files next to the weights: <code>nihonga.py</code> (the bridge '
            'modules, the loader, the scheduler class and the sigma helper) and <code>sigmas.json</code> (the ten values for each standard image '
            'size, for interfaces without code).</dd></div></dl>'] + release_parts()



def release_parts():
    rows = [['Transformer blocks', '32', '30 + 2 bridges'], ['Transformer parameters', '7.115 B (14.2 GB)', '6.729 B (13.5 GB)'],
            ['Sampling steps', '40', '<b>10</b>'], ['Time per 1024 &times; 1024 image, one H100', '6.89 s', '<b>1.75 s</b>'],
            ['Peak GPU memory', '39.6 GB', '39.3 GB'], ['Spelling, 16 hard items', '16', '15'],
            ['General images unlike the 40-step pruned model', '&ndash;', '2 of 25']]
    return ['<h3 id="v1-release">Nihonga v1</h3>',
            '<p class="summary">The release checkpoint uses the adapter of update 2500 (15 of 16 spelling items at 10 steps, against 14 for '
            'update 2000). Built in the shipping format, the transformer was reloaded through the shipped code with every tensor identical; the '
            'scheduler class and the plain sigmas used exactly the intended schedule and give pixel-identical images.</p>',
            table('Qwen-Image-2.1 and Nihonga v1', ['', 'Qwen-Image-2.1', 'Nihonga v1'], rows),
            '<div class="verdict"><p><b>3.9 times faster, the original&rsquo;s spelling level less one item.</b> Pruning gives a small part of '
            'the speed (about 4% per step); the 10-step schedule gives the rest. Memory hardly changes, because the unchanged text encoder holds '
            '17.5 GB; that is for quantization, which is not part of v1. Nihonga v1 makes text-to-image only: editing, reference images and '
            'transparency were not trained or tested. It is a derivative of Qwen-Image-2.1 under the Qwen Research License, for non-commercial '
            'use, distributed as the modified transformer plus code, with the text encoder and VAE loaded from the original.</p></div>']


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
    parser.add_argument('--site', default=str(ROOT.parent / 'josejuanmartinez.github.io/research'))  # the full research log moved here; niwakiai.com/nihonga is the release page
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
    for path in (site / 'nihonga/index.html', site / 'nihonga/build_nihonga.py'):
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
