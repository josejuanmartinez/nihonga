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
             'e2e_cond': 'Same, trained on the final output (no gain; best at its start)', 'e2e_attn': '<b>+ attention branch, trained on the final output</b>'}

    def row(label, params, c):
        o = c['overall']['teacher_relative_l2']['mean']
        g10, w50 = (c['panels'][p]['teacher_relative_l2']['mean'] for p in ('gate10', 'wider50'))
        e, m, l = (c['stages'][s]['teacher_relative_l2']['mean'] for s in ('early', 'middle', 'late'))
        return [label, params, f'<b>{pct(o)}</b>', pct(g10), pct(w50), f'{pct(e, 1)} / {pct(m, 1)} / {pct(l, 1)}']
    rows = [row('Plain, rank 256 (first bridge)', '2.1M', first['candidate'])]
    rows += [row(names.get(n, n), f"{v['parameters'] / 1e6:.1f}M", v) for n, v in r['variants'].items()]
    for later in sorted(list((ROOT / 'data/phase7/block13_bridge_v3').glob('*/summary.json')) +
                        list((ROOT / 'data/phase7/block13_bridge_e2e').glob('*/summary.json'))):
        extra = json.loads(later.read_text(encoding='utf-8'))
        if extra.get('status') == 'passed':
            rows += [row(names.get(n, n), f"{v['parameters'] / 1e6:.1f}M", v) for n, v in extra['variants'].items()]
    return ['<h3 id="phase7-bridge-variants">Better bridges for block 13: conditioning helps, attention helps more</h3>',
            '<p class="summary">Every real block receives the model’s shared timestep modulation (scale and gate values that tell it how noisy the image is); the first bridge ignored it, and its errors were worst at early, high-noise steps. We fitted three variants on identical data: a 4× wider plain bridge, and a bridge that mirrors a block’s MLP branch, h + tanh(gate) · MLP(LN(h) · (1 + scale)), with scale and gate learned mixes of the model’s own modulation, at rank 256 and 1024.</p>',
            table('Block-13 bridge variants inside B: velocity error against the teacher (B with block 13 kept: 5.40% overall)',
                  ['Bridge', 'Parameters', 'Overall', 'Check (10)', 'Wider (50)', 'Early / middle / late'], rows),
            '<div class="verdict"><p><b>Capacity alone does nothing; conditioning does.</b> The conditioned rank-1024 bridge recovers about a third of the gap between the first bridge and B on the wider prompts. Training three times longer, or adding 3×3 local mixing over the image-token grid, moves the error by only about 0.05 points: the per-token fitting objective has reached its ceiling. Training the same bridge on the final velocity instead (gradients through the frozen blocks 14-31) did not help either. Adding a small attention branch (4 heads, image tokens attending to the prompt and to each other, with its own prefix cache) and training it on the final velocity gave the first clear step: 6.42% overall and 6.29% on the wider prompts, still improving when training stopped.</p></div>'] + variant_review_parts()


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
    parts = ['<h3 id="phase7-adaptive">Which block next? Importance measured inside the pruned student</h3>',
             '<p class="summary">The original ranking measured blocks in the intact teacher. After a cut, surviving blocks may compensate or become more redundant, so we measure again inside the smaller student. Each trial <b>bypasses one surviving block</b>, compares the prediction with the teacher on the same <b>60 development prompts at early, middle and late moments</b> (180 comparisons), then restores the block. Bypasses never accumulate and no weight is updated. Before any trial, all 90 intact-teacher controls reproduced exactly and the student reproduced its saved predictions.</p>',
             f'<p class="summary"><b>Correction to the plan.</b> The scan was first run inside candidate A (blocks 4-5 replaced), as the plan specified. A fails the {GATE:.2f}% gate, so its ranking describes a model we have not accepted. The same scan is therefore repeated inside <b>candidate B</b>: the accepted one-block student, 31 blocks plus the pretrained block-5 bridge. Block 4 is eligible again in B.</p>']
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
    parts.append('<h3>The same scan inside candidate A</h3>')
    parts.append(f'<p class="summary">A alone measures <b>{pct(base_a)}</b> on all 180 comparisons ({pct(panels_a["wider50"])} on the 50 wider prompts and {pct(panels_a["gate10"])} on the 10 check prompts, reproducing the earlier 7.41% and 7.97%).</p>')
    parts.append(table('Candidate A scan: velocity error against the teacher with one extra block bypassed (lower is better); A alone measures ' + pct(base_a),
                       ['Block', 'A + bypass', 'Early', 'Middle', 'Late', 'p90'],
                       [[r['block'], f"<b>{pct(r['mean'])}</b>", pct(r['early'], 1), pct(r['middle'], 1), pct(r['late'], 1), pct(r['p90'], 1)] for r in rows_a]))
    parts.append(f'<div class="verdict"><p><b>Inside A, no single bypass stays within the gate.</b> The gentlest, block {best_a["block"]}, moves A from {pct(base_a)} to {pct(best_a["mean"])}. The middle blocks (6-17) cost least; blocks 0 and 31 cannot be skipped. A low bypass score only nominates a block for a learned bridge; it is not a removal.</p></div>')
    parts.append('<h3 id="phase7-latency">Measured speed of candidate A</h3>')
    parts.append(f'<p class="summary">{len(speed["ratios"])} balanced, randomized teacher/A pairs on one H100, 1024×1024, 40 steps, BF16, same prompts and seeds, warm models. Median end-to-end time: <b>{ta["median_seconds"]:.2f} s</b> for the teacher and <b>{tb["median_seconds"]:.2f} s</b> for A. Paired speedup <b>{speed["median"]:.3f}×</b> (bootstrap 95% CI {speed["bootstrap_95pct"][0]:.3f}-{speed["bootstrap_95pct"][1]:.3f}). Peak allocated memory: {ta["peak_allocated_gib"]:.2f} GiB vs {tb["peak_allocated_gib"]:.2f} GiB.</p>')
    adh, txt = q['counts']['adherence'], q['counts']['text']
    pairs = q['pairs']
    wins = sum(r['clear_pass_delta'] > 0 for r in pairs if r['group'] != 'text_rendering')
    losses = sum(r['clear_pass_delta'] < 0 for r in pairs if r['group'] != 'text_rendering')
    text_losses = sum(r['clear_pass_delta'] < 0 for r in pairs if r['group'] == 'text_rendering')
    text_wins = sum(r['clear_pass_delta'] > 0 for r in pairs if r['group'] == 'text_rendering')
    ci = q['paired_bootstrap_95pct']
    parts.append('<h3 id="phase7-blind">Blinded image test: candidate A loses to the teacher</h3>')
    parts.append(f'<p class="summary">Before viewing images we registered <b>159 fresh validation prompts</b> (35 counting, 34 spatial, 30 color, 30 multiple-object, 30 text), one seed each, rendered by the teacher and A at 40 steps. Each image was scored pass, fail or uncertain against one fixed question, with left/right positions randomized and the identity key committed by hash and revealed only after all 318 scores were committed.</p>')
    parts.append(table('Blinded 159-prompt panel: clear passes for the teacher and candidate A',
                       ['Panel', 'Teacher pass', 'A pass', 'A better', 'A worse', 'Gate'],
                       [['Adherence (129)', adh['teacher'].get('pass', 0), adh['candidate_A'].get('pass', 0), wins, losses,
                         f"gain {100*q['adherence_gain_fraction']:+.1f} points, 95% CI {100*ci[0]:+.1f} to {100*ci[1]:+.1f}; needs lower bound ≥ -5: <b>{'pass' if q['adherence_noninferiority_passed'] else 'fail'}</b>"],
                        ['Text (30)', txt['teacher'].get('pass', 0), txt['candidate_A'].get('pass', 0), text_wins, text_losses,
                         f"A must be within 1 of the teacher: <b>{'pass' if q['text_gate_passed'] else 'fail'}</b>"]]))
    parts.append(f'<div class="verdict"><p><b>Candidate A is rejected as a base for further pruning.</b> It fails both image gates as well as the velocity gate; only the speed gate passes. The earlier impression that A matched the teacher came from a small unblinded screen. Further pruning starts from the accepted one-block student.</p></div>')
    parts.append('<p class="note">One model-blind AI reviewer and one seed per prompt. The 10 and 50 validation prompts used for velocity are development data, not an untouched final test.</p>')
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
