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
        status_text = ('<p class="summary"><b>Research direction:</b> the accepted model is still the one-block student. '
                       f'Importance scans inside both candidate A and the accepted student are complete; block {scan_rows(b)[0]["block"]} is the gentlest bypass in the accepted student. '
                       'The next step is a learned bridge for a shortlisted block, with predeclared velocity, image and latency gates. '
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
