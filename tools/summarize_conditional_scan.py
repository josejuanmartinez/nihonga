"""Render a completed conditional scan without changing its ranking or gates."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary', type=Path)
    args = parser.parse_args()
    raw = args.summary.read_bytes()
    report = json.loads(raw)
    expected = set(range(32)) - {4, 5}
    blocks = report['blocks']
    if (report['status'] != 'passed' or len(blocks) != 30
            or {b['original_block_id'] for b in blocks} != expected
            or report['optimizer_updates'] != 0):
        raise RuntimeError('Require a complete frozen 30-block scan')
    blocks = sorted(blocks, key=lambda b: b['overall']['baseline_displacement']['mean'])
    if [b['original_block_id'] for b in blocks] != report['ranking']:
        raise RuntimeError('Saved ranking differs from reported mean displacement')
    rows = []
    for rank, block in enumerate(blocks, 1):
        for scope, metrics in [('overall', block['overall']), *block['panels'].items(),
                               *block['stages'].items()]:
            row = dict(rank=rank, original_block_id=block['original_block_id'],
                       scope=scope, comparisons=metrics['comparisons'])
            for metric, values in metrics.items():
                if isinstance(values, dict):
                    row.update({metric + '_' + key: value for key, value in values.items()})
            rows.append(row)
    destination = args.summary.parent
    with (destination / 'metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        '# Conditional importance inside candidate A', '',
        f"Source SHA-256: `{hashlib.sha256(raw).hexdigest()}`.", '',
        'Each trial independently bypasses one original block in restored, frozen A. '
        'The fixed gate10 and wider50 development panels each use early, middle and late stages. '
        'Ranking uses mean relative displacement from A. No block is accepted by this screen.', '',
        '| Rank | Block | Mean displacement from A (%) | Early (%) | Middle (%) | Late (%) | '
        'Teacher error change (pp) | Worst teacher regression (pp) |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |',
    ]
    for rank, block in enumerate(blocks, 1):
        overall = block['overall']
        values = [100 * overall['baseline_displacement']['mean']]
        values.extend(100 * block['stages'][s]['baseline_displacement']['mean']
                      for s in ('early', 'middle', 'late'))
        values.extend(overall['teacher_error_change_percentage_points'][s]
                      for s in ('mean', 'maximum'))
        lines.append(f"| {rank} | {block['original_block_id']} | "
                     + ' | '.join(f'{v:.4f}' for v in values) + ' |')
    lines += ['', 'Percentiles, panel and stage summaries are in `metrics.csv`. '
              'The source summary retains prompt-group aggregates and the ten worst '
              'prompt/stage regressions for each block and scope.', '',
              'A still fails the historical 6.52% velocity gate. Negative teacher error '
              'changes can reflect cancellation of A\'s existing error. Bypass rankings '
              'do not establish bridge replaceability or image quality.', '']
    (destination / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(destination / 'REPORT.md')


if __name__ == '__main__':
    main()
