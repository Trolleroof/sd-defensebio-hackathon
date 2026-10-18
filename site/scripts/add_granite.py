"""Add self-hosted base Granite 4.0 Micro to the public snapshot, then re-rate.

Reads the nine Granite receipts and process reviews (read-only) and appends them
to lib/results.json before running rate_results.reanalyze, so every Elo is
recomputed over all 81 outcomes. Usage:
  python3 site/scripts/add_granite.py <granite-run-dir>
where <granite-run-dir> contains leaderboard.json and runs/**/receipt.json.
"""
import glob
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rate_results import SITE, reanalyze

COMPETITOR = 'Granite 4.0 Micro (base)'
NAME, EFFORT = 'Granite 4.0 Micro', 'base'


def main(root):
    root = Path(root)
    source = json.loads((SITE / 'lib/results.json').read_text())
    if any(m['model'] == NAME for m in source['models']):
        raise SystemExit('Granite already present')
    rows = []
    for path in sorted(glob.glob(str(root / 'runs/**/receipt.json'), recursive=True)):
        r = json.loads(Path(path).read_text())
        run, score = r['run'], r['score']
        assert run['competitor'] == COMPETITOR and run['suite_sha256'] == source['suite_sha256']
        rows.append({'model': f'{NAME} {EFFORT.capitalize()}', 'task': run['task'], 'seed': run['seed'],
                     'execution_valid': score['execution_valid'], 'primary_loss': score['primary_loss'],
                     'interval_score_90': None, 'failure_reason': score['failure_reason']})
    assert len(rows) == 9
    reviews = json.loads((root / 'leaderboard.json').read_text())['process_review']['reviews']
    criteria = {}
    for c in ('evidence', 'experiment', 'feedback', 'repro', 'selection'):
        scores = [x['score'] for r in reviews for x in r['verdict']['criteria'] if x['id'] == c and x['score'] is not None]
        criteria[c] = {'mean': sum(scores) / len(scores) if scores else None, 'count': len(scores)}
    source['raw_scores'] += rows
    source['models'].append({
        'model': NAME, 'effort': EFFORT, 'suite_elo': None, 'valid': sum(r['execution_valid'] for r in rows),
        'episodes': 9, 'reviews': len(reviews), 'criteria': criteria,
        'tasks': {t: {'elo': None, 'valid': 0, 'mean_loss': None, 'original_elo': None} for t in ('load', 'heat', 'pulse')},
        'overall_elo': None, 'overall_matches': {}, 'self_hosted': True,
    })
    source['episodes'] += 9
    source['valid'] += sum(r['execution_valid'] for r in rows)
    source['reviews'] += len(reviews)
    source['review_target'] += 9
    groups = source['failure_groups']
    groups['Other execution failure'] = groups.get('Other execution failure', 0) + sum(not r['execution_valid'] for r in rows)
    result, audit = reanalyze(source)
    data = json.dumps(result, indent=2, allow_nan=False) + '\n'
    for p in ('lib/results.json', 'public/results/compact-coding-v1.json'):
        (SITE / p).write_text(data)
    (SITE / 'public/results/pooled-task-elo-v1.json').write_text(json.dumps(audit, indent=2, allow_nan=False) + '\n')
    for row in sorted(result['models'], key=lambda r: -r['overall_elo']):
        print(f"{row['model']}: {row['overall_elo']:.1f} {[round(row['tasks'][t]['elo'],1) for t in ('pulse','heat','load')]}")


if __name__ == '__main__':
    main(sys.argv[1])
