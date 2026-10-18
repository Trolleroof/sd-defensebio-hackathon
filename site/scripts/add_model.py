"""Add a self-hosted model to the public snapshot, then re-rate every model.

Reads the nine receipts of one run (read-only) plus its process reviews and
appends them to lib/results.json before running rate_results.reanalyze, so every
Elo is recomputed over all outcomes. Usage:
  python3 site/scripts/add_model.py <run-dir> <reviews-leaderboard.json> <competitor> <model> <effort>
<run-dir> holds runs/**/receipt.json; reviews come from the leaderboard.json
process_review block, filtered to <competitor>. Criteria a judge left unobserved
stay unobserved (null, count excludes them).
Example: add_model.py .../granite4-grpo-step28 .../compare-frontier-plus-granite/leaderboard.json \
  "Granite 4.0 Micro + GRPO (step 28)" "Granite 4.0 Micro + GRPO" "step 28"
"""
import glob
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rate_results import SITE, reanalyze


def main(root, reviews_path, competitor, name, effort):
    root = Path(root)
    source = json.loads((SITE / 'lib/results.json').read_text())
    if any(m['model'] == name for m in source['models']):
        raise SystemExit(f'{name} already present')
    rows = []
    for path in sorted(glob.glob(str(root / 'runs/**/receipt.json'), recursive=True)):
        r = json.loads(Path(path).read_text())
        run, score = r['run'], r['score']
        assert run['competitor'] == competitor and run['suite_sha256'] == source['suite_sha256']
        rows.append({'model': competitor, 'task': run['task'], 'seed': run['seed'],
                     'execution_valid': score['execution_valid'], 'primary_loss': score['primary_loss'],
                     'interval_score_90': None, 'failure_reason': score['failure_reason']})
    assert len(rows) == 9
    reviews = [r for r in json.loads(Path(reviews_path).read_text())['process_review']['reviews'] if r['model'] == competitor]
    criteria = {}
    for c in ('evidence', 'experiment', 'feedback', 'repro', 'selection'):
        scores = [x['score'] for r in reviews for x in r['verdict']['criteria'] if x['id'] == c and x['score'] is not None]
        criteria[c] = {'mean': sum(scores) / len(scores) if scores else None, 'count': len(scores)}
    source['raw_scores'] += rows
    source['models'].append({
        'model': name, 'effort': effort, 'competitor': competitor, 'suite_elo': None, 'valid': sum(r['execution_valid'] for r in rows),
        'episodes': 9, 'reviews': len(reviews), 'criteria': criteria,
        'tasks': {t: {'elo': None, 'valid': sum(r['execution_valid'] for r in rows if r['task'] == t),
                      'mean_loss': (lambda v: sum(v) / len(v) if v else None)([r['primary_loss'] for r in rows if r['task'] == t and r['execution_valid']]),
                      'original_elo': None} for t in ('load', 'heat', 'pulse')},
        'overall_elo': None, 'overall_matches': {}, 'self_hosted': True,
    })
    source['episodes'] += 9
    source['valid'] += sum(r['execution_valid'] for r in rows)
    source['reviews'] += len(reviews)
    source['review_target'] += 9
    groups = source['failure_groups']
    for r in rows:
        if not r['execution_valid']:
            key = 'Response/tool protocol' if 'response attempts exhausted' in r['failure_reason'] else 'Other execution failure'
            groups[key] = groups.get(key, 0) + 1
    result, audit = reanalyze(source)
    data = json.dumps(result, indent=2, allow_nan=False) + '\n'
    for p in ('lib/results.json', 'public/results/compact-coding-v1.json'):
        (SITE / p).write_text(data)
    (SITE / 'public/results/pooled-task-elo-v1.json').write_text(json.dumps(audit, indent=2, allow_nan=False) + '\n')
    for row in sorted(result['models'], key=lambda r: -r['overall_elo']):
        print(f"{row['model']}: {row['overall_elo']:.1f} {[round(row['tasks'][t]['elo'],1) for t in ('pulse','heat','load')]}")


if __name__ == '__main__':
    main(*sys.argv[1:6])
