"""Recompute public task and pooled-task Elo from the 72 sealed outcomes.

Uses the frozen match scorer; the original complete-suite scores remain intact.
Run from any directory: python3 site/scripts/rate_results.py
"""
import copy
import itertools
import json
from pathlib import Path
import sys

SITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SITE.parent / 'benchmarks/compact-coding-v1'))
from research_env.leaderboard import elo

TASKS = ('pulse', 'heat', 'load')
SEEDS = (0, 1, 2)


def reanalyze(source):
    result = copy.deepcopy(source)
    models = sorted(f"{m['model']} {m['effort'].capitalize()}" for m in result['models'])
    cases = {}
    for row in source['raw_scores']:
        key = (row['model'], (row['task'], row['seed']))
        if key in cases:
            raise ValueError(f'Duplicate outcome: {key}')
        cases[key] = row
    slots = list(itertools.product(TASKS, SEEDS))
    if set(cases) != set(itertools.product(models, slots)):
        raise ValueError('Expected all model/task/seed outcomes, including explicit failures')
    overall = elo(models, slots, cases, 'prediction')
    overall_entries = {r['competitor']: r for r in overall['entries']}
    tasks = {task: elo(models, SEEDS, {(m, s): cases[(m, (task, s))]
                                      for m in models for s in SEEDS}, 'prediction')
             for task in TASKS}
    for board in [overall, *tasks.values()]:
        board['entries'].sort(key=lambda row: (-row['elo'], row['competitor']))
        for entry in board['entries']:
            entry['rank'] = 1 + sum(other['elo'] > entry['elo'] + 1e-9 for other in board['entries'])
    for model in result['models']:
        name = f"{model['model']} {model['effort'].capitalize()}"
        entry = overall_entries[name]
        model['overall_elo'] = entry['elo']
        model['overall_matches'] = {key: entry[key] for key in ('wins', 'draws', 'losses', 'skipped')}
        for task in TASKS:
            current = model['tasks'][task]
            current.setdefault('original_elo', current['elo'])
            current['elo'] = next(r['elo'] for r in tasks[task]['entries'] if r['competitor'] == name)
    result['rating_policy'] = {
        'version': 'pooled-task-elo-v1',
        'analysis': 'Post-run reanalysis; original complete-suite Elo retained in suite_elo.',
        'overall': 'All model pairs on each matched task and agent seed; pool the nine task/seed slots.',
        'outcomes': 'Lower MAE wins; valid beats failed; two failures skip; tolerance 1e-9.',
        'initial_elo': 1000, 'k': 32, 'scale': 400, 'match_order_seed': 0,
        'task_order': list(TASKS), 'agent_seeds': list(SEEDS),
        'ranking': 'All participants retain numeric Elo, including those with zero valid attempts.',
        'scheduled_matches': len(overall['matches']),
        'scored_matches': sum(m['outcome'] is not None for m in overall['matches']),
        'audit': '/results/pooled-task-elo-v1.json',
    }
    return result, {'policy': result['rating_policy'], 'overall': overall, 'tasks': tasks}


if __name__ == '__main__':
    result, audit = reanalyze(json.loads((SITE / 'lib/results.json').read_text()))
    data = json.dumps(result, indent=2, allow_nan=False) + '\n'
    for path in ('lib/results.json', 'public/results/compact-coding-v1.json'):
        (SITE / path).write_text(data)
    (SITE / 'public/results/pooled-task-elo-v1.json').write_text(json.dumps(audit, indent=2, allow_nan=False) + '\n')
    for row in sorted(result['models'], key=lambda r: -r['overall_elo']):
        print(f"{row['model']}: {row['overall_elo']:.1f} ({row['valid']}/9 valid)")
