"""Frozen eligible records; window -> bout -> person aggregation."""
from collections import defaultdict
import numpy as np
from research_env.core.common import finite, invalid_score, validate_prediction


def score(reference_records, predictions, scoring_config=None):
    config = {'activities': ['adaptation', 'baseline', 'tsst-prep', 'tsst-speech', 'meditation-1',
                             'screen-reading', 'ssst-prep', 'ssst-sing', 'meditation-2',
                             'keyboard-typing', 'rest-1', 'mobile-typing', 'rest-2', 'standing',
                             'rest-3', 'walking', 'rest-4', 'jogging', 'rest-5', 'running'],
              'epsilon': 1e-9, **(scoring_config or {})}
    invalid = invalid_score
    ids = [r['id'] for r in reference_records]
    if not ids or len(ids) != len(set(ids)) or set(predictions) != set(ids):
        return invalid('Missing, duplicate, or extra eligible predictions')
    bouts = defaultdict(list)
    activities = defaultdict(list)
    try:
        for ref in reference_records:
            p = validate_prediction(predictions[ref['id']])
            y = ref['reference_bpm']
            if not finite(y):
                raise ValueError('Nonfinite reference')
            m, lo, hi = (p[k] for k in ('mean_bpm', 'lower_bpm', 'upper_bpm'))
            error = abs(m-y)
            values = [error, error**2, m-y, float(error > 10),
                      hi-lo + 20*max(lo-y, 0) + 20*max(y-hi, 0),
                      float(lo <= y <= hi), hi-lo]
            if not all(finite(v) for v in values):
                raise ValueError('Metric arithmetic overflow')
            bouts[(ref['person'], ref['session'])].append(values)
            activities[(ref['activity'], ref['person'], ref['session'])].append(values)
    except (ValueError, KeyError, TypeError, OverflowError) as exc:
        return invalid(str(exc))
    people = defaultdict(list)
    for (person, _), values in bouts.items():
        people[person].append(np.mean(values, axis=0))
    per_person = {p: np.mean(v, axis=0) for p, v in people.items()}
    aggregate = np.mean(list(per_person.values()), axis=0)
    slices = defaultdict(lambda: defaultdict(list))
    for (activity, person, _), values in activities.items():
        slices[activity][person].append(np.mean(values, axis=0)[0])
    activity_mae = {a: float(np.mean([np.mean(v) for v in ps.values()])) for a, ps in slices.items()}
    expected = config.get('activities', sorted(activity_mae))
    missing = sorted(set(expected)-set(activity_mae))
    loss = float(aggregate[0])
    baseline = config.get('frozen_baseline_loss')
    return {'execution_valid': True, 'primary_loss': loss, 'units': 'bpm', 'resource_use': None,
            'skill': None if baseline is None else 1-loss/max(baseline, config.get('epsilon', 1e-9)),
            'secondary': {'rmse_bpm': float(np.sqrt(aggregate[1])), 'bias_bpm': float(aggregate[2]),
                          'fraction_error_over_10_bpm': float(aggregate[3]),
                          'activity_mae_bpm': activity_mae, 'missing_activities': missing,
                          'activity_counts': {a: {'people': len(ps), 'windows': sum(len(v) for (activity, _, _), v in activities.items() if activity == a)} for a, ps in slices.items()},
                          'activity_balanced_mae_bpm': None if missing else float(np.mean(list(activity_mae.values()))),
                          'worst_activity_mae_bpm': max(activity_mae.values())},
            'uncertainty': dict(zip(('interval_score_90', 'coverage_90', 'width_bpm'), map(float, aggregate[4:]))),
            'per_person_mae_bpm': {p: float(v[0]) for p, v in per_person.items()},
            'per_bout_mae_bpm': {p+"/"+s: float(np.mean(v, axis=0)[0]) for (p, s), v in bouts.items()},
            'n_people': len(people), 'n_bouts': len(bouts), 'n_windows': len(ids), 'failure_reason': None}


def paired_person_bootstrap(first, second, seed=0, repeats=2000):
    if set(first) != set(second) or not first:
        raise ValueError('Compare identical nonempty people')
    differences = np.array([first[p]-second[p] for p in sorted(first)])
    rng = np.random.default_rng(seed)
    means = differences[rng.integers(0, len(differences), (repeats, len(differences)))].mean(axis=1)
    return {'mean_difference_bpm': float(differences.mean()),
            'ci95_bpm': np.quantile(means, [.025, .975]).tolist(), 'n_people': len(differences), 'seed': seed}
