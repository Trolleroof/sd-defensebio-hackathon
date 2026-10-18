"""Load-task scoring: condition -> person -> equal mean over people. Watts."""
from collections import defaultdict
import numpy as np
from research_env.core.common import finite, invalid_score

FIELDS = ('mean_w', 'lower_w', 'upper_w')


def validate_prediction(prediction):
    if not isinstance(prediction, dict) or set(prediction) != set(FIELDS):
        raise ValueError('Wrong prediction fields: require exactly mean_w, lower_w, upper_w')
    mean, lower, upper = (prediction[k] for k in FIELDS)
    if not all(finite(v) for v in (mean, lower, upper)):
        raise ValueError('Nonfinite or non-numeric prediction')
    if not lower <= mean <= upper:
        raise ValueError('Unordered interval')
    return prediction


def interval_score(lower, upper, y):
    return (upper-lower) + 20*max(lower-y, 0) + 20*max(y-upper, 0)


def ordering_concordance(pairs):
    """pairs: [(predicted, measured)] for one person. Pairs with tied measured values are
    skipped; tied predictions earn 0.5. Returns None if no comparable pair."""
    total, hits = 0, 0.
    for i in range(len(pairs)):
        for j in range(i+1, len(pairs)):
            dm = pairs[j][1]-pairs[i][1]
            if dm == 0:
                continue
            dp = pairs[j][0]-pairs[i][0]
            total += 1
            hits += 0.5 if dp == 0 else float((dp > 0) == (dm > 0))
    return None if total == 0 else hits/total


def score(reference_records, predictions, scoring_config=None):
    config = {'epsilon': 1e-9, **(scoring_config or {})}
    ids = [r['id'] for r in reference_records]
    if not ids or len(ids) != len(set(ids)) or set(predictions) != set(ids):
        return invalid_score('Missing, duplicate, or extra eligible predictions')
    people = defaultdict(list)
    loads = defaultdict(lambda: defaultdict(list))
    sexes = {}
    try:
        for ref in reference_records:
            p = validate_prediction(predictions[ref['id']])
            y = ref['reference_w']
            if not finite(y) or not finite(ref['body_mass_kg']) or ref['body_mass_kg'] <= 0:
                raise ValueError('Nonfinite reference')
            m, lo, hi = (float(p[k]) for k in FIELDS)
            err = abs(m-y)
            values = [err, (m-y)**2, m-y, err/ref['body_mass_kg'], interval_score(lo, hi, y),
                      float(lo <= y <= hi), hi-lo]
            if not all(finite(v) for v in values):
                raise ValueError('Metric arithmetic overflow')
            people[ref['person']].append((values, m, y))
            loads[ref['load_kg']][ref['person']].append(err)
            sexes[ref['person']] = ref.get('sex')
    except (ValueError, KeyError, TypeError, OverflowError) as exc:
        return invalid_score(str(exc))
    per_person = {p: np.mean([v for v, _, _ in rows], axis=0) for p, rows in people.items()}
    aggregate = np.mean(list(per_person.values()), axis=0)
    load_mae = {str(l): float(np.mean([np.mean(e) for e in ps.values()])) for l, ps in sorted(loads.items())}
    ordering = {p: ordering_concordance([(m, y) for _, m, y in rows]) for p, rows in people.items()}
    ordered = [v for v in ordering.values() if v is not None]
    sex_mae = {}
    for s in sorted({s for s in sexes.values() if s}):
        group = [per_person[p][0] for p in per_person if sexes[p] == s]
        sex_mae[s] = {'mae_w': float(np.mean(group)), 'people': len(group)}
    loss = float(aggregate[0])
    baseline = config.get('frozen_baseline_loss')
    return {'execution_valid': True, 'primary_loss': loss, 'units': 'W', 'resource_use': None,
            'skill': None if baseline is None else 1-loss/max(baseline, config['epsilon']),
            'secondary': {'rmse_w': float(np.sqrt(aggregate[1])), 'bias_w': float(aggregate[2]),
                          'mae_w_per_kg': float(aggregate[3]), 'load_mae_w': load_mae,
                          'worst_load_mae_w': max(load_mae.values()),
                          'ordering_concordance': float(np.mean(ordered)) if ordered else None,
                          'ordering_people': len(ordered), 'sex_mae_w': sex_mae},
            'uncertainty': dict(zip(('interval_score_90', 'coverage_90', 'width_w'), map(float, aggregate[4:]))),
            'per_person_mae_w': {p: float(v[0]) for p, v in per_person.items()},
            'n_people': len(people), 'n_conditions': len(ids), 'failure_reason': None}
