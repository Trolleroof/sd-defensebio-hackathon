"""Heat/load scoring: equal slices within bout, bouts within person, equal people."""
from collections import defaultdict
import math
import numpy as np
from research_env.core.common import canonical, finite, invalid_score, validate_prediction


def validate(prediction, task):
    if task == 'load':
        validate_prediction(prediction, ('mean_w','lower_w','upper_w'))
        return {None: (prediction['mean_w'], prediction['lower_w'], prediction['upper_w'])}
    if not isinstance(prediction, dict) or set(prediction) != {'forecasts','state'} or len(canonical(prediction['state'])) > 16384:
        raise ValueError('Heat output requires forecasts and bounded state')
    forecasts = prediction['forecasts']
    if not isinstance(forecasts, list) or len(forecasts) != 3:
        raise ValueError('All three heat horizons required')
    values = {}
    for p, h in zip(forecasts, (5,10,20)):
        if set(p) != {'horizon_min','mean_c','lower_c','upper_c'} or type(p['horizon_min']) is not int or p['horizon_min'] != h:
            raise ValueError('Heat horizons must be ordered 5,10,20')
        m,lo,hi = (p[k] for k in ('mean_c','lower_c','upper_c'))
        if not all(finite(v) for v in (m,lo,hi)) or not lo <= m <= hi:
            raise ValueError('Invalid heat interval')
        values[h] = (m,lo,hi)
    return values


def score(reference_records, predictions, task):
    ids = [r['id'] for r in reference_records]
    if not ids or len(set(ids)) != len(ids) or set(ids) != set(predictions):
        return invalid_score('Missing, duplicate, or extra predictions')
    groups, slices, hot, ordering = defaultdict(list), defaultdict(list), [], defaultdict(list)
    try:
        for r in reference_records:
            output = validate(predictions[r['id']], task)
            for target in r['targets']:
                h, y = target['slice'], target['value']
                m,lo,hi = output[h if task == 'heat' else None]
                if not finite(y):
                    raise ValueError('Invalid reference')
                values = [abs(m-y), (m-y)**2, m-y,
                          hi-lo+20*max(lo-y,0)+20*max(y-hi,0), float(lo<=y<=hi), hi-lo]
                if not all(finite(v) for v in values):
                    raise ValueError('Metric overflow')
                groups[(r['person'],r['session'],h)].append(values)
                slices[(h,r['person'])].append(values)
                if task == 'heat' and y >= 38.5:
                    hot.append((r['person'], max(y-m,0)))
                if task == 'load':
                    ordering[r['person']].append((h,y,m))
        per_bout, per_person = defaultdict(list), defaultdict(list)
        for (person,session,h), values in groups.items():
            per_bout[(person,session)].append(np.mean(values,axis=0))
        for (person,session), values in per_bout.items():
            per_person[person].append(np.mean(values,axis=0))
        means = {p: np.mean(v,axis=0) for p,v in per_person.items()}
        total = np.mean(list(means.values()),axis=0)
        slice_values = defaultdict(list)
        for (h,p), values in slices.items():
            slice_values[h].append(np.mean(values,axis=0))
        slice_means = {str(h): np.mean(v,axis=0) for h,v in slice_values.items()}
    except (ValueError,KeyError,TypeError,OverflowError) as exc:
        return invalid_score(str(exc))
    secondary = {'rmse': math.sqrt(float(total[1])), 'signed_bias': float(total[2]),
                 'per_slice': {h: {'mae': float(v[0]), 'rmse': math.sqrt(float(v[1])), 'bias': float(v[2])} for h,v in slice_means.items()},
                 'worst_slice_mae': max(float(v[0]) for v in slice_means.values())}
    if task == 'heat':
        hot_people = defaultdict(list)
        for p,v in hot:
            hot_people[p].append(v)
        secondary.update(hot_reference_threshold_c=38.5, hot_target_count=len(hot),
                         hottest_underprediction_c=float(np.mean([np.mean(v) for v in hot_people.values()])) if hot else None)
    else:
        secondary['mae_w_per_kg'] = float(np.mean([
            np.mean([abs(predictions[r['id']]['mean_w']-r['targets'][0]['value'])/r['request']['mass_kg']
                     for r in reference_records if r['person']==p]) for p in means]))
        agreement = []
        for rows in ordering.values():
            comparisons = [float(np.sign(a[1]-b[1])==np.sign(a[2]-b[2]))
                           for i,a in enumerate(rows) for b in rows[i+1:]]
            if comparisons:
                agreement.append(np.mean(comparisons))
        secondary['measured_order_agreement'] = float(np.mean(agreement)) if agreement else None
    return {'execution_valid': True, 'primary_loss': float(total[0]), 'units': 'degC' if task=='heat' else 'W',
            'uncertainty': {'interval_score_90': float(total[3]), 'coverage_90': float(total[4]), 'width': float(total[5])},
            'secondary': secondary, 'per_person_mae': {p: float(v[0]) for p,v in means.items()},
            'n_people': len(means), 'n_records': len(ids), 'failure_reason': None, 'resource_use': None, 'skill': None}
