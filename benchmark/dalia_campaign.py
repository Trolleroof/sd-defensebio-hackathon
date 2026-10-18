"""Explicit within-recording activity campaigns on processed DaLiA-shaped data."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .campaign import CampaignWorkspace, STAGES, validate_campaign
from .dalia_synthetic import load_recording, runs, ACTIVITY_NAMES
from .workspace import write

VERSION = 'dalia-activity-campaign-v1'
ACTIVITIES = (1, 7, 2, 4)
FEATURES = ('acc_wrist_enmo', 'acc_wrist_std', 'acc_chest_enmo', 'acc_chest_std', 'temp_wrist')
HISTORY_N, FORECAST_N, GAP_N = 30, 30, 4
REAL_DEVELOPMENT = {'S2', 'S3', 'S7', 'S10'}


def episode(processed_dir, subject, *, origin, case=0):
    if origin not in ('synthetic', 'public'):
        raise ValueError('origin must be synthetic or public')
    if origin == 'public' and subject not in REAL_DEVELOPMENT:
        raise ValueError('Only explicitly permitted real development participants may be inspected')
    if origin == 'synthetic' and not subject.startswith('SYN'):
        raise ValueError('Synthetic source requires a SYN subject identifier')
    path = Path(processed_dir)/f'{subject}.npz'
    r = load_recording(path)
    size = HISTORY_N+FORECAST_N
    required = 3*size+2*GAP_N
    ranges, cursor = {}, 0
    for activity, count in runs(r['activity']):
        if activity in ACTIVITIES and count >= required:
            ranges.setdefault(activity, (cursor, count))
        cursor += count
    if set(ranges) != set(ACTIVITIES):
        raise ValueError('Insufficient disjoint activity coverage for '+str(sorted(set(ACTIVITIES)-set(ranges))))
    sessions, manifest = [], []
    for stage, activity in enumerate(ACTIVITIES):
        name = ACTIVITY_NAMES[activity]
        start, count = ranges[activity]
        # Fixed geometry and order, chosen by coverage alone, never prediction results.
        start += (count-required)//2
        labels = [('support' if stage == 0 else 'acquisition', 'calibration' if stage == 0 else 'diagnostic'),
                  ('development', 'feedback'), ('test', 'private')]
        for slot, (role, label) in enumerate(labels):
            a = start+slot*(size+GAP_N); b = a+size; cutoff = a+HISTORY_N-1
            rows = []
            for i in range(a, b):
                row = {'time_s': float((i-cutoff)*2), 'hr_bpm': float(r['hr_ecg'][i]), 'activity': activity}
                for feature in FEATURES:
                    value = float(r[feature][i])
                    row[feature] = value if np.isfinite(value) else None
                rows.append(row)
            sid = f'{name}-{label}'
            sessions.append({'id': sid, 'participant_id': 'case-participant', 'role': role,
                             'condition': name, 'release_stage': stage, 'cost': 1,
                             'description': f'{name}: 60s observed history and 60s continuation in the same activity; measured movement and skin temperature.',
                             'rows': rows})
            manifest.append({'id': sid, 'source_subject': subject, 'source_index_start': a,
                             'source_index_end_exclusive': b, 'source_cutoff_index': cutoff,
                             'source_time_start_s': float(r['t'][a]), 'source_time_end_s': float(r['t'][b-1])})
    ep = {'schema_version': 1, 'id': f'{VERSION}-{origin}-{case}', 'origin': origin,
          'budgets': {'acquisition': 3, 'evaluations': 8}, 'sessions': sessions,
          'provenance': {'target_provenance': ('Synthetic HR fitted to reference-cohort statistics' if origin == 'synthetic'
                                              else 'PPG-DaLiA ECG-derived reference HR'),
                         'source_dataset': 'PPG-DaLiA' if origin == 'public' else 'PPG-DaLiA-grounded statistical generator',
                         'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                         'source_subject': subject, 'windows': manifest,
                         'claim_boundary': 'Disjoint windows within one source recording; not independent sessions or participants.'},
          'campaign': {'stages': list(STAGES), 'task_profile': VERSION,
                       'context_fields': {'activity': ACTIVITY_NAMES,
                                          **{f: ('degC; skin temperature' if f == 'temp_wrist' else 'g; recorded movement feature') for f in FEATURES}},
                       'briefs': [
                           'Build a personalized HR forecast from sitting calibration. Each request provides 60s of HR, movement and skin-temperature history. Forecast HR for 60s; the recorded activity stays unchanged. Future HR, movement and skin-temperature measurements are not inputs. Null means missing; no imputation is supplied.',
                           'Investigate walking forecasts. You may acquire a complete walking calibration. Decide how to revise your model and retain sitting performance. Activity stays unchanged over each forecast.',
                           'Transfer to stairs. Choose whether to acquire matched calibration, adapt and retain sitting/walking performance. Activity stays unchanged over each forecast.',
                           'Sustain earlier capabilities while adapting to cycling and select your final candidate. These are activity segments within one recording, not a later-day physiological session. Activity stays unchanged over each forecast.']}}
    validate_campaign(ep)
    return ep


def baseline_codes():
    common = '''
import math
MODE = __MODE__
models = {}
def fit(support):
    global models
    pairs = {}
    for recording in support:
        rows = recording['rows']
        for x,y in zip(rows[:-1], rows[1:]):
            key = int(x['activity']) if MODE == 'contextual_ar' else 0
            pairs.setdefault(key, []).append((x['hr_bpm'], y['hr_bpm']))
    models = {}
    for key, data in pairs.items():
        mx = sum(x for x,y in data)/len(data)
        my = sum(y for x,y in data)/len(data)
        var = sum((x-mx)**2 for x,y in data)
        slope = sum((x-mx)*(y-my) for x,y in data)/max(var, 1e-9)
        slope = min(.995, max(0.0, slope))
        center = (mx+my)/2
        residuals = [y-(center+slope*(x-center)) for x,y in data]
        sigma = math.sqrt(sum(e*e for e in residuals)/len(residuals))
        models[key] = (center, slope, sigma)
def predict(history, forecast_times_s):
    last = history[-1]['hr_bpm']
    key = int(history[-1]['activity']) if MODE == 'contextual_ar' else 0
    model = models.get(key)
    deltas = [(y['hr_bpm']-x['hr_bpm'])/2 for x,y in zip(history[-6:-1],history[-5:])]
    trend = sum(deltas)/len(deltas) if deltas else 0
    result = []
    for t in forecast_times_s:
        if MODE == 'damped_trend':
            mean = last+trend*15*(1-math.exp(-t/15))
        elif MODE in ('pooled_ar', 'contextual_ar') and model:
            center,slope,sigma = model
            mean = center+(last-center)*slope**(t/2)
        else:
            mean = last
        width = max(3, (model[2] if model else 2)*math.sqrt(t/2)*1.645)
        result.append({'time_s':t,'mean':mean,'lower':mean-width,'upper':mean+width})
    return result
'''
    codes = {name: common.replace('__MODE__', repr(name)) for name in
             ('persistence', 'damped_trend', 'pooled_ar', 'contextual_ar')}
    codes['contextual_arx'] = r'''import math
models={}
FIELDS=('acc_wrist_enmo','acc_wrist_std','acc_chest_enmo','acc_chest_std','temp_wrist')
def features(row,hr=None):
    hr=row['hr_bpm'] if hr is None else hr
    vals=[1.0,(hr-80)/30]
    for key in FIELDS:
        value=row.get(key)
        if value is None:value=32 if key=='temp_wrist' else 0
        vals.append((value-32)/5 if key=='temp_wrist' else value*5)
    return vals

def solve(A,b):
    M=[list(row)+[v] for row,v in zip(A,b)];n=len(b)
    for i in range(n):
        pivot=max(range(i,n),key=lambda j:abs(M[j][i]));M[i],M[pivot]=M[pivot],M[i]
        scale=M[i][i]
        if abs(scale)<1e-12:continue
        M[i]=[v/scale for v in M[i]]
        for j in range(n):
            if j==i:continue
            scale=M[j][i];M[j]=[a-scale*b for a,b in zip(M[j],M[i])]
    return [row[-1] for row in M]

def fit(support):
    global models
    grouped={-1:[]}
    for recording in support:
        rows=recording['rows']
        for x,y in zip(rows[:-1],rows[1:]):
            pair=(features(x),y['hr_bpm']-x['hr_bpm'])
            grouped[-1].append(pair);grouped.setdefault(int(x['activity']),[]).append(pair)
    models={}
    for key,pairs in grouped.items():
        if not pairs:continue
        n=len(pairs[0][0]);A=[[sum(x[i]*x[j] for x,y in pairs)+(.1 if i==j and i else .001 if i==j else 0) for j in range(n)] for i in range(n)]
        b=[sum(x[i]*y for x,y in pairs) for i in range(n)]
        beta=solve(A,b)
        # Bound autonomous HR feedback; all other coefficients remain learned.
        beta[1]=min(0,max(-15,beta[1]))
        sigma=math.sqrt(sum((y-sum(a*b for a,b in zip(beta,x)))**2 for x,y in pairs)/len(pairs))
        models[key]=(beta,sigma)

def predict(history,forecast_times_s):
    row=history[-1];value=row['hr_bpm'];beta,sigma=models.get(int(row['activity']),models[-1])
    predictions=[];last_t=0
    for t in forecast_times_s:
        for _ in range(int(round((t-last_t)/2))):
            value+=sum(a*b for a,b in zip(beta,features(row,value)))
            value=min(230,max(25,value))
        width=max(3,1.645*sigma*math.sqrt(t/2))
        predictions.append({'time_s':t,'mean':value,'lower':value-width,'upper':value+width});last_t=t
    return predictions
'''
    return codes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('processed'); p.add_argument('subject'); p.add_argument('output')
    p.add_argument('--origin', choices=['synthetic', 'public'], required=True)
    p.add_argument('--case', type=int, default=0)
    args = p.parse_args()
    root = Path(args.output)
    ep = episode(args.processed, args.subject, origin=args.origin, case=args.case)
    w = CampaignWorkspace.create(ep, root/'private', root/'agent')
    print(json.dumps({'control': str(w.root), 'workspace': str(w.visible), 'profile': VERSION}))


if __name__ == '__main__':
    main()
