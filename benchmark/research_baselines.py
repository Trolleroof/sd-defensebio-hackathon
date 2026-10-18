"""Conventional structure search competitors; never injected into agent workspaces."""
from .baselines import persistence_code


SEARCH = '''
import math
_models = {}
def context(rows):
    r = rows[-1]
    return (r['bout_duration_s'], r['load_kg'], r['session_day'])
def curve(t, fast, slow, weight):
    return weight*math.exp(-t/fast)+(1-weight)*math.exp(-t/slow)
def fit_model(sessions):
    best = None
    for fast in (25,40,45,50,60,65,70,90,95,100):
        for slow in (120,180,185,190,260,265,270,360):
            for weight in (.25,.5,.75,1.0):
                error = 0.0
                count = 0
                for session in sessions:
                    rows = session['rows']
                    base = rows[0]['hr_bpm']
                    start = next(r['hr_bpm'] for r in rows if r['time_s']==0)
                    for row in rows:
                        if row['time_s']>0:
                            y = base+(start-base)*curve(row['time_s'], fast, slow, weight)
                            error += (y-row['hr_bpm'])**2
                            count += 1
                item = (error/max(count,1), fast, slow, weight)
                if best is None or item < best:
                    best = item
    return best[1:]
def predict(history, forecast_times_s):
    key = context(history)
    model = _models.get(key, _models.get('fallback'))
    if model is None:
        raise ValueError('fit must run before predict')
    base, start = history[0]['hr_bpm'], history[-1]['hr_bpm']
    out=[]
    for t in forecast_times_s:
        mean=base+(start-base)*curve(t,*model)
        out.append({'time_s':t,'mean':mean,'lower':mean-5,'upper':mean+5})
    return out
'''


def research_codes():
    pooled = SEARCH + '''
def fit(support):
    global _models
    _models={'fallback':fit_model(support)}
'''
    recent = SEARCH + '''
def fit(support):
    global _models
    _models={'fallback':fit_model([support[-1]])}
'''
    contextual = SEARCH + '''
def fit(support):
    global _models
    groups={}
    for session in support:
        groups.setdefault(context(session['rows']), []).append(session)
    _models={key:fit_model(sessions) for key,sessions in groups.items()}
    _models['fallback']=fit_model(support)
'''
    return {'persistence': persistence_code(), 'pooled_structure_search': pooled,
            'recent_only_search': recent, 'contextual_structure_search': contextual}
