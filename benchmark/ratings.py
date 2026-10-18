"""Versioned, reproducible Elo summaries of matched campaign outcomes.

Raw metrics remain authoritative. Ratings summarize this opponent/case pool only.
"""
import itertools
import math
import random


VERSION = 'campaign-elo-v1'


def _valid_metric(run, metric):
    result = run['final']
    if result.get('valid') is not True:
        return None
    value = result.get(metric)
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError('valid runs require finite nonnegative '+metric)
    if result.get('code_execution_verified') is not True:
        raise ValueError('ranked runs require executable-code verification')
    return value


def leaderboard(runs, metric='mae_bpm', *, seed=0, initial=1000.0, k=32.0, tie_tolerance=1e-9):
    if metric not in ('mae_bpm', 'interval_score_90'):
        raise ValueError('unsupported rating metric')
    if not runs:
        raise ValueError('matched runs required')
    if type(seed) is not int or not all(type(x) in (int,float) and math.isfinite(x) for x in (initial,k,tie_tolerance)) or k <= 0 or tie_tolerance < 0:
        raise ValueError('invalid rating configuration')
    cells={}
    for run in runs:
        name=run.get('competitor')
        case=run.get('case')
        if not isinstance(name,str) or not name or type(case) is not int:
            raise ValueError('competitor names and integer case IDs required')
        key=(case,name)
        if key in cells:
            raise ValueError('duplicate competitor/case; use explicit repetition IDs')
        cells[key]=_valid_metric(run,metric)
    names=sorted({name for _,name in cells})
    cases=sorted({case for case,_ in cells})
    if len(names)<2:
        raise ValueError('at least two competitors required')
    if set(cells) != set(itertools.product(cases,names)):
        raise ValueError('all competitors must run the same complete case batch; failures must be explicit')
    ratings={name:float(initial) for name in names}
    counts={name:{'wins':0,'draws':0,'losses':0,'skipped_both_invalid':0,
                  'valid_cases':sum(cells[(case,name)] is not None for case in cases)} for name in names}
    schedule=[(case,a,b) for case in cases for a,b in itertools.combinations(names,2)]
    random.Random(seed).shuffle(schedule)
    matches=[]
    for case,a,b in schedule:
        av,bv=cells[(case,a)],cells[(case,b)]
        if av is None and bv is None:
            counts[a]['skipped_both_invalid']+=1
            counts[b]['skipped_both_invalid']+=1
            matches.append({'case':case,'a':a,'b':b,'outcome':'skipped_both_invalid'})
            continue
        outcome=(0.0 if av is None else 1.0 if bv is None else
                 .5 if abs(av-bv)<=tie_tolerance else 1.0 if av<bv else 0.0)
        expected=1/(1+10**(max(-300,min(300,(ratings[b]-ratings[a])/400))))
        delta=k*(outcome-expected)
        ratings[a]+=delta
        ratings[b]-=delta
        for name,points in ((a,outcome),(b,1-outcome)):
            counts[name]['wins' if points==1 else 'losses' if points==0 else 'draws']+=1
        matches.append({'case':case,'a':a,'b':b,'score_a':outcome,
                        'metric_a':av,'metric_b':bv,'rating_delta_a':delta})
    entries=[]
    for name in names:
        games=sum(counts[name][field] for field in ('wins','draws','losses'))
        entries.append({'competitor':name,'elo':ratings[name],'games':games,
                        'eligible':counts[name]['valid_cases']>0 and games>0,**counts[name]})
    entries.sort(key=lambda row:(not row['eligible'],-row['elo'],row['competitor']))
    for row in entries:
        row['rank']=(1+sum(other['eligible'] and other['elo']>row['elo']+1e-9 for other in entries)
                     if row['eligible'] else None)
    return {'version':VERSION,'metric':metric,'direction':'lower error wins',
            'configuration':{'initial':initial,'k':k,'scale':400,'match_order_seed':seed,
                             'absolute_tie_tolerance':tie_tolerance},
            'scope':{'case_ids':cases,'competitors':names,'unit':'one complete campaign',
                     'status':'provisional; pool-dependent and match-order-dependent; no confidence interval estimated'},
            'entries':entries,'matches':matches}


def campaign_ratings(runs):
    return {'prediction':leaderboard(runs,'mae_bpm'),
            'uncertainty':leaderboard(runs,'interval_score_90')}


def physbench_ratings(results, metadata):
    """Rate one existing PhysBench run without executing models or changing scores."""
    models = [m['name'] for m in metadata['models']]
    subjects = metadata['subjects']
    repeats = metadata['repeats']
    if (len(models) != len(set(models)) or len(subjects) != len(set(subjects))
            or type(repeats) is not int or repeats < 1):
        raise ValueError('invalid run metadata')
    expected = set(itertools.product(subjects, range(repeats), models))
    indexed = {}
    for result in results:
        key = (result['subject'], result['repeat'], result['model'])
        if key in indexed or key not in expected:
            raise ValueError('duplicate or unexpected PhysBench result')
        indexed[key] = result
    if set(indexed) != expected:
        raise ValueError('incomplete PhysBench run; finish all declared jobs before rating')
    rows, excluded, matched = [], [], []
    for subject in sorted(subjects):
        for repeat in range(repeats):
            case_results = [indexed[(subject,repeat,model)] for model in models]
            if any(r.get('infra_error') or r.get('harness_error') for r in case_results):
                excluded.append({'subject':subject,'repeat':repeat,
                                 'reason':'infrastructure or harness failure; exclude matched case for all models'})
                continue
            case = len(matched)
            matched.append({'case':case,'subject':subject,'repeat':repeat})
            for result in case_results:
                valid = (result.get('submitted') is True and result.get('valid_execution') is True
                         and result.get('criteria',{}).get('integrity') is True)
                rows.append({'case':case,'competitor':result['model'],
                             'final':{'valid':valid,'code_execution_verified':valid,
                                      'mae_bpm':result.get('agent_mae')}})
    if not rows:
        return {'version':VERSION,'status':'no_matched_cases','excluded_cases':excluded,'entries':[]}
    board = leaderboard(rows)
    board['scope'].update(source='PhysBench', metric='hidden phase-balanced MAE',
                          split=metadata['split'], run_id=metadata['run_id'],
                          task_config_sha256=metadata.get('task_config_sha256'),
                          criteria_sha256=metadata.get('criteria_sha256'),
                          eligibility='explicit submitted execution plus existing frozen integrity gate',
                          cases=matched)
    board['excluded_cases']=excluded
    board['note']='Subject repeats are correlated. Development Elo is for development; it is not held-out proof. No uncertainty Elo: this source contract does not supply prediction intervals.'
    return board
