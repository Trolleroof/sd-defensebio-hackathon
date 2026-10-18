"""Conventional development-only search using the same runner as research models."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from research_env.core.agent import ResearchEnvironment
from research_env.core.common import read


def search(suite_path,task,run_root):
    suite,root=read(suite_path),Path(run_root)
    environment=ResearchEnvironment(root/'private',root/'workspace',
        Path(suite['datasets'][task]['root'])/'private/development/manifest.json',image=suite['image'])
    seed=read(root/'run.json')['seed']
    if task=='pulse':
        grid=[('median',10,.5),('spectral',10,.5)]+[('motion',10,w) for w in (0,.25,.5,.75,1)]+[('ridge',10**(-4+i*.4),.5) for i in range(23)]
    elif task=='heat':
        grid=[(kind,alpha,0) for kind in ('elapsed','ridge','thermal') for alpha in (.01,.1,1,10,100,1000)]+[('boosting',10**(-3+i*.4),0) for i in range(12)]
    else:
        grid=[('mean_by_load',10,0)]+[('ridge',10**(-4+i*.4),0) for i in range(23)]+[('forest',10,0)]
        # Five seed variants are predeclared, not selected using private labels.
        grid += [('forest',10+i,0) for i in range(5)]
    best=None
    for i,(kind,alpha,motion) in enumerate(grid):
        if environment.session.remaining()<=0:
            break
        evidence=[e['event_id'] for e in environment.session.events() if e['kind']=='development_result']
        decision={'hypothesis':f'Frozen grid candidate {i}: {kind}, alpha={alpha}, motion={motion}',
                  'expected_result':'Measure public development MAE under common budgets',
                  'code_diff':'Only predeclared baseline hyperparameters','evidence_ids':evidence}
        cmd=['python','-m','research_env','baseline','data/train/manifest.json','artifact',
             '--task',task,'--kind',kind,'--alpha',str(alpha),'--motion-weight',str(motion),'--seed',str(seed+(i if kind=='forest' else 0))]
        code='import shutil,subprocess;shutil.rmtree("artifact",ignore_errors=True);raise SystemExit(subprocess.call('+repr(cmd)+'))'
        trained=environment.act({'operation':'command','argv':['python','-c',code],'decision':decision,'seed':seed})
        # Failed training still consumes a development attempt via missing-artifact evaluation.
        result=environment.act({'operation':'evaluate','decision':decision})
        if trained['execution_valid'] and result['execution_valid'] and (best is None or result['primary_loss']<best['primary_loss']):
            best=result
            evidence=[e['event_id'] for e in environment.session.events() if e['kind']=='development_result']
            environment.act({'operation':'select','candidate_id':best['candidate_id'],
                             'reason':'Lowest observed public development MAE','evidence_ids':evidence})
    if best and environment.session.remaining()>0:
        evidence=[e['event_id'] for e in environment.session.events() if e['kind']=='development_result']
        environment.act({'operation':'submit','candidate_id':best['candidate_id'],
                         'reason':'Public development winner of predeclared grid','evidence_ids':evidence})
    return best if environment.session.get('closed') else None


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('suite');p.add_argument('task',choices=('pulse','heat','load'));p.add_argument('run_root')
    a=p.parse_args();search(a.suite,a.task,a.run_root)
