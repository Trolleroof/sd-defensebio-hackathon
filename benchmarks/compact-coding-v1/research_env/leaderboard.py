"""Frozen three-task leaderboard, sealed final receipts, and matched-case Elo."""
import argparse
from datetime import datetime, timezone
import itertools
import os
from pathlib import Path
import random
import re
import subprocess
from research_env.core.common import digest, file_hash, finite, invalid_score, read, tree_hashes, write
from research_env.tasks.registry import TASKS, identify

POLICY = {'version': 'physiology-leaderboard-v1', 'tasks': list(TASKS),
          'agent_seeds': [0,1,2],
          'tier': 'primary-no-reference-calibration', 'wall_s': 7200, 'development_calls': 30,
          'max_cpu': 8, 'max_memory_gib': 32, 'backend': 'docker',
          'initial_elo': 1000, 'k': 32, 'scale': 400, 'match_order_seed': 0,
          'tie_tolerance': 1e-9, 'judge_rubric': 'physiology-research-judge-v1',
          'suite_match': 'equal task votes on matched agent seed; all tasks valid beats any invalid task; both incomplete invalid skipped',
          'failures': 'valid beats invalid; both invalid skipped; zero valid entries unranked',
          'judge': 'separate criterion means/counts; no mixture with prediction Elo',
          'sample_unit': 'one fixed full held-out cohort per task; seeds repeat agents, not biological cohorts'}


def seal(body):
    return {**body, 'sha256': digest(body)}


def verify(value):
    if value.get('sha256') != digest({k:v for k,v in value.items() if k!='sha256'}):
        raise ValueError('Frozen record hash mismatch')
    return value


def code_hashes():
    root = Path(__file__).resolve().parent
    hashes = {str(p.relative_to(root)):file_hash(p) for p in sorted(root.rglob('*'))
              if p.is_file() and p.suffix in ('.py','.md','.lock','.txt')}
    hashes['scripts/search_tasks.py'] = file_hash(root.parent/'scripts/search_tasks.py')
    return hashes


def resource_limits():
    values = subprocess.check_output(['docker','info','--format','{{.NCPU}} {{.MemTotal}}'],
                                     text=True,timeout=15).split()
    return {'cpus':min(8,os.cpu_count() or 1,int(values[0])),
            'memory_bytes':min(32*1024**3,int(values[1]))}


def freeze(data_root, image, output):
    if Path(output).exists():
        raise ValueError('Refuse to overwrite frozen suite')
    if not re.fullmatch(r'sha256:[0-9a-f]{64}', image):
        raise ValueError('Freeze an immutable local Docker image ID')
    datasets = {}
    for task in TASKS:
        root = (Path(data_root)/task).resolve()
        if identify(root/'private/test/manifest.json') != task:
            raise ValueError('Task mismatch')
        datasets[task] = {'root':str(root),'tree_sha256':digest(tree_hashes(root)),
                          'dataset':read(root/'dataset.json')}
    suite = seal({'policy':POLICY,'datasets':datasets,'image':image,'code_hashes':code_hashes(),
                  'resource_limits':resource_limits(),
                  'created_utc':datetime.now(timezone.utc).isoformat(),
                  'scope':'retrospective fixed cohorts; pulse ECG references provisional',
                  'outer_split':'agent_development'})
    write(output,suite)
    return suite


def check_suite(suite, task=None):
    verify(suite)
    if suite['policy'] != POLICY or suite['code_hashes'] != code_hashes():
        raise ValueError('Suite code/policy changed; freeze a new version before comparison')
    if suite['resource_limits'] != resource_limits():
        raise ValueError('Docker CPU/memory allocation differs from the frozen comparison')
    for name in ([task] if task else TASKS):
        dataset = suite['datasets'][name]
        if digest(tree_hashes(dataset['root'])) != dataset['tree_sha256']:
            raise ValueError('Frozen dataset changed')


def initialize(suite_path, task, config_path, seed, run_root):
    from research_env.staging import stage
    suite, config = read(suite_path), read(config_path)
    check_suite(suite,task)
    required = {'competitor','provider','model','decoding','prompt'}
    if not required <= set(config) or any(not isinstance(config[k],str) or not config[k].strip() for k in required-{'decoding'}):
        raise ValueError('Config needs competitor/provider/exact model/prompt and decoding settings')
    if not isinstance(config['decoding'],dict) or not config['decoding'] or type(seed) is not int or seed not in POLICY['agent_seeds']:
        raise ValueError('Explicit decoding settings and integer seed required')
    root = Path(run_root).resolve()
    workspace = stage(Path(suite['datasets'][task]['root'])/'public',root/'workspace')
    meta = seal({'suite_sha256':suite['sha256'],'task':task,'seed':seed,'competitor':config['competitor'],
                 'configuration':config,'configuration_sha256':digest(config),
                 'task_prompt_sha256':file_hash(workspace/f'research_env/tasks/{task}/TASK.md'),
                 'initialized_utc':datetime.now(timezone.utc).isoformat()})
    write(root/'run.json',meta)
    return {'run':str(root),'workspace':str(workspace),'task':task,
            'development_manifest':str(Path(suite['datasets'][task]['root'])/'private/development/manifest.json'),
            'image':suite['image']}


def finalize(suite_path, run_root, failure=None, *, fault=None):
    from research_env.core.session import Session
    from research_env.core.evaluation import evaluate
    from research_env.judging.judge import build_packets, summarize
    suite, root = read(suite_path),Path(run_root)
    meta = verify(read(root/'run.json'))
    task = meta['task']
    check_suite(suite,task)
    if meta['suite_sha256'] != suite['sha256'] or (root/'receipt.json').exists():
        raise ValueError('Wrong suite or final replay already sealed')
    session = Session(root/'private')
    if session.get('wall_s') != POLICY['wall_s'] or session.get('development_calls') != POLICY['development_calls']:
        raise ValueError('Run budgets differ from frozen protocol')
    identity = session.db.execute('SELECT value FROM settings WHERE key="task_identity"').fetchone()
    expected = {'task':task,'manifest_sha256':digest(read(Path(suite['datasets'][task]['root'])/'private/development/manifest.json')),
                'backend':'docker','image':suite['image']}
    if not failure and (not identity or __import__('json').loads(identity[0]) != expected):
        raise ValueError('Run did not use frozen dataset and Docker backend')
    events = session.events()
    final = [e for e in events if e['kind']=='final_selection']
    if failure:
        if not isinstance(failure,str) or not failure.strip():
            raise ValueError('Explicit failure reason required')
        with session.db:
            session.put('closed',True)
            session.append({'kind':'failed_attempt','reason':failure})
        result = invalid_score(failure)
        candidate = None
    else:
        if not session.get('closed') or len(final)!=1 or final[0]['candidate_id'] != session.get('selected'):
            raise ValueError('Replay only the final submitted selection after feedback closes')
        candidate = session.get('selected')
        try:
            result,_ = evaluate(root/'private/candidates'/candidate,
                              Path(suite['datasets'][task]['root'])/'private/test/manifest.json',image=suite['image'])
        except Exception as exc:
            result = invalid_score(f'{type(exc).__name__}: {exc}')
    session.export(root/'trajectory.jsonl',suite['outer_split'])
    packets = build_packets(session.events())
    write(root/'judge-packets.json',packets)
    write(root/'judging.json',summarize(packets,[]))
    final_answer = final[0]['reason'] if final and not failure else None
    status = {'status':'completed' if result['execution_valid'] else 'failed',
              'fault':None if result['execution_valid'] else fault or {
                  'kind':'failed_attempt' if failure else 'invalid_submission',
                  'reason':failure or result.get('failure_reason')},
              'final_answer':final_answer, 'candidate_id':candidate,
              'execution_valid':result['execution_valid'],
              'elapsed_wall_s':max(0,session.clock()-session.get('started')),
              'participant_elapsed_wall_s':max(0,(final[0]['timestamp'] if final else session.clock())-session.get('started')),
              'usage':read(root/'provider-usage.json') if (root/'provider-usage.json').exists() else None}
    with session.db:
        session.transcript.append('run_status',status)
    from research_env.trace import export
    export(root)
    receipt = seal({'run':meta,'score':result,'candidate_id':candidate,
                    'trajectory_sha256':file_hash(root/'trajectory.jsonl'),
                    'transcript_sha256':file_hash(root/'transcript.jsonl'),
                    'run_status_sha256':file_hash(root/'run-status.json'),
                    'judge_packets_sha256':digest(packets),'development_calls_used':session.get('used')})
    write(root/'receipt.json',receipt)
    session.db.close()
    return receipt


def outcome(a,b,metric):
    av,bv = a['execution_valid'],b['execution_valid']
    if not av and not bv:
        return None
    if av != bv:
        return float(av)
    x = a['primary_loss'] if metric=='prediction' else a['uncertainty']['interval_score_90']
    y = b['primary_loss'] if metric=='prediction' else b['uncertainty']['interval_score_90']
    if not finite(x) or not finite(y):
        raise ValueError('Valid results require finite point and interval losses')
    return .5 if abs(x-y)<=POLICY['tie_tolerance'] else float(x<y)


def elo(models,seeds,cases,metric,suite=False):
    ratings = {m:float(POLICY['initial_elo']) for m in models}
    counts = {m:dict(wins=0,draws=0,losses=0,skipped=0,valid_runs=0) for m in models}
    for m in models:
        counts[m]['valid_runs'] = sum(all(cases[(m,s,t)]['execution_valid'] for t in TASKS) if suite else cases[(m,s)]['execution_valid'] for s in seeds)
    matches = list(itertools.product(seeds,list(itertools.combinations(models,2))))
    random.Random(POLICY['match_order_seed']).shuffle(matches)
    exported=[]
    for seed,(a,b) in matches:
        if suite:
            valid_a = all(cases[(a,seed,t)]['execution_valid'] for t in TASKS)
            valid_b = all(cases[(b,seed,t)]['execution_valid'] for t in TASKS)
            if not valid_a and not valid_b:
                win=None
            elif valid_a!=valid_b:
                win=float(valid_a)
            else:
                votes=sum(outcome(cases[(a,seed,t)],cases[(b,seed,t)],metric) for t in TASKS)/3
                win=.5 if abs(votes-.5)<=POLICY['tie_tolerance'] else float(votes>.5)
        else:
            win=outcome(cases[(a,seed)],cases[(b,seed)],metric)
        if win is None:
            counts[a]['skipped']+=1;counts[b]['skipped']+=1
            exported.append({'seed':seed,'a':a,'b':b,'outcome':None,'delta':0})
            continue
        expectation=1/(1+10**((ratings[b]-ratings[a])/POLICY['scale']))
        change=POLICY['k']*(win-expectation)
        ratings[a]+=change;ratings[b]-=change
        for m,w in ((a,win),(b,1-win)):
            counts[m]['draws' if w==.5 else 'wins' if w==1 else 'losses']+=1
        exported.append({'seed':seed,'a':a,'b':b,'outcome':win,'delta':change})
    entries=[]
    for m in sorted(models,key=lambda m:(counts[m]['valid_runs']==0,-ratings[m],m)):
        entries.append({'competitor':m,'elo':ratings[m],**counts[m],
                        'rank':None if not counts[m]['valid_runs'] or len(models)<2 else 1+sum(
                            counts[n]['valid_runs']>0 and ratings[n]>ratings[m]+1e-9 for n in models)})
    return {'entries':entries,'matches':exported}


def build(suite, receipts, judge_reports=None):
    verify(suite)
    rows,configs={},{}
    for receipt in receipts:
        verify(receipt);meta=verify(receipt['run'])
        if meta['suite_sha256']!=suite['sha256'] or meta['task'] not in TASKS or meta['seed'] not in POLICY['agent_seeds']:
            raise ValueError('Receipt from another suite/task')
        name=meta['competitor']
        if name in configs and configs[name]!=meta['configuration_sha256']:
            raise ValueError('Competitor settings changed within pool')
        configs[name]=meta['configuration_sha256']
        key=(name,meta['seed'],meta['task'])
        if key in rows:
            raise ValueError('Duplicate model/seed/task attempt')
        result=receipt['score']
        if type(result.get('execution_valid')) is not bool:
            raise ValueError('Explicit validity required')
        if result['execution_valid']:
            outcome(result,result,'prediction');outcome(result,result,'uncertainty')
        elif result.get('primary_loss') is not None or not result.get('failure_reason'):
            raise ValueError('Invalid attempt requires null loss and explicit failure')
        rows[key]=result
    models=sorted(configs)
    seeds=POLICY['agent_seeds'] if receipts else []
    expected=set(itertools.product(models,seeds,TASKS))
    missing=sorted(expected-set(rows))
    if missing:
        raise ValueError(f'Incomplete matched batch; supply missing attempts or explicit failures: {missing}')
    tasks={}
    for t in TASKS:
        cases={(m,s):rows[(m,s,t)] for m in models for s in seeds}
        raw=[]
        for m in models:
            scores=[rows[(m,s,t)] for s in seeds]
            good=[r for r in scores if r['execution_valid']]
            raw.append({'competitor':m,'validity_rate':len(good)/len(scores),
                        'mean_loss_valid_runs':sum(r['primary_loss'] for r in good)/len(good) if good else None,
                        'attempts':[{'seed':s,'score':rows[(m,s,t)]} for s in seeds]})
        tasks[t]={'prediction':elo(models,seeds,cases,'prediction'),
                  'uncertainty':elo(models,seeds,cases,'uncertainty'),'raw':raw}
    return {'version':POLICY['version'],'suite_sha256':suite['sha256'],
            'status':'empty' if not receipts else 'ranked' if len(models)>1 else 'awaiting opponents',
            'models':models,'agent_seeds':seeds,'tasks':tasks,
            'suite':{'prediction':elo(models,seeds,rows,'prediction',True),
                     'uncertainty':elo(models,seeds,rows,'uncertainty',True)},
            'judging':judge_reports or [],
            'interpretation':'Elo compares this opponent pool only. Agent seeds reuse fixed people. Pulse reference labels provisional; retrospective benchmark.',
            'judge_status':'pending' if not judge_reports else 'see per-run coverage'}


def render(board):
    lines=['# Physiology research-agent leaderboard','',
           f"Status: **{board['status']}**. Three tasks: pulse, core-temperature forecasting, load-carriage metabolic cost.",
           '',board['interpretation'],'',
           'Prediction and uncertainty Elo are separate. LLM process scores are separate and require actual validated judgments.',
           'No prior pilot results are included.','','## Overall suite','',
           '| Model | Prediction Elo | Rank | Valid complete runs |','|---|---:|---:|---:|']
    for row in board['suite']['prediction']['entries']:
        lines.append(f"| {row['competitor']} | {row['elo']:.1f} | {row['rank'] if row['rank'] else '—'} | {row['valid_runs']} |")
    if not board['models']:
        lines+=['','No model runs have been entered.']
    for t,label,unit in [('pulse','Pulse','bpm'),('heat','Core-temperature forecasting','°C'),('load','Load-carriage metabolic cost','W')]:
        lines+=['',f'## {label}','',f'| Model | MAE ({unit}, valid runs) | Prediction Elo | Uncertainty Elo | Validity |',
                '|---|---:|---:|---:|---:|']
        raw={r['competitor']:r for r in board['tasks'][t]['raw']}
        uncertainty={r['competitor']:r['elo'] for r in board['tasks'][t]['uncertainty']['entries']}
        for r in board['tasks'][t]['prediction']['entries']:
            v=raw[r['competitor']];loss=v['mean_loss_valid_runs']
            shown_loss=f'{loss:.4f}' if loss is not None else 'invalid'
            lines.append(f"| {r['competitor']} | {shown_loss} | {r['elo']:.1f} | {uncertainty[r['competitor']]:.1f} | {v['validity_rate']:.0%} |")
    lines+=['','## LLM process judging','','Status: '+board['judge_status']+'.',
            'Criterion means and observable counts are reported per run. Missing judgments never become zero/default scores.']
    for j in board['judging']:
        lines+=['',f"### {j['competitor']} / {j['task']} / seed {j['seed']}",
                '',f"{j['report']['llm_judge_status']}: {j['report']['judged_packets']}/{j['report']['total_packets']} packets; process score {j['report']['process_score']}. Judge: {(j.get('configuration') or {}).get('model','pending')}.",
                '', '| Criterion | Mean | Observable decisions |','|---|---:|---:|']
        for name,v in j['report']['criteria'].items():
            lines.append(f"| {name} | {v['mean'] if v['mean'] is not None else 'pending'} | {v['observable_count']} |")
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('freeze');p.add_argument('data_root');p.add_argument('output');p.add_argument('--image',required=True)
    p=sub.add_parser('init');p.add_argument('suite');p.add_argument('task',choices=TASKS);p.add_argument('config');p.add_argument('run_root');p.add_argument('--seed',type=int,required=True)
    p=sub.add_parser('finalize');p.add_argument('suite');p.add_argument('run_root');p.add_argument('--failure')
    p=sub.add_parser('build');p.add_argument('suite');p.add_argument('output_dir');p.add_argument('runs',nargs='*')
    args=parser.parse_args()
    if args.command=='freeze':
        suite=freeze(args.data_root,args.image,args.output);print(suite['sha256'])
    elif args.command=='init':
        print(__import__('json').dumps(initialize(args.suite,args.task,args.config,args.seed,args.run_root),indent=2))
    elif args.command=='finalize':
        r=finalize(args.suite,args.run_root,args.failure);print(r['score']['execution_valid'])
    else:
        from research_env.judging.judge import summarize
        suite=read(args.suite);check_suite(suite)
        receipts,judges=[],[]
        judge_configuration=None
        for name in args.runs:
            root=Path(name);r=verify(read(root/'receipt.json'))
            if file_hash(root/'trajectory.jsonl')!=r['trajectory_sha256']:
                raise ValueError('Trajectory changed after final replay')
            if (file_hash(root/'transcript.jsonl')!=r['transcript_sha256']
                    or file_hash(root/'run-status.json')!=r['run_status_sha256']):
                raise ValueError('Transcript or terminal status changed after final replay')
            packets=read(root/'judge-packets.json')
            if digest(packets)!=r['judge_packets_sha256']:
                raise ValueError('Judge packets changed')
            judge_root=root/'judge-output'
            verdicts=read(judge_root/'verdicts.json') if (judge_root/'verdicts.json').exists() else []
            config=read(judge_root/'config.json') if verdicts else None
            if config:
                if judge_configuration is not None and config!=judge_configuration:
                    raise ValueError('Use one frozen judge configuration for the entire model pool')
                judge_configuration=config
            report=summarize(packets,verdicts)
            judges.append({'competitor':r['run']['competitor'],'task':r['run']['task'],'seed':r['run']['seed'],
                           'report':report,'configuration':config,'configuration_sha256':digest(config) if config else None})
            receipts.append(r)
        board=build(suite,receipts,judges)
        output=Path(args.output_dir);write(output/'leaderboard.json',board)
        (output/'LEADERBOARD.md').write_text(render(board))
        print(board['status'])


if __name__=='__main__':
    main()
