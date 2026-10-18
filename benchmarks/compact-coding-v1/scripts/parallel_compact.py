"""Parallel API episodes with a single shared spend ledger and serialized Docker."""
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import signal
import threading
import time
from research_env import compact
from research_env.core import evaluation,session
from research_env.core.common import digest,file_hash,read,write
from research_env.core.spend import SpendLimitReached
from research_env.leaderboard import check_suite
from scripts.resume_compact import ProviderAwareLedger


def run(key,workers=3):
    if workers not in (3,4):raise ValueError('Authorized parallel configuration is three or four workers')
    root=Path('.leaderboard-runs/compact-eight');plan=read('leaderboard/sweeps/compact-eight/plan.json')
    lock=(root/'runner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    configs=[read(p) for p in plan['configs']];judge=read(plan['judge_config']);suite=read(plan['suite'])
    check_suite(suite)
    if [c['model'] for c in configs]!=list(compact.EXPANDED_MODELS) or plan['approved_fresh_budget_usd']!=16 or plan['seeds']!=[0,1,2]:raise ValueError('Unexpected parallel scope')
    fingerprint=digest({'plan':plan,'configs':configs,'judge':judge,'suite':suite['sha256']})
    state=read(root/'status.json')
    if state['fingerprint']!=fingerprint:raise ValueError('Original pool changed')
    amendment={'version':'parallel-orchestration-v1','workers':workers,'docker_concurrency':1,
      'implementation_sha256':file_hash(Path(__file__)),'accounting_amendment_sha256':file_hash(root/'accounting-amendment.json'),
      'scope':'Same frozen requests, trial seeds, cohorts, episode paths, receipts and $16 ledger; concurrent episode orchestration only.'}
    path=root/'parallel-amendment.json'
    if path.exists():
        prior=read(path)
        if {k:v for k,v in prior.items() if k!='previous_amendment_sha256'}==amendment:
            amendment=prior
        elif prior['workers']==3 and workers==4:
            parent=digest(prior);write(root/f'parallel-amendment-{parent[:12]}.json',prior)
            amendment['previous_amendment_sha256']=parent
        else:raise ValueError('Parallel amendment changed')
    write(path,amendment)
    gate=threading.BoundedSemaphore(1);original=session.execute
    def execute_serial(*args,**kwargs):
        started=time.monotonic()
        with gate:
            # Waiting for another case must not extend this case's wall budget.
            if 'timeout_s' in kwargs:kwargs['timeout_s']=max(.01,kwargs['timeout_s']-(time.monotonic()-started))
            return original(*args,**kwargs)
    session.execute=execute_serial;evaluation.execute=execute_serial
    stop=threading.Event();active={};errors=[];mutex=threading.Lock();ledger=ProviderAwareLedger(root/'spend.sqlite',16)
    def update(status=None):
        with mutex:
            if status:state['status']=status
            state.update(completed_episodes=len(list(root.glob('runs/**/receipt.json'))),
              judged_episodes=len(list(root.glob('runs/**/episode-judge/report.json'))),
              active_episodes=list(active.values()),parallel_workers=workers,runner_pid=os.getpid(),updated_at=time.time(),spend=ledger.summary())
            state.pop('active_model',None);state.pop('active_task',None);state.pop('active_seed',None);state.pop('error',None)
            if errors:state['parallel_errors']=list(errors)
            write(root/'status.json',state)
    previous=signal.signal(signal.SIGINT,lambda *args:stop.set())
    jobs=[]
    for seed in plan['seeds']:
        for task in compact.TASKS:
            for config,path in zip(configs,plan['configs']):
                p=compact.run_path(root,config,task,seed,plan['seeds'])
                if not (p/'receipt.json').exists() or not (p/'episode-judge/report.json').exists():jobs.append((config,path,task,seed,p))
    def work(job):
        if stop.is_set():return
        c,path,task,seed,p=job;identifier=f'{c["slug"]}/{task}/{seed}'
        spend=ProviderAwareLedger(root/'spend.sqlite',16)
        with mutex:active[identifier]={'model':c['model'],'task':task,'seed':seed}
        try:
            if not (p/'receipt.json').exists():compact.episode(plan['suite'],task,c,path,p,key,spend,seed)
            if not stop.is_set() and not (p/'episode-judge/report.json').exists():
                try:compact.review(p,judge,key,spend)
                except SpendLimitReached:raise
                except Exception as exc:write(p/'episode-judge/failure.json',{'error':type(exc).__name__,'message':compact.trim(str(exc),180)})
        except SpendLimitReached:
            with mutex:errors.append({'episode':identifier,'error':'SpendLimitReached'})
            stop.set()
        except Exception as exc:
            with mutex:errors.append({'episode':identifier,'error':type(exc).__name__,'message':compact.trim(str(exc),180)})
        finally:
            spend.close()
            with mutex:active.pop(identifier,None)
    try:
        update('running')
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            pending={};jobs=iter(jobs)
            def fill():
                while len(pending)<workers and not stop.is_set():
                    try:job=next(jobs)
                    except StopIteration:break
                    pending[pool.submit(work,job)]=job
            fill()
            while pending:
                done,_=concurrent.futures.wait(pending,timeout=2,return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:pending.pop(future);future.result()
                fill();update('stopping' if stop.is_set() else 'running')
        if stop.is_set():update('paused_spend_cap' if any(e['error']=='SpendLimitReached' for e in errors) else 'paused_by_user')
        else:
            result=compact.board(root,configs,judge,plan['seeds'])
            if result is None:update('needs_attention')
            else:
                result['orchestration_amendment']=amendment;result['accounting_amendment']=read(root/'accounting-amendment.json')
                write(root/'leaderboard.json',result);(root/'LEADERBOARD.md').write_text(compact.render(result))
                update('complete' if result['process_review']['coverage']==72 else 'complete_with_missing_judgments')
    finally:
        session.execute=original;evaluation.execute=original;signal.signal(signal.SIGINT,previous);ledger.close();lock.close()
    return read(root/'status.json')
