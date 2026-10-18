"""Small matched coding sweep: two experiments, selection, one blind episode review."""
import argparse
import fcntl
from copy import deepcopy
import json
from pathlib import Path
import shutil
import time
import urllib.request
from research_env.core.common import canonical,digest,file_hash,read,write
from research_env.core.spend import SpendLedger,SpendLimitReached
from research_env.core.transcript import Transcript
from research_env.core.agent import ResearchEnvironment
from research_env.leaderboard import initialize,finalize,check_suite,verify,elo
from research_env.sweep import credential

MODELS=('google/gemini-4-argon','anthropic/claude-sonnet-5.5','x-ai/grok-4.6',
        'openai/gpt-6-astra','google/gemini-3.8-flash')
EXPANDED_MODELS=('google/gemini-4-argon','anthropic/claude-sonnet-5.5','openai/gpt-6-astra',
 'anthropic/claude-opus-5.5','x-ai/grok-4.7','google/gemini-3.8-flash','moonshotai/kimi-k3','deepseek/deepseek-v4-flash')
TASKS=('load','heat','pulse')
CRITERIA=('evidence','experiment','feedback','repro','selection')
ACTION={'type':'function','function':{'name':'respond','description':'Return the requested experiment or final selection.',
 'parameters':{'type':'object','properties':{'code':{'type':'string'},'candidate':{'type':'integer'},
 'reason':{'type':'string','maxLength':160}},'required':['reason']}}}
GRADE={'type':'function','function':{'name':'grade','description':'Review the supplied episode evidence.',
 'parameters':{'type':'object','properties':{'criteria':{'type':'array','items':{'type':'object','properties':{
 'id':{'type':'string','enum':list(CRITERIA)},'score':{'type':['integer','null']},'reason':{'type':'string'},
 'evidence_ids':{'type':'array','items':{'type':'string'}}},'required':['id','score','reason','evidence_ids']}}},'required':['criteria']}}}
SYSTEM=('Develop an executable predictor in offline /work. You get two Python training experiments, '
 'then select one valid evaluated artifact. Call respond once. code must create artifact/inference.py '
 'with predict(artifact_dir,request), fitted files and requirements.lock. Use only staged public data; '
 'no network or private labels. reason is a brief evidence-based explanation, <=160 characters. '
 'You may use the provided baseline helper or write your own code. Use the supplied trial seed. '
 'Available: numpy, scipy, sklearn, joblib; pandas is unavailable. Keep code concise.')
BRIEFS={
 'load':'Predict gross metabolic W at 4 km/h, loads 0/5/10/15kg for unseen people. Inputs mass_kg,height_cm,age_years,load_kg. Output mean_w,lower_w,upper_w,state. Primary: person MAE.',
 'heat':'Forecast rectal temperature at +5/+10/+20min for unseen people using causal history of skin_c,hr_bpm,ambient_c,humidity_pct and demographics. Output forecasts with horizon_min,mean_c,lower_c,upper_c plus state. Primary: person MAE.',
 'pulse':'Estimate heart rate for unseen people using trailing PPG/acceleration windows. Output mean_bpm,lower_bpm,upper_bpm,state. Primary: person MAE. ECG-derived reference labels are provisional.'}


def trim(value,n):
    return str(value).encode()[:n].decode(errors='ignore')


def score_summary(value):
    return {'execution_valid':value.get('execution_valid'),'primary_loss':value.get('primary_loss'),
            'failure_reason':trim(value.get('failure_reason') or '',80) or None}


def request_body(config,observation,judge=False):
    options={k:deepcopy(v) for k,v in config['decoding'].items() if k in ('provider','reasoning','verbosity')}
    system=('Score only the blind supplied evidence. Five criteria: evidence use, experiment choice, feedback integration, '
            'reproducibility, final selection. Each score 0-4 or null if unobservable. Cite supplied event IDs for scores. '
            'Do not infer hidden test performance or identify models. Call grade once; concise reasons.') if judge else SYSTEM
    body={'model':config['model'],**options,'max_tokens':1025 if judge else 1536,
          'tools':[GRADE if judge else ACTION],'tool_choice':'auto',
          'messages':[{'role':'system','content':system},{'role':'user','content':canonical(observation).decode()}]}
    if not judge and config['seed_mode']=='send':body['seed']=observation.get('seed',0)
    limit=3072 if judge else 4096
    if len(canonical(body))>limit:raise ValueError('Compact request exceeds frozen byte limit')
    return body


def worst_case(configs,judge,seeds=(0,),include_probes=False):
    participant=sum(18*len(seeds)*((4096+4096)*float(c['pricing_ceiling']['prompt'])+1536*float(c['pricing_ceiling']['completion'])) for c in configs)
    review=6*len(configs)*len(seeds)*((3072+4096)*float(judge['pricing_ceiling']['prompt'])+1025*float(judge['pricing_ceiling']['completion']))
    probes=0
    if include_probes:
        probes=sum(2*(8192*float(c['pricing_ceiling']['prompt'])+1536*float(c['pricing_ceiling']['completion'])) for c in configs)
        probes+=2*(7168*float(judge['pricing_ceiling']['prompt'])+1025*float(judge['pricing_ceiling']['completion']))
    normal=12*len(configs)*len(seeds)+(len(configs)+1 if include_probes else 0)
    return {'participant_usd':participant,'judge_usd':review,'probe_usd':probes,'total_usd':participant+review+probes,
            'normal_calls':normal,'maximum_calls_including_repairs':2*normal}


def validate_action(action,valid=None):
    if not isinstance(action,dict) or not isinstance(action.get('reason'),str) or not action['reason'].strip() or len(action['reason'])>160:
        raise ValueError('Provide a brief nonempty reason, at most 160 characters')
    if valid is None:
        if not isinstance(action.get('code'),str) or not action['code'].strip() or len(action['code'].encode())>12000:
            raise ValueError('Provide bounded executable training code')
    elif type(action.get('candidate')) is not int or action['candidate'] not in valid:
        raise ValueError('Select an index from valid_candidates')
    return action


def validate_grade(value,ids):
    rows=value.get('criteria',[]) if isinstance(value,dict) else []
    if len(rows)!=5 or {r.get('id') for r in rows}!=set(CRITERIA):raise ValueError('Review all five criteria exactly once')
    for r in rows:
        s=r.get('score');citations=r.get('evidence_ids')
        if not isinstance(r.get('reason'),str) or not r['reason'].strip():raise ValueError('Explain each judgment')
        if s is not None and (type(s) is not int or not 0<=s<=4):raise ValueError('Scores must be 0-4 or null')
        if not isinstance(citations,list) or any(i not in ids for i in citations) or (s is not None and not citations):
            raise ValueError('Scored criteria require genuine evidence IDs')
    return value


def call(config,observation,key,root,spend,phase,validate,*,judge=False):
    api=root/'api';api.mkdir(parents=True,exist_ok=True)
    trace=Transcript(root/'private')
    try:
        for attempt in range(2):
            path=api/f'{phase}-{attempt}.json'
            if path.exists():
                prior=read(path)
                if prior.get('observation_sha256')!=digest(observation):raise ValueError('Cannot change resumed API observation')
                if prior.get('status')=='completed':return validate(prior['action'])
                observation={**observation,'correction':prior.get('correction','Previous reply failed; use the specified tool.')}
                continue
            body=request_body(config,observation,judge)
            charge=spend.reserve(body,config['pricing_ceiling'])
            record={'status':'running','request':body,'observation_sha256':digest(observation),'started_at':time.time()}
            write(path,record);request_id=trace.append('model_request',{'call_id':phase,'phase':phase,'attempt':attempt,'request':body})
            try:
                req=urllib.request.Request(config['endpoint'],data=canonical(body),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
                with urllib.request.urlopen(req,timeout=min(600,max(30,config.get('request_timeout_s',120)))) as response:
                    raw=response.read(2_000_001)
                    if len(raw)>2_000_000:raise ValueError('Oversized provider response')
                    record['response_text']=raw.decode();result=json.loads(raw)
                record['usage']=result.get('usage') or {};spend.settle(charge,record['usage'])
                message=result['choices'][0]['message'];calls=message.get('tool_calls') or []
                expected='grade' if judge else 'respond'
                if len(calls)!=1 or calls[0].get('function',{}).get('name')!=expected:raise ValueError('Use exactly one '+expected+' tool call')
                action=json.loads(calls[0]['function']['arguments']);validate(action)
                record.update(status='completed',action=action,response_model=result.get('model'),response_provider=result.get('provider'))
                return action
            except SpendLimitReached:raise
            except Exception as exc:
                record.update(status='failed',error=type(exc).__name__)
                if hasattr(exc,'code'):
                    record['http_status']=exc.code
                    record['response_text']=exc.read(4000).decode(errors='replace').replace(key,'[REDACTED]')
                correction=str(exc) if isinstance(exc,(ValueError,json.JSONDecodeError)) else 'Provider failed; retry once.'
                record['correction']=trim(correction,180)
                observation={**observation,'correction':record['correction']}
            finally:
                record['finished_at']=time.time();write(path,record)
                trace.append('model_response',{'call_id':phase,'request_trace_id':request_id,**{k:v for k,v in record.items() if k!='request'}})
                records=[read(p) for p in api.glob('*.json')]
                write(root/'provider-usage.json',{'calls':len(records),'tokens':sum(r.get('usage',{}).get('total_tokens') if type(r.get('usage',{}).get('total_tokens')) is int else len(canonical(r['request']))+r['request']['max_tokens']+4096 for r in records)})
        raise ValueError('Two response attempts exhausted for '+phase)
    finally:trace.close()


def observation(task,round_index,experiments,seed=0):
    guide=("from research_env.tasks.pulse.baselines import train; train('data/train/manifest.json',None,'artifact',0,kind='ridge',alpha=10)"
           if task=='pulse' else "from research_env.tasks.baselines import train; train('data/train/manifest.json','artifact',kind='ridge',seed=0,alpha=10)")
    guide=guide.replace("'artifact',0,",f"'artifact',{seed},").replace('seed=0',f'seed={seed}')
    prior=[{'index':x['index'],'evidence_id':x['evidence_id'],'reason':trim(x['reason'],200),**score_summary(x['score']),
            'stderr':trim(x.get('stderr',''),180)} for x in experiments]
    return {'task':BRIEFS[task],'seed':seed,'phase':round_index,'instruction':'Return training code for experiment '+str(round_index+1)+'.'
            if round_index<2 else 'Choose candidate index from valid_candidates; ground reason in development evidence.',
            'helper_example':guide,'files':'data/train/manifest.json; research_env/tasks/'+task+'/DATA_DICTIONARY.md; previous code trial0.py',
            'experiments':prior,'valid_candidates':[x['index'] for x in experiments if x['score']['execution_valid']]}


def episode(suite_path,task,config,config_path,root,key,spend,seed=0):
    suite=read(suite_path)
    if not (root/'run.json').exists():initialize(suite_path,task,config_path,seed,root)
    meta=verify(read(root/'run.json'))
    if meta['suite_sha256']!=suite['sha256'] or meta['configuration_sha256']!=digest(config) or meta['task']!=task or meta['seed']!=seed:
        raise ValueError('Cannot change compact attempt identity')
    env=ResearchEnvironment(root/'private',root/'workspace',Path(suite['datasets'][task]['root'])/'private/development/manifest.json',image=suite['image'])
    experiments=read(root/'experiments.json') if (root/'experiments.json').exists() else []
    failure=None
    try:
        for i in range(len(experiments),2):
            a=call(config,observation(task,i,experiments,seed),key,root,spend,f'experiment{i}',validate_action)
            target=env.workspace/'artifact'
            if target.is_symlink():target.unlink()
            elif target.exists():shutil.rmtree(target)
            (env.workspace/f'trial{i}.py').write_text(a['code'])
            ids=[e['event_id'] for e in env.session.events()]
            decision={'hypothesis':a['reason'],'expected_result':'Valid predictor improves person MAE','evidence_ids':ids,'code_diff':f'trial{i}.py'}
            command=env.act({'operation':'command','argv':['python',f'trial{i}.py'],'seed':seed,'decision':decision},model_call_id=f'experiment{i}')
            if target.is_dir() and not target.is_symlink():
                source=target/'training_code.py'
                if source.exists() or source.is_symlink():source.unlink()
                source.write_text(a['code'])
            decision['evidence_ids']=[e['event_id'] for e in env.session.events()]
            score=env.act({'operation':'evaluate','decision':decision},model_call_id=f'experiment{i}')
            event=env.session.events()[-1]['event_id']
            experiments.append({'index':i,'reason':a['reason'],'score':score,'evidence_id':event,'stderr':command.get('stderr','')[-200:]})
            write(root/'experiments.json',experiments)
        valid=[x['index'] for x in experiments if x['score']['execution_valid']]
        if not valid:raise ValueError('Neither experiment produced a valid development artifact')
        selection=call(config,observation(task,2,experiments,seed),key,root,spend,'selection',lambda a:validate_action(a,valid))
        candidate=experiments[selection['candidate']]['score']['candidate_id']
        env.act({'operation':'submit','candidate_id':candidate,'reason':selection['reason'],
                 'evidence_ids':[x['evidence_id'] for x in experiments]},model_call_id='selection')
    except SpendLimitReached:raise
    except Exception as exc:failure=type(exc).__name__+': '+trim(str(exc),180)
    finally:env.session.db.close()
    return finalize(suite_path,root,failure)


def review(root,judge,key,spend):
    out=root/'episode-judge';out.mkdir(exist_ok=True)
    receipt=verify(read(root/'receipt.json'));events=[json.loads(x)['event'] for x in (root/'trajectory.jsonl').read_text().splitlines()]
    selected=[]
    for e in events:
        if e['kind']=='experiment':selected.append({'id':e['event_id'],'kind':e['kind'],'decision':{k:trim(v,80) for k,v in e.get('decision',{}).items() if k in ('hypothesis','code_diff')}})
        if e['kind']=='development_result':selected.append({'id':e['event_id'],'kind':e['kind'],**score_summary(e['result'])})
        if e['kind']=='final_selection':selected.append({'id':e['event_id'],'kind':e['kind'],'reason':trim(e['reason'],80),'evidence_ids':e['evidence_ids']})
    if not selected:selected=[{'id':'no-evidence','kind':'failed_before_experiment'}]
    packet={'rubric':'compact-process-v1','events':selected,'trajectory_sha256':receipt['trajectory_sha256']}
    write(out/'packet.json',packet);ids={e['id'] for e in selected}
    verdict=call(judge,packet,key,out,spend,'review',lambda a:validate_grade(a,ids),judge=True)
    report={'packet_sha256':digest(packet),'judge_configuration_sha256':digest(judge),'verdict':verdict}
    write(out/'report.json',report);return report


def run_path(root,config,task,seed,seeds):
    base=Path(root)/'runs'/config['slug']/task
    return base if list(seeds)==[0] else base/f'seed-{seed}'


def board(root,configs,judge=None,seeds=(0,)):
    rows=[];reviews=[];suite_hash=None
    for c in configs:
        for task in TASKS:
            for seed in seeds:
                base=run_path(root,c,task,seed,seeds);p=base/'receipt.json'
                if not p.exists():return None
                r=verify(read(p));meta=verify(r['run'])
                if meta['competitor']!=c['competitor'] or meta['configuration_sha256']!=digest(c) or meta['task']!=task or meta['seed']!=seed:
                    raise ValueError('Receipt identity differs from compact pool')
                if suite_hash is not None and suite_hash!=meta['suite_sha256']:raise ValueError('Mixed compact suites')
                suite_hash=meta['suite_sha256']
                for f,k in [('trajectory.jsonl','trajectory_sha256'),('transcript.jsonl','transcript_sha256'),('run-status.json','run_status_sha256')]:
                    if file_hash(base/f)!=r[k]:raise ValueError('Sealed evidence changed')
                report=base/'episode-judge/report.json'
                if report.exists() and judge is not None:
                    packet=read(base/'episode-judge/packet.json');reviewed=read(report)
                    if reviewed['packet_sha256']!=digest(packet) or packet['trajectory_sha256']!=r['trajectory_sha256'] or reviewed['judge_configuration_sha256']!=digest(judge):
                        raise ValueError('Judge evidence/configuration changed')
                    validate_grade(reviewed['verdict'],{e['id'] for e in packet['events']})
                    reviews.append({'model':c['competitor'],'task':task,'seed':seed,**reviewed})
                rows.append(r)
    models=sorted(c['competitor'] for c in configs)
    cases={(r['run']['competitor'],r['run']['seed'],r['run']['task']):r['score'] for r in rows}
    return {'protocol':'physiology-compact-coding-v1','scope':f'{len(models)} models; {len(seeds)} agent repeats over fixed cohorts; two code experiments per task',
            'suite_sha256':suite_hash,'seeds':list(seeds),
            'suite':elo(models,seeds,cases,'prediction',suite=True),
            'tasks':{t:elo(models,seeds,{(m,s):cases[m,s,t] for m in models for s in seeds},'prediction') for t in TASKS},
            'uncertainty':elo(models,seeds,cases,'uncertainty',suite=True),
            'process_review':{'coverage':len(reviews),'expected':len(rows),'reviews':reviews},
            'raw':[{'model':r['run']['competitor'],'task':r['run']['task'],'seed':r['run']['seed'],'score':r['score']} for r in rows]}


def render(result):
    n=len(result['seeds']);models=len(result['suite']['entries'])
    lines=['# Compact physiology leaderboard','',f'{models} models × three tasks × {n} repeats. Two code experiments per episode. Repeats use the same held-out people; population validation remains open.','', '| Model | Suite Elo | Valid episodes |','|---|---:|---:|']
    for row in result['suite']['entries']:
        valid=sum(r['score']['execution_valid'] for r in result['raw'] if r['model']==row['competitor'])
        rating=f"{row['elo']:.1f}" if row['rank'] is not None else 'unranked'
        lines.append(f"| {row['competitor']} | {rating} | {valid} / {3*n} |")
    for task in TASKS:
        lines+=['',f'## {task.title()}','', '| Model | Prediction Elo | Valid trials |','|---|---:|---:|']
        for row in result['tasks'][task]['entries']:
            rating=f"{row['elo']:.1f}" if row['rank'] is not None else 'unranked'
            lines.append(f"| {row['competitor']} | {rating} | {row['valid_runs']} / {n} |")
    lines+=['','## Process review','',f"Validated episode reviews: {result['process_review']['coverage']} / {result['process_review']['expected']}. Scores use a separate 0–4 rubric.",'', '| Model | Evidence | Experiment | Feedback | Reproducibility | Selection |','|---|---:|---:|---:|---:|---:|']
    for row in result['suite']['entries']:
        name=row['competitor'];cells=[]
        for criterion in CRITERIA:
            values=[v['score'] for r in result['process_review']['reviews'] if r['model']==name for v in r['verdict']['criteria'] if v['id']==criterion and v['score'] is not None]
            cells.append(f'{sum(values)/len(values):.2f} ({len(values)})' if values else 'unobserved')
        lines.append('| '+name+' | '+' | '.join(cells)+' |')
    lines+=['','Numerical losses, uncertainty Elo, matches and individual cited reviews are in leaderboard.json. Pulse ECG references remain provisional.']
    return '\n'.join(lines)+'\n'


def preflight(configs,judge,root,key,spend):
    for config in configs:
        p=root/'preflight'/config['slug']
        observation={'seed':0,'instruction':"Compatibility check only. Call respond once with code print('ok') and a brief reason. No research or execution needed."}
        call(config,observation,key,p,spend,'probe',validate_action)
    packet={'rubric':'compact-process-v1','events':[{'id':'probe','kind':'compatibility_check'}]}
    call(judge,packet,key,root/'preflight/judge',spend,'probe',lambda a:validate_grade(a,{'probe'}),judge=True)


def execute(plan_path,root,key):
    plan=read(plan_path);root=Path(root);root.mkdir(parents=True,exist_ok=True)
    lock=(root/'runner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    configs=[read(p) for p in plan['configs']];judge=read(plan['judge_config']);check_suite(read(plan['suite']))
    seeds=plan.get('seeds',[0]);cap=plan['approved_fresh_budget_usd'];probes=plan.get('preflight',False)
    expanded=[c['model'] for c in configs]==list(EXPANDED_MODELS)
    if expanded:
        if cap!=16 or seeds!=[0,1,2] or not probes:raise ValueError('Expanded authorization requires $16, three repeats, and preflight')
        for c in configs:
            expected='high' if c['model'] in EXPANDED_MODELS[-2:] else 'medium'
            if c['decoding']['reasoning']['effort']!=expected:raise ValueError('Effort differs from user authorization')
    elif [c['model'] for c in configs]!=list(MODELS) or cap!=10 or seeds!=[0]:raise ValueError('Unexpected compact roster or authorization')
    bound=worst_case(configs,judge,seeds,probes)
    if bound['total_usd']>cap:raise ValueError('Frozen maximum request cost exceeds approved retry budget')
    fingerprint=digest({'plan':plan,'configs':configs,'judge':judge,'suite':read(plan['suite'])['sha256']})
    if (root/'status.json').exists() and read(root/'status.json')['fingerprint']!=fingerprint:raise ValueError('Compact pool changed on resume')
    spend=SpendLedger(root/'spend.sqlite',cap);target=len(configs)*len(TASKS)*len(seeds)
    state={'fingerprint':fingerprint,'status':'running','target_episodes':target,'completed_episodes':0,'judged_episodes':0,'worst_case':bound}
    def update(**values):state.update(values);write(root/'status.json',{**state,'spend':spend.summary()})
    try:
        if probes:
            update(status='preflight');preflight(configs,judge,root,key,spend)
            update(status='running',preflight_passed=True)
        for seed in seeds:
            for task in TASKS:
                for c,path in zip(configs,plan['configs']):
                    p=run_path(root,c,task,seed,seeds);update(active_model=c['model'],active_task=task,active_seed=seed)
                    if not (p/'receipt.json').exists():episode(plan['suite'],task,c,path,p,key,spend,seed)
                    update(completed_episodes=len(list(root.glob('runs/**/receipt.json'))))
                    if not (p/'episode-judge/report.json').exists():
                        try:review(p,judge,key,spend)
                        except SpendLimitReached:raise
                        except Exception as exc:write(p/'episode-judge/failure.json',{'error':type(exc).__name__,'message':trim(str(exc),180)})
                    update(judged_episodes=len(list(root.glob('runs/**/episode-judge/report.json'))))
        result=board(root,configs,judge,seeds);write(root/'leaderboard.json',result)
        (root/'LEADERBOARD.md').write_text(render(result))
        update(status='complete' if state['judged_episodes']==target else 'complete_with_missing_judgments')
    except SpendLimitReached:update(status='paused_spend_cap')
    except BaseException as exc:
        update(status='needs_attention',error=type(exc).__name__);raise
    finally:spend.close();lock.close()
    return read(root/'status.json')


def main():
    p=argparse.ArgumentParser();p.add_argument('plan');p.add_argument('output');p.add_argument('--env-file');p.add_argument('--key-env',default='OPENROUTER_API_KEY');a=p.parse_args()
    print(json.dumps(execute(a.plan,a.output,credential(a.key_env,a.env_file)),indent=2))


if __name__=='__main__':main()
