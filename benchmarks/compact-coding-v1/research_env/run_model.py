"""Run an explicitly configured HTTPS model through the isolated research controller."""
import argparse
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request
from research_env.core.agent import ResearchEnvironment
from research_env.core.common import canonical, digest, read, write
from research_env.leaderboard import check_suite, finalize, initialize, verify
from research_env.core.transcript import Transcript
from research_env.core.spend import SpendLimitReached


class ActionProtocolError(ValueError):
    """A reply that must be corrected before any controller action is executed."""


ACTION_TOOL = {'type':'function','function':{
    'name':'take_action','description':'Perform one research controller action and receive its observed result on the next turn.',
    'parameters':{'type':'object','properties':{
        'operation':{'type':'string','enum':['edit','command','evaluate','select','submit']},
        'path':{'type':'string'},'text':{'type':'string'},
        'argv':{'type':'array','items':{'type':'string'}},'seed':{'type':'integer'},
        'candidate_id':{'type':'string'},'reason':{'type':'string'},
        'evidence_ids':{'type':'array','items':{'type':'string'}},
        'decision':{'type':'object','properties':{
            'hypothesis':{'type':'string'},'expected_result':{'type':'string'},
            'code_diff':{'type':'string'},'evidence_ids':{'type':'array','items':{'type':'string'}}},
            'required':['hypothesis','expected_result','code_diff','evidence_ids']}},
        'required':['operation']}}}


def provider_fault(exc):
    if isinstance(exc, ActionProtocolError):
        return {'kind':'invalid_action_response','type':type(exc).__name__}
    if isinstance(exc, urllib.error.HTTPError):
        return {'kind':'provider_http_error','type':type(exc).__name__,'status_code':exc.code}
    if isinstance(exc, (TimeoutError, __import__('socket').timeout)):
        return {'kind':'provider_timeout','type':type(exc).__name__}
    if isinstance(exc, json.JSONDecodeError):
        return {'kind':'invalid_response_json','type':type(exc).__name__}
    if isinstance(exc, (KeyboardInterrupt, SystemExit)):
        return {'kind':'interrupted','type':type(exc).__name__}
    return {'kind':'provider_error','type':type(exc).__name__}

SYSTEM = '''You are a research participant. Develop an executable predictor for the task in your
observation using only the public staged workspace. Choose experiments and final code yourself.
Return one JSON controller action per turn, never Markdown. You cannot use host tools or see private
labels. command runs argv in an offline CPU container with current directory /work. Use python.
All edits and commands act in the staged workspace. Artifacts go in artifact/ and inference.py
implements predict(artifact_dir, request). Copy a lock and save source/config/seed/reproduction.
Use public training files under data/train and TASK.md under research_env/tasks/TASKNAME/.
Actions: {"operation":"edit","path":"relative/path","text":"file contents"};
{"operation":"command","argv":["python","-c","..."],"seed":0,"decision":DECISION};
{"operation":"evaluate","decision":DECISION};
{"operation":"select","candidate_id":"from evaluation","reason":"public justification","evidence_ids":["e00000"]};
{"operation":"submit","candidate_id":"from evaluation","reason":"public justification","evidence_ids":["e00000"]}.
DECISION has hypothesis, expected_result, evidence_ids (prior event IDs), code_diff (substantive
change description). Use observed results for revision. Submit before your time/call budget ends.
Use finite predictions and all required outputs. Do not infer hidden scores. Explain substantive
experiment decisions and limitations in records; do not provide private chain of thought.'''


class ModelProvider:
    def __init__(self, config, key, run_root, spend=None):
        self.config,self.key,self.root=config,key,Path(run_root)
        self.spend=spend
        if not config.get('endpoint','').startswith('https://'):
            raise ValueError('Explicit HTTPS chat-completions endpoint required')
        for name in ('max_calls','max_total_tokens'):
            if type(config.get(name)) is not int or config[name]<=0:
                raise ValueError('Explicit positive inference budgets required')
        if type(config['decoding'].get('max_tokens')) is not int or config['decoding']['max_tokens']<=0:
            raise ValueError('Explicit max_tokens required')
        if {'model','messages','seed','stream'} & set(config['decoding']):
            raise ValueError('Decoding cannot override identity/messages/seed/stream')
        if config.get('seed_mode','send') not in ('send','omit'):
            raise ValueError('seed_mode must explicitly send or omit the provider seed')
        if config.get('action_transport','json') not in ('json','tools'):
            raise ValueError('Explicit JSON or native tools action transport required')
        if config.get('action_transport')=='tools' and (config['decoding'].get('tools')!=[ACTION_TOOL] or config['decoding'].get('tool_choice')!='auto'):
            raise ValueError('Native transport requires the frozen take_action schema and auto tool choice')
        if type(config.get('max_consecutive_protocol_errors',1)) is not int or not 1<=config.get('max_consecutive_protocol_errors',1)<=3:
            raise ValueError('Allow at most three consecutive malformed replies')
        p=self.root/'provider-usage.json'
        self.usage=read(p) if p.exists() else {'calls':0,'tokens':0}
        self.last_call_id = None
        self.last_fault = None

    def __call__(self, observation, seed):
        # Fresh context containing the complete public ledger. Conservative byte-based
        # token reservation makes failed/unaccounted calls consume the inference budget.
        system=SYSTEM+'\n'+self.config['prompt']
        if self.config.get('action_transport')=='tools':
            system=system.replace('Return one JSON controller action per turn, never Markdown.',
                'Call take_action exactly once per turn with the controller action as arguments. Do not invent tool results or emit actions in prose.')
        observation=dict(observation)
        remaining_calls=self.config['max_calls']-self.usage['calls']
        remaining_tokens=self.config['max_total_tokens']-self.usage['tokens']
        observation['inference_budget']={
            'calls_remaining_including_this_call':remaining_calls,
            'total_tokens_remaining_before_this_call':remaining_tokens,
            'max_response_tokens':self.config['decoding']['max_tokens'],
            'accounting':'provider usage when available; conservative reservation otherwise',
            'submission_required':True}
        estimate=len(canonical(observation))+len(system.encode())+self.config['decoding']['max_tokens']+2048
        observation['inference_budget']['near_limit']=remaining_calls<=3 or remaining_tokens<=3*estimate
        observation['inference_budget']['instruction']='If near_limit is true, select and submit your best valid evaluated candidate now; do not start more experiments.'
        reservation=len(canonical(observation))+len(system.encode())+self.config['decoding']['max_tokens']+1024
        if self.usage['calls']>=self.config['max_calls'] or self.usage['tokens']+reservation>self.config['max_total_tokens']:
            self.last_fault = {'kind':'budget_exhausted','budget':'provider_calls_or_tokens'}
            raise ValueError('Participant inference budget exhausted')
        body={'model':self.config['model'],**self.config['decoding'],
              'messages':[{'role':'system','content':system},{'role':'user','content':canonical(observation).decode()}]}
        if self.config.get('seed_mode','send')=='send':
            body['seed']=seed
        charge=self.spend.reserve(body,self.config['pricing_ceiling']) if self.spend else None
        self.usage['calls']+=1;self.usage['tokens']+=reservation
        write(self.root/'provider-usage.json',self.usage)
        self.last_call_id = self.usage['calls']
        self.last_fault = None
        started = time.monotonic()
        trace = Transcript(self.root/'private')
        record={'call':self.usage['calls'],'observation_sha256':digest(observation),
                'system_sha256':digest(system),'model':self.config['model'],'seed':seed,'decoding':self.config['decoding'],
                'request':body,'started_at':time.time(),'status':'running',
                'token_accounting':'conservative_reservation','accounted_tokens':reservation}
        write(self.root/f"provider-call-{self.usage['calls']:04}.json",record)
        request_id = trace.append('model_request', {'call_id':self.last_call_id,'endpoint':self.config['endpoint'],
                                                   'request':body,'observation_sha256':digest(observation)})
        try:
            req=urllib.request.Request(self.config['endpoint'],data=canonical(body),
                                       headers={'Authorization':'Bearer '+self.key,'Content-Type':'application/json'})
            timeout=min(self.config.get('request_timeout_s',60),observation.get('remaining_wall_s',600))
            with urllib.request.urlopen(req,timeout=max(1,timeout)) as response:
                raw=response.read(2_000_001)
                if len(raw)>2_000_000:
                    raise ValueError('Model response exceeds 2 MB')
                record['response_text'] = raw.decode('utf-8',errors='replace')
                result=json.loads(raw)
            usage=result.get('usage',{})
            used=usage.get('total_tokens')
            if type(used) is int and used>=0:
                self.usage['tokens']+=used-reservation
                record['token_accounting']='provider_reported'
                record['accounted_tokens']=used
            record['usage']=usage
            if self.spend:
                self.spend.settle(charge,usage)
            record['response_model']=result.get('model')
            record['response_id']=result.get('id')
            record['response_provider']=result.get('provider')
            message=result['choices'][0]['message']
            text=message.get('content')
            record['content']=text
            if self.config.get('action_transport')=='tools':
                calls=message.get('tool_calls') or []
                record['tool_calls']=calls
                function=calls[0].get('function',{}) if isinstance(calls,list) and len(calls)==1 and isinstance(calls[0],dict) else {}
                if not isinstance(function,dict) or function.get('name')!='take_action' or calls[0].get('type')!='function' or not isinstance(function.get('arguments'),str):
                    raise ActionProtocolError('Call take_action exactly once; no action from this reply was executed')
                action=json.loads(calls[0]['function']['arguments'])
            else:
                action=json.loads(text)
            if not isinstance(action,dict):
                raise ActionProtocolError('Controller action must be an object; no action was executed')
            record['status']='completed'
            record['action']=action
            return action
        except BaseException as exc:
            self.last_fault = provider_fault(exc)
            record['status']='cancelled' if self.last_fault['kind']=='interrupted' else 'error'
            record['fault']=self.last_fault
            record['error']=f'{type(exc).__name__}: provider call failed'
            raise
        finally:
            record['finished_at']=time.time()
            record['wall_s']=time.monotonic()-started
            write(self.root/'provider-usage.json',self.usage)
            write(self.root/f"provider-call-{self.usage['calls']:04}.json",record)
            trace.append('model_response', {'call_id':self.last_call_id,'request_trace_id':request_id,
                         **{k:v for k,v in record.items() if k not in ('request','call')}})
            trace.close()


def run(suite_path,task,config_path,seed,root,key,spend=None):
    suite,config=read(suite_path),read(config_path)
    check_suite(suite,task)
    root=Path(root)
    if not (root/'run.json').exists():
        initialize(suite_path,task,config_path,seed,root)
    meta=verify(read(root/'run.json'))
    if (meta['configuration_sha256']!=digest(config) or meta['suite_sha256']!=suite['sha256']
            or meta['task']!=task or meta['seed']!=seed):
        raise ValueError('Cannot change run configuration on resume')
    if (root/'receipt.json').exists():
        raise ValueError('Attempt already finalized')
    provider=ModelProvider(config,key,root,spend)
    env=ResearchEnvironment(root/'private',root/'workspace',
                            Path(suite['datasets'][task]['root'])/'private/development/manifest.json',image=suite['image'])
    error, fault = None, None
    try:
        with env.session.db:
            env.session.db.execute('INSERT OR IGNORE INTO settings VALUES (?,?)',('response_protocol_errors','0'))
            env.session.transcript.append('run_status', {'status':'running','fault':None,'final_answer':None,
                                                        'source':'model_provider'})
        while not env.observe()['closed'] and env.session.remaining()>0:
            try:
                action=provider(env.observe(),seed)
            except SpendLimitReached:
                raise
            except (json.JSONDecodeError,ActionProtocolError) as exc:
                with env.session.db:
                    consecutive=env.session.get('response_protocol_errors')+1
                    env.session.put('response_protocol_errors',consecutive)
                    env.session.append({'kind':'protocol_error','layer':'model_response',
                        'consecutive_errors':consecutive,'error':str(exc),
                        'instruction':'The last response was not executed. Return one valid controller action using the configured transport; do not invent results.'})
                if consecutive>=config.get('max_consecutive_protocol_errors',1):
                    error='Participant action format failed after consecutive malformed replies'
                    fault=provider.last_fault
                    break
                continue
            except Exception as exc:
                error=f'{type(exc).__name__}: participant provider failed or exhausted'
                fault=provider.last_fault
                break
            with env.session.db:
                env.session.put('response_protocol_errors',0)
            try:
                env.act(action, model_call_id=provider.last_call_id)
            except (ValueError,KeyError,TypeError) as exc:
                with env.session.db:
                    env.session.append({'kind':'protocol_error','error':str(exc)})
        if not env.session.get('closed'):
            error=error or 'No submission before wall budget'
            fault=fault or {'kind':'budget_exhausted','budget':'wall_time'}
        return finalize(suite_path,root,error,fault=fault)
    except SpendLimitReached:
        with env.session.db:
            env.session.transcript.append('run_status', {'status':'paused','fault':{'kind':'sweep_spend_cap'},
                                                        'final_answer':None,'resumable':True})
        from research_env.trace import export
        export(root)
        raise
    except (KeyboardInterrupt, SystemExit):
        with env.session.db:
            env.session.transcript.append('run_status', {'status':'cancelled','fault':{'kind':'interrupted'},
                                                        'final_answer':None,'resumable':True})
        from research_env.trace import export
        export(root)
        raise
    finally:
        env.session.db.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('suite');p.add_argument('task',choices=('pulse','heat','load'))
    p.add_argument('config');p.add_argument('run_root');p.add_argument('--seed',type=int,required=True)
    p.add_argument('--key-env',required=True)
    a=p.parse_args()
    result=run(a.suite,a.task,a.config,a.seed,a.run_root,os.environ[a.key_env])
    print(json.dumps({'execution_valid':result['score']['execution_valid'],'receipt':str(Path(a.run_root)/'receipt.json')}))


if __name__=='__main__':
    main()
