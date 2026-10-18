"""Explicit OpenRouter sweep: fixed roster, credential check, probes, runs and judging."""
import argparse
import json
import os
from pathlib import Path
import shlex
import urllib.request
from research_env.core.common import digest, read, write
from research_env.core.spend import SpendLedger, SpendLimitReached
from research_env.leaderboard import POLICY, build, check_suite, render, verify
from research_env.run_model import ModelProvider, run
from research_env.tasks.registry import TASKS
from research_env.judging.judge import OpenAICompatibleJudge, judge_packet, summarize


def load_plan(manifest_path):
    manifest_path=Path(manifest_path).resolve()
    manifest=read(manifest_path)
    root=manifest_path.parents[3]
    configs=[]
    for entry in manifest['models']:
        path=root/entry['config']
        config=read(path)
        if config['model']!=entry['model'] or digest(config)!=entry['configuration_sha256']:
            raise ValueError('Model roster/configuration changed')
        configs.append((entry,path,config))
    if len(configs)!=10 or len({c['model'] for _,_,c in configs})!=10:
        raise ValueError('Require exactly ten distinct models')
    judge=root/manifest['judge_config']
    if digest(read(judge))!=manifest['judge_configuration_sha256']:
        raise ValueError('Judge configuration changed')
    return root,manifest,configs,judge


def check_access(key):
    request=urllib.request.Request('https://openrouter.ai/api/v1/key',headers={'Authorization':'Bearer '+key})
    with urllib.request.urlopen(request,timeout=30) as response:
        data=json.load(response)['data']
    return {k:data.get(k) for k in ('limit','limit_remaining','usage','is_free_tier','effective_budget')}


def credential(name,env_file=None):
    if os.environ.get(name):return os.environ[name]
    if env_file:
        path=Path(env_file)
        if not path.is_file():raise ValueError('Credential file does not exist: '+str(path))
        for line in path.read_text().splitlines():
            line=line.strip()
            if line.startswith('export '):line=line[7:].lstrip()
            key,separator,value=line.partition('=')
            if separator and key.strip()==name:
                tokens=shlex.split(value,comments=True)
                if len(tokens)==1 and tokens[0]:return tokens[0]
    raise ValueError('No configured '+name+' credential; key values are never printed')


def validate_catalog(config,models):
    if config['model'] not in models:
        raise ValueError('Configured model is no longer listed')
    supported=set(models[config['model']].get('supported_parameters',[]))
    requested=set(config['decoding'])-{'provider'}
    if config['seed_mode']=='send':requested.add('seed')
    if requested-supported:
        raise ValueError('Unsupported model parameters: '+str(sorted(requested-supported)))
    if config['decoding']['provider'].get('allow_fallbacks') is not False:
        raise ValueError('Scored sweep must not silently fall back to another provider')


def check_resume_window(root):
    status_path=root/'run-status.json'
    if not status_path.exists() or not read(status_path).get('resumable'):
        return
    from research_env.core.session import Session
    session=Session(root/'private')
    try:
        if not session.get('closed') and session.remaining()<=0:
            raise RuntimeError('Interrupted attempt expired during pause; preserve it as infrastructure interruption rather than silently retrying/scoring it')
    finally:
        session.db.close()


def judge_run(root,config_path,key,spend):
    out=root/'judge-output';out.mkdir(exist_ok=True)
    config=read(config_path)
    if (out/'config.json').exists() and read(out/'config.json')!=config:
        raise ValueError('Cannot change judge settings on resume')
    write(out/'config.json',config)
    provider=OpenAICompatibleJudge(api_key=key,spend=spend,**config)
    if (out/'usage.json').exists():
        usage=read(out/'usage.json');provider.calls=usage['calls'];provider.tokens=usage['tokens']
    packets=read(root/'judge-packets.json')
    verdicts=read(out/'verdicts.json') if (out/'verdicts.json').exists() else []
    summarize(packets,verdicts)  # Validate resumed verdict identities before skipping.
    completed={v['packet_sha256'] for v in verdicts}
    for packet in packets:
        if packet['packet_sha256'] in completed:continue
        try:
            result=judge_packet(packet,provider,out/(packet['event_id']+'-attempts.json'))
            if result.get('verdict') is not None:
                verdicts.append(result['verdict'])
                write(out/'verdicts.json',verdicts)
        finally:
            write(out/'usage.json',{'calls':provider.calls,'tokens':provider.tokens})
    write(out/'report.json',summarize(packets,verdicts))
    return {'competitor':read(root/'run.json')['competitor'],'task':read(root/'run.json')['task'],
            'seed':read(root/'run.json')['seed'],'configuration':config,
            'configuration_sha256':digest(config),'report':summarize(packets,verdicts)}


def execute(manifest_path,output,key,max_usd,*,probes_only=False):
    repo,manifest,configs,judge=load_plan(manifest_path)
    if max_usd>manifest.get('approved_shared_api_cap_usd',0):
        raise ValueError('Requested cap exceeds recorded user authorization')
    suite_path=repo/manifest['suite']
    suite=read(suite_path);check_suite(suite)
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    fingerprint=digest({'manifest':manifest,'suite_sha256':suite['sha256']})
    if (output/'status.json').exists() and read(output/'status.json')['fingerprint']!=fingerprint:
        raise ValueError('Existing sweep belongs to another roster or suite')
    spend=SpendLedger(output/'spend.sqlite',max_usd)
    state={'fingerprint':fingerprint,'status':'preflight','completed_runs':0,'total_runs':90,
           'manifest':str(Path(manifest_path).resolve()),'spend':spend.summary()}
    def update(**values):
        state.update(values);state['spend']=spend.summary();write(output/'status.json',state)
    update()
    try:
        write(output/'key-budget.json',check_access(key))
        with urllib.request.urlopen('https://openrouter.ai/api/v1/models',timeout=30) as response:
            catalog={m['id']:m for m in json.load(response)['data']}
        for entry,config_path,config in configs:
            validate_catalog(config,catalog)
            probe=output/'preflight'/entry['slug'];probe.mkdir(parents=True,exist_ok=True)
            if (probe/'passed.json').exists():
                if read(probe/'passed.json')['configuration_sha256']!=digest(config):raise ValueError('Probe settings changed')
                continue
            update(active_model=entry['label'],active_task='provider preflight')
            provider=ModelProvider(config,key,probe,spend)
            native=config.get('action_transport')=='tools'
            expected={'operation':'edit','path':'probe.txt','text':'ok'} if native else {'operation':'probe','ok':True}
            action=provider({'interface':'provider-health-check','remaining_wall_s':600,'history':[],
                'task':'Health check only. No action will be executed. Return exactly '+json.dumps(expected)+
                       (' through take_action.' if native else ' as JSON.')},0)
            if action!=expected:
                raise ValueError('Provider probe did not return the requested action; no benchmark launched')
            write(probe/'passed.json',{'configuration_sha256':digest(config),'passed':True})
        if probes_only:
            update(status='preflight_passed');return state
        paths=[]
        for seed in POLICY['agent_seeds']:
            for task in ('load','heat','pulse'):
                for entry,config_path,config in configs:
                    root=output/'runs'/entry['slug']/f'{task}-seed-{seed}';paths.append(root)
                    if (root/'receipt.json').exists():
                        receipt=verify(read(root/'receipt.json'))
                        if receipt['run']['suite_sha256']!=suite['sha256'] or receipt['run']['configuration']!=config:
                            raise ValueError('Cannot reuse another model configuration/suite receipt')
                    else:
                        check_resume_window(root)
                        update(status='running',active_model=entry['label'],active_task=task,active_seed=seed,active_run=str(root))
                        run(suite_path,task,config_path,seed,root,key,spend)
                    update(completed_runs=len(list((output/'runs').rglob('receipt.json'))))
        receipts=[read(p/'receipt.json') for p in paths]
        board=build(suite,receipts)
        write(output/'leaderboard.json',board);(output/'LEADERBOARD.md').write_text(render(board))
        judges=[]
        for root in paths:
            update(status='judging',active_run=str(root))
            judges.append(judge_run(root,judge,key,spend))
            update(judged_runs=len(judges))
        board=build(suite,receipts,judges)
        write(output/'leaderboard.json',board);(output/'LEADERBOARD.md').write_text(render(board))
        coverage_complete=all(j['report']['llm_judge_status']=='complete' for j in judges)
        update(status='complete' if coverage_complete else 'complete_with_missing_judgments',
               active_run=None,active_model=None,active_task=None,
               judge_coverage_complete=coverage_complete)
    except SpendLimitReached as exc:
        update(status='paused_spend_cap',error=str(exc))
    except BaseException as exc:
        update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'needs_attention',error=type(exc).__name__)
        raise
    finally:
        spend.close()
    return state


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest');p.add_argument('output');p.add_argument('--key-env',default='OPENROUTER_API_KEY')
    p.add_argument('--env-file',help='Explicit authorized .env path; parsed as data, never sourced as shell code')
    p.add_argument('--max-usd',type=float,required=True);p.add_argument('--probes-only',action='store_true')
    a=p.parse_args()
    print(json.dumps(execute(a.manifest,a.output,credential(a.key_env,a.env_file),a.max_usd,probes_only=a.probes_only),indent=2))


if __name__=='__main__':main()
