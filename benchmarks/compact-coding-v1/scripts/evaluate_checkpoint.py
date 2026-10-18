"""Nine compact episodes per declared self-hosted checkpoint; no model allocation."""
import argparse
import fcntl
from pathlib import Path
import re
from research_env import compact
from research_env.core.common import digest,read,write
from research_env.sweep import credential
from scripts.verify_release import verify_release
from scripts.resume_compact import ProviderAwareLedger


def validate_config(c):
    if not all(isinstance(c.get(k),str) and c[k].strip() and 'REPLACE' not in c[k] for k in ('competitor','model','slug','endpoint','prompt')):raise ValueError('Fill exact checkpoint identity and endpoint')
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,95}',c['slug']):raise ValueError('Use a simple unique slug')
    if c['endpoint'].startswith('https://openrouter.ai/') or any(float(v)!=0 for v in c['pricing_ceiling'].values()):raise ValueError('This adapter is for self-hosted inference; GPU rental is accounted separately')
    if c['decoding'].get('max_tokens')!=1536 or c.get('seed_mode') not in ('send','omit'):raise ValueError('Preserve the compact token and seed contract')
    if 'provider' in c['decoding']:raise ValueError('Do not send OpenRouter routing to a self-hosted endpoint')
    provenance=c.get('checkpoint',{})
    if not re.fullmatch(r'[0-9a-f]{64}',provenance.get('manifest_sha256','')):raise ValueError('Record immutable checkpoint-file manifest SHA-256')
    for k in ('base_model_revision','tokenizer_revision','chat_template_sha256','dtype','quantization','serving_image_digest','effort_support'):
        if not isinstance(provenance.get(k),str) or not provenance[k] or 'REPLACE' in provenance[k]:raise ValueError('Record checkpoint/serving provenance: '+k)
    unsupported=set(c['decoding'])-{'max_tokens','reasoning','verbosity'}
    if unsupported:raise ValueError('Compact transport would omit unsupported fields: '+str(sorted(unsupported)))
    return c


def evaluate(suite_path,config_paths,judge_path,output,key,judge_key,max_usd,replica=False):
    verification=verify_release(suite_path=suite_path,replica=replica)
    configs=[validate_config(read(p)) for p in config_paths];judge=read(judge_path)
    if len({c['slug'] for c in configs})!=len(configs) or len({c['competitor'] for c in configs})!=len(configs):raise ValueError('Distinct checkpoint identities required')
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    lock=(root/'runner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    spend=ProviderAwareLedger(root/'spend.sqlite',max_usd)
    fingerprint=digest({'suite_sha256':read(suite_path)['sha256'],'configs':configs,'judge':judge,'cap':max_usd,'replica':replica})
    status=root/'checkpoint-status.json'
    if status.exists() and read(status)['fingerprint']!=fingerprint:raise ValueError('Checkpoint pool changed; preserve the existing output and ledger')
    state={'fingerprint':fingerprint,'status':'running','verification':verification,'target_episodes':9*len(configs)}
    def update(**values):
        state.update(values);write(status,{**state,'completed_episodes':len(list(root.glob('runs/**/receipt.json'))),'judged_episodes':len(list(root.glob('runs/**/episode-judge/report.json'))),'spend':spend.summary()})
    try:
        update()
        for c in configs:
            compact.call(c,{'seed':0,'instruction':"Compatibility check: call respond with code print('ok') and a brief reason."},key,root/'preflight'/c['slug'],spend,'probe',compact.validate_action)
        for seed in (0,1,2):
            for task in compact.TASKS:
                for c,path in zip(configs,config_paths):
                    p=compact.run_path(root,c,task,seed,[0,1,2]);update(active_model=c['model'],active_task=task,active_seed=seed)
                    if not (p/'receipt.json').exists():compact.episode(suite_path,task,c,path,p,key,spend,seed)
                    if not (p/'episode-judge/report.json').exists():compact.review(p,judge,judge_key,spend)
                    update()
        result=compact.board(root,configs,judge,[0,1,2]);result['checkpoint_comparison_mode']=verification
        write(root/'leaderboard.json',result);(root/'LEADERBOARD.md').write_text(compact.render(result));update(status='complete')
    except BaseException as exc:
        update(status='needs_attention',error=type(exc).__name__);raise
    finally:spend.close();lock.close()
    return read(status)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--suite',required=True);p.add_argument('--config',action='append',required=True);p.add_argument('--judge',default='leaderboard/sweeps/compact-eight/judge.json');p.add_argument('--output',required=True);p.add_argument('--max-usd',type=float,required=True);p.add_argument('--key-env',default='CHECKPOINT_API_KEY');p.add_argument('--judge-key-env',default='OPENROUTER_API_KEY');p.add_argument('--replica',action='store_true');a=p.parse_args()
    print(evaluate(a.suite,a.config,a.judge,a.output,credential(a.key_env),credential(a.judge_key_env),a.max_usd,a.replica))
