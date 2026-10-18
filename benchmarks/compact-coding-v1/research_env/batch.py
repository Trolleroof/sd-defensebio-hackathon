"""Explicitly launch one configured competitor on all frozen tasks and seeds."""
import argparse
import os
from pathlib import Path
from research_env.core.common import read
from research_env.leaderboard import POLICY, check_suite, finalize, initialize, verify
from research_env.tasks.registry import TASKS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('suite');p.add_argument('config');p.add_argument('run_root')
    p.add_argument('--key-env');p.add_argument('--conventional',action='store_true')
    args=p.parse_args()
    suite=read(args.suite);check_suite(suite)
    if not args.conventional and not args.key_env:
        p.error('Participant model calls require --key-env')
    if args.conventional:
        from scripts.search_tasks import search
    else:
        from research_env.run_model import run
        key=os.environ[args.key_env]
    for seed in POLICY['agent_seeds']:
        for task in TASKS:
            root=Path(args.run_root)/f'{task}-seed-{seed}'
            if (root/'receipt.json').exists():
                receipt=verify(read(root/'receipt.json'))
                if receipt['run']['suite_sha256']!=suite['sha256'] or receipt['run']['configuration']!=read(args.config):
                    raise ValueError('Cannot resume another competitor or suite')
                continue
            print(f'Running {task} seed {seed}',flush=True)
            if args.conventional:
                if (root/'run.json').exists():
                    raise ValueError('Interrupted conventional grid: finalize as failed; do not reset budget')
                initialize(args.suite,task,args.config,seed,root)
                best=search(args.suite,task,root)
                finalize(args.suite,root,None if best else 'Conventional search did not submit')
            else:
                run(args.suite,task,args.config,seed,root,key)


if __name__=='__main__':
    main()
