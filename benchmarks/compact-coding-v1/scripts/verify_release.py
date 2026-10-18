"""Check this release and, optionally, a controller's prepared data/suite."""
import argparse
from pathlib import Path
from research_env.core.common import digest,read,tree_hashes
from research_env.leaderboard import code_hashes,check_suite

REFERENCE=Path(__file__).resolve().parents[1]/'reference.json'

def verify_release(data_root=None,suite_path=None,replica=False):
    reference=read(REFERENCE)
    if code_hashes()!=reference['code_hashes']:raise ValueError('Release differs from the running benchmark source')
    if data_root:
        for task,expected in reference['datasets'].items():
            if digest(tree_hashes(Path(data_root)/task))!=expected['tree_sha256']:raise ValueError('Prepared data differ: '+task)
    if suite_path:
        suite=read(suite_path);check_suite(suite)
        for task,expected in reference['datasets'].items():
            if suite['datasets'][task]['tree_sha256']!=expected['tree_sha256']:raise ValueError('Suite uses different data: '+task)
        if suite['policy']!=reference['policy']:raise ValueError('Suite uses different numerical policy')
        if not replica and suite['sha256']!=reference['reference_suite_sha256']:raise ValueError('Use the original suite, or explicitly select replica mode for a new pool')
    return {'code_verified':True,'data_verified':bool(data_root or suite_path),'mode':'replica-new-pool' if replica else 'reference','reference_suite_sha256':reference['reference_suite_sha256']}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root');p.add_argument('--suite');p.add_argument('--replica',action='store_true');a=p.parse_args()
    print(verify_release(a.data_root,a.suite,a.replica))
