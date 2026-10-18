import os,tempfile,threading,time,unittest
from pathlib import Path
from unittest.mock import patch
from research_env import compact
from research_env.core import session
from research_env.core.common import read,write,digest
from scripts.parallel_compact import run
from scripts.resume_compact import ProviderAwareLedger

class Parallel(unittest.TestCase):
 def exercise(self,workers):
  original=Path.cwd();plan=read('leaderboard/sweeps/compact-eight/plan.json')
  configs=[read(p) for p in plan['configs']];judge=read(plan['judge_config']);suite={'sha256':'fixture-suite'}
  fingerprint=digest({'plan':plan,'configs':configs,'judge':judge,'suite':suite['sha256']})
  with tempfile.TemporaryDirectory() as d:
   try:
    os.chdir(d);root=Path('.leaderboard-runs/compact-eight')
    write('leaderboard/sweeps/compact-eight/plan.json',plan)
    for p,c in zip(plan['configs'],configs):write(p,c)
    write(plan['judge_config'],judge);write(plan['suite'],suite)
    write(root/'status.json',{'fingerprint':fingerprint,'target_episodes':72})
    write(root/'accounting-amendment.json',{'preserved':True})
    ledger=ProviderAwareLedger(root/'spend.sqlite',16)
    charge=ledger.reserve({'model':'fixture','max_tokens':1},{'prompt':.000001,'completion':.000001});ledger.settle(charge,{'cost':.002});ledger.close()
    state={'api':0,'api_peak':0,'docker':0,'docker_peak':0};guard=threading.Lock()
    def worker(*args,**kw):
     with guard:state['docker']+=1;state['docker_peak']=max(state['docker_peak'],state['docker'])
     time.sleep(.003)
     with guard:state['docker']-=1
     return {}
    def episode(suite,task,c,path,p,key,spend,seed):
     with guard:state['api']+=1;state['api_peak']=max(state['api_peak'],state['api'])
     charge=spend.reserve({'model':'fixture','max_tokens':1},{'prompt':.000001,'completion':.000001})
     spend.settle(charge,{'cost':.001});time.sleep(.005)
     session.execute(None,None,timeout_s=1)
     write(p/'receipt.json',{'fixture':True})
     with guard:state['api']-=1
    def review(p,*args):write(p/'episode-judge/report.json',{'fixture':True})
    with patch('scripts.parallel_compact.check_suite'),patch('research_env.core.session.execute',side_effect=worker),patch('research_env.compact.episode',side_effect=episode),patch('research_env.compact.review',side_effect=review),patch('research_env.compact.board',return_value={'process_review':{'coverage':72}}),patch('research_env.compact.render',return_value='fixture'):
     result=run('SECRET',workers)
    self.assertEqual(state['api_peak'],workers);self.assertEqual(state['docker_peak'],1)
    self.assertEqual(result['completed_episodes'],72);self.assertEqual(result['judged_episodes'],72)
    self.assertAlmostEqual(result['spend']['accounted_usd'],.074)
    self.assertEqual(result['spend']['requests'],73);self.assertEqual(result['status'],'complete')
   finally:os.chdir(original)

 def test_three_workers_share_budget_and_serialize_docker(self):self.exercise(3)
 def test_four_workers_share_budget_and_serialize_docker(self):self.exercise(4)

if __name__=='__main__':unittest.main()
