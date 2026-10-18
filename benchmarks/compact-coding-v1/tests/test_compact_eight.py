import json,shutil,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from research_env.compact import *
from research_env.leaderboard import seal
from test_transcript import make_suite
from test_compact_sweep import response

P=read('leaderboard/sweeps/compact-eight/plan.json')
CS=[read(x) for x in P['configs']];J=read(P['judge_config'])

class Eight(unittest.TestCase):
 def test_counts_budget_efforts_and_probe_allowance(self):
  b=worst_case(CS,J,[0,1,2],True)
  self.assertLess(b['total_usd'],16)
  self.assertEqual(b['normal_calls'],297);self.assertEqual(b['maximum_calls_including_repairs'],594)
  self.assertGreater(b['probe_usd'],0)
  self.assertEqual([c['model'] for c in CS],list(EXPANDED_MODELS))
  for i,c in enumerate(CS):self.assertEqual(c['decoding']['reasoning']['effort'],'high' if i>=6 else 'medium')

 def test_trial_seed_reaches_request_helper_and_path(self):
  for seed in [0,1,2]:
   obs=observation('load',0,[],seed)
   self.assertIn(f'seed={seed}',obs['helper_example'])
   self.assertEqual(request_body(CS[0],obs)['seed'],seed)
   self.assertEqual(run_path(Path('/r'),CS[0],'load',seed,[0,1,2]).name,f'seed-{seed}')
  self.assertNotIn('seed',request_body(CS[1],observation('load',0,[],2)))

 def test_seed_two_receipt_and_complete_72_case_board(self):
  with tempfile.TemporaryDirectory() as d,patch('research_env.leaderboard.resource_limits',return_value={'cpus':1,'memory_bytes':1024**3}):
   root=Path(d);make_suite(root);cfg=CS[0];write(root/'config.json',cfg);run=root/'source';ledger=SpendLedger(root/'spend.sqlite',16)
   def worker(workspace,command,**kw):
    artifact=Path(workspace)/'artifact';artifact.mkdir(exist_ok=True);(artifact/'inference.py').write_text('def predict(a,r):return {}\n')
    return {'execution_valid':True,'stderr':''}
   score={'execution_valid':True,'primary_loss':1,'uncertainty':{'interval_score_90':2}}
   action={'code':'print(1)','reason':'Fit a reproducible candidate'}
   with patch('urllib.request.urlopen',side_effect=[response(action),response(action),response({'candidate':0,'reason':'Observed valid loss'})]),patch('research_env.core.session.execute',side_effect=worker),patch('research_env.core.agent.evaluate',return_value=(score.copy(),{})),patch('research_env.core.evaluation.evaluate',return_value=(score.copy(),{})):
    receipt=episode(root/'suite.json','load',cfg,root/'config.json',run,'SECRET',ledger,2)
   self.assertEqual(receipt['run']['seed'],2);self.assertTrue(receipt['score']['execution_valid'])
   ledger.close()
   for c in CS:
    for task in TASKS:
     for seed in [0,1,2]:
      target=run_path(root,c,task,seed,[0,1,2]);target.mkdir(parents=True)
      for f in ('trajectory.jsonl','transcript.jsonl','run-status.json'):shutil.copyfile(run/f,target/f)
      m={k:v for k,v in receipt['run'].items() if k!='sha256'};m.update(task=task,seed=seed,competitor=c['competitor'],configuration=c,configuration_sha256=digest(c))
      r={k:v for k,v in receipt.items() if k!='sha256'};r['run']=seal(m);write(target/'receipt.json',seal(r))
   result=board(root,CS,J,[0,1,2]);self.assertEqual(len(result['raw']),72)
   self.assertEqual(len(result['suite']['entries']),8);self.assertTrue(all(r['valid_runs']==3 for r in result['suite']['entries']))
   self.assertEqual(result['process_review']['expected'],72)
   self.assertIn('8 models × three tasks × 3 repeats',render(result))
   wrong=run_path(root,CS[0],'load',1,[0,1,2])/'receipt.json';r=read(wrong);r['run']=receipt['run'];write(wrong,seal({k:v for k,v in r.items() if k!='sha256'}))
   with self.assertRaisesRegex(ValueError,'identity'):board(root,CS,J,[0,1,2])

if __name__=='__main__':unittest.main()
