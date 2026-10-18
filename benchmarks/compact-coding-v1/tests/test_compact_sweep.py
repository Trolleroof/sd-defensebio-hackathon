import io,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from research_env.compact import *
from test_transcript import make_suite

PLAN=read('leaderboard/sweeps/compact-five/plan.json')
CONFIGS=[read(p) for p in PLAN['configs']]
JUDGE=read(PLAN['judge_config'])

def response(action,name='respond'):
 return io.BytesIO(canonical({'usage':{'cost':0.001,'total_tokens':None},'choices':[{'message':{'tool_calls':[{'function':{'name':name,'arguments':json.dumps(action)}}]}}]}))

class Compact(unittest.TestCase):
 def test_budget_and_maximum_observations(self):
  self.assertLess(worst_case(CONFIGS,JUDGE)['total_usd'],10)
  self.assertEqual(worst_case(CONFIGS,JUDGE)['maximum_calls_including_repairs'],120)
  xs=[dict(index=i,evidence_id=f'e{i:05}',reason='🙂'*160,score={'execution_valid':True,'primary_loss':1},stderr='x'*180) for i in range(2)]
  for c in CONFIGS:
   self.assertEqual(c['decoding']['reasoning']['effort'],'medium')
   for task in TASKS:
    for phase in range(3):
     body=request_body(c,observation(task,phase,xs))
     self.assertLessEqual(len(canonical(body)),4096)
  with self.assertRaises(ValueError):request_body(CONFIGS[0],{'huge':'x'*5000})

 def test_paid_repair_and_resume_do_not_repeat_network(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);ledger=SpendLedger(root/'spend.sqlite',10)
   good={'code':'print(1)','reason':'Test a valid predictor'}
   with patch('urllib.request.urlopen',side_effect=[response({}),response(good)]) as net:
    self.assertEqual(call(CONFIGS[0],{'x':1},'SECRET',root,ledger,'test',validate_action),good)
    self.assertEqual(net.call_count,2)
   with patch('urllib.request.urlopen') as net:
    self.assertEqual(call(CONFIGS[0],{'x':1},'SECRET',root,ledger,'test',validate_action),good)
    net.assert_not_called()
   self.assertEqual(ledger.summary()['requests'],2)
   self.assertAlmostEqual(ledger.summary()['accounted_usd'],.002)
   self.assertNotIn('SECRET',(root/'api/test-0.json').read_text());ledger.close()

 def test_episode_seals_two_experiments_and_one_blind_review(self):
  with tempfile.TemporaryDirectory() as d,patch('research_env.leaderboard.resource_limits',return_value={'cpus':1,'memory_bytes':1024**3}):
   root=Path(d);make_suite(root);cfg=CONFIGS[0];write(root/'config.json',cfg);run=root/'run';ledger=SpendLedger(root/'spend.sqlite',10)
   def worker(workspace,command,**kw):
    target=Path(workspace)/'artifact';target.mkdir(exist_ok=True)
    (target/'inference.py').write_text('def predict(a,r):return {}\n')
    return {'execution_valid':True,'stderr':''}
   action={'code':'print(1)','reason':'Fit ridge as a reproducible baseline'}
   score={'execution_valid':True,'primary_loss':1,'uncertainty':{'interval_score_90':2}}
   with patch('urllib.request.urlopen',side_effect=[response(action),response(action),response({'candidate':0,'reason':'Select the valid low-loss candidate'})]),patch('research_env.core.session.execute',side_effect=worker),patch('research_env.core.agent.evaluate',return_value=(score.copy(),{})),patch('research_env.core.evaluation.evaluate',return_value=(score.copy(),{})):
    receipt=episode(root/'suite.json','load',cfg,root/'config.json',run,'SECRET',ledger)
   self.assertTrue(receipt['score']['execution_valid']);self.assertEqual(len(read(run/'experiments.json')),2)
   events=[json.loads(x)['event'] for x in (run/'trajectory.jsonl').read_text().splitlines()]
   ids=[e['event_id'] for e in events if e['kind']=='experiment']
   verdict={'criteria':[{'id':c,'score':1,'reason':'Observed a recorded experiment','evidence_ids':[ids[0]]} for c in CRITERIA]}
   with patch('urllib.request.urlopen',return_value=response(verdict,'grade')):
    review(run,JUDGE,'SECRET',ledger)
   packet=read(run/'episode-judge/packet.json')
   self.assertNotIn('final_score',canonical(packet).decode());self.assertEqual(ledger.summary()['requests'],4)
   self.assertLess(len(canonical(request_body(JUDGE,packet,True))),3072)
   self.assertIsNone(board(root,CONFIGS));ledger.close()

 def test_bad_grade_citations_rejected(self):
  v={'criteria':[{'id':c,'score':1,'reason':'r','evidence_ids':['invented']} for c in CRITERIA]}
  with self.assertRaises(ValueError):validate_grade(v,{'real'})

if __name__=='__main__':unittest.main()
