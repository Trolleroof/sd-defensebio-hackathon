"""Exact messages, correlation, durability, privacy and terminal-state logging."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from research_env.core.common import canonical, digest, file_hash, read, write
from research_env.core.transcript import Transcript
from research_env.core.agent import ResearchEnvironment
from research_env.leaderboard import freeze
from research_env.run_model import ModelProvider, run
from research_env.trace import export


def config():
    return {'competitor':'fixture-model','provider':'mock','endpoint':'https://example.invalid/chat/completions',
            'model':'exact-fixture-v1','decoding':{'temperature':0,'max_tokens':512},
            'prompt':'PRIVATE-PROMPT-MARKER','max_calls':10,'max_total_tokens':1000000}


def make_suite(root):
    for task in ('pulse','heat','load'):
        manifest = {'version':'galaxy-pulse-causal-v1'} if task=='pulse' else {'task':task,'version':task+'-v1'}
        write(root/'data'/task/'dataset.json',{'task':task})
        for role in ('train','development','test'):
            write(root/'data'/task/'private'/role/'manifest.json',manifest)
            if role!='test':
                write(root/'data'/task/'public'/role/'manifest.json',manifest)
    freeze(root/'data','sha256:'+'0'*64,root/'suite.json')
    write(root/'config.json',config())


class Logging(unittest.TestCase):
    def test_participant_receives_remaining_inference_budget_and_submission_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=config();cfg['max_calls']=4
            provider=ModelProvider(cfg,'SECRET',tmp)
            provider.usage={'calls':1,'tokens':850000}
            sent=[]
            def call(request,**kwargs):
                sent.append(json.loads(json.loads(request.data)['messages'][1]['content']))
                return io.BytesIO(canonical({'usage':{'total_tokens':100},'choices':[{'message':{'content':'{"operation":"submit"}'}}]}))
            observation={'remaining_wall_s':100,'history':[]}
            with patch('urllib.request.urlopen',side_effect=call):provider(observation,0)
            budget=sent[0]['inference_budget']
            self.assertEqual(budget['calls_remaining_including_this_call'],3)
            self.assertEqual(budget['total_tokens_remaining_before_this_call'],150000)
            self.assertTrue(budget['near_limit'])
            self.assertIn('submit',budget['instruction'])
            self.assertNotIn('inference_budget',observation)
            cfg['max_calls']=10
            provider.usage={'calls':1,'tokens':990000}
            with patch('urllib.request.urlopen',side_effect=call):provider(observation,0)
            self.assertEqual(sent[-1]['inference_budget']['calls_remaining_including_this_call'],9)
            self.assertTrue(sent[-1]['inference_budget']['near_limit'])

    def test_hash_chain_persists_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            log=Transcript(tmp)
            log.append('model_request',{'request':{'messages':[{'role':'user','content':'hello'}]}})
            log.close()
            log=Transcript(tmp)
            log.append('model_response',{'content':'hi'})
            self.assertEqual(log.records()[1]['trace_id'],'t000001')
            with log.db:
                log.db.execute("UPDATE transcript SET payload=replace(payload,'hello','altered') WHERE seq=0")
            with self.assertRaises(ValueError):log.records()
            log.close()

    def test_exact_request_response_timing_and_no_auth_headers(self):
        with tempfile.TemporaryDirectory() as tmp:
            provider=ModelProvider(config(),'PRIVATE-API-KEY',tmp)
            response={'id':'r1','model':'resolved-fixture-version','usage':{'total_tokens':20},
                      'choices':[{'message':{'role':'assistant','content':'{"operation":"edit","path":"x","text":"hi"}'},'finish_reason':'stop'}]}
            sent=[]
            def call(request,**kwargs):
                sent.append(json.loads(request.data))
                return io.BytesIO(canonical(response))
            with patch('urllib.request.urlopen',side_effect=call):
                action=provider({'remaining_wall_s':123.456,'history':[]},2)
            record=read(Path(tmp)/'provider-call-0001.json')
            self.assertEqual(record['request'],sent[0])
            self.assertEqual(json.loads(record['response_text']),response)
            self.assertEqual(record['action'],action)
            self.assertEqual(record['usage']['total_tokens'],20)
            self.assertEqual(record['token_accounting'],'provider_reported')
            self.assertEqual(record['accounted_tokens'],20)
            self.assertGreaterEqual(record['wall_s'],0)
            log=Transcript(Path(tmp)/'private');events=log.records();log.close()
            self.assertEqual(events[1]['data']['request_trace_id'],events[0]['trace_id'])
            self.assertNotIn('PRIVATE-API-KEY',json.dumps(events)+json.dumps(record))
            self.assertEqual(json.loads(events[0]['data']['request']['messages'][1]['content'])['remaining_wall_s'],123.456)

    def test_invalid_response_retains_content_and_structured_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            provider=ModelProvider(config(),'SECRET',tmp)
            value={'choices':[{'message':{'content':'not valid JSON'}}]}
            with patch('urllib.request.urlopen',return_value=io.BytesIO(canonical(value))):
                with self.assertRaises(json.JSONDecodeError):provider({},0)
            result=read(Path(tmp)/'provider-call-0001.json')
            self.assertEqual(result['content'],'not valid JSON')
            self.assertEqual(result['fault']['kind'],'invalid_response_json')
            self.assertEqual(result['status'],'error')

    def test_edits_and_action_errors_logged_without_entering_observations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            write(root/'development.json',{'task':'load','version':'load-v1'})
            env=ResearchEnvironment(root/'private',root/'workspace',root/'development.json',backend='macos-development')
            action={'operation':'edit','path':'x.py','text':'# EXACT-EDIT-CONTENTS'}
            env.act(action,model_call_id=4)
            with self.assertRaises(ValueError):env.act({'operation':'unknown'},model_call_id=5)
            trace=env.session.transcript.records()
            edits=[r for r in trace if r['kind']=='action_start']
            self.assertEqual(edits[0]['data']['action'],action)
            self.assertEqual(edits[0]['data']['call_id'],4)
            self.assertEqual(trace[-1]['data']['fault']['kind'],'controller_error')
            self.assertNotIn('EXACT-EDIT-CONTENTS',json.dumps(env.observe()))
            env.session.db.close()

    def test_full_scripted_loop_receipt_viewer_and_judge_isolation(self):
        with tempfile.TemporaryDirectory() as tmp, patch('research_env.leaderboard.resource_limits',return_value={'cpus':1,'memory_bytes':1024**3}):
            root=Path(tmp);make_suite(root)
            requests=[]
            def provider(request,**kwargs):
                body=json.loads(request.data);requests.append(body)
                obs=json.loads(body['messages'][1]['content'])
                ids=[e['event_id'] for e in obs['history']]
                decision={'hypothesis':'Verify trace','expected_result':'Valid predictor','evidence_ids':ids,'code_diff':'Fixture edit'}
                if len(requests)==1:
                    action={'operation':'edit','path':'artifact/inference.py','text':'# <script>alert("x")</script>\ndef predict(a,r):return {}\n'}
                elif len(requests)==2:
                    action={'operation':'command','argv':['python','-c','print("ok")'],'decision':decision,'seed':0}
                elif len(requests)==3:
                    action={'operation':'evaluate','decision':decision}
                else:
                    candidate=[e['candidate_id'] for e in obs['history'] if e['kind']=='candidate'][-1]
                    action={'operation':'submit','candidate_id':candidate,'reason':'Fixture selected from public evidence','evidence_ids':ids}
                return io.BytesIO(canonical({'model':'resolved-fixture','usage':{'total_tokens':30},'choices':[{'message':{'content':json.dumps(action)}}]}))
            score={'execution_valid':True,'primary_loss':1,'uncertainty':{'interval_score_90':2}}
            worker={'execution_valid':True,'stdout':'ok\n','stderr':'','exit_code':0,'wall_s':.01}
            with patch('urllib.request.urlopen',side_effect=provider), \
                 patch('research_env.core.session.execute',return_value=worker), \
                 patch('research_env.core.evaluation.evaluate',return_value=({**score,'primary_loss':987654.321},{})), \
                 patch('research_env.core.agent.evaluate',return_value=(score,{})):
                receipt=run(root/'suite.json','load',root/'config.json',0,root/'run','SECRET-KEY')
            records=[json.loads(line) for line in (root/'run/transcript.jsonl').read_text().splitlines()]
            self.assertEqual(len([r for r in records if r['kind']=='model_request']),4)
            results=[r for r in records if r['kind']=='action_result']
            self.assertEqual([r['data']['call_id'] for r in results],[1,2,3,4])
            self.assertEqual(results[1]['data']['result']['stdout'],'ok\n')
            self.assertTrue(all(r['data']['ledger_event_ids'] for r in results))
            self.assertEqual(receipt['transcript_sha256'],file_hash(root/'run/transcript.jsonl'))
            status=read(root/'run/run-status.json')
            self.assertEqual(status['status'],'completed')
            self.assertEqual(status['usage']['tokens'],120)
            self.assertEqual(status['final_answer'],'Fixture selected from public evidence')
            html=(root/'run/transcript.html').read_text()
            self.assertNotIn('<script>',html);self.assertIn('&lt;script&gt;',html)
            self.assertNotIn('PRIVATE-PROMPT-MARKER',(root/'run/judge-packets.json').read_text())
            self.assertNotIn('SECRET-KEY',(root/'run/transcript.jsonl').read_text())
            self.assertEqual(receipt['score']['primary_loss'],987654.321)
            self.assertNotIn('987654.321',(root/'run/transcript.jsonl').read_text())
            self.assertNotIn('987654.321',(root/'run/judge-packets.json').read_text())
            export(root/'run')
            self.assertEqual(receipt['run_status_sha256'],file_hash(root/'run/run-status.json'))

    def test_interrupted_action_retains_correlation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);write(root/'development.json',{'task':'load','version':'load-v1'})
            env=ResearchEnvironment(root/'private',root/'workspace',root/'development.json')
            decision={'hypothesis':'Interrupt fixture','expected_result':'Cancelled','evidence_ids':['e00000'],'code_diff':'No change'}
            with patch('research_env.core.session.execute',side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    env.act({'operation':'command','argv':['python','-c','pass'],'decision':decision},model_call_id=3)
            last=env.session.transcript.records()[-1]
            self.assertEqual(last['data']['status'],'cancelled')
            self.assertEqual(last['data']['call_id'],3)
            self.assertTrue(last['data']['ledger_event_ids'])
            env.session.db.close()

    def test_interruption_is_durable_and_resumable(self):
        with tempfile.TemporaryDirectory() as tmp, patch('research_env.leaderboard.resource_limits',return_value={'cpus':1,'memory_bytes':1024**3}):
            root=Path(tmp);make_suite(root)
            with patch('urllib.request.urlopen',side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):run(root/'suite.json','load',root/'config.json',0,root/'run','SECRET')
            status=read(root/'run/run-status.json')
            self.assertEqual(status['status'],'cancelled')
            self.assertTrue(status['resumable'])
            self.assertFalse((root/'run/receipt.json').exists())
            self.assertEqual(read(root/'run/provider-usage.json')['calls'],1)

    def test_failed_provider_seals_failure_and_preserves_error_record(self):
        with tempfile.TemporaryDirectory() as tmp, patch('research_env.leaderboard.resource_limits',return_value={'cpus':1,'memory_bytes':1024**3}):
            root=Path(tmp);make_suite(root)
            with patch('urllib.request.urlopen',side_effect=OSError('authentication SECRET must not be copied')):
                receipt=run(root/'suite.json','load',root/'config.json',0,root/'run','SECRET')
            status=read(root/'run/run-status.json')
            self.assertEqual(status['status'],'failed')
            self.assertEqual(status['fault']['kind'],'provider_error')
            self.assertFalse(receipt['score']['execution_valid'])
            self.assertIsNone(receipt['score']['primary_loss'])
            self.assertNotIn('SECRET',(root/'run/transcript.jsonl').read_text())


if __name__ == '__main__':
    unittest.main()
