import copy
import tempfile
import unittest
from pathlib import Path
from benchmark.agent import AgentSession
from benchmark.campaign import CampaignWorkspace, open_workspace
from benchmark.campaign_examples import campaign_episode
from benchmark.judge import build_blind_judge_packet
from benchmark.research_baselines import research_codes
from benchmark.workspace import read, score


def decision(action):
    return {'evidence':'Released recordings and feedback', 'hypothesis':'Condition-matched calibration may help',
            'action':action,'expected_result':'Measure error and earlier-condition retention'}


def fake_execute(code, support, requests):
    return {'execution':{'valid':True,'executor':'test-double'},
            'predictions':{r['id']:[{'time_s':t,'mean':100,'lower':90,'upper':110}
                                  for t in r['forecast_times_s']] for r in requests}}


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.ep=campaign_episode()
        self.w=CampaignWorkspace.create(self.ep,self.root/'private',self.root/'agent')

    def checkpoint(self):
        return self.w.checkpoint_code('pass',decision('checkpoint'),executor=fake_execute)

    def test_staged_evidence_and_early_submission_guard(self):
        public=self.w.public_observation()
        self.assertEqual(public['brief']['stage'],'Build')
        self.assertEqual({s['id'] for s in public['recordings']},{'short-calibration'})
        self.assertEqual({r['id'] for r in public['requests']},{'short-feedback'})
        self.assertEqual(public['brief']['acquisition_menu'],[])
        with self.assertRaises(ValueError):
            self.w.acquire('later-diagnostic',decision('future'))
        with self.assertRaises(ValueError):
            self.w.submit_code('pass',decision('early'),executor=fake_execute)
        for r in public['requests']:
            self.assertTrue(all(row['time_s']<=0 for row in r['history']))
        self.assertNotIn('hidden',str(public))

    def test_checkpoint_keeps_budget_code_and_history_without_private_scores(self):
        self.w.evaluate_code('pass',decision('evaluate'),executor=fake_execute)
        result=self.checkpoint()
        self.assertNotIn('score',result)
        self.assertNotIn('mae_bpm',result)
        self.assertEqual(result['next_stage'],'Investigate')
        reopened=open_workspace(self.w.root)
        self.assertIsInstance(reopened,CampaignWorkspace)
        self.assertEqual(reopened.state['remaining']['evaluations'],7)
        self.assertEqual(reopened._load_candidate(result['artifact_sha256']),'pass')
        self.assertEqual(len(reopened.state['events']),2)
        self.assertFalse(reopened.state['submitted'])
        self.assertEqual(len(read(self.w.root/'checkpoint-0.json')['score']['conditions']),4)
        # The public event contains a receipt, not private checkpoint scores.
        self.assertNotIn('conditions',read(self.w.root/'state.json')['events'][-1]['result'])

    def test_executor_only_receives_available_support(self):
        seen=[]
        def capture(code,support,requests):
            seen.append((support,requests))
            return fake_execute(code,support,requests)
        self.w.evaluate_code('pass',decision('evaluate'),executor=capture)
        self.assertEqual([s['id'] for s in seen[0][0]],['short-calibration'])
        self.assertEqual([r['id'] for r in seen[0][1]],['short-feedback'])
        self.checkpoint()
        self.w.acquire('long-diagnostic',decision('acquire'))
        self.w.evaluate_code('pass',decision('evaluate'),executor=capture)
        self.assertEqual({s['id'] for s in seen[1][0]},{'short-calibration','long-diagnostic'})
        self.assertEqual({r['id'] for r in seen[1][1]},{'short-feedback','long-feedback'})
        self.assertTrue(all(row['time_s']<=0 for r in seen[1][1] for row in r['history']))

    def test_no_budget_reset_or_reacquisition_on_round_transition(self):
        self.checkpoint()
        self.w.acquire('long-diagnostic',decision('acquire'))
        self.checkpoint()
        with self.assertRaises(ValueError):
            self.w.acquire('long-diagnostic',decision('repeat'))
        self.assertEqual(read(self.w.root/'state.json')['remaining']['acquisition'],2)
        self.checkpoint()
        with self.assertRaises(ValueError):
            self.checkpoint()
        receipt=self.w.submit_code('pass',decision('finish'),executor=fake_execute)
        self.assertTrue(receipt['submitted'])
        with self.assertRaises(ValueError):
            self.w.evaluate_code('pass',decision('late'),executor=fake_execute)

    def test_chronological_judge_snapshots_do_not_gain_later_evidence(self):
        self.w.evaluate_code('pass',decision('evaluate'),executor=fake_execute)
        self.checkpoint()
        self.w.acquire('long-diagnostic',decision('acquire'))
        self.w.evaluate_code('pass',decision('evaluate'),executor=fake_execute)
        packet=build_blind_judge_packet(self.w.root/'state.json',self.w.visible)
        first=packet['event_packets'][0]
        self.assertNotIn('long-diagnostic',str(first))
        self.assertNotIn('long-feedback',str(first))
        self.assertNotIn('loaded',str(first))
        last=packet['event_packets'][-1]
        self.assertIn('long-diagnostic',str(last))
        self.assertNotIn('later-diagnostic',str(last))

    def test_agent_protocol_discovers_checkpoint_and_reopens_same_campaign(self):
        session=AgentSession(self.w,max_steps=48)
        self.assertIn('checkpoint',session.observe()['protocol'])
        edit=session.act({'operation':'edit','code':'pass','decision':decision('edit')})
        self.checkpoint()
        restarted=AgentSession(open_workspace(self.w.root))
        self.assertEqual(restarted.observe()['brief']['stage'],'Investigate')
        self.assertIn(edit['candidate_id'],restarted.observe()['candidate_ids'])
        self.assertEqual(restarted.observe()['current_code'],'pass')

    def test_structure_search_retains_earlier_condition_but_recent_only_forgets(self):
        support=[s for s in self.ep['sessions'] if s['role'] in ('support','acquisition') and '-diagnostic' in s['id'] or s['role']=='support']
        targets=[s for s in self.ep['sessions'] if s['role']=='test']
        scores={}
        for name in ('pooled_structure_search','recent_only_search','contextual_structure_search'):
            namespace={}
            exec(research_codes()[name],namespace)
            namespace['fit'](copy.deepcopy(support))
            errors={}
            for s in targets:
                history=[r for r in s['rows'] if r['time_s']<=0]
                predictions=namespace['predict'](history,[r['time_s'] for r in s['rows'] if r['time_s']>0])
                errors[s['condition']]=score(s['rows'],predictions)['mae_bpm']
            scores[name]=errors
        self.assertLess(max(scores['contextual_structure_search'].values()),1e-8)
        self.assertGreater(scores['recent_only_search']['short'],10)
        self.assertLess(scores['recent_only_search']['later'],1e-8)
        self.assertGreater(sum(scores['pooled_structure_search'].values()),5)

    def test_acquisition_choices_provide_different_information(self):
        self.checkpoint()
        self.w.acquire('repeat-short-1',decision('repeat original'))
        known=self.w._support()
        self.assertEqual({s['condition'] for s in known},{'short'})
        self.assertEqual(read(self.w.root/'state.json')['remaining']['acquisition'],2)


if __name__=='__main__':
    unittest.main()
