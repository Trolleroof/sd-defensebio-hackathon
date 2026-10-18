import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from benchmark.agent import AgentSession
from benchmark.campaign import CampaignWorkspace
from benchmark.campaign_examples import campaign_episode
from benchmark.examples import episode
from benchmark.executor import execute
from benchmark.workspace import Workspace, read, write, score


class ContractTests(unittest.TestCase):
    def test_device_temperature_is_excluded_from_public_and_execution_inputs(self):
        for cls, ep in [(Workspace, episode()), (CampaignWorkspace, campaign_episode())]:
            with self.subTest(workspace=cls.__name__), tempfile.TemporaryDirectory() as tmp:
                for recording in ep['sessions']:
                    recording['sensor_metadata'] = {'imu_chip_temp_c': 99, 'imuDieTempC': 98}
                    for row in recording['rows']:
                        row.update(imu_chip_temp_c=99, temp_wrist=32)
                root = Path(tmp)
                w = cls.create(ep, root/'private', root/'agent')
                observation = AgentSession(w).observe()
                import json
                public = json.dumps(observation)
                self.assertNotIn('imu_chip_temp_c', public)
                self.assertNotIn('imuDieTempC', public)
                self.assertIn('temp_wrist', public)
                for filename in ('brief.json', 'support.json', 'requests.json'):
                    self.assertNotIn('imu_chip_temp_c', (w.visible/filename).read_text())
                seen = []
                def executor(code, support, requests):
                    payload = json.dumps([support, requests])
                    self.assertNotIn('imu_chip_temp_c', payload)
                    self.assertNotIn('imuDieTempC', payload)
                    self.assertIn('temp_wrist', payload)
                    seen.append(True)
                    return {'execution': {'valid': False}, 'predictions': {}}
                w._execute('pass', 'development', executor)
                self.assertEqual(seen, [True])

    def test_contract_is_published_for_both_workspaces(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name, cls, ep in [('episode', Workspace, episode()),
                                  ('campaign', CampaignWorkspace, campaign_episode())]:
                with self.subTest(name=name):
                    root = Path(tmp)/name
                    w = cls.create(ep, root/'private', root/'agent')
                    observation = AgentSession(w).observe()
                    contract = observation['brief']['predictor_contract']
                    self.assertEqual(read(w.visible/'brief.json')['predictor_contract'], contract)
                    self.assertEqual(contract['required_callable'], 'predict(history, forecast_times_s)')
                    self.assertEqual(contract['optional_callable'], 'fit(support)')
                    self.assertEqual('checkpoint' in contract['actions'], name == 'campaign')
                    self.assertEqual(observation['current_code'], None)
                    temperatures = contract['temperature_semantics']
                    self.assertIn('skin temperature', temperatures['temp_wrist'])
                    self.assertNotIn('imu_chip_temp_c', temperatures)
                    self.assertNotIn('estimated_skin_temp_c', temperatures)
                    self.assertEqual(contract['return_value']['units'], 'time_s in seconds; mean, lower, upper in bpm.')
                    if name == 'campaign':
                        def no_execution(code, support, requests):
                            return {'execution': {'valid': False}, 'predictions': {}}
                        w.checkpoint_code('pass', {
                            'evidence': 'Initial observation', 'hypothesis': 'Interface is stable across stages',
                            'action': 'Advance stage', 'expected_result': 'Same contract in Investigate',
                        }, executor=no_execution)
                        advanced = AgentSession(w).observe()
                        self.assertEqual(advanced['brief']['stage'], 'Investigate')
                        self.assertEqual(advanced['brief']['predictor_contract'], contract)

    def test_saved_episode_without_contract_gets_it_on_observation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            w = Workspace.create(episode(), root/'private', root/'agent')
            brief = read(w.visible/'brief.json')
            del brief['predictor_contract']
            write(w.visible/'brief.json', brief)
            self.assertIn('predictor_contract', AgentSession(Workspace(w.root)).observe()['brief'])

    @unittest.skipUnless(sys.platform == 'darwin' and shutil.which('sandbox-exec'),
                         'macOS sandbox required for actual contract execution')
    def test_documented_interface_runs_fits_and_scores(self):
        # An interface probe, never a model supplied to benchmark participants.
        code = '''
level = None
def fit(support):
    global level
    level = support[0]['rows'][0]['hr_bpm']
    return 'ignored'
def predict(history, forecast_times_s):
    return [{'time_s': t, 'mean': level, 'lower': level - 5, 'upper': level + 5}
            for t in forecast_times_s]
'''
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            w = Workspace.create(episode(), root/'private', root/'agent')
            contract = AgentSession(w).observe()['brief']['predictor_contract']
        requests = [{'id': 'a', 'history': [{'time_s': 0, 'hr_bpm': 120}],
                     'forecast_times_s': [30, 60]},
                    {'id': 'b', 'history': [{'time_s': 0, 'hr_bpm': 130}],
                     'forecast_times_s': [10]}]
        for level in (100, 110):
            result = execute(code, [{'rows': [{'time_s': 0, 'hr_bpm': level}]}], requests)
            self.assertTrue(result['execution']['valid'], result)
            for request in requests:
                forecasts = result['predictions'][request['id']]
                self.assertEqual([p['time_s'] for p in forecasts], request['forecast_times_s'])
                for row in forecasts:
                    self.assertEqual(set(row), set(contract['return_value']['required_exact_fields']))
                    self.assertEqual(row['mean'], level)
                reference = [{'time_s': t, 'hr_bpm': level + 1} for t in request['forecast_times_s']]
                metrics = score(reference, forecasts)
                self.assertEqual(metrics['mae_bpm'], 1)
                self.assertEqual(metrics['interval_score_90'], 10)
                self.assertEqual(metrics['coverage_90'], 1)
