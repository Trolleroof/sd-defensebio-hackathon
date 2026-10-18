import tempfile
import unittest
from pathlib import Path
from benchmark.examples import episode, decision, predict, run
from benchmark.workspace import Workspace, read, score


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.w = Workspace.create(episode(), self.root/'private', self.root/'agent')

    def test_hidden_labels_not_exported(self):
        requests = read(self.root/'agent'/'requests.json')
        for r in requests:
            self.assertTrue(all(row['time_s'] <= 0 for row in r['history']))
            self.assertNotIn('rows', r)
        self.assertFalse((self.root/'agent'/'episode.json').exists())

    def test_restart_budget_and_invalid_call_charged(self):
        self.assertFalse(self.w.evaluate({}, decision('Bad prediction'))['valid'])
        reopened = Workspace(self.root/'private')
        self.assertEqual(reopened.state['remaining']['evaluations'], 1)
        reopened.evaluate({}, decision('Bad prediction again'))
        with self.assertRaises(ValueError):
            self.w.evaluate({}, decision('Exhausted'))

    def test_acquisition_once_and_menu_only(self):
        with self.assertRaises(ValueError):
            self.w.acquire('held-out', decision('Forbidden'))
        self.w.acquire('long-bout', decision('Acquire'))
        with self.assertRaises(ValueError):
            self.w.acquire('long-bout', decision('Repeat'))
        self.assertEqual(Workspace(self.root/'private').state['remaining']['acquisition'], 0)

    def test_submit_terminates_and_hides_score(self):
        receipt = self.w.submit('pass', {}, decision('Submit'))
        self.assertNotIn('mae_bpm', receipt)
        self.assertFalse(read(self.root/'private'/'final-score.json')['valid'])
        with self.assertRaises(ValueError):
            self.w.evaluate({}, decision('After submission'))

    def test_score_and_bad_predictions(self):
        s = episode()['sessions'][0]
        preds = [{'time_s': r['time_s'], 'mean': r['hr_bpm'], 'lower': r['hr_bpm']-5,
                  'upper': r['hr_bpm']+5} for r in s['rows'][1:]]
        self.assertEqual(score(s['rows'], preds)['interval_score_90'], 10)
        preds[0]['mean'] = float('nan')
        with self.assertRaises(ValueError):
            score(s['rows'], preds)

    def test_full_scripted_example(self):
        result = run(self.root/'demo')
        self.assertEqual(result['initial']['mae_bpm'], 20)
        self.assertEqual(result['revised']['mae_bpm'], 0)
        self.assertFalse(result['private_final_score']['code_execution_verified'])

    def test_duplicate_sessions_and_nested_workspace_rejected(self):
        ep = episode()
        ep['sessions'].append(ep['sessions'][0])
        with self.assertRaises(ValueError):
            Workspace.create(ep, self.root/'a', self.root/'b')
        with self.assertRaises(ValueError):
            Workspace.create(episode(), self.root/'nested', self.root/'nested'/'agent')


if __name__ == '__main__':
    unittest.main()
