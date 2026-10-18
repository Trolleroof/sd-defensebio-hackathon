import tempfile
import unittest
from pathlib import Path
from benchmark.agent import AgentSession
from benchmark.examples import episode, decision
from benchmark.workspace import Workspace, read, write


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.w = Workspace.create(episode(), self.root/'private', self.root/'agent')

    def test_code_execution_receives_no_hidden_targets(self):
        seen = []
        def executor(code, support, requests):
            seen.append((code, support, requests))
            self.assertEqual({s['id'] for s in support}, {'calibration'})
            self.assertTrue(all(row['time_s'] <= 0 for r in requests for row in r['history']))
            return {'execution': {'valid': False, 'error': 'intentional failure'}, 'predictions': {}}
        result = self.w.evaluate_code('pass', decision('run'), executor)
        self.assertFalse(result['valid'])
        self.assertEqual(read(self.w.root/'state.json')['remaining']['evaluations'], 1)
        receipt = self.w.submit_code('pass', decision('submit'), executor)
        self.assertNotIn('execution', receipt)
        self.assertNotIn('mae_bpm', receipt)
        self.assertEqual(len(seen), 2)
        self.assertFalse(read(self.w.root/'final-score.json')['valid'])

    def test_interrupted_submission_recovers_as_explicit_failure(self):
        self.w._begin('submit_code', decision('submit'))
        self.w.state['submitted'] = True
        self.w._save()
        reopened = Workspace(self.w.root)
        score = read(reopened.root/'final-score.json')
        self.assertFalse(score['valid'])
        self.assertTrue(reopened.state['events'][0]['interrupted'])
        with self.assertRaises(ValueError):
            reopened._begin('evaluate_code', decision('late'))

    def test_agent_reconciles_completed_edit_after_agent_state_loss(self):
        session = AgentSession(self.w)
        before = read(session.path)
        session.act({'operation': 'edit', 'code': 'pass', 'decision': decision('edit')})
        write(session.path, before)
        reopened = AgentSession(Workspace(self.w.root))
        self.assertEqual(reopened.observe()['current_code'], 'pass')

    def test_artifact_tampering_is_detected(self):
        event = self.w._begin('edit', decision('edit'))
        digest = self.w._freeze_candidate('pass', event)
        (self.w.root/'candidates'/(digest+'.py')).write_text('tampered')
        with self.assertRaises(ValueError):
            self.w._load_candidate(digest)

    def test_oversized_action_is_rejected_before_log_persistence(self):
        session = AgentSession(self.w)
        before = session.path.read_text()
        with self.assertRaises(ValueError):
            session.act({'operation': 'edit', 'code': 'x'*300001, 'decision': decision('oversized')})
        self.assertEqual(session.path.read_text(), before)


if __name__ == '__main__':
    unittest.main()
