import tempfile
import unittest
from pathlib import Path
from benchmark.agent import AgentSession
from benchmark.examples import episode, decision
from benchmark.workspace import Workspace, read


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.w = Workspace.create(episode(), self.root/'private', self.root/'agent')
        self.session = AgentSession(self.w, max_steps=4)

    def test_observation_has_no_future_labels(self):
        observation = self.session.observe()
        for r in observation['requests']:
            self.assertEqual(len(r['history']), 1)
            self.assertNotIn('rows', r)
        self.assertEqual([r['id'] for r in observation['recordings']], ['calibration'])

    def test_edit_revert_keeps_evaluation_budget(self):
        a = self.session.act({'operation': 'edit', 'code': 'pass', 'decision': decision('edit')})
        self.session.act({'operation': 'edit', 'code': 'x=1', 'decision': decision('edit again')})
        self.session.act({'operation': 'revert', 'candidate_id': a['candidate_id'], 'decision': decision('revert')})
        self.assertEqual(self.session.observe()['current_code'], 'pass')
        self.assertEqual(self.session.observe()['remaining']['evaluations'], 2)

    def test_restart_does_not_reset_step_cap(self):
        for _ in range(4):
            self.session.act({'operation': 'unknown', 'decision': decision('invalid')})
        reopened = AgentSession(self.w, max_steps=100)
        self.assertTrue(reopened.observe()['terminated'])
        with self.assertRaises(ValueError):
            reopened.act({'operation': 'edit', 'code': 'pass', 'decision': decision('late')})

    def test_invalid_decision_cannot_change_code(self):
        result = self.session.act({'operation': 'edit', 'code': 'pass', 'decision': {}})
        self.assertFalse(result['valid'])
        self.assertIsNone(self.session.observe()['current_code'])

    def test_no_filesystem_path_tool(self):
        result = self.session.act({'operation': 'read', 'path': '../private/episode.json', 'decision': decision('read')})
        self.assertFalse(result['valid'])


if __name__ == '__main__':
    unittest.main()
