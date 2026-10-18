"""Trainer rollout glue against the REAL PhysiologyResearchEnv (not a mock) on prepared heat data.

Skips unless `.research-data/heat/prepared` exists and a local sandbox backend is available
(macOS sandbox-exec, or linux-pod on Linux). No GPU, no model: a policy that reads the env's
tool results stands in for the LLM.
"""
import asyncio
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

from research_env.rl.api import get_family
from training import env_client
from training.config import DEFAULT_CONFIG, load_config, validate
from training.rollout import AssistantTurn, PolicyExhausted, run_group
from training.trajectory_io import validate_record

HEAT = Path('.research-data/heat/prepared')
BACKEND = 'macos-development' if sys.platform == 'darwin' and shutil.which('sandbox-exec') else \
    'linux-pod' if sys.platform.startswith('linux') else None
DECISION = {'hypothesis': 'starter artifact is a valid baseline', 'expected_result': 'valid development score',
            'evidence_ids': [], 'code_diff': 'none'}


def call(name, args, cid):
    return {'id': cid, 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}


class ReadingPolicy:
    """Sample 0: evaluate, then submit the candidate id the env returned. Sample 1: one malformed call, then stop."""
    async def step(self, messages, tools, *, sample_index):
        turn = sum(m['role'] == 'assistant' for m in messages)
        if sample_index == 1:
            if turn == 0:
                return AssistantTurn('', [call('evaluate', '{not json', 'c0')])
            raise PolicyExhausted('context exhausted')
        if turn == 0:
            return AssistantTurn('evaluate starter', [call('evaluate', {'decision': DECISION}, 'c0')])
        status, body = messages[-1]['content'].split('\n', 1)
        self_check = json.loads(status)
        assert 'development_calls_remaining' in self_check and 'history' not in messages[-1]['content']
        result = json.loads(body)
        return AssistantTurn('submit', [call('submit', {'candidate_id': result['candidate_id'], 'reason': 'only candidate',
                                                        'evidence_ids': []}, 'c1')])


@unittest.skipUnless(HEAT.exists() and BACKEND, 'needs prepared heat data and a local sandbox backend')
class RealEnvGroup(unittest.TestCase):
    def test_group_against_real_env(self):
        task = next(t for t in get_family('heat').build_pool(HEAT, 0) if t.outer_split == 'agent_training')
        with tempfile.TemporaryDirectory() as tmp:
            settings = env_client.EnvSettings(prepared_roots={'heat': str(HEAT.resolve())}, run_root=tmp,
                                              backend=BACKEND, max_turns=6, tool_result_chars=2500)
            records = asyncio.run(run_group(task=task, seed=0, group_size=2, policy=ReadingPolicy(),
                                            settings=settings))
        self.assertEqual([r['reward'] for r in records], [0.0, -2.0])   # starter == frozen baseline; unsubmitted
        self.assertEqual(records[0]['reward_components']['tier'], 'valid_submission')
        self.assertGreater(records[0]['rollout']['advantage'], 0)
        self.assertLess(records[1]['rollout']['advantage'], 0)
        self.assertEqual(records[0]['budgets'], records[1]['budgets'])
        for record in records:
            validate_record(record)
            masked = [m for m, keep in zip(record['messages'], record['loss_mask']) if keep]
            self.assertTrue(masked and all(m['role'] == 'assistant' for m in masked))
            self.assertTrue(all(len(m['content']) <= 2500+300 for m in record['messages'] if m['role'] == 'tool'))


class ContextBudget(unittest.TestCase):
    def test_overflowing_turn_budget_is_rejected(self):
        cfg = load_config(DEFAULT_CONFIG)
        cfg['env']['max_turns'] = 40
        with self.assertRaisesRegex(ValueError, 'Context budget'):
            validate(cfg)


if __name__ == '__main__':
    unittest.main()
