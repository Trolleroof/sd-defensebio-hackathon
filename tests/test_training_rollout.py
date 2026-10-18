"""CPU-only tests for training/ (GRPO rollout glue). No GPU, no network, no verl/vLLM.

The env and policy are mocks implementing RL_ENV_CONTRACT.md v1; expected numbers are worked by
hand in comments, not recomputed with the code under test.
"""
import asyncio
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import yaml

from training import env_client
from training.advantage import batch_advantages, group_advantages, token_advantages
from training.build_dataset import rows_for
from training.config import base_model_is_placeholder, load_config, validate, SMOKE_CONFIG, DEFAULT_CONFIG
from training.rollout import (AssistantTurn, OpenAIChatPolicy, PolicyExhausted, parse_tool_calls, run_batch,
                              run_group, to_api_messages)
from training.trajectory_io import build_record, loss_mask_for, message, read_jsonl, validate_record, write_jsonl
from training.verl_overrides import AGENT_LOOP_YAML, AGENT_NAME, overrides

OPS = ('edit', 'command', 'evaluate', 'select', 'submit')
SUBMIT_REWARD = {'good': 1.0, 'meh': 0.0, 'bad': -1.0}


def task(split='agent_training', tid='pulse-t1'):
    return env_client.TaskSpec(task_family='pulse', task_id=tid, outer_split=split, fit_people=('p1', 'p2'),
                               feedback_people=('p3',), reward_people=('p4',), profile='fast', options={})


class MockEnv:
    """Implements the contract API. Reward: submit(candidate_id) -> SUBMIT_REWARD; budget out -> -2.0."""
    constructed = []

    def __init__(self, prepared_roots, run_root, *, backend='docker', image=None,
                 reward_version='physiology-reward-v1', max_turns=40, history_chars=60_000):
        self.kwargs = dict(prepared_roots=prepared_roots, backend=backend, image=image,
                           reward_version=reward_version, max_turns=max_turns, history_chars=history_chars)
        self.run_root = run_root
        self.max_turns = max_turns
        self.actions, self.turns, self.invalid, self.done, self.closed = [], 0, 0, False, False
        MockEnv.constructed.append(self)

    def budgets(self):
        return {'max_turns': self.max_turns, 'development_calls': 8, 'wall_s': 1200, 'profile': 'fast'}

    def reset(self, task, seed):
        self.task, self.seed = task, seed
        return {'task_id': task.task_id, 'budgets': self.budgets(), 'history': []}

    def tool_schemas(self):
        return [{'type': 'function', 'function': {'name': op, 'parameters': {'type': 'object'}}} for op in OPS]

    def render_observation(self, obs):
        return json.dumps(obs, sort_keys=True)

    def _info(self, extra=None):
        return {'invalid_actions': self.invalid, 'ledger_head_sha256': 'ab' * 32,
                'reward_components': {'reward_version': 'physiology-reward-v1', 'invalid_actions': self.invalid,
                                      'turns': self.turns, **(extra or {})}}

    def step(self, action):
        assert not self.done, 'step after done'
        self.actions.append(action)
        self.turns += 1
        op = action.get('operation')
        if op not in OPS:
            self.invalid += 1
            obs, reward, done = {'error': f'illegal action: {action.get("error", op)}'}, 0.0, False
        elif op == 'submit':
            obs, reward, done = {'submitted': action['candidate_id']}, SUBMIT_REWARD[action['candidate_id']], True
        else:
            obs, reward, done = {'ok': op}, 0.0, False
        if not done and self.turns >= self.max_turns:
            obs, reward, done = {'closed': 'max_turns'}, -2.0, True
        self.done = done
        return obs, reward, done, self._info({'tier': 'mock'} if done else None)

    def close(self):
        self.closed = True


def call(name, args, cid=None):
    return {'id': cid, 'type': 'function',
            'function': {'name': name, 'arguments': args if isinstance(args, str) else json.dumps(args)}}


def submit(candidate):
    return AssistantTurn('submitting', [call('submit', {'candidate_id': candidate, 'reason': 'r', 'evidence_ids': []}, 'c-sub')])


class ScriptedPolicy:
    def __init__(self, scripts):
        self.scripts = {i: list(s) for i, s in scripts.items()}
        self.in_flight = self.max_in_flight = 0
        self.seen_tools = []

    async def step(self, messages, tools, *, sample_index):
        self.seen_tools.append(tools)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        await asyncio.sleep(0.01)
        self.in_flight -= 1
        script = self.scripts[sample_index]
        if not script:
            raise PolicyExhausted('script exhausted')
        return script.pop(0)


def settings(tmp, max_turns=6):
    return env_client.EnvSettings(prepared_roots={'pulse': str(Path(tmp) / 'prep')}, run_root=str(Path(tmp) / 'runs'),
                                  backend='docker', max_turns=max_turns)


def run(coro):
    return asyncio.run(coro)


class GroupConstruction(unittest.TestCase):
    def setUp(self):
        MockEnv.constructed = []
        self.tmp = tempfile.mkdtemp()

    def test_group_of_four_identical_budgets_and_hand_advantages(self):
        pol = ScriptedPolicy({0: [submit('good')], 1: [submit('meh')], 2: [submit('bad')], 3: [submit('meh')]})
        recs = run(run_group(task=task(), seed=7, group_size=4, policy=pol, settings=settings(self.tmp),
                             env_factory=MockEnv))
        self.assertEqual(len(recs), 4)
        self.assertEqual(len({r['group_id'] for r in recs}), 1)
        self.assertEqual({r['task_id'] for r in recs}, {'pulse-t1'})
        self.assertEqual([r['rollout']['sample_index'] for r in recs], [0, 1, 2, 3])
        self.assertEqual([r['seed'] for r in recs], [7, 7, 7, 7])          # shared env seed
        self.assertEqual([e.seed for e in MockEnv.constructed], [7, 7, 7, 7])
        self.assertEqual([r['reward'] for r in recs], [1.0, 0.0, -1.0, 0.0])
        # Identical env settings/budgets; only the scratch run directory differs.
        self.assertEqual(len(MockEnv.constructed), 4)
        self.assertTrue(all(e.kwargs == MockEnv.constructed[0].kwargs for e in MockEnv.constructed))
        self.assertEqual(len({str(e.run_root) for e in MockEnv.constructed}), 4)
        self.assertEqual(len({json.dumps(r['budgets'], sort_keys=True) for r in recs}), 1)
        self.assertTrue(all(e.closed for e in MockEnv.constructed))
        self.assertGreater(pol.max_in_flight, 1)                         # episodes ran concurrently
        # Hand: mean 0; sample std = sqrt((1+0+1+0)/3) = sqrt(2/3) = 0.8164966; 1/0.8164966 = 1.2247449
        adv = [r['rollout']['advantage'] for r in recs]
        for got, want in zip(adv, [1.2247449, 0.0, -1.2247449, 0.0]):
            self.assertAlmostEqual(got, want, places=5)
        for r in recs:
            validate_record(r)
            self.assertEqual(r['schema'], 'physiology-rl-trajectory-v1')
            self.assertEqual(r['ledger_head_sha256'], 'ab' * 32)

    def test_budget_mismatch_within_group_is_an_error(self):
        class Drifting(MockEnv):
            def budgets(self):
                return {**super().budgets(), 'development_calls': 8 + len(MockEnv.constructed)}
        pol = ScriptedPolicy({0: [submit('good')], 1: [submit('good')]})
        with self.assertRaisesRegex(RuntimeError, 'Budgets differ'):
            run(run_group(task=task(), seed=0, group_size=2, policy=pol, settings=settings(self.tmp),
                          env_factory=Drifting))

    def test_group_size_one_rejected(self):
        with self.assertRaises(ValueError):
            run(run_group(task=task(), seed=0, group_size=1, policy=ScriptedPolicy({0: []}),
                          settings=settings(self.tmp), env_factory=MockEnv))

    def test_batch_of_two_tasks_gives_two_groups(self):
        cfg = load_config(SMOKE_CONFIG, environ={})
        cfg['grpo']['group_size'] = 2
        pol = ScriptedPolicy({0: [submit('good'), submit('bad')], 1: [submit('bad'), submit('good')]})
        # Each sample_index script is shared by both groups' sample i; order of pops is irrelevant here.
        recs = run(run_batch(tasks=[task(tid='a'), task(tid='b')], seeds=[0, 1], cfg=cfg, policy=pol,
                             settings=settings(self.tmp), env_factory=MockEnv))
        self.assertEqual(len(recs), 4)
        self.assertEqual(len({r['group_id'] for r in recs}), 2)
        by_group = {}
        for r in recs:
            by_group.setdefault(r['group_id'], []).append(r['task_id'])
        self.assertTrue(all(len(set(v)) == 1 and len(v) == 2 for v in by_group.values()))


class MalformedToolCalls(unittest.TestCase):
    def setUp(self):
        MockEnv.constructed = []
        self.tmp = tempfile.mkdtemp()

    def test_parse_variants(self):
        names = set(OPS)
        bad_json = parse_tool_calls(AssistantTurn('', [call('edit', '{"path": ')]), names, 0)
        unknown = parse_tool_calls(AssistantTurn('', [call('rm_rf', {})]), names, 0)
        none = parse_tool_calls(AssistantTurn('just text', []), names, 0)
        not_obj = parse_tool_calls(AssistantTurn('', [call('edit', '[1, 2]')]), names, 0)
        good = parse_tool_calls(AssistantTurn('', [call('evaluate', {'decision': 'd'}, 'x1')]), names, 0)
        for parsed in (bad_json, unknown, none, not_obj):
            self.assertEqual(parsed[0].action['operation'], env_client.MALFORMED_OPERATION)
            self.assertIsNotNone(parsed[0].error)
        self.assertEqual(none[0].tool_call_id, '')
        self.assertEqual(bad_json[0].tool_call_id, 'call_0_0')
        self.assertEqual(good[0].action, {'operation': 'evaluate', 'decision': 'd'})
        self.assertIsNone(good[0].error)

    def test_malformed_calls_become_env_error_observations_and_episode_continues(self):
        script = [AssistantTurn('', [call('edit', '{not json', 'k1')]),
                  AssistantTurn('', [call('delete_everything', {}, 'k2')]),
                  AssistantTurn('I think I am done', []),
                  submit('good')]
        pol = ScriptedPolicy({0: list(script), 1: list(script)})
        recs = run(run_group(task=task(), seed=0, group_size=2, policy=pol, settings=settings(self.tmp),
                             env_factory=MockEnv))
        env = MockEnv.constructed[0]
        self.assertEqual([a['operation'] for a in env.actions],
                         [env_client.MALFORMED_OPERATION] * 3 + ['submit'])
        self.assertEqual(env.invalid, 3)
        rec = recs[0]
        self.assertEqual(rec['reward'], 1.0)
        self.assertEqual(rec['reward_components']['invalid_actions'], 3)
        self.assertEqual(rec['rollout']['parse_errors'], 3)
        roles = [m['role'] for m in rec['messages']]
        # system, user(obs), [assistant, tool] x2, assistant(no call), user(error feedback), assistant, tool
        self.assertEqual(roles, ['system', 'user', 'assistant', 'tool', 'assistant', 'tool',
                                 'assistant', 'user', 'assistant', 'tool'])
        self.assertIn('illegal action', rec['messages'][3]['content'])
        self.assertEqual(rec['messages'][3]['tool_call_id'], 'k1')
        self.assertIn('No tool call', rec['messages'][7]['content'])

    def test_env_exception_is_caught_by_adapter(self):
        class Raising(MockEnv):
            def step(self, action):
                if action.get('operation') == 'command':
                    raise OSError('worker died')
                return super().step(action)
        script = [AssistantTurn('', [call('command', {'argv': ['ls'], 'decision': 'd'}, 'c1')]), submit('meh')]
        recs = run(run_group(task=task(), seed=0, group_size=2, policy=ScriptedPolicy({0: list(script), 1: list(script)}),
                             settings=settings(self.tmp), env_factory=Raising))
        self.assertIn('OSError: worker died', recs[0]['messages'][3]['content'])
        self.assertEqual(recs[0]['reward'], 0.0)

    def test_policy_exhaustion_lets_env_decide_terminal_reward(self):
        pol = ScriptedPolicy({0: [AssistantTurn('', [call('evaluate', {'decision': 'd'}, 'e1')])], 1: []})
        recs = run(run_group(task=task(), seed=0, group_size=2, policy=pol, settings=settings(self.tmp, max_turns=3),
                             env_factory=MockEnv))
        self.assertTrue(all(r['rollout']['forced_termination'] for r in recs))
        self.assertEqual([r['reward'] for r in recs], [-2.0, -2.0])   # MockEnv: budget exhausted -> -2.0
        # Hand: identical rewards -> mean -2, deviations 0 -> advantages 0
        self.assertEqual([r['rollout']['advantage'] for r in recs], [0.0, 0.0])


class LossMask(unittest.TestCase):
    def test_assistant_only(self):
        msgs = [message('system', 's'), message('user', 'u'), message('assistant', 'a', [call('edit', {})]),
                message('tool', 'file contents + scores', tool_call_id='t'), message('assistant', 'b'),
                message('tool', 'x', tool_call_id='t2')]
        self.assertEqual(loss_mask_for(msgs), [0, 0, 1, 0, 1, 0])

    def test_record_rejects_wrong_mask_and_roles(self):
        rec = build_record(task=task(), seed=0, group_id='g', reward=0.5, reward_version='physiology-reward-v1',
                           reward_components={}, messages=[message('system', 's'), message('assistant', 'a')],
                           ledger_head_sha256=None, budgets={})
        self.assertEqual(rec['loss_mask'], [0, 1])
        bad = dict(rec, loss_mask=[1, 1])
        with self.assertRaises(ValueError):
            validate_record(bad)
        with self.assertRaises(ValueError):
            message('critic', 'x')

    def test_token_advantages_zero_on_tool_tokens(self):
        # response tokens: 3 assistant, 4 tool-observation, 2 assistant
        mask = [1, 1, 1, 0, 0, 0, 0, 1, 1]
        self.assertEqual(token_advantages(0.5, mask), [0.5, 0.5, 0.5, 0, 0, 0, 0, 0.5, 0.5])

    def test_jsonl_roundtrip(self):
        rec = build_record(task=task(), seed=1, group_id='g', reward=-1.25, reward_version='physiology-reward-v1',
                           reward_components={'tier': 'invalid'}, messages=[message('user', 'u')],
                           ledger_head_sha256='00', budgets={'max_turns': 3})
        with tempfile.TemporaryDirectory() as d:
            p = write_jsonl([rec], Path(d) / 'sub' / 't.jsonl')
            self.assertEqual(read_jsonl(p), [rec])


class Advantages(unittest.TestCase):
    def test_hand_values(self):
        # [1, 0, -1, 0]: mean 0, sample std sqrt(2/3) -> +-1.2247449
        got = group_advantages([1, 0, -1, 0])
        for g, w in zip(got, [1.2247449, 0.0, -1.2247449, 0.0]):
            self.assertAlmostEqual(g, w, places=5)
        # mean-centering only (Dr. GRPO): [1, 0, -1, 0] - 0
        self.assertEqual(group_advantages([1, 0, -1, 0], norm_by_std=False), [1.0, 0.0, -1.0, 0.0])
        # [0.2, -2.0]: mean -0.9; sample std = sqrt((1.1^2 + 1.1^2)/1) = 1.5556349; 1.1/1.5556349 = 0.7071068
        a, b = group_advantages([0.2, -2.0])
        self.assertAlmostEqual(a, 0.7071068, places=5)
        self.assertAlmostEqual(b, -0.7071068, places=5)
        # identical rewards -> zero advantage (no signal)
        self.assertEqual(group_advantages([-1.5, -1.5, -1.5]), [0.0, 0.0, 0.0])
        # verl convention for a singleton group: mean 0, std 1
        self.assertAlmostEqual(group_advantages([0.3])[0], 0.3, places=5)

    def test_batch_groups_are_independent(self):
        got = batch_advantages(['a', 'b', 'a', 'b'], [1.0, 5.0, 0.0, 5.0])
        # group a: [1, 0] mean .5, std sqrt(.5) = .7071068 -> +-.7071068; group b identical -> 0
        self.assertAlmostEqual(got[0], 0.7071068, places=5)
        self.assertAlmostEqual(got[2], -0.7071068, places=5)
        self.assertEqual((got[1], got[3]), (0.0, 0.0))


class RefuseFinalEvaluation(unittest.TestCase):
    def setUp(self):
        MockEnv.constructed = []

    def test_everywhere(self):
        t = task(split='agent_final_evaluation')
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(env_client.FinalEvaluationRefused):
                run(run_group(task=t, seed=0, group_size=2, policy=ScriptedPolicy({0: [], 1: []}),
                              settings=settings(d), env_factory=MockEnv))
        self.assertEqual(MockEnv.constructed, [])  # no env was ever created
        with self.assertRaises(env_client.FinalEvaluationRefused):
            build_record(task=t, seed=0, group_id='g', reward=0, reward_version='v', reward_components={},
                         messages=[], ledger_head_sha256=None, budgets={})
        with self.assertRaises(env_client.FinalEvaluationRefused):
            rows_for([task(), t])
        with self.assertRaises(env_client.FinalEvaluationRefused):
            env_client.reset(MockEnv({}, '.'), t, 0)
        with self.assertRaises(ValueError):
            validate({**load_config(DEFAULT_CONFIG, environ={}), 'tasks': {'outer_split': 'agent_final_evaluation'}})
        rec = build_record(task=task(), seed=0, group_id='g', reward=0, reward_version='v', reward_components={},
                           messages=[], ledger_head_sha256=None, budgets={})
        with self.assertRaises(ValueError):
            validate_record(dict(rec, outer_split='agent_final_evaluation'))
        with self.assertRaises(ValueError):
            env_client.refuse_final_evaluation(task(split='test'))


class ConfigAndTrainerWiring(unittest.TestCase):
    def test_placeholder_and_overrides(self):
        cfg = load_config(DEFAULT_CONFIG, environ={})
        self.assertTrue(base_model_is_placeholder(cfg))
        cfg = load_config(DEFAULT_CONFIG, environ={'PHYS_GRPO__model__base_model': 'org/any-model',
                                                   'PHYS_GRPO__grpo__group_size': '16'})
        self.assertFalse(base_model_is_placeholder(cfg))
        self.assertEqual(cfg['model']['base_model'], 'org/any-model')
        self.assertEqual(cfg['grpo']['group_size'], 16)
        with self.assertRaises(ValueError):
            load_config(DEFAULT_CONFIG, environ={'PHYS_GRPO__grpo__group_size': '1'})

    def test_smoke_extends_base(self):
        smoke = load_config(SMOKE_CONFIG, environ={})
        base = load_config(DEFAULT_CONFIG, environ={})
        self.assertEqual(smoke['grpo']['group_size'], 2)
        self.assertEqual(smoke['optim']['kl_loss_type'], base['optim']['kl_loss_type'])  # inherited
        self.assertFalse(base_model_is_placeholder(smoke))

    def test_verl_overrides_map_group_and_batch(self):
        cfg = load_config(DEFAULT_CONFIG, environ={'PHYS_GRPO__model__base_model': 'org/m'})
        ov = dict(o.split('=', 1) for o in overrides(cfg, '/data'))
        self.assertEqual(ov['algorithm.adv_estimator'], 'grpo')
        self.assertEqual(ov['actor_rollout_ref.rollout.n'], str(cfg['grpo']['group_size']))
        self.assertEqual(ov['data.train_batch_size'], str(cfg['grpo']['tasks_per_batch']))
        self.assertEqual(ov['actor_rollout_ref.model.path'], 'org/m')
        self.assertEqual(ov['actor_rollout_ref.rollout.multi_turn.max_assistant_turns'], str(cfg['env']['max_turns']))
        self.assertEqual(ov['actor_rollout_ref.rollout.agent.default_agent_loop'], AGENT_NAME)
        registry = yaml.safe_load(Path(ov['actor_rollout_ref.rollout.agent.agent_loop_config_path']).read_text())
        self.assertEqual(Path(ov['actor_rollout_ref.rollout.agent.agent_loop_config_path']), AGENT_LOOP_YAML)
        self.assertEqual(registry, [{'name': AGENT_NAME, '_target_': 'training.verl_agent_loop.PhysiologyEnvAgentLoop'}])

    def test_dataset_rows(self):
        rows = rows_for([task(tid='x'), task(tid='y')], env_seed=3)
        self.assertEqual([r['agent_name'] for r in rows], [AGENT_NAME] * 2)
        spec = env_client.task_from_dict(rows[1]['extra_info']['task_spec'])
        self.assertEqual((spec.task_id, spec.reward_people, rows[1]['extra_info']['env_seed']), ('y', ('p4',), 3))


class OpenAIPolicyOffline(unittest.TestCase):
    def test_fake_client_tool_calls(self):
        captured = {}

        class Completions:
            async def create(self, **kw):
                captured.update(kw)
                tc = SimpleNamespace(id='abc', function=SimpleNamespace(name='evaluate', arguments='{"decision": "d"}'))
                msg = SimpleNamespace(content=None, tool_calls=[tc])
                return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason='tool_calls')])

        client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
        pol = OpenAIChatPolicy('http://unused', 'org/m', client=client, max_tokens=7)
        msgs = [message('system', 's'), message('assistant', 'a'), message('tool', 'o', tool_call_id='t')]
        turn = run(pol.step(msgs, [{'type': 'function', 'function': {'name': 'evaluate'}}], sample_index=0))
        self.assertEqual(turn.tool_calls[0]['function'], {'name': 'evaluate', 'arguments': '{"decision": "d"}'})
        self.assertEqual(captured['max_tokens'], 7)
        self.assertEqual(captured['messages'], to_api_messages(msgs))
        self.assertNotIn('tool_calls', captured['messages'][1])          # empty tool_calls dropped
        self.assertEqual(captured['messages'][2]['tool_call_id'], 't')
        self.assertTrue(math.isfinite(captured['temperature']))


if __name__ == '__main__':
    unittest.main()
