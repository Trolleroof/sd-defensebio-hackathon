"""RL environment contract v1 checks on a tiny synthetic prepared pulse fixture.

Synthetic signals exercise wiring only; they carry no physiological evidence.
"""
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
import numpy as np
from research_env.core.common import digest, file_hash, read, tree_hashes, write
from research_env.rl.api import (OUTER_SPLITS, TaskSpec, get_family, validate_against_splits,
                                 validate_pool, validate_spec)
from research_env.rl.reward import compute_reward
from research_env.rl import trajectory

HR = {'T1': 66, 'T2': 72, 'T3': 78, 'T4': 84, 'T5': 90, 'D1': 75, 'D2': 81, 'X1': 69, 'X2': 87}
SPLITS = {'train': ['T1', 'T2', 'T3', 'T4', 'T5'], 'development': ['D1', 'D2'], 'test': ['X1', 'X2']}
BOUTS = {'b000': [10., 12., 14.], 'b001': [24., 26., 28.]}


def make_prepared(root):
    """Prepared-root layout of research_env.tasks.pulse.data.prepare, with synthetic sinusoids."""
    root = Path(root)
    t = np.arange(0, 32.001, .04)
    for role, people in SPLITS.items():
        folder = root/'private'/role
        folder.mkdir(parents=True)
        entries = []
        for person in people:
            ppg = np.column_stack((t, np.sin(2*np.pi*HR[person]/60*t)))
            acc = np.column_stack((t, .1*np.sin(2*np.pi*.3*t), .1*np.cos(2*np.pi*.3*t), np.zeros_like(t)))
            name = digest(['recording', person])[:16]+'.npz'
            np.savez_compressed(folder/name, ppg=ppg, acceleration=acc)
            records = [{'id': digest([person, s, c])[:24], 'person': person, 'session': s, 'activity': 'walking',
                        'cutoff': c, 'reference_bpm': float(HR[person])} for s, cs in BOUTS.items() for c in cs]
            entries.append({'person': person, 'signals': name, 'sha256': file_hash(folder/name), 'records': records})
        write(folder/'manifest.json', {'version': 'synthetic-fixture', 'role': role, 'rules': {}, 'sources': {},
                                       'entries': entries})
    write(root/'private'/'splits.json', SPLITS)
    return root


STATEFUL = '''CALLS = [0]
def predict(artifact_dir, request):
    CALLS[0] += 1
    n = 0 if request['state'] is None else request['state']['n'] + 1
    # Equals 60+n only if the module and state are fresh at every bout start.
    m = 60.0 + n + 100*(CALLS[0]-1-n)
    return {'mean_bpm': m, 'lower_bpm': m-1, 'upper_bpm': m+1, 'state': {'n': n}}
'''
TRAIN_SCRIPT = '''import shutil
from research_env.tasks.pulse.baselines import train
shutil.rmtree('artifact', ignore_errors=True)
train('data/train/manifest.json', None, 'artifact', 0, 'spectral')
'''
DECISION = {'hypothesis': 'Spectral peak tracks the pulse', 'expected_result': 'Valid development MAE',
            'evidence_ids': [], 'code_diff': 'Add train_model.py fitting the spectral baseline'}


class Reward(unittest.TestCase):
    def call(self, **kw):
        base = dict(submitted=False, reward_result=None, baseline_loss=None, any_valid_evaluation=False)
        return compute_reward(**{**base, **kw})

    def test_hand_computed_tiers(self):
        ok = lambda loss: {'execution_valid': True, 'primary_loss': loss}
        # 1 - 6/8 = 0.25
        r, c = self.call(submitted=True, reward_result=ok(6.), baseline_loss=8.)
        self.assertAlmostEqual(r, .25)
        self.assertEqual((c['tier'], c['loss'], c['baseline_loss']), ('valid_submission', 6., 8.))
        self.assertEqual(c['reward_version'], 'physiology-reward-v1')
        # 1 - 20/8 = -1.5 -> clipped to -1
        self.assertEqual(self.call(submitted=True, reward_result=ok(20.), baseline_loss=8.)[0], -1.)
        self.assertAlmostEqual(self.call(submitted=True, reward_result=ok(20.), baseline_loss=8.)[1]['skill'], -1.5)
        self.assertEqual(self.call(submitted=True, reward_result=ok(0.), baseline_loss=8.)[0], 1.)
        # baseline 0 -> eps; 1 - 1/1e-9 clipped to -1
        self.assertEqual(self.call(submitted=True, reward_result=ok(1.), baseline_loss=0.)[0], -1.)
        bad = {'execution_valid': False, 'primary_loss': None, 'failure_reason': 'nan'}
        self.assertEqual(self.call(submitted=True, reward_result=bad, baseline_loss=8., any_valid_evaluation=True)[0], -1.25)
        # A "valid" result with a missing loss never becomes zero loss.
        self.assertEqual(self.call(submitted=True, reward_result={'execution_valid': True, 'primary_loss': None})[0], -1.25)
        self.assertEqual(self.call(any_valid_evaluation=True)[0], -1.5)
        r, c = self.call(invalid_actions=3, turns=7, development_calls_used=2)
        self.assertEqual(r, -2.)
        self.assertEqual({k: c[k] for k in ('invalid_actions', 'turns', 'development_calls_used', 'eps')},
                         {'invalid_actions': 3, 'turns': 7, 'development_calls_used': 2, 'eps': 1e-9})
        with self.assertRaises(ValueError):
            self.call(reward_version='physiology-reward-v2')


class Splits(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = make_prepared(Path(self.tmp.name)/'prepared')
        self.family = get_family('pulse')

    def tearDown(self):
        self.tmp.cleanup()

    def test_pool_isolation_and_determinism(self):
        pool = self.family.build_pool(self.root, seed=3)
        self.assertEqual(pool, self.family.build_pool(self.root, seed=3))
        self.assertNotEqual([s.task_id for s in pool], [s.task_id for s in self.family.build_pool(self.root, seed=4)])
        test = set(SPLITS['test'])
        splits = {s.outer_split for s in pool}
        self.assertEqual(splits, {'agent_training', 'agent_development'})
        for spec in pool:
            people = [*spec.fit_people, *spec.feedback_people, *spec.reward_people]
            self.assertEqual(len(people), len(set(people)))
            self.assertFalse(set(people) & test)
            if spec.outer_split == 'agent_training':
                self.assertLessEqual(set(people), set(SPLITS['train']))
                # 5 inner training people -> 3 fit / 1 feedback / 1 reward
                self.assertEqual((len(spec.fit_people), len(spec.feedback_people), len(spec.reward_people)), (3, 1, 1))
            else:
                self.assertEqual(set(spec.reward_people), set(SPLITS['development']))
                self.assertLessEqual(set(spec.fit_people) | set(spec.feedback_people), set(SPLITS['train']))
        final = self.family.final_evaluation_pool(self.root, seed=3)
        self.assertEqual([set(s.reward_people) for s in final], [test])
        from research_env.tasks.pulse.rl_pool import partition_sizes
        self.assertEqual(partition_sizes(13), (8, 2, 3))

    def test_validators_reject_leaks(self):
        spec = lambda split, fit, fb, rw: TaskSpec('pulse', 'x', split, fit, fb, rw, 'fast')
        with self.assertRaisesRegex(ValueError, 'test people'):
            validate_against_splits(spec('agent_training', ('T1',), ('T2',), ('X1',)), SPLITS)
        with self.assertRaisesRegex(ValueError, 'test people'):
            validate_against_splits(spec('agent_development', ('T1', 'X2'), ('T2',), ('D1',)), SPLITS)
        with self.assertRaisesRegex(ValueError, 'inner training'):
            validate_against_splits(spec('agent_training', ('T1',), ('T2',), ('D1',)), SPLITS)
        with self.assertRaisesRegex(ValueError, 'overlap'):
            validate_spec(spec('agent_training', ('T1',), ('T1',), ('T2',)))
        with self.assertRaises(ValueError):
            validate_spec(spec('random_split', ('T1',), ('T2',), ('T3',)))
        with self.assertRaises(ValueError):
            validate_pool([spec('agent_training', ('T1',), ('T2',), ('T3',))]*2, SPLITS)
        validate_against_splits(spec('agent_final_evaluation', ('T1',), ('D1',), ('X1',)), SPLITS)
        self.assertEqual(OUTER_SPLITS, ('agent_training', 'agent_development', 'agent_final_evaluation'))

    def test_workspace_has_no_private_labels_or_reward_people(self):
        spec = self.family.build_pool(self.root, seed=0)[0]
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)/'workspace'
            self.family.stage_workspace(spec, self.root, workspace)
            train = read(workspace/'data/train/manifest.json')
            development = read(workspace/'data/development/manifest.json')
            self.assertEqual({e['person'] for e in train['entries']}, set(spec.fit_people))
            self.assertEqual({e['person'] for e in development['entries']}, set(spec.feedback_people))
            self.assertNotIn('reference_bpm', json.dumps(development))
            text = ''.join(p.read_text(errors='ignore') for p in workspace.rglob('*.json'))
            for person in spec.reward_people:
                self.assertNotIn(f'"{person}"', text)
            self.assertFalse((workspace/'research_env/core/evaluation.py').exists())
            short = TaskSpec.from_dict({**spec.to_dict(), 'options': {'max_bouts_per_person': 1, 'max_windows_per_bout': 2}})
            manifest = read(self.family.private_manifest(short, self.root, 'reward', Path(tmp)/'private'))
            self.assertEqual([len(e['records']) for e in manifest['entries']], [2])
            self.assertEqual([r['cutoff'] for r in manifest['entries'][0]['records']][1]-
                             [r['cutoff'] for r in manifest['entries'][0]['records']][0], 2.)


@unittest.skipUnless(sys.platform == 'darwin', 'Local sandbox tests require macOS')
class Replay(unittest.TestCase):
    def test_batched_resets_per_bout_and_matches_strict(self):
        from research_env.core.evaluation import evaluate
        from research_env.tasks.pulse.baselines import replay, train
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            root = make_prepared(tmp/'prepared')
            manifest = root/'private/train/manifest.json'
            artifact = tmp/'stateful'
            artifact.mkdir()
            (artifact/'inference.py').write_text(STATEFUL)
            strict, sp = evaluate(artifact, manifest, backend='macos-development', wall_s=120)
            batched, bp = evaluate(artifact, manifest, backend='macos-development', wall_s=120, replay='batched')
            # Each bout predicts 60, 61, 62 against constant HR h >= 66: bout MAE = h - 61.
            expected = np.mean([HR[p] - 61 for p in SPLITS['train']])
            self.assertTrue(strict['execution_valid'], strict)
            self.assertAlmostEqual(strict['primary_loss'], expected)
            self.assertEqual(sp, bp)
            self.assertEqual(batched['primary_loss'], strict['primary_loss'])
            self.assertEqual(len(batched['resource_use']['workers']), 5)   # one per person
            self.assertEqual(len(strict['resource_use']['workers']), 10)   # one per bout
            self.assertFalse(batched['scored_resource_isolation'])
            # Pulse results unchanged through the adapter path vs trusted in-process replay.
            train(manifest, None, tmp/'spectral', 0, 'spectral')
            default, _ = evaluate(tmp/'spectral', manifest, backend='macos-development', wall_s=120)
            trusted, _ = replay(tmp/'spectral', manifest)
            self.assertAlmostEqual(default['primary_loss'], trusted['primary_loss'])
            with self.assertRaisesRegex(ValueError, 'Unknown replay adapter'):
                evaluate(tmp/'spectral', manifest, backend='macos-development', adapter='heat-unregistered')


@unittest.skipUnless(sys.platform == 'darwin', 'Local sandbox tests require macOS')
class Episode(unittest.TestCase):
    def setUp(self):
        from research_env.rl.env import PhysiologyResearchEnv
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.root = make_prepared(base/'prepared')
        self.envs = []
        self.make = lambda **kw: self.envs.append(PhysiologyResearchEnv(
            {'pulse': self.root}, base/'runs', backend='macos-development', **kw)) or self.envs[-1]
        self.spec = get_family('pulse').build_pool(self.root, seed=0)[0]

    def tearDown(self):
        for env in self.envs:
            env.close()
        self.tmp.cleanup()

    def test_scripted_episode_reward_and_trajectory(self):
        env = self.make(max_turns=10)
        obs = env.reset(self.spec, seed=0)
        names = [t['function']['name'] for t in env.tool_schemas()]
        self.assertEqual(names, ['edit', 'command', 'evaluate', 'select', 'submit'])
        self.assertEqual(obs['budgets'], {'max_turns': 10, 'wall_s': 1200, 'development_calls': 8,
                                          'replay': 'batched', 'profile': 'fast'})
        workspace = env.episode_dir/'workspace'
        for private in (env.episode_dir/'controller', env.episode_dir/'private'):
            self.assertFalse(private.resolve().is_relative_to(workspace.resolve()))
        self.assertNotIn('reward', [p.name for p in (env.episode_dir/'private').iterdir()])
        steps = [{'operation': 'edit', 'path': 'train_model.py', 'text': TRAIN_SCRIPT},
                 {'operation': 'command', 'argv': [sys.executable, 'train_model.py'], 'decision': DECISION, 'seed': 0},
                 {'operation': 'evaluate', 'decision': DECISION}]
        rewards = []
        for action in steps:
            obs, reward, done, info = env.step(action)
            rewards.append(reward)
            self.assertIsNone(obs['error'], obs)
            self.assertFalse(done)
        self.assertEqual(rewards, [0., 0., 0.])   # terminal-only reward
        evaluation = obs['last_result']
        self.assertTrue(evaluation['execution_valid'], evaluation)
        self.assertEqual(set(evaluation['per_person_mae_bpm']), set(self.spec.feedback_people))
        obs, reward, done, info = env.step({'operation': 'submit', 'candidate_id': evaluation['candidate_id'],
                                            'reason': 'Only valid candidate', 'evidence_ids': []})
        self.assertTrue(done)
        c = info['reward_components']
        self.assertEqual(c['tier'], 'valid_submission')
        self.assertEqual((c['turns'], c['invalid_actions'], c['development_calls_used']), (4, 0, 1))
        self.assertEqual(info['done_reason'], 'submitted')
        self.assertGreater(c['baseline_loss'], 0)
        self.assertEqual(reward, max(-1., min(1., 1-c['loss']/c['baseline_loss'])))
        self.assertTrue((env.episode_dir/'private'/'reward'/'manifest.json').exists())
        # After the end, steps are no-ops with zero reward.
        self.assertEqual(env.step({'operation': 'evaluate', 'decision': {}})[1:3], (0.0, True))
        record = trajectory.build_record(env, group_id='g0')
        roles = [m['role'] for m in record['messages']]
        self.assertEqual(roles[:2], ['system', 'user'])
        self.assertEqual(roles[2:], ['assistant', 'tool']*4)
        self.assertEqual(record['loss_mask'], [0, 0] + [1, 0]*4)
        self.assertEqual(record['ledger_head_sha256'], env.environment.session.events()[-1]['sha256'])
        self.assertEqual(record['messages'][3]['tool_call_id'], record['messages'][2]['tool_calls'][0]['id'])
        out = Path(self.tmp.name)/'traj.jsonl'
        trajectory.write_jsonl(out, [record])
        self.assertEqual(trajectory.read_jsonl(out)[0]['reward'], reward)
        self.assertEqual(record['schema'], 'physiology-rl-trajectory-v1')
        rendered = env.render_observation(obs)
        self.assertNotIn('reference_bpm', rendered)
        for person in self.spec.reward_people:
            self.assertNotIn(person, rendered)
        env.close()

    def test_errors_become_observations_and_budget_exhaustion(self):
        env = self.make(max_turns=6, profiles={'fast': {'development_calls': 1}})
        env.reset(self.spec, seed=1)
        bad = ['not a dict', {'operation': 'rm -rf'}, {'operation': 'edit', 'path': '../escape.py', 'text': 'x'},
               {'operation': 'command', 'argv': 'python', 'decision': DECISION}]
        for action in bad:
            obs, reward, done, info = env.step(action)
            self.assertIsNotNone(obs['error'])
            self.assertEqual(reward, 0.)
            self.assertFalse(done)
        self.assertEqual(info['invalid_actions'], 4)
        self.assertFalse((env.episode_dir/'escape.py').exists())
        obs, reward, done, info = env.step({'operation': 'evaluate', 'decision': {'hypothesis': 'no artifact'}})
        self.assertIsNone(obs['error'])
        self.assertFalse(obs['last_result']['execution_valid'])     # a failed evaluation consumes the call
        self.assertEqual(obs['development_calls_remaining'], 0)
        obs, reward, done, info = env.step({'operation': 'evaluate', 'decision': {}})
        self.assertIn('Development budget exhausted', obs['error'])
        self.assertTrue(done)
        self.assertEqual(info['done_reason'], 'max_turns')
        self.assertEqual(reward, -2.0)
        self.assertEqual(info['reward_components']['invalid_actions'], 5)

    def test_wall_budget_and_identical_group_budgets(self):
        env = self.make(profiles={'fast': {'wall_s': 1}})
        budgets = [env.reset(self.spec, seed=s)['budgets'] for s in (0, 1)]
        self.assertEqual(budgets[0], budgets[1])
        time.sleep(1.1)
        obs, reward, done, info = env.step({'operation': 'edit', 'path': 'a.py', 'text': 'x'})
        self.assertTrue(done)
        self.assertEqual(info['done_reason'], 'wall_budget_exhausted')
        self.assertEqual((reward, info['invalid_actions']), (-2.0, 0))

    def test_terminate_scores_unsubmitted_and_reports_ledger_head(self):
        env = self.make()
        env.reset(self.spec, seed=2)
        obs, reward, done, info = env.step({'operation': 'edit', 'path': 'a.py', 'text': 'x'})
        self.assertEqual(info['ledger_head_sha256'], env.environment.session.events()[-1]['sha256'])
        obs, reward, done, info = env.terminate('context_exhausted')
        self.assertEqual((reward, done, info['done_reason']), (-2.0, True, 'terminated: context_exhausted'))
        self.assertEqual(env.terminate('again')[1:3], (0.0, True))
        self.assertEqual(env.step({'operation': 'evaluate', 'decision': {}})[1:3], (0.0, True))

    def test_final_evaluation_export_refused(self):
        final = get_family('pulse').final_evaluation_pool(self.root, seed=0)[0]
        env = self.make(max_turns=1)
        env.reset(final, seed=0)
        self.assertEqual(env.budgets['development_calls'], 30)
        obs, reward, done, info = env.step({'operation': 'evaluate', 'decision': {}})
        self.assertTrue(done)
        with self.assertRaisesRegex(ValueError, 'agent_final_evaluation'):
            trajectory.build_record(env, group_id='g')
        record = {'schema': 'physiology-rl-trajectory-v1', 'outer_split': 'agent_final_evaluation',
                  'messages': [], 'loss_mask': []}
        with self.assertRaisesRegex(ValueError, 'agent_final_evaluation'):
            trajectory.write_jsonl(Path(self.tmp.name)/'x.jsonl', [record])


if __name__ == '__main__':
    unittest.main()
