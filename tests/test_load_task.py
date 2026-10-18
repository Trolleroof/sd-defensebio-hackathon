"""Set 3 load task checks. Expected values are hand-computed, not re-implemented.

Run: .venv-research/bin/python -m unittest tests.test_load_task -v
"""
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from research_env.tasks.load import baseline_model, data, tasks
from research_env.tasks.load.scoring import ordering_concordance, score
from research_env.tasks.load.evaluation import replay_in_process

ROOT = Path(__file__).resolve().parents[1]
ORIGINALS = ROOT/'.research-data'/'load'/'originals'
HAVE_DATA = (ORIGINALS/data.WORKBOOK).exists()
PREPARED = None


def prepared():
    global PREPARED
    if PREPARED is None:
        PREPARED = Path(tempfile.mkdtemp(prefix='load-prepared-'))
        data.prepare(ORIGINALS, PREPARED)
    return PREPARED


class Units(unittest.TestCase):
    def test_kcal_per_day_to_watts_hand_examples(self):
        self.assertAlmostEqual(data.kcal_per_day_to_w(86400), 4184.0, places=9)
        self.assertAlmostEqual(data.kcal_per_day_to_w(1000), 48.425925926, places=8)   # 4,184,000/86,400
        self.assertAlmostEqual(data.kcal_per_day_to_w(4855), 235.107870370, places=8)  # 20,313,320/86,400
        # A rate, not "calories burned in the 11-minute bout" (that would be 37.09 kcal).
        self.assertAlmostEqual(4855*11/1440, 37.0868, places=4)

    def test_speed_and_grade(self):
        self.assertAlmostEqual(data.SPEED_M_S, 1.1111111111, places=9)  # 4000 m / 3600 s
        self.assertEqual(data.GRADE_PERCENT, 0.0)

    def test_pandolf_hand_examples(self):
        # W=70, L=0, V=10/9, G=0: 105 + 70*1.5*100/81 = 105 + 129.6296
        self.assertAlmostEqual(baseline_model.pandolf_w(70, 0, 10/9, 0), 234.6296, places=3)
        # L=10: 105 + 2*80/49 (3.2653) + 80*1.5*100/81 (148.1481)
        self.assertAlmostEqual(baseline_model.pandolf_w(70, 10, 10/9, 0), 256.4134, places=3)


class Parsing(unittest.TestCase):
    def test_parse_number_explicit(self):
        self.assertEqual(data.parse_number(1, '172.3', 'Height'), (172.3, None))
        self.assertEqual(data.parse_number(2, 74.0, 'Weight'), (74.0, None))
        self.assertEqual(data.parse_number(1, '-', 'EE_WW15'), (None, 'source_missing_dash'))
        for bad in ('1,5', 'abc', '', '12 kg'):
            with self.assertRaises(ValueError):
                data.parse_number(1, bad, 'x')
        with self.assertRaises(ValueError):
            data.parse_number(0, '', 'x')  # empty cell never silently becomes zero


REFS = [{'id': 'a0', 'person': 'A', 'load_kg': 0., 'reference_w': 200., 'body_mass_kg': 50., 'sex': 'F'},
        {'id': 'a5', 'person': 'A', 'load_kg': 5., 'reference_w': 220., 'body_mass_kg': 50., 'sex': 'F'},
        {'id': 'b0', 'person': 'B', 'load_kg': 0., 'reference_w': 300., 'body_mass_kg': 100., 'sex': 'M'}]
PREDS = {'a0': {'mean_w': 210., 'lower_w': 200., 'upper_w': 220.},
         'a5': {'mean_w': 220., 'lower_w': 215., 'upper_w': 225.},
         'b0': {'mean_w': 280., 'lower_w': 270., 'upper_w': 290.}}


class Scoring(unittest.TestCase):
    def test_person_balanced_hand_example(self):
        r = score(REFS, PREDS)
        self.assertTrue(r['execution_valid'])
        self.assertAlmostEqual(r['primary_loss'], 12.5)  # (mean(10,0)+20)/2, not 30/3
        self.assertEqual(r['per_person_mae_w'], {'A': 5.0, 'B': 20.0})
        self.assertAlmostEqual(r['secondary']['bias_w'], -7.5)        # (5 + -20)/2
        self.assertAlmostEqual(r['secondary']['mae_w_per_kg'], 0.15)  # A (0.2+0)/2, B 0.2
        self.assertAlmostEqual(r['secondary']['load_mae_w']['0.0'], 15.0)
        self.assertAlmostEqual(r['secondary']['load_mae_w']['5.0'], 0.0)
        self.assertAlmostEqual(r['secondary']['worst_load_mae_w'], 15.0)
        self.assertAlmostEqual(r['uncertainty']['interval_score_90'], 117.5)  # A (20+10)/2, B 20+20*10
        self.assertAlmostEqual(r['uncertainty']['coverage_90'], 0.5)

    def test_ordering_concordance(self):
        self.assertAlmostEqual(ordering_concordance([(1, 1), (2, 2), (2, 3)]), 2.5/3)
        self.assertIsNone(ordering_concordance([(1, 5), (2, 5)]))  # tied measured costs: not forced

    def test_invalid_or_missing_outputs_never_score(self):
        mutations = [lambda p: p.pop('b0'),
                     lambda p: p.update(extra={'mean_w': 1., 'lower_w': 0., 'upper_w': 2.}),
                     lambda p: p['a0'].update(mean_w=math.nan),
                     lambda p: p['a0'].update(upper_w=math.inf),
                     lambda p: p['a0'].update(lower_w=215.),
                     lambda p: p['a0'].update(mean_w='210'),
                     lambda p: p['a0'].update(state=None),
                     lambda p: p['a0'].pop('upper_w')]
        for i, mutate in enumerate(mutations):
            preds = json.loads(json.dumps(PREDS))
            mutate(preds)
            r = score(REFS, preds)
            with self.subTest(i=i):
                self.assertFalse(r['execution_valid'])
                self.assertIsNone(r['primary_loss'])
                self.assertTrue(r['failure_reason'])


@unittest.skipUnless(HAVE_DATA, 'EVOBREATH originals absent; run scripts/prepare_load.py --fetch')
class Dataset(unittest.TestCase):
    def test_all_48_people_accounted_with_exclusions(self):
        root = prepared()
        audit = json.loads((root/'private'/'audit.json').read_text())
        records = data.load_records(root)
        self.assertEqual(audit['source_people'], 48)
        self.assertEqual(audit['excluded_people'], [])
        self.assertEqual(len({r['person'] for r in records}), 48)
        self.assertEqual((len(records), audit['possible_condition_observations']), (189, 192))
        self.assertEqual(sorted((e['person'], e['cell'], e['original']) for e in audit['excluded_conditions']),
                         [('BE0002', 'N20', '-'), ('BE0058', 'N38', '-'), ('BE0063', 'N44', '-')])
        first = next(r for r in records if r['id'] == 'BE0026-L00')  # DATA row 2: K2 = 4978 kcal/day
        self.assertEqual((first['source_cell'], first['source_kcal_per_day']), ('K2', 4978.0))
        self.assertAlmostEqual(first['gross_w'], 241.0642593, places=6)  # 20,827,952/86,400
        self.assertEqual((first['stature_cm'], first['body_mass_kg']), (172.3, 66.3))  # stored as text
        self.assertEqual(first['target_convention'], 'gross')
        self.assertAlmostEqual(first['net_w_secondary'], 149.6845370, places=6)  # 3091*4184/86400

    def test_person_split_grouping(self):
        root = prepared()
        splits = data.load_splits(root)
        records = data.load_records(root)
        self.assertEqual({k: len(v) for k, v in splits.items()}, {'train': 28, 'development': 10, 'test': 10})
        every = [p for v in splits.values() for p in v]
        self.assertEqual(len(every), len(set(every)))
        self.assertEqual(data.make_splits(sorted(every), {r['person']: r['sex'] for r in records}), splits)
        with tempfile.TemporaryDirectory() as tmp:
            for role in ('development', 'test'):
                m = tasks.write_manifest(Path(tmp)/f'{role}.json', records, splits[role], tier='primary',
                                         variant='person', role=role, salt='s')
                self.assertEqual({e['source_person'] for e in m['records']}, set(splits[role]))
                # every condition of a person stays in that person's split
                self.assertEqual(len(m['records']), sum(r['person'] in splits[role] for r in records))

    def test_hidden_unloaded_targets_cannot_enter_primary_inference(self):
        root = prepared()
        records = data.load_records(root)
        people = data.load_splits(root)['development']
        before = [req for _, req in tasks.scored_records(records, people, 'primary', 'person')]
        poisoned = [dict(r, gross_w=-999.0) if r['load_kg'] == 0 else r for r in records]
        after = [req for _, req in tasks.scored_records(poisoned, people, 'primary', 'person')]
        self.assertEqual(before, after)
        for req in before:
            self.assertEqual(set(req), set(tasks.REQUEST_FIELDS))
            self.assertIsNone(req['calibration'])
            self.assertNotIn('gross_w', json.dumps(req))
            self.assertNotIn('rmr', json.dumps(req).lower())
        cal = tasks.scored_records(records, people, 'calibration', 'person')
        unloaded = {r['person']: r['gross_w'] for r in records if r['load_kg'] == 0}
        self.assertTrue(cal)
        for r, req in cal:
            self.assertNotEqual(r['load_kg'], 0)  # revealed condition is not scored
            self.assertEqual(req['calibration'], {'load_kg': 0.0, 'gross_w': unloaded[r['person']]})

    def test_heldout15_variant_is_separate(self):
        root = prepared()
        records, splits = data.load_records(root), data.load_splits(root)
        rows = tasks.fit_rows(records, splits['train'], 'heldout15', 's')
        self.assertTrue(rows and all(r['load_kg'] != 15 for r in rows))
        scored = tasks.scored_records(records, splits['test'], 'primary', 'heldout15')
        self.assertTrue(scored and all(r['load_kg'] == 15 for r, _ in scored))

    def test_all_baselines_fit_and_predict_valid(self):
        from research_env.tasks.load.baselines import train_rows
        root = prepared()
        records, splits = data.load_records(root), data.load_splits(root)
        rows = tasks.fit_rows(records, splits['train'], 'person', 's')
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            tasks.write_manifest(tmp/'dev.json', records, splits['development'], tier='primary',
                                 variant='person', role='dev', salt='s')
            for kind in baseline_model.KINDS:
                train_rows(rows, tmp/kind, 0, kind, 'primary')
                r, _ = replay_in_process(tmp/kind, tmp/'dev.json', baseline_model.predict)
                self.assertTrue(r['execution_valid'], kind)
                self.assertTrue(0 < r['primary_loss'] < 200, kind)

    def test_rl_pool_workspace_and_replay(self):
        from research_env.tasks.load.rl_pool import FAMILY
        root = prepared()
        pool = FAMILY.build_pool(root, seed=0)
        self.assertEqual(pool, FAMILY.build_pool(root, seed=0))  # deterministic
        splits = data.load_splits(root)
        for spec in pool:
            sets = [set(spec.fit_people), set(spec.feedback_people), set(spec.reward_people)]
            self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
            if spec.outer_split == 'agent_training':
                self.assertLessEqual(set().union(*sets), set(splits['train']))
                self.assertEqual([len(s) for s in sets], [18, 5, 5])
            elif spec.outer_split == 'agent_development':
                self.assertEqual(sets[2], set(splits['development']))
            else:
                self.assertEqual(spec.outer_split, 'agent_final_evaluation')
                self.assertEqual(sets[2], set(splits['test']))
            if spec.outer_split != 'agent_final_evaluation':
                self.assertFalse(set().union(*sets) & set(splits['test']))
        self.assertEqual(len({s.task_id for s in pool}), len(pool))
        spec = next(s for s in pool if s.outer_split == 'agent_training' and s.options['tier'] == 'primary')
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            workspace = tmp/'workspace'
            FAMILY.stage_workspace(spec, root, workspace)
            text = ''.join(p.read_text() for p in workspace.rglob('*')
                           if p.is_file() and p.suffix in ('.json', '.md', '.py'))
            for forbidden in ('BE00', 'rmr_w', 'reference_w', 'net_w', 'fat_mass'):
                self.assertNotIn(forbidden, text)
            for p in spec.reward_people:
                self.assertNotIn(tasks.pseudonym(spec.task_id, p), text)
            manifest = FAMILY.private_manifest(spec, root, 'reward', tmp/'controller')
            self.assertFalse(manifest.resolve().is_relative_to(workspace.resolve()))
            in_process, _ = replay_in_process(workspace/'artifact', manifest, baseline_model.predict)
            # Starter artifact is ridge; the reward baseline is the dev-selected Pandolf equation for primary.
            from research_env.tasks.load import rl_pool
            self.assertEqual(rl_pool.REWARD_BASELINES[('primary', spec.options['variant'])], 'pandolf')
            rl_pool.train_rows(rl_pool.fit_rows(rl_pool.load_records(root), spec.fit_people, spec.options['variant'],
                                                spec.task_id), tmp/'pandolf', 0, 'pandolf', 'primary')
            pandolf, _ = replay_in_process(tmp/'pandolf', manifest, baseline_model.predict)
            self.assertAlmostEqual(pandolf['primary_loss'], FAMILY.baseline_loss(spec, root), places=9)
            self.assertNotAlmostEqual(in_process['primary_loss'], pandolf['primary_loss'], places=3)
            if sys.platform == 'darwin' and shutil.which('sandbox-exec'):
                result = FAMILY.score(workspace/'artifact', manifest, backend='macos-development', image=None,
                                      wall_s=300, profile='fast')
                self.assertTrue(result['execution_valid'], result['failure_reason'])
                self.assertAlmostEqual(result['primary_loss'], in_process['primary_loss'], places=9)
                broken = tmp/'broken'
                shutil.copytree(workspace/'artifact', broken)
                (broken/'inference.py').write_text('def predict(artifact_dir, request):\n    return {"mean_w": 1.0}\n')
                bad = FAMILY.score(broken, manifest, backend='macos-development', image=None, wall_s=300, profile='fast')
                self.assertFalse(bad['execution_valid'])
                self.assertIsNone(bad['primary_loss'])


if __name__ == '__main__':
    unittest.main()
