import tempfile
import unittest
from pathlib import Path
import numpy as np
from benchmark.dalia_synthetic import CHANNELS, KEYS, fit, generate, load_recording, summarize


class DaliaSyntheticTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        rng = np.random.default_rng(4)
        self.n = 450
        for subject in ('S1', 'S4'):
            activity = np.repeat(np.arange(9), 50).astype(float)
            noise = rng.normal(size=self.n)
            r = {k: 10+j+activity*.2+np.cumsum(noise*.01)+rng.normal(0, .03, self.n)
                 for j, k in enumerate(CHANNELS)}
            r.update(t=8+2*np.arange(self.n), activity=activity)
            r['eda_wrist'][3] = np.nan
            np.savez_compressed(self.root/f'{subject}.npz', **r)

    def test_fit_never_reads_unselected_or_reserved_subjects(self):
        (self.root/'S5.npz').write_bytes(b'inaccessible malformed final-test recording')
        model = fit(self.root, ['S1', 'S4'])
        self.assertEqual(model['fit_subjects'], ['S1', 'S4'])
        self.assertEqual(set(model['fit_files']), {'S1', 'S4'})
        with self.assertRaises(ValueError):fit(self.root, ['S1', 'S5'])
        with self.assertRaises(ValueError):fit(self.root, ['S2'])
        with self.assertRaises(ValueError):fit(self.root, [])

    def test_synthetic_contract_determinism_and_new_realization(self):
        model = fit(self.root, ['S1', 'S4'])
        a, meta = generate(model, 1); b, _ = generate(model, 1); c, _ = generate(model, 2)
        self.assertEqual(set(a), KEYS)
        self.assertEqual(meta['origin'], 'synthetic')
        self.assertEqual(meta['skin_temperature']['quantity'], 'wrist_skin_temperature')
        self.assertNotIn('imu_chip_conversion', meta['skin_temperature'])
        self.assertEqual(model['skin_temperature'], meta['skin_temperature'])
        self.assertLess(meta['latent_cap_fraction'], 1)
        for k in KEYS:
            self.assertEqual(a[k].shape, (self.n,))
            np.testing.assert_equal(a[k], b[k])
        self.assertFalse(np.array_equal(a['hr_ecg'], c['hr_ecg']))
        for k in CHANNELS:
            self.assertGreaterEqual(np.nanmin(a[k]), np.min(model['quantiles'][CHANNELS.index(k)]))
            self.assertLessEqual(np.nanmax(a[k]), np.max(model['quantiles'][CHANNELS.index(k)]))
        np.savez_compressed(self.root/'SYN0001.npz', **a)
        loaded = load_recording(self.root/'SYN0001.npz')
        stats = summarize([loaded])
        self.assertEqual(stats['samples'], self.n)
        self.assertTrue(np.isfinite(stats['channels']['hr_ecg']['lag1']))
        self.assertEqual(len(stats['channel_correlation']), len(CHANNELS))

    def test_bad_shapes_and_grid_fail(self):
        r = load_recording(self.root/'S1.npz'); r['t'][5] += 1
        np.savez_compressed(self.root/'bad.npz', **r)
        with self.assertRaises(ValueError):load_recording(self.root/'bad.npz')
