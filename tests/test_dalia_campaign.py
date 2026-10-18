import tempfile
import unittest
from pathlib import Path
import numpy as np
from benchmark.dalia_campaign import episode, ACTIVITIES, FEATURES
from benchmark.campaign import CampaignWorkspace
from benchmark.agent import AgentSession
from benchmark.workspace import read


class DaliaCampaignTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        from benchmark.dalia_synthetic import CHANNELS
        activity=np.repeat(ACTIVITIES,240);n=len(activity)
        r={k:np.linspace(60,90,n) if k=='hr_ecg' else np.ones(n) for k in CHANNELS}
        r.update(t=8+2*np.arange(n),activity=activity)
        np.savez_compressed(self.root/'SYN0001.npz',**r)

    def test_disjoint_source_windows_and_causal_history(self):
        ep=episode(self.root,'SYN0001',origin='synthetic')
        windows=ep['provenance']['windows']
        for a,b in zip(sorted(windows,key=lambda x:x['source_index_start'])[:-1],sorted(windows,key=lambda x:x['source_index_start'])[1:]):
            self.assertLessEqual(a['source_index_end_exclusive']+4,b['source_index_start'])
        w=CampaignWorkspace.create(ep,self.root/'private',self.root/'agent')
        ob=AgentSession(w).observe()
        self.assertEqual([s['id'] for s in ob['recordings']],['sitting-calibration'])
        self.assertEqual([r['id'] for r in ob['requests']],['sitting-feedback'])
        self.assertEqual(len(ob['requests'][0]['history']),30)
        self.assertEqual(len(ob['requests'][0]['forecast_times_s']),30)
        self.assertTrue(all(t>0 for t in ob['requests'][0]['forecast_times_s']))
        self.assertTrue(all(r['time_s']<=0 for r in ob['requests'][0]['history']))
        self.assertTrue(set(FEATURES)<=set(ob['requests'][0]['history'][0]))
        self.assertNotIn('future_covariates',ob['requests'][0])
        self.assertNotIn('source_index',str(ob))
        self.assertNotIn('source_subject',str(ob))
        self.assertNotIn('chip',str(ob).lower())

    def test_real_final_subject_is_rejected_before_reading(self):
        with self.assertRaisesRegex(ValueError,'development'):
            episode(self.root,'S5',origin='public')

    def test_insufficient_activity_coverage_is_explicit(self):
        from benchmark.dalia_synthetic import load_recording
        r=load_recording(self.root/'SYN0001.npz');r['activity'][:]=1
        np.savez_compressed(self.root/'SYN0001.npz',**r)
        with self.assertRaisesRegex(ValueError,'coverage'):
            episode(self.root,'SYN0001',origin='synthetic')
