import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from research_env.core.common import read,write
from scripts.verify_release import verify_release
from scripts.evaluate_checkpoint import validate_config,evaluate


def config():
    c=read('configs/checkpoint.example.json');c.update(competitor='fixture-finetuned',model='fixture-v1',slug='fixture-v1')
    c['checkpoint']={k:('a'*64 if 'sha256' in k else 'fixture-recorded') for k in c['checkpoint']}
    return c

class Release(unittest.TestCase):
 def test_snapshot_matches_and_changed_code_rejected(self):
    self.assertTrue(verify_release()['code_verified'])
    with patch('scripts.verify_release.code_hashes',return_value={}):
        with self.assertRaisesRegex(ValueError,'source'):verify_release()
 def test_changed_data_rejected(self):
    with tempfile.TemporaryDirectory() as d:
        with self.assertRaisesRegex(ValueError,'data differ'):verify_release(data_root=d)
 def test_placeholders_missing_checkpoint_and_ignored_parameters_rejected(self):
    with self.assertRaises(ValueError):validate_config(read('configs/checkpoint.example.json'))
    self.assertEqual(validate_config(config())['model'],'fixture-v1')
    c=config();c['decoding']['temperature']=.2
    with self.assertRaisesRegex(ValueError,'omit'):validate_config(c)
    c=config();c['checkpoint']['manifest_sha256']=''
    with self.assertRaisesRegex(ValueError,'SHA-256'):validate_config(c)
 def test_exactly_nine_episodes_and_resume_does_not_repeat_them(self):
    with tempfile.TemporaryDirectory() as d:
        root=Path(d);write(root/'suite.json',{'sha256':'fixture-suite'});write(root/'config.json',config())
        paths=['leaderboard/sweeps/compact-eight/judge.json'];seen=[]
        def episode(suite,task,c,path,p,key,spend,seed):
            seen.append((task,seed));write(p/'receipt.json',{'fixture':True})
        def review(p,*args):write(p/'episode-judge/report.json',{'fixture':True})
        with patch('scripts.evaluate_checkpoint.verify_release',return_value={'mode':'reference'}),patch('research_env.compact.call'),patch('research_env.compact.episode',side_effect=episode),patch('research_env.compact.review',side_effect=review),patch('research_env.compact.board',return_value={}),patch('research_env.compact.render',return_value='fixture'):
            for _ in range(2):
                result=evaluate(root/'suite.json',[root/'config.json'],paths[0],root/'output','SECRET','JUDGE_SECRET',1)
        self.assertEqual(set(seen),{(t,s) for t in ('load','heat','pulse') for s in (0,1,2)})
        self.assertEqual(len(seen),9);self.assertEqual(result['completed_episodes'],9);self.assertEqual(result['judged_episodes'],9)
        self.assertEqual(result['spend']['accounted_usd'],0);self.assertEqual(result['status'],'complete')

if __name__=='__main__':unittest.main()
