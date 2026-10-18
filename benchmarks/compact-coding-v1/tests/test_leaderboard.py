"""Independent unit, causality, matching and failure-policy checks for the suite."""
import copy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from research_env.core.common import digest, read, write
from research_env.tasks.heat.data import make_request, minute
from research_env.tasks.load.data import numeric, watts
from research_env.tasks.scoring import score, validate
from research_env.tasks.registry import identify
from research_env.leaderboard import POLICY, build, seal, verify
from research_env.judging.judge import build_packets


def pred(m,lo=None,hi=None):
    return {'mean_w':m,'lower_w':m if lo is None else lo,'upper_w':m if hi is None else hi,'state':None}


def load_ref(i,p,y,load=0):
    return {'id':i,'person':p,'session':'walking','request':{'mass_kg':50},'targets':[{'slice':load,'value':y}]}


class TaskContracts(unittest.TestCase):
    def test_independent_load_units(self):
        self.assertAlmostEqual(watts(86400),4184)
        self.assertAlmostEqual(watts(1000),48.425925925925924)
        self.assertEqual(numeric('66.3'),66.3)
        for bad in ('-',float('nan'),True):
            with self.assertRaises(ValueError):numeric(bad)

    def test_load_equal_people_and_intervals(self):
        refs=[load_ref('a','p',100),load_ref('b','p',100,5),load_ref('c','q',100)]
        result=score(refs,{'a':pred(100),'b':pred(104),'c':pred(120)},'load')
        self.assertEqual(result['primary_loss'],11)  # ((0+4)/2 +20)/2
        self.assertEqual(result['per_person_mae'],{'p':2.,'q':20.})
        r=score([load_ref('a','p',70)],{'a':pred(60,55,65)},'load')
        self.assertEqual(r['uncertainty']['interval_score_90'],110)
        wide=score([load_ref('a','p',70)],{'a':pred(60,-1000,1000)},'load')
        self.assertGreater(wide['uncertainty']['interval_score_90'],110)
        for values in ({},{'a':pred(float('nan'))},{'a':pred(5,6,7)}):
            r=score([load_ref('a','p',70)],values,'load')
            self.assertFalse(r['execution_valid']);self.assertIsNone(r['primary_loss'])

    def test_heat_equal_horizons_and_people(self):
        refs=[{'id':'a','person':'p','session':'cycling',
               'targets':[{'slice':h,'value':37} for h in (5,10,20)]},
              {'id':'b','person':'q','session':'cycling',
               'targets':[{'slice':h,'value':37} for h in (5,10,20)]}]
        def heat(values):
            return {'forecasts':[{'horizon_min':h,'mean_c':v,'lower_c':v-1,'upper_c':v+1} for h,v in zip((5,10,20),values)],'state':None}
        result=score(refs,{'a':heat((37,38,39)),'b':heat((40,40,40))},'heat')
        self.assertEqual(result['primary_loss'],2)  # ((0+1+2)/3 +3)/2
        self.assertEqual(result['secondary']['per_slice']['5']['mae'],1.5)
        self.assertIsNone(result['secondary']['hottest_underprediction_c'])
        bad=heat((37,38,39));bad['forecasts'][1]['horizon_min']=20
        with self.assertRaises(ValueError):validate(bad,'heat')

    def test_heat_causality_forbidden_fields(self):
        row={'time_min':10,'hr_bpm':120,'skin_c':35,'ambient_c':36,'humidity_pct':20,'rectal_c':37,'vendor_core_c':99}
        future={**row,'time_min':11}
        demo={'age_years':25,'mass_kg':70,'height_cm':170,'VO2peak':99}
        first=make_request([row,future],demo,10)
        future['hr_bpm']=1e9;future['skin_c']=-1e9
        row['rectal_c']=-100
        self.assertEqual(first,make_request([row,future],demo,10))
        self.assertNotIn('rectal_c',str(first));self.assertNotIn('VO2peak',str(first))
        self.assertEqual(minute('6-7'),7)

    def test_unknown_manifest_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'manifest.json';write(p,{'version':'fake'})
            with self.assertRaises(ValueError):identify(p)

    def test_exported_artifact_compiles_and_predicts(self):
        from research_env.tasks.baselines import train
        import importlib.util
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            records=[]
            for mass in (60,80):
                for load in (0,5,10,15):
                    records.append({'request':{'mass_kg':mass,'load_kg':load,'height_cm':175,'age_years':30},
                                    'targets':[{'slice':load,'value':mass*3+load*2}]})
            write(root/'train.json',{'role':'train','task':'load','records':records})
            train(root/'train.json',root/'artifact','ridge')
            code=(root/'artifact/inference.py').read_text()
            compile(code,'inference.py','exec')
            self.assertNotIn('research_env.core',code)
            spec=importlib.util.spec_from_file_location('artifact',root/'artifact/inference.py')
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            validate(module.predict(root/'artifact',records[0]['request']),'load')

    def test_tabular_splits_and_private_targets(self):
        from research_env.tasks.tabular import export_dataset, iter_windows
        records=[{'id':str(i),'person':str(i),'session':'x','request':{'x':i},'targets':[{'slice':0,'value':i}]} for i in range(12)]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'data'
            manifest=export_dataset(root,'heat',records,list(map(str,range(12))),{}, {})
            self.assertEqual(sum(map(len,manifest['splits'].values())),12)
            public=read(root/'public/development/manifest.json')
            self.assertTrue(all('targets' not in r for r in public['records']))
            self.assertFalse((root/'public/test').exists())
            with self.assertRaises(ValueError):list(iter_windows(root/'public/development/manifest.json',True))


def sample_receipts():
    suite=seal({'policy':POLICY})
    receipts=[]
    for model in ('a','b'):
        for seed in (0,1,2):
            for task in ('pulse','heat','load'):
                config={'competitor':model}
                meta=seal({'suite_sha256':suite['sha256'],'task':task,'seed':seed,'competitor':model,
                           'configuration':config,'configuration_sha256':digest(config)})
                result={'execution_valid':True,'primary_loss':1 if model=='a' else 2,
                        'uncertainty':{'interval_score_90':2 if model=='a' else 1}}
                receipts.append(seal({'run':meta,'score':result}))
    return suite,receipts


class Leaderboard(unittest.TestCase):
    def test_pending_judge_renders_without_configuration(self):
        from research_env.leaderboard import render
        from research_env.judging.judge import summarize
        suite,receipts=sample_receipts()
        board=build(suite,receipts,[{'competitor':'a','task':'pulse','seed':0,
                                   'configuration':None,'report':summarize([],[])}])
        self.assertIn('Judge: pending',render(board))

    def test_empty_and_matched_order_invariance(self):
        suite,receipts=sample_receipts()
        self.assertEqual(build(suite,[])['status'],'empty')
        one=build(suite,receipts)
        self.assertEqual(one,build(suite,list(reversed(receipts))))
        self.assertEqual(one['tasks']['pulse']['prediction']['entries'][0]['competitor'],'a')
        self.assertEqual(one['tasks']['pulse']['uncertainty']['entries'][0]['competitor'],'b')
        self.assertEqual(one['suite']['prediction']['entries'][0]['wins'],3)
        self.assertEqual(one['judge_status'],'pending')

    def test_failures_no_free_draws_or_zero_losses(self):
        suite,receipts=sample_receipts()
        for i,r in enumerate(receipts):
            if r['run']['competitor']=='b':
                receipts[i]=seal({**{k:v for k,v in r.items() if k!='sha256'},
                                  'score':{'execution_valid':False,'primary_loss':None,'failure_reason':'no submission'}})
        board=build(suite,receipts)
        b=board['suite']['prediction']['entries'][-1]
        self.assertIsNone(b['rank']);self.assertEqual(b['losses'],3)
        self.assertIsNone(board['tasks']['pulse']['raw'][1]['mean_loss_valid_runs'])

    def test_missing_duplicate_different_suite_rejected(self):
        suite,receipts=sample_receipts()
        for bad in (receipts[:-1],receipts+[receipts[0]],[r for r in receipts if r['run']['seed']!=2]):
            with self.assertRaises(ValueError):build(suite,bad)
        mutated=copy.deepcopy(receipts)
        mutated[0]['run']['suite_sha256']='old-results'
        mutated[0]['run']=seal({k:v for k,v in mutated[0]['run'].items() if k!='sha256'})
        mutated[0]=seal({k:v for k,v in mutated[0].items() if k!='sha256'})
        with self.assertRaises(ValueError):build(suite,mutated)

    def test_hash_tampering_and_configuration_drift(self):
        suite,receipts=sample_receipts()
        tampered=copy.deepcopy(receipts[0]);tampered['score']['primary_loss']=0
        with self.assertRaises(ValueError):verify(tampered)
        meta={k:v for k,v in receipts[0]['run'].items() if k!='sha256'}
        meta['configuration_sha256']='different-model'
        receipts[0]=seal({'run':seal(meta),'score':receipts[0]['score']})
        with self.assertRaises(ValueError):build(suite,receipts)

    def test_blind_tabular_person_scores(self):
        events=[{'event_id':'e00000','kind':'development_result',
                 'result':{'per_person_mae':{'SecretIdentity':1},'primary_loss':1}},
                {'event_id':'e00001','kind':'final_selection','reason':'submit','candidate_id':'code','evidence_ids':['e00000']}]
        self.assertNotIn('SecretIdentity',str(build_packets(events)))

    def test_provider_no_live_network_and_failed_call_budget(self):
        from research_env.run_model import ModelProvider
        config={'endpoint':'https://example.invalid/v1/chat/completions','model':'frozen-model','prompt':'task',
                'max_calls':1,'max_total_tokens':100000,'decoding':{'temperature':0,'max_tokens':100}}
        with tempfile.TemporaryDirectory() as tmp:
            provider=ModelProvider(config,'SECRET',tmp)
            with patch('urllib.request.urlopen',side_effect=OSError('offline')):
                with self.assertRaises(OSError):provider({},0)
            self.assertEqual(read(Path(tmp)/'provider-usage.json')['calls'],1)
            resumed=ModelProvider(config,'SECRET',tmp)
            with self.assertRaises(ValueError):resumed({},0)
            self.assertNotIn('SECRET',(Path(tmp)/'provider-call-0001.json').read_text())

    def test_frozen_suite_and_final_selection_receipt(self):
        from research_env.leaderboard import freeze, initialize, finalize, check_suite
        from research_env.core.agent import ResearchEnvironment
        with tempfile.TemporaryDirectory() as tmp, patch('research_env.leaderboard.resource_limits', return_value={'cpus':1,'memory_bytes':1024**3}):
            root=Path(tmp)
            for task in ('pulse','heat','load'):
                manifest={'version':'galaxy-pulse-causal-v1'} if task=='pulse' else {'task':task,'version':task+'-v1'}
                write(root/'data'/task/'dataset.json',{'task':task})
                for role in ('train','development','test'):
                    write(root/'data'/task/'private'/role/'manifest.json',manifest)
                    if role!='test':write(root/'data'/task/'public'/role/'manifest.json',manifest)
            image='sha256:'+'0'*64
            suite_path=root/'suite.json'
            suite=freeze(root/'data',image,suite_path)
            config={'competitor':'test','provider':'mock','model':'mock-v1','decoding':{'temperature':0},'prompt':'fixture'}
            write(root/'config.json',config)
            run=root/'run'
            initialize(suite_path,'load',root/'config.json',0,run)
            env=ResearchEnvironment(run/'private',run/'workspace',root/'data/load/private/development/manifest.json',image=image)
            with env.session.db:
                env.session.append({'kind':'candidate','candidate_id':'fixture'})
            env.session.select('fixture',['e00000'],'Fixture final selection',True)
            result={'execution_valid':True,'primary_loss':1,'uncertainty':{'interval_score_90':2}}
            with patch('research_env.core.evaluation.evaluate',return_value=(result,{})) as replay:
                receipt=finalize(suite_path,run)
            self.assertEqual(receipt['candidate_id'],'fixture')
            self.assertEqual(replay.call_args.args[1].resolve(), (root/'data/load/private/test/manifest.json').resolve())
            self.assertEqual(read(run/'judging.json')['llm_judge_status'],'pending')
            with self.assertRaises(ValueError):finalize(suite_path,run)
            write(root/'data/load/dataset.json',{'changed':True})
            with self.assertRaises(ValueError):check_suite(suite)
            env.session.db.close()


if __name__=='__main__':
    unittest.main()
