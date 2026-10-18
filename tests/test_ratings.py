import unittest
from benchmark.ratings import leaderboard, physbench_ratings


def run(case,name,value,valid=True):
    return {'case':case,'competitor':name,'final':{'valid':valid,'code_execution_verified':valid,
            'mae_bpm':value,'interval_score_90':value}}


class RatingTests(unittest.TestCase):
    def test_better_outcomes_win_and_points_are_conserved(self):
        board=leaderboard([run(i,n,v) for i in range(3) for n,v in [('base',5),('tuned',1)]])
        self.assertEqual(board['entries'][0]['competitor'],'tuned')
        self.assertEqual(board['entries'][0]['wins'],3)
        self.assertAlmostEqual(sum(r['elo'] for r in board['entries']),2000)

    def test_model_name_never_grants_a_win(self):
        board=leaderboard([run(0,'base',0),run(0,'our_finetuned_model',5)])
        self.assertEqual(board['entries'][0]['competitor'],'base')

    def test_equal_predictions_draw(self):
        board=leaderboard([run(0,'a',1),run(0,'b',1)])
        self.assertEqual([r['elo'] for r in board['entries']],[1000,1000])
        self.assertEqual([r['rank'] for r in board['entries']],[1,1])

    def test_failure_loses_but_two_failures_do_not_earn_draws(self):
        board=leaderboard([run(0,'a',None,False),run(0,'b',1),run(0,'c',None,False)])
        self.assertEqual(board['entries'][0]['competitor'],'b')
        self.assertFalse(next(r for r in board['entries'] if r['competitor']=='a')['eligible'])
        self.assertEqual(sum(m['outcome']=='skipped_both_invalid' for m in board['matches'] if 'outcome' in m),1)

    def test_missing_duplicates_and_nonfinite_results_rejected(self):
        for runs in ([run(0,'a',1),run(1,'b',1)],
                     [run(0,'a',1),run(0,'a',2),run(0,'b',1)],
                     [run(0,'a',float('nan')),run(0,'b',1)]):
            with self.assertRaises(ValueError):
                leaderboard(runs)

    def test_reproducible_from_input_regardless_of_input_order(self):
        runs=[run(i,n,v) for i in range(3) for n,v in [('a',1),('b',2),('c',3)]]
        self.assertEqual(leaderboard(runs),leaderboard(list(reversed(runs))))


class PhysBenchRatingsTests(unittest.TestCase):
    def fixture(self):
        meta={'models':[{'name':'base'},{'name':'tuned'}], 'subjects':['s1','s2'],
              'repeats':1,'split':'dev','run_id':'test'}
        results=[{'model':m,'subject':s,'repeat':0,'submitted':True,'valid_execution':True,
                  'agent_mae':error,'criteria':{'integrity':True}}
                 for s in meta['subjects'] for m,error in [('base',5),('tuned',1)]]
        return results,meta

    def test_existing_integrity_gate_applies_without_anchor_margin(self):
        results,meta=self.fixture()
        results[1]['criteria']['integrity']=False
        board=physbench_ratings(results,meta)
        self.assertEqual(sum(row['wins'] for row in board['entries']),2)
        self.assertTrue(all(row['wins']==1 for row in board['entries']))

    def test_infrastructure_failure_excludes_whole_matched_case(self):
        results,meta=self.fixture()
        results[1]['infra_error']='provider unavailable'
        board=physbench_ratings(results,meta)
        self.assertEqual(len(board['excluded_cases']),1)
        self.assertEqual(board['entries'][0]['games'],1)
        self.assertEqual(board['scope']['split'],'dev')

    def test_incomplete_run_is_not_a_leaderboard(self):
        results,meta=self.fixture()
        with self.assertRaises(ValueError):
            physbench_ratings(results[:-1],meta)


if __name__=='__main__':
    unittest.main()
