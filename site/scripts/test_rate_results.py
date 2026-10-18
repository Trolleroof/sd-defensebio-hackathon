import copy
import json
import unittest
from rate_results import SITE, reanalyze


class Ratings(unittest.TestCase):
    def setUp(self):
        self.source = json.loads((SITE / 'lib/results.json').read_text())

    def test_preserves_evidence_and_is_reproducible(self):
        original = copy.deepcopy(self.source)
        result, audit = reanalyze(self.source)
        self.assertEqual(original, self.source)
        self.assertEqual(result['raw_scores'], original['raw_scores'])
        for before, after in zip(original['models'], result['models']):
            self.assertEqual(before['suite_elo'], after['suite_elo'])
            self.assertEqual(before['criteria'], after['criteria'])
            for task in before['tasks']:
                self.assertEqual(before['tasks'][task]['mean_loss'], after['tasks'][task]['mean_loss'])
        reversed_source = copy.deepcopy(self.source)
        reversed_source['raw_scores'].reverse()
        reversed_result, reversed_audit = reanalyze(reversed_source)
        self.assertEqual(result["models"], reversed_result["models"])
        self.assertEqual(audit, reversed_audit)
        self.assertEqual((result, audit), reanalyze(result))
        self.assertEqual(len(audit['overall']['matches']), 252)
        self.assertAlmostEqual(sum(m['overall_elo'] for m in result['models']), 8000)
        self.assertTrue(all(isinstance(m['overall_elo'], float) for m in result['models']))
        for model in result['models']:
            if model['valid'] == 0:
                self.assertLess(model['overall_elo'], 1000)
                self.assertEqual(model['overall_matches']['wins'], 0)
        for board in [audit['overall'], *audit['tasks'].values()]:
            self.assertTrue(all(isinstance(row['rank'], int) for row in board['entries']))
            for match in board['matches']:
                if match['outcome'] is None:
                    self.assertEqual(match['delta'], 0)

    def test_missing_and_duplicate_attempts_rejected(self):
        for rows in [self.source['raw_scores'][:-1], self.source['raw_scores'] + self.source['raw_scores'][:1]]:
            with self.assertRaises(ValueError):
                reanalyze({**self.source, 'raw_scores': rows})

    def test_better_losses_and_validity_raise_rating(self):
        for index, row in enumerate(self.source['raw_scores']):
            if row['model'].startswith('GPT-6 Astra'):
                self.source['raw_scores'][index] = {**row, 'execution_valid': True, 'primary_loss': -1}
        result, _ = reanalyze(self.source)
        by_name = {m['model']: m for m in result['models']}
        self.assertGreater(by_name['GPT-6 Astra']['overall_elo'], max(m['overall_elo'] for m in result['models'] if m['model'] != 'GPT-6 Astra'))


if __name__ == '__main__':
    unittest.main()
