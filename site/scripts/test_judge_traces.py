import copy
import json
import unittest
from export_judge_traces import SITE, validate_packet, validate_report


class JudgeTraces(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((SITE / 'lib/judge-traces.json').read_text())
        self.record = next(r for r in self.data['records'] if r['report'])

    def test_coverage_identity_and_saved_reports(self):
        self.assertEqual(len(self.data['records']), 72)
        self.assertEqual(len({r['id'] for r in self.data['records']}), 72)
        self.assertEqual(sum(r['status'] == 'reviewed' for r in self.data['records']), 65)
        self.assertEqual(sum(r['status'] == 'missing' for r in self.data['records']), 7)
        self.assertEqual((SITE / 'lib/judge-traces.json').read_bytes(), (SITE / 'public/results/judge-traces.json').read_bytes())
        for record in self.data['records']:
            if record['packet']:
                validate_packet(record['packet'], record['provenance'])
            if record['report']:
                validate_report(record['report'], record['packet'], self.data['judge']['configuration_sha256'])
            else:
                self.assertEqual(record['status'], 'missing')

    def test_changed_packet_or_configuration_rejected(self):
        changed = copy.deepcopy(self.record['packet'])
        changed['events'][0]['id'] = 'changed'
        for packet, config in [(changed, self.data['judge']['configuration_sha256']), (self.record['packet'], 'wrong')]:
            with self.assertRaises(ValueError):
                validate_report(self.record['report'], packet, config)

    def test_unknown_fields_and_false_citations_rejected(self):
        for key in ['hidden_test_score', 'api_key']:
            packet = {**self.record['packet'], key: 'unexpected'}
            with self.assertRaises(ValueError):
                validate_packet(packet, self.record['provenance'])
        report = copy.deepcopy(self.record['report'])
        report['verdict']['criteria'][0]['evidence_ids'] = ['invented-event']
        with self.assertRaises(ValueError):
            validate_report(report, self.record['packet'], self.data['judge']['configuration_sha256'])


if __name__ == '__main__':
    unittest.main()
