"""Publish validated compact judge packets and verdicts, never raw API transcripts."""
import argparse
import json
from pathlib import Path
import sys

SITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SITE.parent / 'benchmarks/compact-coding-v1'))
from research_env.compact import request_body
from research_env.core.common import digest, file_hash
from research_env.leaderboard import verify

CRITERIA = ('evidence', 'experiment', 'feedback', 'repro', 'selection')
EVENT_KEYS = {
    'experiment': {'id', 'kind', 'decision'},
    'development_result': {'id', 'kind', 'execution_valid', 'primary_loss', 'failure_reason'},
    'final_selection': {'id', 'kind', 'reason', 'evidence_ids'},
    'failed_before_experiment': {'id', 'kind'},
}


def validate_packet(packet, receipt):
    if set(packet) != {'rubric', 'events', 'trajectory_sha256'} or packet['rubric'] != 'compact-process-v1':
        raise ValueError('Unexpected compact packet schema')
    if packet['trajectory_sha256'] != receipt['trajectory_sha256']:
        raise ValueError('Packet trajectory mismatch')
    ids = set()
    for event in packet['events']:
        if event.get('kind') not in EVENT_KEYS or set(event) != EVENT_KEYS[event['kind']]:
            raise ValueError('Unexpected event fields; refuse to publish')
        if event['id'] in ids:
            raise ValueError('Duplicate event ID')
        ids.add(event['id'])
        if event['kind'] == 'experiment' and set(event['decision']) - {'hypothesis', 'code_diff'}:
            raise ValueError('Unexpected decision content')
    return ids


def validate_report(report, packet, judge_hash):
    if set(report) != {'packet_sha256', 'judge_configuration_sha256', 'verdict'}:
        raise ValueError('Unexpected report fields')
    if report['packet_sha256'] != digest(packet) or report['judge_configuration_sha256'] != judge_hash:
        raise ValueError('Judge report hash mismatch')
    if set(report['verdict']) != {'criteria'}:
        raise ValueError('Unexpected verdict fields')
    rows = report['verdict']['criteria']
    if len(rows) != 5 or {r['id'] for r in rows} != set(CRITERIA):
        raise ValueError('Expected five distinct criteria')
    ids = {event['id'] for event in packet['events']}
    for row in rows:
        if set(row) != {'id', 'score', 'reason', 'evidence_ids'}:
            raise ValueError('Unexpected criterion fields')
        score = row['score']
        if score is not None and (type(score) is not int or not 0 <= score <= 4):
            raise ValueError('Invalid criterion score')
        if not isinstance(row['reason'], str) or not row['reason'].strip():
            raise ValueError('Missing criterion rationale')
        if not isinstance(row['evidence_ids'], list) or not set(row['evidence_ids']) <= ids or (score is not None and not row['evidence_ids']):
            raise ValueError('Invalid evidence citation')


def export(root, judge):
    source = json.loads((SITE / 'lib/results.json').read_text())
    names = {f"{m['model']} {m['effort'].capitalize()}": m for m in source['models']}
    episodes, seen = [], set()
    for path in sorted(root.glob('runs/*/*/seed-*/receipt.json')):
        receipt = verify(json.loads(path.read_text()))
        run = verify(receipt['run'])
        model = names[run['competitor']]
        key = (model['model'], run['task'], run['seed'])
        if key in seen or run['suite_sha256'] != source['suite_sha256']:
            raise ValueError('Duplicate episode or different suite')
        seen.add(key)
        folder = path.parent
        for name, field in [('trajectory.jsonl', 'trajectory_sha256'), ('transcript.jsonl', 'transcript_sha256'), ('run-status.json', 'run_status_sha256')]:
            if file_hash(folder / name) != receipt[field]:
                raise ValueError('Sealed episode evidence changed')
        raw = next(r for r in source['raw_scores'] if (r['model'], r['task'], r['seed']) == (run['competitor'], run['task'], run['seed']))
        if any(raw[k] != receipt['score'][k] for k in ('execution_valid', 'primary_loss')):
            raise ValueError('Public outcome differs from receipt')
        packet_path = folder / 'episode-judge/packet.json'
        report_path = folder / 'episode-judge/report.json'
        packet = json.loads(packet_path.read_text()) if packet_path.exists() else None
        report = json.loads(report_path.read_text()) if report_path.exists() else None
        if packet is not None:
            validate_packet(packet, receipt)
        if report is not None:
            if packet is None:
                raise ValueError('Report without evidence')
            validate_report(report, packet, digest(judge))
        episodes.append({
            'id': '-'.join(path.parent.relative_to(root / 'runs').parts),
            'model': model['model'], 'effort': model['effort'], 'task': run['task'], 'seed': run['seed'],
            'execution_valid': receipt['score']['execution_valid'],
            'status': 'reviewed' if report else 'missing',
            'packet': packet, 'report': report,
            'provenance': {'receipt_sha256': receipt['sha256'], 'trajectory_sha256': receipt['trajectory_sha256'],
                           'report_sha256': file_hash(report_path) if report else None},
        })
    if len(episodes) != source['episodes'] or sum(e['status'] == 'reviewed' for e in episodes) != source['reviews']:
        raise ValueError('Review coverage differs from public results')
    for model in source['models']:
        for criterion in CRITERIA:
            values = [row['score'] for e in episodes if e['model'] == model['model'] and e['report']
                      for row in e['report']['verdict']['criteria'] if row['id'] == criterion and row['score'] is not None]
            published = model['criteria'][criterion]
            if len(values) != published['count'] or (sum(values) / len(values) if values else None) != published['mean']:
                raise ValueError('Verdicts do not reproduce published judge means')
    judge_request = request_body(judge, episodes[0]['packet'], judge=True)
    result = {'protocol': source['protocol'], 'suite_sha256': source['suite_sha256'], 'rubric': 'compact-process-v1',
              'judge': {'model': judge['model'], 'effort': judge['decoding']['reasoning']['effort'], 'configuration_sha256': digest(judge),
                        'instructions': judge_request['messages'][0]['content'], 'grade_tool': judge_request['tools'][0]},
              'episodes': len(episodes), 'reviewed': source['reviews'], 'missing': len(episodes) - source['reviews'],
              'scope': 'Exact compressed public decision/development packets and saved judge verdicts. Hidden test scores, API transcripts and private model reasoning are not included.',
              'records': episodes}
    encoded = json.dumps(result, indent=2, allow_nan=False) + '\n'
    # Fail closed if an unexpected secret or local account path appears in public text.
    for marker in ('sk-or-', 'rpa_', 'Bearer ', '/Users/', 'api_key', 'Authorization'):
        if marker in encoded:
            raise ValueError('Unexpected sensitive content in export')
    for name in ('lib/judge-traces.json', 'public/results/judge-traces.json'):
        (SITE / name).write_text(encoded)
    print(f"Exported {len(episodes)} episodes: {result['reviewed']} reviews, {result['missing']} missing; {len(encoded.encode())} bytes")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_root', type=Path)
    parser.add_argument('judge_config', type=Path)
    args = parser.parse_args()
    export(args.run_root, json.loads(args.judge_config.read_text()))
