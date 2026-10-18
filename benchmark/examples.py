"""Run: python3 -m benchmark.examples /tmp/physiology-demo"""
import json
import sys
from pathlib import Path
from .workspace import Workspace, read


def episode():
    sessions = []
    for name, role, values in [('calibration', 'support', [120, 110, 100, 90]),
                                ('long-bout', 'acquisition', [130, 120, 110, 100]),
                                ('feedback', 'development', [125, 115, 105, 95]),
                                ('held-out', 'test', [128, 118, 108, 98])]:
        sessions.append({'id': name, 'role': role, 'participant_id': 'synthetic-person-1',
                         'cost': 1, 'description': 'Synthetic rest after recorded activity',
                         'rows': [{'time_s': i*30, 'hr_bpm': v} for i, v in enumerate(values)]})
    return {'schema_version': 1, 'id': 'recovery-example-v0', 'origin': 'synthetic',
            'budgets': {'acquisition': 1, 'evaluations': 2}, 'sessions': sessions}


def decision(action):
    return {'evidence': 'support.json calibration trajectory',
            'hypothesis': 'The observed decline may generalize to another recovery session.',
            'action': action, 'expected_result': 'Development MAE decreases if the decline generalizes.'}


def predict(request, decline):
    start = request['history'][-1]['hr_bpm']
    return [{'time_s': t, 'mean': start-decline*t/30,
             'lower': start-decline*t/30-5, 'upper': start-decline*t/30+5}
            for t in request['forecast_times_s']]


def run(root):
    root = Path(root)
    w = Workspace.create(episode(), root/'private', root/'agent')
    requests = read(root/'agent'/'requests.json')
    dev = next(r for r in requests if r['role'] == 'development')
    test = next(r for r in requests if r['role'] == 'test')
    first = w.evaluate({dev['id']: predict(dev, 0)}, decision('Evaluate persistence'))
    w.acquire('long-bout', decision('Acquire longer-bout recording'))
    w = Workspace(root/'private')
    revised = w.evaluate({dev['id']: predict(dev, 10)}, decision('Evaluate fitted decline'))
    receipt = w.submit('# Scripted demonstration candidate; not an LLM run\n',
                       {test['id']: predict(test, 10)}, decision('Select revised candidate'))
    return {'origin': 'synthetic', 'scripted_example': True, 'initial': first,
            'revised': revised, 'receipt': receipt,
            'private_final_score': read(root/'private'/'final-score.json')}


if __name__ == '__main__':
    print(json.dumps(run(sys.argv[1]), indent=2))
