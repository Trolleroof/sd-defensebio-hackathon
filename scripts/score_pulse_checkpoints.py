"""Private controller replay; output must never be returned to participant."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_env.core.session import Session
from research_env.core.evaluation import evaluate
from research_env.core.common import write

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('controller')
parser.add_argument('test_manifest')
parser.add_argument('private_report')
parser.add_argument('--backend', choices=('docker', 'macos-development'), default='docker')
parser.add_argument('--image')
args = parser.parse_args()
session = Session(args.controller)
results = []
for event in session.events():
    if event['kind'] not in ('checkpoint', 'final_selection'):
        continue
    candidate = event.get('candidate_id')
    if candidate is None:
        result = {'execution_valid': False, 'primary_loss': None, 'skill': None, 'failure_reason': 'no_declared_incumbent'}
    else:
        result, _ = evaluate(session.root/'candidates'/candidate, args.test_manifest, backend=args.backend, image=args.image)
    results.append({'event_id': event['event_id'], 'minute': event.get('minute'), 'candidate_id': candidate, 'private_score': result})
write(args.private_report, {'private_only': True, 'checkpoints': results,
                           'selection_policy': 'Use declared final selection, never private checkpoint winner'})
