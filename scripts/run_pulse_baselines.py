"""Trusted builder profiling, never an untrusted/strict-budget participant score."""
import argparse
from pathlib import Path
import sys
import time
import resource
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_env.core.common import read, write
from research_env.tasks.pulse.baselines import KINDS, train, replay

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('dataset')
parser.add_argument('output')
args = parser.parse_args()
root, out = Path(args.dataset), Path(args.output)
out.mkdir(parents=True, exist_ok=False)
results = {}
for kind in KINDS:
    start = time.monotonic()
    artifact = out/'artifacts'/kind
    train(root/'public/train/manifest.json', None, artifact, 0, kind)
    fit_s = time.monotonic()-start
    # Development only. Private-test labels never choose the candidate.
    result, _ = replay(artifact, root/'private/development/manifest.json')
    result.update(training_wall_s=fit_s, execution_context='trusted_builder_only',
                  scored_resource_isolation=False, reference_status='provisional')
    result['resource_use']['builder_process_peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024)
    results[kind] = result
    write(out/f'{kind}.json', result)
    print(kind, result['primary_loss'], 'bpm', flush=True)
selection = min(results, key=lambda k: results[k]['primary_loss'])
# Freeze the choice before accessing private test results.
write(out/'selection.json', {'selected_on_development': selection, 'criterion': 'development primary loss only'})
test_results = {}
for kind in KINDS:
    result, _ = replay(out/'artifacts'/kind, root/'private/test/manifest.json')
    result.update(execution_context='trusted_builder_only', scored_resource_isolation=False, reference_status='provisional')
    test_results[kind] = result
    write(out/f'test-{kind}.json', result)
write(out/'report.json', {'baselines': results, 'private_test_baselines': test_results, 'selected_on_development': selection,
                         'live_agent': 'pending', 'live_judge': 'pending',
                         'reference_audit': 'provisional', 'strict_container_replay': 'pending'})
