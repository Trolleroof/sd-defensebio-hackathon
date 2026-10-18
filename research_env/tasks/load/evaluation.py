"""Trusted load replay. Labels stay on the controller; the worker sees only requests.

Mirrors research_env.core.evaluation (which is pulse-specific) and reuses the shared
worker backend ``research_env.core.worker.execute``.
"""
from itertools import groupby
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
from research_env.core.common import digest, invalid_score, read, tree_hashes
from research_env.core.worker import execute, worker_python
from research_env.tasks.load.scoring import score, validate_prediction


def _check_artifact(artifact_dir):
    artifact_dir = Path(artifact_dir)
    hashes = tree_hashes(artifact_dir)
    if re.fullmatch(r'[0-9a-f]{64}', artifact_dir.name) and digest(hashes) != artifact_dir.name:
        raise ValueError('Frozen artifact hash mismatch')
    if 'inference.py' not in hashes:
        raise ValueError('Artifact must include inference.py with predict(artifact_dir, request)')
    return hashes


def evaluate(artifact_dir, private_manifest, *, backend='docker', image=None, wall_s=7200,
             on_tick=None, profile='benchmark'):
    """One fresh worker per person (both profiles; the task is stateless and tiny)."""
    hashes = _check_artifact(artifact_dir)
    manifest = read(private_manifest)
    references = manifest['records']
    start = time.monotonic()
    predictions, resources = {}, []
    for _, group in groupby(references, key=lambda r: r['person']):
        group = list(group)
        with tempfile.TemporaryDirectory(prefix='load-replay-') as tmp:
            root = Path(tmp)
            shutil.copytree(artifact_dir, root/'artifact')
            shutil.copyfile(Path(__file__).with_name('inference_entry.py'), root/'entry.py')
            python = worker_python(backend)
            remaining = wall_s-(time.monotonic()-start)
            if remaining <= 0:
                return invalid_score('replay_wall_budget'), {}
            execution = execute(root, [python, '-I', 'entry.py'], timeout_s=remaining, backend=backend,
                                image=image, on_tick=on_tick, requests=(r['request'] for r in group))
            resources.append({k: v for k, v in execution.items() if k not in ('responses', 'stdout', 'stderr')})
            if not execution['execution_valid'] or len(execution['responses']) != len(group):
                return {**invalid_score(execution['failure_reason'] or 'incomplete_replay'),
                        'worker_stderr_tail': execution.get('stderr', '')[-2000:]}, {}
            try:
                for ref, prediction in zip(group, execution['responses']):
                    predictions[ref['id']] = validate_prediction(prediction)
            except (TypeError, ValueError) as exc:
                return invalid_score(str(exc)), {}
    result = score(references, predictions)
    result.update(resource_use={'wall_s': time.monotonic()-start, 'workers': resources, 'profile': profile},
                  artifact_hashes=hashes, scored_resource_isolation=backend == 'docker')
    return result, predictions


def replay_in_process(artifact_dir, private_manifest, predict):
    """Trusted builder profiling of *our own* JSON baselines only (no untrusted code)."""
    manifest = read(private_manifest)
    predictions = {}
    for ref in manifest['records']:
        predictions[ref['id']] = predict(artifact_dir, dict(ref['request']))
    return score(manifest['records'], predictions), predictions
