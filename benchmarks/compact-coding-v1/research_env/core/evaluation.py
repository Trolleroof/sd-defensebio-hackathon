"""Trusted replay: no source recordings, labels or evaluator code in worker."""
from itertools import groupby
from pathlib import Path
import shutil
import tempfile
import sys
import time
import re
from research_env.core.common import digest, invalid_score, tree_hashes
from research_env.core.worker import execute
from research_env.tasks.registry import identify, iter_windows, score, validate


def evaluate(artifact_dir, private_manifest, *, backend='docker', image=None, wall_s=7200, on_tick=None):
    artifact_dir = Path(artifact_dir)
    hashes = tree_hashes(artifact_dir)
    if re.fullmatch(r'[0-9a-f]{64}', artifact_dir.name) and digest(hashes) != artifact_dir.name:
        raise ValueError('Frozen artifact hash mismatch')
    if 'inference.py' not in hashes:
        raise ValueError('Artifact must include inference.py with predict(artifact_dir, request)')
    start = time.monotonic()
    task = identify(private_manifest)
    references, predictions, resources = [], {}, []
    windows = iter_windows(private_manifest, with_reference=True)
    key = lambda pair: (pair[1]['person'], pair[1]['session'])
    for _, bout in groupby(windows, key=key):
        # Hold only this bout on the trusted side. Worker sees each update only
        # after responding to the preceding update, and resets for the next bout.
        bout = list(bout)
        with tempfile.TemporaryDirectory(prefix='pulse-replay-') as tmp:
            root = Path(tmp)
            shutil.copytree(artifact_dir, root/'artifact')
            shutil.copyfile(Path(__file__).with_name('inference_entry.py'), root/'entry.py')
            python = sys.executable if backend == 'macos-development' else 'python'
            remaining = wall_s-(time.monotonic()-start)
            if remaining <= 0:
                return invalid_score('replay_wall_budget'), {}
            execution = execute(root, [python, '-I', 'entry.py'], timeout_s=remaining, backend=backend,
                                image=image, on_tick=on_tick, requests=(req for req, _ in bout))
            resources.append({k: v for k, v in execution.items() if k not in ('responses', 'stdout', 'stderr')})
            if not execution['execution_valid'] or len(execution['responses']) != len(bout):
                return {**invalid_score(execution['failure_reason'] or 'incomplete_replay'), 'worker': execution}, {}
            try:
                for (_, ref), prediction in zip(bout, execution['responses']):
                    validate(prediction, task)
                    references.append(ref)
                    predictions[ref['id']] = prediction
            except (TypeError, ValueError) as exc:
                return invalid_score(str(exc)), {}
    result = score(references, predictions, task)
    result.update(resource_use={'wall_s': time.monotonic()-start, 'workers': resources},
                  artifact_hashes=hashes, scored_resource_isolation=backend == 'docker')
    return result, predictions
