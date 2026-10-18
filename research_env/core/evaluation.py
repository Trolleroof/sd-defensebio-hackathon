"""Trusted replay: no source recordings, labels or evaluator code in worker.

The replay loop is task-neutral and dispatches through a ``ReplayAdapter``. The
default adapter is pulse, so existing callers and pulse results are unchanged.

Replay modes:
- ``strict`` (default, benchmark profile): one fresh worker process per bout.
- ``batched`` (fast RL-training profile): one worker per person; at every bout
  start the worker reloads the candidate module and resets causal state. This is
  an in-process reset, weaker than a fresh process (files written to the worker
  scratch area or interpreter-level caches can persist across that person's
  bouts). It never spans people, and it is not used for benchmark scoring.
"""
from itertools import groupby
from pathlib import Path
import shutil
import tempfile
import sys
import time
import re
from research_env.core.common import digest, invalid_score, tree_hashes, validate_prediction
from research_env.core.worker import execute, worker_python

REPLAY_MODES = ('strict', 'batched')
BOUT_START = '__bout_start__'

# Written into the worker scratch root for batched replay. The bout-start marker
# is removed before the candidate sees the request, so request schemas match strict mode.
BATCHED_ENTRY = '''"""Copied into a per-person worker; module and state reset at each bout start."""
import importlib.util
import json
from pathlib import Path
import sys

MARKER = %r


def load():
    spec = importlib.util.spec_from_file_location('candidate', Path('artifact')/'inference.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


module, state = None, None
for line in sys.stdin:
    request = json.loads(line)
    if request.pop(MARKER, False) or module is None:
        sys.modules.pop('candidate', None)
        module, state = load(), None
    request['state'] = state
    result = module.predict('artifact', request)
    state = result['state']
    sys.stdout.write(json.dumps(result, allow_nan=False)+'\\n')
    sys.stdout.flush()
''' % BOUT_START


class PulseReplayAdapter:
    """Pulse request stream, prediction validation and scoring (galaxy-pulse-causal-v1)."""
    name = 'pulse'

    def bouts(self, private_manifest):
        """Yield (person, [(request, reference), ...]) per bout, in manifest order."""
        from research_env.tasks.pulse.data import iter_windows
        windows = iter_windows(private_manifest, with_reference=True)
        for (person, _), bout in groupby(windows, key=lambda pair: (pair[1]['person'], pair[1]['session'])):
            yield person, list(bout)

    def reference_id(self, reference):
        return reference['id']

    def validate(self, prediction):
        return validate_prediction(prediction)

    def score(self, references, predictions):
        from research_env.tasks.pulse.scoring import score
        return score(references, predictions)


ADAPTERS = {'pulse': PulseReplayAdapter()}


def register_adapter(adapter):
    """Task families (heat/load) register their replay adapter under their family name."""
    ADAPTERS[adapter.name] = adapter
    return adapter


def get_adapter(adapter):
    if adapter is None:
        return ADAPTERS['pulse']
    if isinstance(adapter, str):
        if adapter not in ADAPTERS:
            raise ValueError(f'Unknown replay adapter {adapter!r}; refusing to score it as another task')
        return ADAPTERS[adapter]
    return adapter


def _stage(root, artifact_dir, replay):
    shutil.copytree(artifact_dir, root/'artifact')
    if replay == 'strict':
        shutil.copyfile(Path(__file__).with_name('inference_entry.py'), root/'entry.py')
    else:
        (root/'entry.py').write_text(BATCHED_ENTRY)


def evaluate(artifact_dir, private_manifest, *, backend='docker', image=None, wall_s=7200, on_tick=None,
             adapter=None, replay='strict'):
    if replay not in REPLAY_MODES:
        raise ValueError('Unknown replay mode')
    adapter = get_adapter(adapter)
    artifact_dir = Path(artifact_dir)
    hashes = tree_hashes(artifact_dir)
    if re.fullmatch(r'[0-9a-f]{64}', artifact_dir.name) and digest(hashes) != artifact_dir.name:
        raise ValueError('Frozen artifact hash mismatch')
    if 'inference.py' not in hashes:
        raise ValueError('Artifact must include inference.py with predict(artifact_dir, request)')
    start = time.monotonic()
    references, predictions, resources = [], {}, []
    python = worker_python(backend)
    if replay == 'strict':
        groups = ([bout] for _, bout in adapter.bouts(private_manifest))
    else:
        groups = ([bout for _, bout in person_bouts]
                  for _, person_bouts in groupby(adapter.bouts(private_manifest), key=lambda item: item[0]))
    for group in groups:
        # Hold only this group on the trusted side. Worker sees each update only
        # after responding to the preceding update; state resets for each bout.
        pairs = [pair for bout in group for pair in bout]
        if replay == 'strict':
            requests = (req for req, _ in pairs)
        else:
            requests = ({**req, BOUT_START: True} if i == 0 else req
                        for bout in group for i, (req, _) in enumerate(bout))
        with tempfile.TemporaryDirectory(prefix='replay-') as tmp:
            root = Path(tmp)
            _stage(root, artifact_dir, replay)
            remaining = wall_s-(time.monotonic()-start)
            if remaining <= 0:
                return invalid_score('replay_wall_budget'), {}
            execution = execute(root, [python, '-I', 'entry.py'], timeout_s=remaining, backend=backend,
                                image=image, on_tick=on_tick, requests=requests)
            resources.append({k: v for k, v in execution.items() if k not in ('responses', 'stdout', 'stderr')})
            if not execution['execution_valid'] or len(execution['responses']) != len(pairs):
                return {**invalid_score(execution['failure_reason'] or 'incomplete_replay'), 'worker': execution}, {}
            try:
                for (_, ref), prediction in zip(pairs, execution['responses']):
                    adapter.validate(prediction)
                    references.append(ref)
                    predictions[adapter.reference_id(ref)] = prediction
            except (TypeError, ValueError) as exc:
                return invalid_score(str(exc)), {}
    result = adapter.score(references, predictions)
    result.update(resource_use={'wall_s': time.monotonic()-start, 'workers': resources, 'replay': replay},
                  artifact_hashes=hashes, scored_resource_isolation=replay == 'strict' and bool(resources)
                  and all(r.get('scored_resource_isolation') for r in resources))
    return result, predictions
