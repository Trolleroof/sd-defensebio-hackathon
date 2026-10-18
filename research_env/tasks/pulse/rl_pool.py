"""Pulse TaskFamily for GRPO post-training (contract v1).

Inner partition of the dataset's training people (13 in the audited cohort):
8 fit / 2 feedback / 3 reward. Smaller cohorts scale proportionally with at least
one person per role. Agent-development tasks reuse fit/feedback from training
people and use the dataset's development people as reward people. Private test
people only appear in `final_evaluation_pool` (agent_final_evaluation).

Fast-profile options subsample bouts per person and windows per bout
deterministically from the task ID, for both feedback and reward manifests and the
frozen ridge baseline, so agent and baseline are scored on identical records.
"""
from __future__ import annotations
from pathlib import Path
import shutil
import tempfile
from research_env.core.common import digest, file_hash, invalid_score, read, write
from research_env.rl.api import TaskSpec, make_task_id, register, validate_pool

PARTITION_13 = (8, 2, 3)
FAST_OPTIONS = {'max_bouts_per_person': 4, 'max_windows_per_bout': 24}
BASELINE = {'kind': 'ridge', 'seed': 0, 'alpha': 10.}
_STAGED_SOURCES = ('__init__.py', '__main__.py', 'requirements.lock', 'core/__init__.py', 'core/common.py',
                   'tasks/__init__.py', 'tasks/pulse/__init__.py', 'tasks/pulse/data.py',
                   'tasks/pulse/baselines.py', 'tasks/pulse/TASK.md', 'tasks/pulse/DATA_DICTIONARY.md',
                   'tasks/pulse/example-request.json')


def partition_sizes(n):
    """(fit, feedback, reward) for n inner training people; 13 -> 8/2/3."""
    if n < 3:
        raise ValueError('Need at least three inner training people')
    reward = max(1, round(n*PARTITION_13[2]/13))
    feedback = max(1, round(n*PARTITION_13[1]/13))
    return n-feedback-reward, feedback, reward


def _ordered(people, *salt):
    return sorted(people, key=lambda p: digest(['pulse-rl-pool-v1', *salt, p]))


def _private_role(prepared_root, person, splits):
    role = next(r for r in ('train', 'development', 'test') if person in splits[r])
    return Path(prepared_root)/'private'/role/'manifest.json'


def _subsample(records, spec, person):
    """Deterministic bout/window subsample preserving causal order within each bout."""
    max_bouts = spec.options.get('max_bouts_per_person')
    max_windows = spec.options.get('max_windows_per_bout')
    sessions = list(dict.fromkeys(r['session'] for r in records))
    if max_bouts:
        keep = set(_ordered(sessions, spec.task_id, person)[:max_bouts])
        sessions = [s for s in sessions if s in keep]
    result = []
    for session in sessions:
        bout = [r for r in records if r['session'] == session]
        if max_windows and len(bout) > max_windows:
            step = len(bout)/max_windows
            bout = [bout[int(i*step)] for i in range(max_windows)]
        result.extend(bout)
    return result


def _write_manifest(prepared_root, splits, people, out_dir, role, spec=None, strip_labels=False):
    """Copy the named people's signals + records into out_dir/manifest.json (opaque filenames kept)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    entries, template = [], None
    for person in people:
        source = _private_role(prepared_root, person, splits)
        manifest = read(source)
        template = template or manifest
        entry = next(e for e in manifest['entries'] if e['person'] == person)
        shutil.copyfile(source.parent/entry['signals'], out_dir/entry['signals'])
        records = [dict(r) for r in entry['records']]
        if spec is not None:
            records = _subsample(records, spec, person)
        if strip_labels:
            for r in records:
                r.pop('reference_bpm', None)
                r.pop('activity', None)
        if file_hash(out_dir/entry['signals']) != entry['sha256']:
            raise ValueError('Signal hash mismatch while staging')
        entries.append({**entry, 'records': records})
    write(out_dir/'manifest.json', {**{k: v for k, v in template.items() if k != 'entries'},
                                    'role': role, 'entries': entries})
    return out_dir/'manifest.json'


class PulseFamily:
    name = 'pulse'
    primary_unit = 'bpm'
    task_path = 'research_env/tasks/pulse/TASK.md'

    def __init__(self, n_training_tasks=16, n_development_tasks=2):
        self.n_training_tasks, self.n_development_tasks = n_training_tasks, n_development_tasks
        self._baselines = {}

    def dataset_splits(self, prepared_root):
        return read(Path(prepared_root)/'private'/'splits.json')

    def _spec(self, seed, split, profile, fit, feedback, reward, options):
        fit, feedback, reward = (tuple(sorted(x)) for x in (fit, feedback, reward))
        return TaskSpec('pulse', make_task_id('pulse', seed, split, profile, fit, feedback, reward, options),
                        split, fit, feedback, reward, profile, dict(options))

    def build_pool(self, prepared_root, seed, profile='fast'):
        splits = self.dataset_splits(prepared_root)
        train = sorted(splits['train'])
        n_fit, n_feedback, n_reward = partition_sizes(len(train))
        options = dict(FAST_OPTIONS) if profile == 'fast' else {}
        pool = []
        for k in range(self.n_training_tasks):
            order = _ordered(train, seed, 'agent_training', k)
            pool.append(self._spec(seed, 'agent_training', profile, order[:n_fit],
                                   order[n_fit:n_fit+n_feedback], order[n_fit+n_feedback:], {**options, 'index': k}))
        for k in range(self.n_development_tasks):
            order = _ordered(train, seed, 'agent_development', k)
            pool.append(self._spec(seed, 'agent_development', profile, order[:-n_feedback], order[-n_feedback:],
                                   splits['development'], {**options, 'index': k}))
        return validate_pool(pool, splits)

    def final_evaluation_pool(self, prepared_root, seed):
        """Benchmark-profile final evaluation: fit=train, feedback=development, reward=private test."""
        splits = self.dataset_splits(prepared_root)
        return validate_pool([self._spec(seed, 'agent_final_evaluation', 'benchmark', splits['train'],
                                         splits['development'], splits['test'], {'index': 0})], splits)

    def stage_workspace(self, spec, prepared_root, workspace):
        """Public fit data with labels; feedback signals/update list without labels; no reward people."""
        splits = self.dataset_splits(prepared_root)
        workspace = Path(workspace)
        _write_manifest(prepared_root, splits, spec.fit_people, workspace/'data'/'train', 'train')
        _write_manifest(prepared_root, splits, spec.feedback_people, workspace/'data'/'development',
                        'development', spec=spec if spec.profile == 'fast' else None, strip_labels=True)
        package = Path(__file__).resolve().parents[2]
        for name in _STAGED_SOURCES:
            target = workspace/'research_env'/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(package/name, target)

    def private_manifest(self, spec, prepared_root, people, out_dir):
        if people not in ('feedback', 'reward'):
            raise ValueError("people must be 'feedback' or 'reward'")
        group = spec.feedback_people if people == 'feedback' else spec.reward_people
        return _write_manifest(prepared_root, self.dataset_splits(prepared_root), group, Path(out_dir)/people,
                               people, spec=spec if spec.profile == 'fast' else None)

    def score(self, artifact_dir, manifest, *, backend, image, wall_s, profile, on_tick=None):
        from research_env.core.evaluation import evaluate
        try:
            result, _ = evaluate(artifact_dir, manifest, backend=backend, image=image, wall_s=wall_s,
                                 on_tick=on_tick, adapter='pulse',
                                 replay='batched' if profile == 'fast' else 'strict')
        except Exception as exc:
            result = invalid_score(f'{type(exc).__name__}: {exc}')
        result.setdefault('resource_use', None)
        result['metrics'] = {k: result.get(k) for k in ('secondary', 'uncertainty', 'per_person_mae_bpm',
                                                          'n_people', 'n_bouts', 'n_windows')}
        return result

    def baseline_loss(self, spec, prepared_root):
        """Frozen ridge fitted on fit_people, replayed (trusted builder code) on reward_people."""
        key = (str(Path(prepared_root).resolve()), spec.task_id)
        if key not in self._baselines:
            from research_env.tasks.pulse.baselines import replay, train
            splits = self.dataset_splits(prepared_root)
            with tempfile.TemporaryDirectory(prefix='pulse-baseline-') as tmp:
                tmp = Path(tmp)
                fit = _write_manifest(prepared_root, splits, spec.fit_people, tmp/'fit', 'train')
                reward = self.private_manifest(spec, prepared_root, 'reward', tmp)
                train(fit, None, tmp/'ridge', BASELINE['seed'], BASELINE['kind'], BASELINE['alpha'])
                result, _ = replay(tmp/'ridge', reward)
            if not result['execution_valid']:
                raise ValueError(f'Frozen baseline invalid: {result["failure_reason"]}')
            self._baselines[key] = float(result['primary_loss'])
        return self._baselines[key]


FAMILY = PulseFamily()
register(FAMILY)
