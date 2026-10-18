"""Load TaskFamily for GRPO post-training (contract: docs/RL_ENV_CONTRACT.md v1)."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
import hashlib
from pathlib import Path
import shutil
import tempfile
from research_env.core.common import digest, invalid_score, write
from research_env.tasks.load import baseline_model
from research_env.tasks.load.baselines import train_rows
from research_env.tasks.load.data import load_records, load_splits
from research_env.tasks.load.evaluation import evaluate, replay_in_process
from research_env.tasks.load.tasks import fit_rows, public_requests, write_manifest

try:
    from research_env.rl.api import OUTER_SPLITS, TaskSpec, register
except ImportError:  # TODO: remove once research_env/rl/api.py (rl-core) lands.
    OUTER_SPLITS = ('agent_training', 'agent_development', 'agent_final_evaluation')
    register = None

    @dataclass(frozen=True)
    class TaskSpec:  # structurally identical to the contract's TaskSpec
        task_family: str
        task_id: str
        outer_split: str
        fit_people: tuple[str, ...]
        feedback_people: tuple[str, ...]
        reward_people: tuple[str, ...]
        profile: str
        options: dict = field(default_factory=dict)

HERE = Path(__file__).parent
INNER = (18, 5, 5)            # fit / feedback / reward from the 28 inner training people
OUTER_FEEDBACK = 5            # development/final tasks: 23 fit / 5 feedback from training people
PARTITIONS = 6
CONFIGS = (('primary', 'person'), ('calibration', 'person'), ('primary', 'heldout15'))
PROFILES = {'fast': {'wall_s': 1200, 'development_calls': 8},
            'benchmark': {'wall_s': 7200, 'development_calls': 30}}
BASELINE_KIND = 'ridge'  # starter artifact staged in the workspace
# Frozen reward baselines: the fixed baseline selected on development loss per (tier, variant)
# (docs/SET3_LOAD_ENVIRONMENT.md). Pandolf 1977 beats every fitted model on the primary tier, so
# skill against ridge would reward rediscovering a published equation.
REWARD_BASELINES = {('primary', 'person'): 'pandolf', ('calibration', 'person'): 'ridge',
                    ('primary', 'heldout15'): 'pandolf'}
EPSILON = 1e-9


def _order(people, *salt):
    key = lambda p: hashlib.sha256(':'.join(map(str, (*salt, p))).encode()).hexdigest()
    return sorted(people, key=key)


def _spec(outer, fit, feedback, reward, profile, options):
    fit, feedback, reward = (tuple(sorted(x)) for x in (fit, feedback, reward))
    if set(fit) & set(feedback) or set(fit) & set(reward) or set(feedback) & set(reward):
        raise ValueError('People sets must be disjoint')
    body = {'family': 'load', 'outer': outer, 'fit': fit, 'feedback': feedback, 'reward': reward,
            'profile': profile, 'options': options}
    return TaskSpec('load', 'load-' + digest(body)[:16], outer, fit, feedback, reward, profile, dict(options))


class LoadFamily:
    name = 'load'
    primary_unit = 'W'

    def build_pool(self, prepared_root: Path, seed: int) -> list:
        splits = load_splits(prepared_root)
        train, dev, test = splits['train'], splits['development'], splits['test']
        if sum(INNER) != len(train):
            raise ValueError('Inner partition does not match training cohort')
        pool = []
        for k in range(PARTITIONS):
            perm = _order(train, 'load-pool-v1', seed, k)
            fit, feedback, reward = perm[:INNER[0]], perm[INNER[0]:INNER[0]+INNER[1]], perm[INNER[0]+INNER[1]:]
            for tier, variant in CONFIGS:
                pool.append(_spec('agent_training', fit, feedback, reward, 'fast',
                                  {'tier': tier, 'variant': variant, 'partition': k, 'pool_seed': seed}))
        perm = _order(train, 'load-outer-v1', seed)
        fit, feedback = perm[:-OUTER_FEEDBACK], perm[-OUTER_FEEDBACK:]
        for outer, reward in (('agent_development', dev), ('agent_final_evaluation', test)):
            for tier, variant in CONFIGS:
                pool.append(_spec(outer, fit, feedback, reward, 'benchmark',
                                  {'tier': tier, 'variant': variant, 'pool_seed': seed}))
        return pool

    def stage_workspace(self, spec, prepared_root: Path, workspace: Path) -> None:
        workspace, prepared_root = Path(workspace).resolve(), Path(prepared_root).resolve()
        if workspace.is_relative_to(prepared_root) or prepared_root.is_relative_to(workspace):
            raise ValueError('Workspace and prepared (private) root must be disjoint')
        records = load_records(prepared_root)
        tier, variant = spec.options['tier'], spec.options['variant']
        rows = fit_rows(records, spec.fit_people, variant, spec.task_id)
        for name in ('TASK.md', 'DATA_DICTIONARY.md', 'example-request.json'):
            (workspace/name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(HERE/name, workspace/name)
        write(workspace/'data'/'train.json', {'role': 'train', 'tier': tier, 'variant': variant, 'units': 'W', 'rows': rows})
        with tempfile.TemporaryDirectory() as tmp:
            dev = write_manifest(Path(tmp)/'m.json', records, spec.feedback_people, tier=tier,
                                 variant=variant, role='feedback', salt=spec.task_id)
        write(workspace/'data'/'development_requests.json', {'tier': tier, 'variant': variant,
                                                            'requests': public_requests(dev)})
        (workspace/'baseline').mkdir(exist_ok=True)
        shutil.copyfile(baseline_model.__file__, workspace/'baseline'/'baseline_model.py')
        train_rows(rows, workspace/'artifact', 0, BASELINE_KIND, tier)
        write(workspace/'task.json', {'task_family': 'load', 'task_id': spec.task_id, 'tier': tier,
                                      'variant': variant, 'profile': spec.profile, 'budgets': PROFILES[spec.profile],
                                      'units': 'W', 'primary_metric': 'person-balanced MAE (W)',
                                      'starter_artifact': BASELINE_KIND,
                                      'frozen_reward_baseline': REWARD_BASELINES[(tier, variant)], 'epsilon': EPSILON,
                                      'n_fit_people': len(spec.fit_people), 'n_feedback_people': len(spec.feedback_people)})

    def private_manifest(self, spec, prepared_root: Path, people: str, out_dir: Path) -> Path:
        if people not in ('feedback', 'reward'):
            raise ValueError("people must be 'feedback' or 'reward'")
        group = spec.feedback_people if people == 'feedback' else spec.reward_people
        path = Path(out_dir)/f'{spec.task_id}-{people}.json'
        write_manifest(path, load_records(prepared_root), group, tier=spec.options['tier'],
                       variant=spec.options['variant'], role=people, salt=spec.task_id)
        return path

    def score(self, artifact_dir: Path, manifest: Path, *, backend: str, image, wall_s: float, profile: str) -> dict:
        try:
            result, _ = evaluate(artifact_dir, manifest, backend=backend, image=image, wall_s=wall_s, profile=profile)
        except Exception as exc:  # invalid never becomes zero loss
            result = invalid_score(f'{type(exc).__name__}: {exc}')
        metrics = {k: result.get(k) for k in ('secondary', 'uncertainty', 'per_person_mae_w', 'n_people', 'n_conditions')}
        return {'execution_valid': bool(result.get('execution_valid')),
                'primary_loss': result.get('primary_loss') if result.get('execution_valid') else None,
                'failure_reason': result.get('failure_reason'), 'units': 'W', 'metrics': metrics,
                'resource_use': result.get('resource_use') or {}}

    def baseline_loss(self, spec, prepared_root: Path) -> float:
        """Frozen dev-selected baseline (REWARD_BASELINES) fitted on fit_people, scored in-process on reward_people."""
        records = load_records(prepared_root)
        tier, variant = spec.options['tier'], spec.options['variant']
        with tempfile.TemporaryDirectory(prefix='load-baseline-') as tmp:
            tmp = Path(tmp)
            train_rows(fit_rows(records, spec.fit_people, variant, spec.task_id), tmp/'artifact', 0,
                       REWARD_BASELINES[(tier, variant)], tier)
            write_manifest(tmp/'reward.json', records, spec.reward_people, tier=tier,
                           variant=variant, role='reward', salt=spec.task_id)
            result, _ = replay_in_process(tmp/'artifact', tmp/'reward.json', baseline_model.predict)
        if not result['execution_valid']:
            raise RuntimeError(f'Frozen baseline invalid: {result["failure_reason"]}')
        return float(result['primary_loss'])


FAMILY = LoadFamily()
if register is not None:
    register(FAMILY)


def spec_to_dict(spec):
    return asdict(spec)
