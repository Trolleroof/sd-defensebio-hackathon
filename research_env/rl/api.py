"""RL task-family interface (contract v1, docs/RL_ENV_CONTRACT.md) and split-safety validators."""
from __future__ import annotations
from dataclasses import dataclass, field
import importlib
from pathlib import Path
from typing import Protocol, runtime_checkable
from research_env.core.common import digest

OUTER_SPLITS = ('agent_training', 'agent_development', 'agent_final_evaluation')
PEOPLE_SETS = ('fit_people', 'feedback_people', 'reward_people')

# Frozen profile budgets. Every episode of a task (hence every GRPO group member)
# receives exactly these budgets; overrides are environment-wide, never per episode.
PROFILES = {
    'fast': {'wall_s': 1200, 'development_calls': 8, 'replay': 'batched'},
    'benchmark': {'wall_s': 7200, 'development_calls': 30, 'replay': 'strict'},
}


@dataclass(frozen=True)
class TaskSpec:
    task_family: str               # 'pulse' | 'heat' | 'load'
    task_id: str                   # deterministic from (family, seed, people, options)
    outer_split: str               # one of OUTER_SPLITS
    fit_people: tuple[str, ...]    # public training data the agent may fit on
    feedback_people: tuple[str, ...]  # scored by the agent's `evaluate` calls (visible feedback)
    reward_people: tuple[str, ...]    # hidden; trainer reward only; disjoint from the two above
    profile: str                   # 'fast' | 'benchmark'
    options: dict = field(default_factory=dict)  # family-specific: bouts, channels, horizons, loads, tier

    def to_dict(self):
        return {'task_family': self.task_family, 'task_id': self.task_id, 'outer_split': self.outer_split,
                'fit_people': list(self.fit_people), 'feedback_people': list(self.feedback_people),
                'reward_people': list(self.reward_people), 'profile': self.profile, 'options': dict(self.options)}

    @classmethod
    def from_dict(cls, value):
        return cls(**{**value, **{k: tuple(value[k]) for k in PEOPLE_SETS}})


@runtime_checkable
class TaskFamily(Protocol):
    name: str
    primary_unit: str              # 'bpm' | 'degC' | 'W'

    def build_pool(self, prepared_root: Path, seed: int) -> list[TaskSpec]: ...
    def stage_workspace(self, spec: TaskSpec, prepared_root: Path, workspace: Path) -> None: ...
    def private_manifest(self, spec: TaskSpec, prepared_root: Path, people: str, out_dir: Path) -> Path: ...
    def score(self, artifact_dir: Path, manifest: Path, *, backend: str, image: str | None,
              wall_s: float, profile: str) -> dict: ...
    def baseline_loss(self, spec: TaskSpec, prepared_root: Path) -> float: ...
    # Optional: dataset_splits(prepared_root) -> {'train': [...], 'development': [...], 'test': [...]}.
    # When present, the environment validates every TaskSpec against it at reset.


_REGISTRY: dict[str, TaskFamily] = {}


def register(family: TaskFamily) -> None:
    if not isinstance(family, TaskFamily):
        raise TypeError('Family does not implement the TaskFamily protocol')
    _REGISTRY[family.name] = family


def get_family(name: str) -> TaskFamily:
    """Registered family, else lazily import research_env.tasks.<name>.rl_pool (which registers itself)."""
    if name not in _REGISTRY:
        if not name.isidentifier():
            raise KeyError(f'Unknown task family {name!r}')
        try:
            importlib.import_module(f'research_env.tasks.{name}.rl_pool')
        except ModuleNotFoundError as exc:
            raise KeyError(f'Unknown task family {name!r}') from exc
    if name not in _REGISTRY:
        raise KeyError(f'Unknown task family {name!r}')
    return _REGISTRY[name]


def make_task_id(task_family, seed, outer_split, profile, fit_people, feedback_people, reward_people, options):
    return f'{task_family}-' + digest(['physiology-rl-task-v1', task_family, seed, outer_split, profile,
                                       sorted(fit_people), sorted(feedback_people), sorted(reward_people),
                                       options])[:20]


def validate_spec(spec: TaskSpec) -> TaskSpec:
    """Structural checks: known split/profile, nonempty and pairwise-disjoint people sets."""
    if spec.outer_split not in OUTER_SPLITS:
        raise ValueError(f'Unknown outer split {spec.outer_split!r}')
    if spec.profile not in PROFILES:
        raise ValueError(f'Unknown profile {spec.profile!r}')
    groups = [getattr(spec, k) for k in PEOPLE_SETS]
    if any(not g or any(not isinstance(p, str) or not p for p in g) for g in groups):
        raise ValueError('Every people set must be a nonempty tuple of person IDs')
    people = [p for g in groups for p in g]
    if len(people) != len(set(people)):
        raise ValueError('fit/feedback/reward people overlap')
    return spec


def validate_against_splits(spec: TaskSpec, splits: dict) -> TaskSpec:
    """Person-level outer split safety against the dataset's train/development/test split.

    agent_training: all three sets from inner training people only.
    agent_development: fit/feedback from training people; reward = development people.
    agent_final_evaluation: reward people are the dataset's private test people.
    Test people never appear in agent_training or agent_development tasks.
    """
    validate_spec(spec)
    train, development, test = (set(splits[k]) for k in ('train', 'development', 'test'))
    if train & development or train & test or development & test:
        raise ValueError('Dataset splits overlap')
    fit, feedback, reward = (set(getattr(spec, k)) for k in PEOPLE_SETS)
    everyone = fit | feedback | reward
    if not everyone <= train | development | test:
        raise ValueError('TaskSpec names people outside the dataset cohort')
    if spec.outer_split != 'agent_final_evaluation' and everyone & test:
        raise ValueError('Private test people used by a training/development task')
    if spec.outer_split == 'agent_training' and not everyone <= train:
        raise ValueError('agent_training tasks must draw only inner training people')
    if spec.outer_split == 'agent_development' and (not fit | feedback <= train or not reward <= development):
        raise ValueError('agent_development tasks: fit/feedback from training, reward from development people')
    if spec.outer_split == 'agent_final_evaluation' and (not reward <= test or (fit | feedback) & test):
        raise ValueError('agent_final_evaluation reward people must be private test people only')
    return spec


def validate_pool(specs, splits) -> list[TaskSpec]:
    ids = [s.task_id for s in specs]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate task IDs in pool')
    return [validate_against_splits(s, splits) for s in specs]
