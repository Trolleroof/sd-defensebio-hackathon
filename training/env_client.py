"""Thin adapter between the trainer and ``research_env.rl`` (RL_ENV_CONTRACT.md v1).

This is the only module that imports ``research_env.rl``. If rl-core's real API differs from
the contract, change this file only. Everything here is defensive: the contract promises the
env never raises on bad actions, but a crash in one episode must not kill a GRPO group, so
exceptions from ``step`` are converted into error observations here as well.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Callable

OUTER_SPLITS = ('agent_training', 'agent_development', 'agent_final_evaluation')
FORBIDDEN_TRAINING_SPLIT = 'agent_final_evaluation'
MALFORMED_OPERATION = '__malformed_tool_call__'

try:  # rl-core's TaskSpec when available; the local mirror keeps CPU tests independent of it.
    from research_env.rl.api import TaskSpec  # type: ignore
except Exception:  # pragma: no cover - exercised only before rl-core lands
    @dataclass(frozen=True)
    class TaskSpec:  # mirror of the contract dataclass
        task_family: str
        task_id: str
        outer_split: str
        fit_people: tuple = ()
        feedback_people: tuple = ()
        reward_people: tuple = ()
        profile: str = 'fast'
        options: dict = field(default_factory=dict)


class FinalEvaluationRefused(ValueError):
    """Raised when an ``agent_final_evaluation`` task is offered as training data."""


def refuse_final_evaluation(task: Any) -> None:
    split = task_field(task, 'outer_split')
    if split not in OUTER_SPLITS:
        raise ValueError(f'Unknown outer_split {split!r}')
    if split == FORBIDDEN_TRAINING_SPLIT:
        raise FinalEvaluationRefused(
            f'Task {task_field(task, "task_id")!r} is agent_final_evaluation; it may never be used for training rollouts')


def task_field(task: Any, name: str) -> Any:
    return task[name] if isinstance(task, dict) else getattr(task, name)


def task_to_dict(task: Any) -> dict:
    if isinstance(task, dict):
        return dict(task)
    if is_dataclass(task):
        d = asdict(task)
        return {k: list(v) if isinstance(v, tuple) else v for k, v in d.items()}
    raise TypeError(f'Cannot serialise task of type {type(task).__name__}')


def task_from_dict(data: dict | str) -> Any:
    if isinstance(data, str):
        data = json.loads(data)
    names = {f.name for f in fields(TaskSpec)}
    kwargs = {k: tuple(v) if k.endswith('_people') else v for k, v in data.items() if k in names}
    return TaskSpec(**kwargs)


@dataclass(frozen=True)
class EnvSettings:
    """Constructor arguments for ``PhysiologyResearchEnv``; identical for every episode in a group."""
    prepared_roots: dict
    run_root: str
    backend: str = 'docker'
    image: str | None = None
    reward_version: str = 'physiology-reward-v1'
    max_turns: int = 40
    history_chars: int = 60_000


def make_env(settings: EnvSettings, factory: Callable[..., Any] | None = None) -> Any:
    """Create one environment. ``factory`` lets tests (or a remote client) replace the real class."""
    if factory is None:
        from research_env.rl.env import PhysiologyResearchEnv  # deferred: GPU image / rl-core only
        factory = PhysiologyResearchEnv
    return factory({k: Path(v) for k, v in settings.prepared_roots.items()}, Path(settings.run_root),
                   backend=settings.backend, image=settings.image, reward_version=settings.reward_version,
                   max_turns=settings.max_turns, history_chars=settings.history_chars)


def reset(env: Any, task: Any, seed: int) -> dict:
    refuse_final_evaluation(task)
    return env.reset(task, seed)


def step(env: Any, action: dict) -> tuple[dict, float, bool, dict]:
    """Contract ``step``; converts any exception into an error observation (non-terminal, reward 0)."""
    try:
        obs, reward, done, info = env.step(action)
    except Exception as exc:  # contract says this never happens; keep the group alive if it does
        obs = {'error': f'{type(exc).__name__}: {exc}', 'adapter_caught_exception': True}
        reward, done, info = 0.0, False, {}
    return obs, float(reward), bool(done), dict(info or {})


def malformed_action(error: str, raw: Any) -> dict:
    """Action dict for a tool call that could not be parsed. Sent to the env so it counts toward
    ``info['invalid_actions']`` and returns its own error observation."""
    return {'operation': MALFORMED_OPERATION, 'error': error, 'raw': raw if isinstance(raw, str) else json.dumps(raw)}


def action_from_tool_call(name: str, arguments: dict, tool_names: set[str]) -> dict:
    """Tool name == ResearchEnvironment operation (contract: one tool per operation)."""
    if name not in tool_names:
        raise ValueError(f'Unknown tool {name!r}; available: {sorted(tool_names)}')
    if not isinstance(arguments, dict):
        raise ValueError('Tool arguments must be a JSON object')
    if 'operation' in arguments and arguments['operation'] != name:
        raise ValueError('Argument "operation" conflicts with tool name')
    return {**arguments, 'operation': name}


def tool_schemas(env: Any) -> list[dict]:
    return list(env.tool_schemas())


def tool_names(schemas: list[dict]) -> set[str]:
    return {s['function']['name'] if 'function' in s else s['name'] for s in schemas}


def render(env: Any, obs: dict) -> str:
    try:
        return env.render_observation(obs)
    except Exception as exc:
        return json.dumps({'error': f'render failed: {type(exc).__name__}: {exc}', 'observation_keys': sorted(obs)})


def budgets_of(obs: dict, info: dict, settings: EnvSettings, task: Any) -> dict:
    """Budgets every episode in a group must share. Prefers env-reported values."""
    reported = obs.get('budgets') or info.get('budgets')
    if reported:
        return dict(reported)
    return {'profile': task_field(task, 'profile'), 'max_turns': settings.max_turns,
            'development_calls': obs.get('development_calls_remaining'),
            'history_chars': settings.history_chars}


def terminate(env: Any, reason: str, max_steps: int) -> tuple[dict, float, bool, dict]:
    """End an episode the policy can no longer continue (e.g. context window exhausted).

    Uses ``env.terminate(reason)`` if rl-core provides it (requested in the handoff); otherwise
    sends malformed actions until the env's own turn budget ends the episode, so the env alone
    decides the terminal reward."""
    if hasattr(env, 'terminate'):
        return step_like(env.terminate(reason))
    obs, reward, done, info = {}, 0.0, False, {}
    for _ in range(max_steps + 1):
        obs, reward, done, info = step(env, malformed_action(f'policy stopped: {reason}', ''))
        if done:
            break
    return obs, reward, done, info


def step_like(result: Any) -> tuple[dict, float, bool, dict]:
    obs, reward, done, info = result
    return obs, float(reward), bool(done), dict(info or {})


def ledger_head(obs: dict, info: dict) -> str | None:
    return info.get('ledger_head_sha256') or obs.get('ledger_head_sha256')


def build_pools(families: list[str], prepared_roots: dict, seed: int) -> list[Any]:
    """TaskSpecs from rl-core's registered task families (contract ``TaskFamily.build_pool``)."""
    from research_env.rl.api import get_family  # deferred: rl-core only
    pool = []
    for name in families:
        pool.extend(get_family(name).build_pool(Path(prepared_roots[name]), seed))
    return pool


def close(env: Any) -> None:
    try:
        env.close()
    except Exception:
        pass
