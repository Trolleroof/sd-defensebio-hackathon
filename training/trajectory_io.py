"""Build, validate and write ``physiology-rl-trajectory-v1`` records (RL_ENV_CONTRACT.md).

rl-core owns ``research_env/rl/trajectory.py`` (the env-side exporter). This module produces the
same record shape from the trainer side and refuses ``agent_final_evaluation`` tasks.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from training.env_client import refuse_final_evaluation, task_field

SCHEMA = 'physiology-rl-trajectory-v1'
ROLES = ('system', 'user', 'assistant', 'tool')
REQUIRED = ('schema', 'task_id', 'task_family', 'outer_split', 'seed', 'group_id', 'reward', 'reward_version',
            'reward_components', 'messages', 'loss_mask', 'ledger_head_sha256', 'budgets')


def message(role: str, content: str | None, tool_calls: list | None = None, tool_call_id: str | None = None) -> dict:
    if role not in ROLES:
        raise ValueError(f'Bad role {role!r}')
    return {'role': role, 'content': content if content is not None else '',
            'tool_calls': list(tool_calls or []), 'tool_call_id': tool_call_id}


def loss_mask_for(messages: Iterable[dict]) -> list[int]:
    """1 only for assistant (model-written) messages; system, user, tool outputs are masked out."""
    return [1 if m['role'] == 'assistant' else 0 for m in messages]


def build_record(*, task: Any, seed: int, group_id: str, reward: float, reward_version: str,
                 reward_components: dict, messages: list[dict], ledger_head_sha256: str | None,
                 budgets: dict, rollout: dict | None = None) -> dict:
    refuse_final_evaluation(task)
    rec = {'schema': SCHEMA, 'task_id': task_field(task, 'task_id'), 'task_family': task_field(task, 'task_family'),
           'outer_split': task_field(task, 'outer_split'), 'seed': int(seed), 'group_id': group_id,
           'reward': float(reward), 'reward_version': reward_version, 'reward_components': dict(reward_components),
           'messages': messages, 'loss_mask': loss_mask_for(messages),
           'ledger_head_sha256': ledger_head_sha256, 'budgets': dict(budgets)}
    if rollout:
        rec['rollout'] = dict(rollout)  # trainer-side metadata; not part of the frozen v1 fields
    validate_record(rec)
    return rec


def validate_record(rec: dict) -> None:
    missing = [k for k in REQUIRED if k not in rec]
    if missing:
        raise ValueError(f'Trajectory record missing {missing}')
    if rec['schema'] != SCHEMA:
        raise ValueError(f'Unexpected schema {rec["schema"]!r}')
    if rec['outer_split'] == 'agent_final_evaluation':
        raise ValueError('agent_final_evaluation trajectories may not be exported as training data')
    if len(rec['loss_mask']) != len(rec['messages']):
        raise ValueError('loss_mask length must equal messages length')
    for m, bit in zip(rec['messages'], rec['loss_mask']):
        if m['role'] not in ROLES:
            raise ValueError(f'Bad role {m["role"]!r}')
        if bit != (1 if m['role'] == 'assistant' else 0):
            raise ValueError('loss_mask must be 1 exactly on assistant messages')


def write_jsonl(records: Iterable[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as fh:
        for rec in records:
            validate_record(rec)
            fh.write(json.dumps(rec, sort_keys=True) + '\n')
    return path


def read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
