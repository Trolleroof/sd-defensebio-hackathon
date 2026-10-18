"""`physiology-rl-trajectory-v1` JSONL export for the GRPO trainer.

loss_mask[i] = 1 only for assistant messages; system/user/tool content (file
contents, logs, scores) is masked out. Final-evaluation tasks are refused.
"""
from __future__ import annotations
import json
from pathlib import Path
from research_env.core.common import canonical, digest

SCHEMA = 'physiology-rl-trajectory-v1'


def verify_ledger(events):
    previous = None
    for event in events:
        body = {k: v for k, v in event.items() if k != 'sha256'}
        if event['previous_sha256'] != previous or digest(body) != event['sha256']:
            raise ValueError('Ledger chain corrupted')
        previous = event['sha256']
    return previous


def build_record(env, *, group_id):
    """Record for a finished episode of `env` (call before env.close())."""
    task = env.task
    if task.outer_split == 'agent_final_evaluation':
        raise ValueError('Refusing to export agent_final_evaluation episodes as training data')
    if not env.done or env.reward_info is None:
        raise ValueError('Episode not finished')
    head = verify_ledger(env.environment.session.events())
    messages = [{'role': m['role'], 'content': m['content'], 'tool_calls': list(m.get('tool_calls') or []),
                 'tool_call_id': m.get('tool_call_id')} for m in env.messages]
    return {'schema': SCHEMA, 'task_id': task.task_id, 'task_family': task.task_family,
            'outer_split': task.outer_split, 'seed': env.seed, 'group_id': str(group_id),
            'reward': env.reward_info['reward'], 'reward_version': env.reward_version,
            'reward_components': env.reward_info['components'], 'messages': messages,
            'loss_mask': [1 if m['role'] == 'assistant' else 0 for m in messages],
            'ledger_head_sha256': head, 'budgets': dict(env.budgets)}


def validate_record(record):
    if record.get('schema') != SCHEMA:
        raise ValueError('Wrong trajectory schema')
    if record['outer_split'] == 'agent_final_evaluation':
        raise ValueError('Refusing to export agent_final_evaluation episodes as training data')
    if record['outer_split'] not in ('agent_training', 'agent_development'):
        raise ValueError('Unknown outer split')
    if len(record['loss_mask']) != len(record['messages']) or any(
            mask != (m['role'] == 'assistant') for mask, m in zip(record['loss_mask'], record['messages'])):
        raise ValueError('loss_mask must mark exactly the assistant messages')
    return record


def write_jsonl(path, records, *, append=True):
    lines = [canonical(validate_record(r)).decode()+'\n' for r in records]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a' if append else 'w') as stream:
        stream.writelines(lines)
    return path


def read_jsonl(path):
    return [validate_record(json.loads(line)) for line in Path(path).read_text().splitlines() if line.strip()]
