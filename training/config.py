"""Load and validate ``grpo_config*.yaml``.

* ``extends: other.yaml`` deep-merges this file over ``other.yaml`` (relative to this file).
* Environment overrides: ``PHYS_GRPO__<section>__<key>=<yaml value>``, e.g.
  ``PHYS_GRPO__model__base_model=meta-llama/Llama-3.1-8B-Instruct`` or
  ``PHYS_GRPO__grpo__group_size=16``. Values are parsed as YAML scalars.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Mapping

import yaml

PLACEHOLDER_PREFIX = 'CHOOSE_BASE_MODEL'
ENV_PREFIX = 'PHYS_GRPO__'
DEFAULT_CONFIG = Path(__file__).with_name('grpo_config.yaml')
SMOKE_CONFIG = Path(__file__).with_name('grpo_config.smoke.yaml')
REQUIRED_SECTIONS = ('model', 'env', 'tasks', 'grpo', 'generation', 'optim', 'trainer')


def deep_merge(base: dict, over: Mapping) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _read(path: Path, seen: tuple = ()) -> dict:
    path = path.resolve()
    if path in seen:
        raise ValueError(f'Cyclic extends: {path}')
    data = yaml.safe_load(path.read_text()) or {}
    parent = data.pop('extends', None)
    if parent:
        return deep_merge(_read(path.parent / parent, seen + (path,)), data)
    return data


def apply_env_overrides(cfg: dict, environ: Mapping[str, str]) -> dict:
    cfg = copy.deepcopy(cfg)
    for name, raw in environ.items():
        if not name.startswith(ENV_PREFIX):
            continue
        keys = name[len(ENV_PREFIX):].split('__')
        node = cfg
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = yaml.safe_load(raw)
    return cfg


def validate(cfg: dict) -> dict:
    missing = [s for s in REQUIRED_SECTIONS if s not in cfg]
    if missing:
        raise ValueError(f'Config missing sections: {missing}')
    g = cfg['grpo']
    if int(g['group_size']) < 2:
        raise ValueError('grpo.group_size must be >= 2 (group-relative advantage is zero otherwise)')
    if int(g['tasks_per_batch']) < 1:
        raise ValueError('grpo.tasks_per_batch must be >= 1')
    if g.get('env_seed_policy', 'shared') not in ('shared', 'per_episode'):
        raise ValueError('grpo.env_seed_policy must be shared or per_episode')
    for key in ('outer_split', 'dev_outer_split'):
        if cfg['tasks'].get(key) == 'agent_final_evaluation':
            raise ValueError(f'tasks.{key}=agent_final_evaluation is forbidden for training')
    if int(cfg['env']['max_turns']) < 1:
        raise ValueError('env.max_turns must be >= 1')
    if not isinstance(cfg['model'].get('base_model'), str) or not cfg['model']['base_model']:
        raise ValueError('model.base_model must be a Hugging Face id or local path')
    return cfg


def base_model_is_placeholder(cfg: dict) -> bool:
    return str(cfg['model']['base_model']).startswith(PLACEHOLDER_PREFIX)


def load_config(path: str | Path = DEFAULT_CONFIG, environ: Mapping[str, str] | None = None) -> dict:
    cfg = _read(Path(path))
    cfg = apply_env_overrides(cfg, os.environ if environ is None else environ)
    return validate(cfg)


def episodes_per_step(cfg: dict) -> int:
    return int(cfg['grpo']['group_size']) * int(cfg['grpo']['tasks_per_batch'])


def get(cfg: dict, dotted: str, default: Any = None) -> Any:
    node: Any = cfg
    for k in dotted.split('.'):
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node
