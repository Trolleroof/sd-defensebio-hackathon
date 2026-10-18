"""Group-relative advantages from terminal env rewards.

Matches verl v0.9.1 ``compute_grpo_outcome_advantage`` (verl/trainer/ppo/core_algos.py):
per group, ``(r - mean) / (std + eps)`` with ``torch.std`` (Bessel-corrected, n-1); a group of
size 1 uses mean 0 and std 1. With ``norm_by_std=False`` it is plain mean-centering (Dr. GRPO).
The trainer uses verl's implementation; this pure-Python copy is used by the standalone
rollout path for logging and by the CPU tests.
"""
from __future__ import annotations

import math
from collections import OrderedDict
from typing import Hashable, Sequence


def group_advantages(rewards: Sequence[float], *, norm_by_std: bool = True, eps: float = 1e-6) -> list[float]:
    r = [float(x) for x in rewards]
    if not r:
        return []
    if len(r) == 1:
        mean, std = 0.0, 1.0
    else:
        mean = sum(r) / len(r)
        std = math.sqrt(sum((x - mean) ** 2 for x in r) / (len(r) - 1))
    if norm_by_std:
        return [(x - mean) / (std + eps) for x in r]
    return [x - mean for x in r]


def batch_advantages(group_ids: Sequence[Hashable], rewards: Sequence[float], **kw) -> list[float]:
    """Advantages for a flat batch whose episodes are tagged with their GRPO group id."""
    if len(group_ids) != len(rewards):
        raise ValueError('group_ids and rewards differ in length')
    groups: 'OrderedDict[Hashable, list[int]]' = OrderedDict()
    for i, g in enumerate(group_ids):
        groups.setdefault(g, []).append(i)
    out = [0.0] * len(rewards)
    for idx in groups.values():
        for i, a in zip(idx, group_advantages([rewards[i] for i in idx], **kw)):
            out[i] = a
    return out


def token_advantages(advantage: float, response_mask: Sequence[int]) -> list[float]:
    """Broadcast an episode advantage over its tokens; tool/observation tokens (mask 0) get 0."""
    return [advantage if m else 0.0 for m in response_mask]
