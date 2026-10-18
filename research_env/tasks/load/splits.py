"""Deterministic person-level split. Generated once, then frozen in config/split_v1.json.

People are ordered by sha256(salt + source ID) within each sex and dealt into
train/development/test in proportion, so the split depends on neither row
order nor any target value. Regenerating with the same salt reproduces the file.
"""
from __future__ import annotations

import hashlib

SPLIT_ID = "evobreath-person-split-v1"
SALT = "load-evobreath-v1/person-split"
SIZES = {"train": 28, "development": 10, "test": 10}


def make_split(people: list[tuple[str, str]], salt: str = SALT, sizes=SIZES) -> dict:
    """people: (source_person_id, sex). Returns {split_id, salt, sizes, splits}."""
    total = sum(sizes.values())
    if len(people) != total:
        raise ValueError(f"expected {total} people, got {len(people)}")
    order = lambda pid: hashlib.sha256((salt + pid).encode()).hexdigest()
    by_sex: dict[str, list[str]] = {}
    for pid, sex in people:
        by_sex.setdefault(sex, []).append(pid)
    # Interleave sexes in hash order so each split's sex mix tracks the cohort.
    queues = {s: sorted(ids, key=order) for s, ids in sorted(by_sex.items())}
    dealt: list[str] = []
    counts = {s: 0 for s in queues}
    while len(dealt) < total:
        sex = min((s for s in queues if counts[s] < len(queues[s])),
                  key=lambda s: ((counts[s] + 0.5) / len(queues[s]), s))
        dealt.append(queues[sex][counts[sex]])
        counts[sex] += 1
    splits = {name: [] for name in sizes}
    for pid, name in zip(dealt, _interleave(sizes)):
        splits[name].append(pid)
    return {"split_id": SPLIT_ID, "salt": salt, "sizes": dict(sizes),
            "splits": {k: sorted(v) for k, v in splits.items()}}


def _interleave(sizes: dict) -> list[str]:
    """Spread split labels evenly over positions (largest-remainder schedule)."""
    total = sum(sizes.values())
    out, given = [], {k: 0 for k in sizes}
    for i in range(total):
        name = max(sizes, key=lambda k: ((i + 1) * sizes[k] / total - given[k], -list(sizes).index(k)))
        out.append(name)
        given[name] += 1
    return out
