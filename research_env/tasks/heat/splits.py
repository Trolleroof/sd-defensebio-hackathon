"""Deterministic person-level 14/5/5 split, stratified by recorded sex.

The committed ``split_manifest.json`` is the frozen artifact; this function
documents and reproduces how it was generated. Subjects are split before any
windows or features exist, so every minute of one person stays in one group.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

SEED = 20261017
SIZES = {"train": 14, "development": 5, "private_test": 5}
# Per-group sex quotas for 13 M / 11 F; development and test get one each of
# 3M2F and 2M3F so neither held-out group is single-sex dominated.
QUOTAS = {"train": {"M": 8, "F": 6}, "development": {"M": 3, "F": 2}, "private_test": {"M": 2, "F": 3}}


def generate(sex_by_subject: dict[int, str]) -> dict:
    rng = random.Random(SEED)
    pools = {s: sorted(k for k, v in sex_by_subject.items() if v == s) for s in ("M", "F")}
    for pool in pools.values():
        rng.shuffle(pool)
    groups = {name: [] for name in SIZES}
    for sex, pool in pools.items():
        for name in ("development", "private_test", "train"):
            n = QUOTAS[name][sex]
            groups[name].extend(pool[:n])
            del pool[:n]
    for name in groups:
        groups[name].sort()
        assert len(groups[name]) == SIZES[name]
    return {"split_version": "heat-split-v1", "seed": SEED, "unit": "person",
            "stratified_by": "Demographics!B (sex)", "quotas": QUOTAS,
            "frozen": "2026-10-17, after quality audit heat-quality-v1 and before any model comparison",
            "exclusions": [], "groups": groups}


def load(path: Path) -> dict:
    return json.loads(Path(path).read_text())


SPLIT_PATH = Path(__file__).with_name("split_manifest.json")
