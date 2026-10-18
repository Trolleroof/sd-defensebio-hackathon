"""Shared v1 contracts. Controller objects never enter the participant worker."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_bytes(canonical(value) + b'\n')
    tmp.replace(path)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_splits(splits):
    if set(splits) != {'train', 'development', 'test'}:
        raise ValueError('Require train/development/test person lists')
    people = [p for group in splits.values() for p in group]
    if any(not isinstance(p, str) or not p for p in people) or len(people) != len(set(people)):
        raise ValueError('Person overlap or invalid person')
    if any(not group for group in splits.values()):
        raise ValueError('Empty split')
    return splits


def validate_prediction(prediction, names=('mean_bpm', 'lower_bpm', 'upper_bpm')):
    if not isinstance(prediction, dict) or set(prediction) != {*names, 'state'}:
        raise ValueError('Wrong prediction fields')
    mean, lower, upper = (prediction[k] for k in names)
    if not all(finite(v) for v in (mean, lower, upper)) or not lower <= mean <= upper:
        raise ValueError('Nonfinite or unordered interval')
    if len(canonical(prediction['state'])) > 16384:
        raise ValueError('State exceeds 16 KiB')
    return prediction


def tree_hashes(root):
    root = Path(root)
    result = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError('Symlinks forbidden in submission/assets')
        if path.is_file():
            result[str(path.relative_to(root))] = file_hash(path)
    return result


def invalid_score(reason, resource_use=None):
    return {'execution_valid': False, 'primary_loss': None, 'skill': None,
            'secondary': None, 'uncertainty': None, 'resource_use': resource_use,
            'failure_reason': reason}
