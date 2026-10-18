"""Upload prepared public vs private manifests to SEPARATE Beam volumes.

Dry-run by default: prints the exact ``beam`` CLI commands. ``--execute`` runs
them (requires ``beam login``; volume storage is billed per the pricing page).

    python -m research_env.beam.upload_data PREPARED_ROOT --dataset pulse \
        [--private-extra DIR ...] [--execute]

PREPARED_ROOT is the output of ``research_env prepare`` and must contain
``public/`` and ``private/``. Public files go to ``beam://PUBLIC_VOLUME/DATASET/``;
private files (labels, per-task feedback/reward manifests from
``TaskFamily.private_manifest``) go to ``beam://PRIVATE_VOLUME/DATASET/``.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

from research_env.beam.config import BeamConfig

LABEL_KEY = re.compile(r'^reference', re.I)
_DATASET = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]{0,63}')


def _has_label_keys(value):
    if isinstance(value, dict):
        return any(LABEL_KEY.match(str(k)) or _has_label_keys(v) for k, v in value.items())
    if isinstance(value, list):
        return any(_has_label_keys(v) for v in value)
    return False


def check_public_tree(public: Path):
    """Refuse to publish anything that looks private."""
    problems = []
    for path in sorted(public.rglob('*')):
        rel = path.relative_to(public)
        if path.is_symlink():
            problems.append(f'symlink: {rel}')
        elif 'private' in (p.lower() for p in rel.parts):
            problems.append(f'private path component: {rel}')
        elif path.suffix == '.json' and path.is_file():
            try:
                if _has_label_keys(json.loads(path.read_text())):
                    problems.append(f'reference/label keys in public JSON: {rel}')
            except (ValueError, UnicodeDecodeError):
                problems.append(f'unreadable JSON: {rel}')
    return problems


def plan_uploads(prepared_root, dataset, cfg: BeamConfig, private_extra=()):
    cfg.validate()
    if not _DATASET.fullmatch(dataset):
        raise ValueError('dataset must be a simple name')
    root = Path(prepared_root).resolve()
    public, private = root/'public', root/'private'
    if not public.is_dir() or not private.is_dir():
        raise ValueError('PREPARED_ROOT must contain public/ and private/')
    extras = [Path(p).resolve() for p in private_extra]
    for extra in extras:
        if extra.is_relative_to(public) or public.is_relative_to(extra):
            raise ValueError(f'private extra overlaps public tree: {extra}')
    problems = check_public_tree(public)
    if problems:
        raise ValueError('Public tree failed checks:\n' + '\n'.join(problems))
    cmds = [['beam', 'volume', 'create', cfg.public_volume],
            ['beam', 'volume', 'create', cfg.private_volume],
            ['beam', 'cp', str(public), f'beam://{cfg.public_volume}/{dataset}'],
            ['beam', 'cp', str(private), f'beam://{cfg.private_volume}/{dataset}']]
    for extra in extras:
        cmds.append(['beam', 'cp', str(extra), f'beam://{cfg.private_volume}/{dataset}/tasks/{extra.name}'])
    for cmd in cmds:  # invariant: private sources never target the public volume
        if cmd[1] == 'cp' and cmd[3].startswith(f'beam://{cfg.public_volume}/') and cmd[2] != str(public):
            raise AssertionError('private source routed to public volume')
    return cmds


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('prepared_root')
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--private-extra', action='append', default=[])
    ap.add_argument('--execute', action='store_true')
    args = ap.parse_args(argv)
    cmds = plan_uploads(args.prepared_root, args.dataset, BeamConfig.from_env(), args.private_extra)
    for cmd in cmds:
        print(' '.join(cmd))
    if not args.execute:
        print('# dry run: re-run with --execute to upload', file=sys.stderr)
        return 0
    if not shutil.which('beam'):
        raise SystemExit('beam CLI not found')
    for cmd in cmds:
        code = subprocess.run(cmd, check=False).returncode
        if code != 0 and cmd[1] != 'volume':  # volume create fails harmlessly if it exists
            return code
    return 0


if __name__ == '__main__':
    sys.exit(main())
