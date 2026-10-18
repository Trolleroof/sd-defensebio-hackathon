"""Fast reset between GRPO episodes via Beam filesystem images.

``create_reset_image`` stages a task's initial workspace once, captures the
sandbox root filesystem with ``SandboxInstance.create_image_from_filesystem()``
and records the workspace hashes locally. Passing the returned id as ``image=``
to ``worker.execute(..., backend='beam')`` then uploads only the files that an
episode changed relative to that snapshot. Every episode still gets a fresh
sandbox from the immutable image, so episodes never share mutable state.

Memory snapshots (``snapshot_memory``) are deliberately unused: a restored process
tree could carry state from the staging run into an episode.
"""
from __future__ import annotations
import json
from pathlib import Path
import tempfile

from research_env.beam.config import BeamConfig
from research_env.beam.sandbox import (HELPER_SOURCE, _IMAGE_ID, _load_sdk, _pack, sandbox_env,
                                       sandbox_kwargs)
from research_env.core.common import tree_hashes


def create_reset_image(workspace, base_image_id, *, config: BeamConfig | None = None, sdk=None,
                       timeout_s=1800):
    """Billable when run against real Beam. Returns the new image id."""
    cfg = (config or BeamConfig.from_env()).validate()
    if not _IMAGE_ID.fullmatch(base_image_id or ''):
        raise ValueError('Explicit base Beam image id required')
    sdk = sdk or _load_sdk()
    root = Path(workspace).resolve()
    hashes = tree_hashes(root)
    env = sandbox_env(1, cfg)
    ctl = cfg.control_dir
    sb = None
    try:
        sb = sdk.Sandbox(**sandbox_kwargs(cfg, sdk, image_id=base_image_id, cpus=1, memory_gb=2,
                                          timeout_s=timeout_s, env=env)).create()
        with tempfile.TemporaryDirectory(prefix='beam-snapshot-') as tmp:
            tar = Path(tmp)/'in.tar.gz'
            _pack(root, list(hashes), tar)
            (Path(tmp)/'deleted.json').write_text('[]')
            sb.fs.create_directory(ctl)
            sb.fs.write_text(ctl + '/helper.py', HELPER_SOURCE.read_text())
            sb.fs.upload_file(str(tar), ctl + '/in.tar.gz')
            sb.fs.upload_file(str(Path(tmp)/'deleted.json'), ctl + '/deleted.json')
        p = sb.process.exec('python', '-I', ctl + '/helper.py', 'apply', ctl + '/in.tar.gz',
                            ctl + '/deleted.json', cfg.workdir, cwd='/tmp', env=env)
        if p.wait(timeout=timeout_s) != 0:
            raise RuntimeError('workspace staging failed in snapshot sandbox')
        image_id = sb.create_image_from_filesystem()
    finally:
        if sb is not None:
            sb.terminate()
    if not _IMAGE_ID.fullmatch(image_id or ''):
        raise RuntimeError('Beam returned an unexpected image id')
    registry = Path(cfg.snapshot_registry)
    registry.mkdir(parents=True, exist_ok=True)
    (registry/f'{image_id.replace(":", "_")}.json').write_text(json.dumps(
        {'image_id': image_id, 'base_image_id': base_image_id, 'workspace_hashes': hashes}, indent=1))
    return image_id
