"""Trusted Beam scorer: the only Beam workload that mounts the private volume.

It receives a frozen artifact (tar bytes + content digest) and the *name* of a
private manifest on its own volume, runs the trusted streaming replay, and
returns only the score dict (no predictions, no references). Candidate code is
never executed inside the scorer container: replay launches network-blocked
participant sandboxes (backend='beam') that mount only public data, so the
scorer needs a Beam token, bound as a plaintext function secret by NAME
(``BeamConfig.scorer_token_secret``). Participant sandboxes get no secrets.

Controller usage (billable): ``submit_artifact(score_artifact, artifact_dir, 'pulse/<task>/reward.json',
family='pulse', profile='fast', wall_s=1200)``; ``score_artifact.remote`` runs the function on Beam.
"""
from __future__ import annotations
import io
import json
import math
from pathlib import Path, PurePosixPath
import tarfile
import tempfile

from research_env.beam.config import BeamConfig
from research_env.core.common import digest, invalid_score, tree_hashes

REQUIRED_KEYS = ('execution_valid', 'primary_loss', 'failure_reason', 'metrics', 'resource_use')
_DROP_KEYS = {'predictions', 'references', 'responses'}


def pack_artifact(artifact_dir) -> tuple[bytes, str]:
    """Controller side: freeze an artifact into (tar bytes, content digest)."""
    root = Path(artifact_dir)
    hashes = tree_hashes(root)
    if 'inference.py' not in hashes:
        raise ValueError('Artifact must include inference.py')
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as tar:
        for rel in hashes:
            tar.add(root/rel, arcname=rel, recursive=False)
    return buf.getvalue(), digest(hashes)


def _unpack_artifact(data: bytes, expected_digest: str, dest: Path) -> Path:
    out = dest/expected_digest
    out.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        for m in tar.getmembers():
            p = PurePosixPath(m.name)
            if p.is_absolute() or '..' in p.parts or not m.isfile():
                raise ValueError('Artifact archive may contain only regular relative files')
        tar.extractall(out, filter='data')
    if digest(tree_hashes(out)) != expected_digest:
        raise ValueError('Frozen artifact hash mismatch')
    return out


def _resolve_manifest(private_root: Path, manifest_rel: str) -> Path:
    root = Path(private_root).resolve()
    path = (root/manifest_rel).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('Manifest must be a file on the private volume')
    return path


def _default_score_fn(family, artifact_dir, manifest, *, backend, image, wall_s, profile):
    try:
        from research_env.rl.api import get_family  # rl-core contract, when present
    except ImportError:
        get_family = None
    if get_family is not None:
        return get_family(family).score(artifact_dir, manifest, backend=backend, image=image,
                                        wall_s=wall_s, profile=profile)
    if family != 'pulse':
        raise ValueError(f'No task adapter available for {family!r}')
    from research_env.core.evaluation import evaluate
    result, _predictions = evaluate(artifact_dir, manifest, backend=backend, image=image, wall_s=wall_s)
    return result


def public_score(result: dict) -> dict:
    """Score dict only: strip prediction/reference payloads, force JSON-safe values."""
    out = {k: v for k, v in dict(result).items() if k not in _DROP_KEYS}
    for key in REQUIRED_KEYS:
        out.setdefault(key, None if key not in ('metrics', 'resource_use') else {})
    if out['execution_valid'] is not True:
        out['execution_valid'], out['primary_loss'] = False, None
    elif not isinstance(out['primary_loss'], (int, float)) or not math.isfinite(out['primary_loss']):
        out.update(execution_valid=False, primary_loss=None, failure_reason='nonfinite_loss')
    return json.loads(json.dumps(out, default=str, allow_nan=False))


def score_payload(artifact_tar: bytes, artifact_digest: str, manifest_rel: str, *, family: str,
                  profile: str, wall_s: float, private_root, participant_image_id: str | None,
                  score_fn=None) -> dict:
    score_fn = score_fn or _default_score_fn
    try:
        manifest = _resolve_manifest(Path(private_root), manifest_rel)
        with tempfile.TemporaryDirectory(prefix='scorer-artifact-') as tmp:
            artifact_dir = _unpack_artifact(artifact_tar, artifact_digest, Path(tmp))
            result = score_fn(family, artifact_dir, manifest, backend='beam', image=participant_image_id,
                              wall_s=wall_s, profile=profile)
    except Exception as exc:
        result = invalid_score(f'scorer_{exc.__class__.__name__}: {exc}'[:500])
    return public_score(result)


def _remote_entry(artifact_tar: bytes, artifact_digest: str, manifest_rel: str, family: str,
                  profile: str, wall_s: float) -> dict:  # pragma: no cover - runs on Beam
    cfg = BeamConfig.from_env()
    return score_payload(artifact_tar, artifact_digest, manifest_rel, family=family, profile=profile,
                         wall_s=wall_s, private_root=cfg.private_mount_path,
                         participant_image_id=cfg.participant_image_id)


def build_scorer(cfg: BeamConfig, sdk):
    """Beam function definition for the trusted scorer (private volume attached)."""
    from research_env.beam.image import participant_image
    cfg.validate()
    env = {'PHYSIOLOGY_BEAM_PARTICIPANT_IMAGE_ID': cfg.participant_image_id or '',
           'PHYSIOLOGY_BEAM_PUBLIC_MOUNT_MODE': cfg.public_mount_mode,
           'PHYSIOLOGY_BEAM_PUBLIC_VOLUME': cfg.public_volume,
           'PHYSIOLOGY_BEAM_PRIVATE_VOLUME': cfg.private_volume,
           'PHYSIOLOGY_BEAM_PRIVATE_MOUNT_PATH': cfg.private_mount_path}
    return sdk.function(app=cfg.scorer_app, name=cfg.scorer_name, cpu=cfg.scorer_cpu,
                        memory=int(cfg.scorer_memory_gib*1024), image=participant_image(cfg, sdk),
                        volumes=[sdk.Volume(name=cfg.private_volume, mount_path=cfg.private_mount_path)],
                        secrets=[cfg.scorer_token_secret], env=env, timeout=cfg.scorer_timeout_s,
                        retries=0)(_remote_entry)


def submit_artifact(scorer, artifact_dir, manifest_rel, *, family, profile, wall_s) -> dict:
    """Controller side: send a frozen artifact to the scorer; accept only a score dict."""
    data, artifact_digest = pack_artifact(artifact_dir)
    result = scorer.remote(data, artifact_digest, manifest_rel, family, profile, wall_s)
    if not isinstance(result, dict) or not all(k in result for k in REQUIRED_KEYS):
        return invalid_score('scorer_returned_malformed_result')
    return public_score(result)


try:  # module-level object for `beam deploy research_env/beam/scorer_app.py:score_artifact`
    import beam as _beam
    score_artifact = build_scorer(BeamConfig.from_env(), _beam)
except Exception:  # beam-client not installed (CI, local tests): definition stays lazy
    score_artifact = None
