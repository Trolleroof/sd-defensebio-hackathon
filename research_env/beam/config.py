"""Beam backend configuration. Names and resources only; never secret values.

Every value can be overridden through environment variables (``PHYSIOLOGY_BEAM_*``)
so the trainer/controller can be configured without code changes. Secrets are
referenced by *Beam workspace secret name* or *local environment variable name*;
this module never reads a secret value.
"""
from __future__ import annotations
from dataclasses import dataclass, field, replace
import os
from pathlib import Path

ENV_PREFIX = 'PHYSIOLOGY_BEAM_'
REPO_ROOT = Path(__file__).resolve().parents[2]

# Same base as research_env/Dockerfile: Python 3.13.3, manifest digest pinned.
DOCKERFILE_BASE_IMAGE = ('docker.io/library/python:3.13.3-slim-bookworm'
                         '@sha256:56a11364ffe0fee3bd60af6d6d5209eba8a99c2c16dc4c7c5861dc06261503cc')
PIP_VERSION = '26.2.1'

PUBLIC_MOUNT_MODES = ('volume', 'cloud_bucket', 'none')


@dataclass(frozen=True)
class BeamConfig:
    # Participant sandbox (untrusted agent code)
    participant_image_id: str | None = None      # output of `python -m research_env.beam.image`
    base_image: str = DOCKERFILE_BASE_IMAGE
    requirements_lock: Path = REPO_ROOT/'research_env'/'requirements.lock'
    max_cpus: float = 8.0                         # contract ceiling
    max_memory_gib: float = 32.0                  # contract ceiling
    ttl_margin_s: int = 120                       # sandbox TTL = timeout + margin (backstop if controller dies)
    workdir: str = '/work'
    control_dir: str = '/tmp/.physiology-control'  # helper/channel files inside the sandbox
    max_sync_bytes: int = 2*1024**3               # per direction
    stream_poll_s: float = 0.5                    # fs.wait_for slice while awaiting a streaming response
    max_file_bytes: int = 16*1024*1024            # RLIMIT_FSIZE, same as docker/local backends
    # Public data mount
    public_mount_mode: str = 'volume'             # 'volume' (not RO-enforced) | 'cloud_bucket' (RO) | 'none'
    public_volume: str = 'physiology-public'
    public_mount_path: str = '/public'
    public_bucket: str | None = None              # existing S3-compatible bucket name (cloud_bucket mode)
    public_bucket_access_key_secret: str = 'PHYSIOLOGY_PUBLIC_BUCKET_ACCESS_KEY'  # Beam secret NAME
    public_bucket_secret_key_secret: str = 'PHYSIOLOGY_PUBLIC_BUCKET_SECRET_KEY'  # Beam secret NAME
    public_bucket_endpoint: str | None = None
    public_bucket_region: str | None = None
    # Trusted scorer (holds private labels)
    scorer_app: str = 'physiology-scorer'
    scorer_name: str = 'score-artifact'
    private_volume: str = 'physiology-private'
    private_mount_path: str = '/private'
    scorer_cpu: float = 2.0
    scorer_memory_gib: float = 8.0
    scorer_timeout_s: int = 3*3600
    scorer_token_secret: str = 'BEAM_TOKEN'       # Beam secret NAME bound to the scorer only
    # Snapshot (filesystem image) registry for fast episode reset
    snapshot_registry: Path = REPO_ROOT/'.research-runs'/'beam-snapshots'
    # Gate: set true only after running the container resource gates on Beam.
    resource_isolation_verified: bool = False
    extra_env: dict = field(default_factory=dict)

    def validate(self):
        if self.public_mount_mode not in PUBLIC_MOUNT_MODES:
            raise ValueError(f'public_mount_mode must be one of {PUBLIC_MOUNT_MODES}')
        if not 0 < self.max_cpus <= 8 or not 0 < self.max_memory_gib <= 32:
            raise ValueError('Contract caps: <=8 CPU, <=32 GiB')
        if self.public_volume == self.private_volume:
            raise ValueError('Public and private volumes must be distinct')
        mounts = [self.public_mount_path, self.private_mount_path, self.workdir, self.control_dir]
        for i, a in enumerate(mounts):
            for b in mounts[i+1:]:
                if _overlaps(a, b):
                    raise ValueError(f'Overlapping sandbox paths: {a} {b}')
        if self.public_mount_mode == 'cloud_bucket' and not self.public_bucket:
            raise ValueError('cloud_bucket mode requires public_bucket')
        return self

    @property
    def public_mount_read_only(self) -> bool:
        # beam.Volume(name, mount_path) has no read-only flag; CloudBucketConfig does.
        return self.public_mount_mode in ('cloud_bucket', 'none')

    @classmethod
    def from_env(cls, environ=None, **overrides) -> 'BeamConfig':
        environ = os.environ if environ is None else environ
        values = {}
        for name, f in cls.__dataclass_fields__.items():
            key = ENV_PREFIX + name.upper()
            if key not in environ or name == 'extra_env':
                continue
            raw, default = environ[key], f.default
            if isinstance(default, bool):
                values[name] = raw.strip().lower() in ('1', 'true', 'yes')
            elif isinstance(default, int):
                values[name] = int(raw)
            elif isinstance(default, float):
                values[name] = float(raw)
            elif isinstance(default, Path):
                values[name] = Path(raw)
            else:
                values[name] = raw or None
        return replace(cls(), **values, **overrides).validate()


def _overlaps(a, b):
    a, b = a.rstrip('/') + '/', b.rstrip('/') + '/'
    return a.startswith(b) or b.startswith(a)
