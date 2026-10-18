"""Beam participant-sandbox backend for research_env.core.worker.execute.

One fresh Beam Sandbox per call: outbound network blocked, no secrets, no local
directory sync, CPU/memory from the contract caps, only the *public* mount
attached. The staged workspace is synced in, the command runs with a wall
timeout, changed files are synced back, and the sandbox is terminated on every
path (including exceptions). The private volume is never referenced here.
"""
from __future__ import annotations
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import tempfile
import time

from research_env.beam.config import BeamConfig
from research_env.core.common import tree_hashes

HELPER_SOURCE = Path(__file__).with_name('remote_helper.py')
_IMAGE_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9._:-]{0,200}')
OUTPUT_TAIL = 65536


def _load_sdk():
    import beam  # beam-client; imported lazily so other backends never need it
    return beam


def sandbox_env(cpus, cfg: BeamConfig):
    threads = str(max(1, int(cpus)))
    return {'PATH': '/usr/local/bin:/usr/bin:/bin', 'HOME': cfg.workdir, 'TMPDIR': '/tmp',
            'CUDA_VISIBLE_DEVICES': '-1', 'PYTHONDONTWRITEBYTECODE': '1', 'MPLCONFIGDIR': '/tmp/matplotlib',
            **{k: threads for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
            **cfg.extra_env}


def public_mounts(cfg: BeamConfig, sdk):
    if cfg.public_mount_mode == 'none':
        return []
    if cfg.public_mount_mode == 'volume':
        mounts = [sdk.Volume(name=cfg.public_volume, mount_path=cfg.public_mount_path)]
    else:
        bucket_cfg = sdk.CloudBucketConfig(read_only=True,
                                           access_key=cfg.public_bucket_access_key_secret,
                                           secret_key=cfg.public_bucket_secret_key_secret,
                                           endpoint=cfg.public_bucket_endpoint,
                                           region=cfg.public_bucket_region)
        mounts = [sdk.CloudBucket(name=cfg.public_bucket, mount_path=cfg.public_mount_path, config=bucket_cfg)]
    for m in mounts:
        if getattr(m, 'name', None) == cfg.private_volume or _under(getattr(m, 'mount_path', ''), cfg.private_mount_path):
            raise RuntimeError('Refusing to mount private storage into a participant sandbox')
    return mounts


def _under(path, parent):
    path, parent = str(path).rstrip('/') + '/', str(parent).rstrip('/') + '/'
    return path.startswith(parent)


def sandbox_kwargs(cfg: BeamConfig, sdk, *, image_id, cpus, memory_gb, timeout_s, env):
    return dict(cpu=cpus, memory=int(memory_gb*1024),  # int memory is MiB in beam-client
                image=sdk.Image.from_id(image_id),
                keep_warm_seconds=int(math.ceil(timeout_s)) + cfg.ttl_margin_s,
                volumes=public_mounts(cfg, sdk), secrets=[], env=dict(env),
                block_network=True, sync_local_dir=False, docker_enabled=False, ports=[],
                authorized=False)


def baseline_manifest(cfg: BeamConfig, image_id):
    """Workspace hashes already present in a reset image (see snapshots.py)."""
    path = Path(cfg.snapshot_registry)/f'{image_id.replace(":", "_")}.json'
    if path.is_file():
        return json.loads(path.read_text())['workspace_hashes']
    return {}


def _pack(root: Path, rels, out: Path):
    with tarfile.open(out, 'w:gz') as tar:
        for rel in rels:
            tar.add(root/rel, arcname=rel, recursive=False)


def _safe_rel(name):
    p = PurePosixPath(name)
    if p.is_absolute() or not p.parts or any(x in ('..', '') for x in p.parts):
        raise ValueError(f'Unsafe path from sandbox: {name!r}')
    return Path(*p.parts)


def _unpack_back(tar_path: Path, deleted, root: Path, max_bytes):
    with tempfile.TemporaryDirectory(prefix='beam-sync-') as staging:
        staging = Path(staging)
        with tarfile.open(tar_path) as tar:
            members = tar.getmembers()
            if sum(m.size for m in members) > max_bytes:
                raise ValueError('workspace_sync_too_large')
            for m in members:
                _safe_rel(m.name)
                if not m.isfile():
                    raise ValueError('Only regular files may sync back')
            tar.extractall(staging, filter='data')
        for m in members:
            rel = _safe_rel(m.name)
            target = root/rel
            for parent in [target, *target.parents]:
                if parent == root:
                    break
                if parent.is_symlink():
                    raise ValueError('Symlink in workspace')
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(staging/rel, target)
    for name in deleted:
        target = root/_safe_rel(name)
        if target.is_file() and not target.is_symlink():
            target.unlink()


def _read_stream(stream):
    try:
        data = stream.read()
    except Exception:
        return ''
    if isinstance(data, bytes):
        data = data.decode(errors='replace')
    return (data or '')[-OUTPUT_TAIL:]


def execute_beam(workspace, command, *, timeout_s, cpus=8, memory_gb=32, image=None,
                 on_tick=None, requests=None, config: BeamConfig | None = None, sdk=None):
    cfg = (config or BeamConfig.from_env()).validate()
    root = Path(workspace).resolve()
    local_hashes = tree_hashes(root)  # rejects symlinks
    if not command or any(not isinstance(x, str) for x in command):
        raise ValueError('Command must be a nonempty argv list')
    cpus = min(cpus, cfg.max_cpus, 8)
    memory_gb = min(memory_gb, cfg.max_memory_gib, 32)
    if timeout_s <= 0 or cpus < 1 or memory_gb <= 0:
        raise ValueError('Invalid limits')
    image_id = image or cfg.participant_image_id
    if not image_id or not _IMAGE_ID.fullmatch(image_id):
        raise ValueError('Explicit Beam image id required (see research_env.beam.image)')
    sdk = sdk or _load_sdk()
    env = sandbox_env(cpus, cfg)
    start = time.monotonic()
    deadline = start + timeout_s
    remaining = lambda: deadline - time.monotonic()
    ctl, chan = cfg.control_dir, cfg.control_dir + '/chan'
    helper = ctl + '/helper.py'
    fsize, cpu_s = str(cfg.max_file_bytes), str(max(1, int(timeout_s*cpus)))
    streaming = requests is not None
    sb = proc = None
    timed_out, error, exit_code, cleanup_error = False, None, None, None
    responses, stdout, stderr, sandbox_id = [], '', '', None

    def tick():
        if on_tick:
            on_tick()

    def run_helper(*args, limit=300):
        p = sb.process.exec('python', '-I', helper, *args, cwd='/tmp', env=env)
        code = p.wait(timeout=limit)
        if code != 0:
            raise RuntimeError(f'helper {args[0]} failed: {_read_stream(p.stderr)[-2000:] or _read_stream(p.stdout)[-2000:]}')
        return _read_stream(p.stdout)

    with tempfile.TemporaryDirectory(prefix='beam-worker-') as tmp:
        tmp = Path(tmp)
        try:
            sb = sdk.Sandbox(**sandbox_kwargs(cfg, sdk, image_id=image_id, cpus=cpus, memory_gb=memory_gb,
                                              timeout_s=timeout_s, env=env)).create()
            sandbox_id = getattr(sb, 'sandbox_id', None)
            # Sync workspace in: only files absent from / different to the reset image.
            base = baseline_manifest(cfg, image_id)
            upload = [rel for rel, h in local_hashes.items() if base.get(rel) != h]
            if sum((root/rel).stat().st_size for rel in upload) > cfg.max_sync_bytes:
                raise ValueError('workspace_sync_too_large')
            _pack(root, upload, tmp/'in.tar.gz')
            (tmp/'deleted.json').write_text(json.dumps(sorted(set(base)-set(local_hashes))))
            sb.fs.create_directory(ctl)
            sb.fs.write_text(helper, HELPER_SOURCE.read_text())
            sb.fs.upload_file(str(tmp/'in.tar.gz'), ctl + '/in.tar.gz')
            sb.fs.upload_file(str(tmp/'deleted.json'), ctl + '/deleted.json')
            run_helper('apply', ctl + '/in.tar.gz', ctl + '/deleted.json', cfg.workdir)
            if remaining() <= 0:
                timed_out = True
                raise TimeoutError
            if streaming:
                sb.fs.create_directory(chan)
                proc = sb.process.exec('python', '-I', helper, 'bridge', chan, fsize, cpu_s, '--', *command,
                                       cwd=cfg.workdir, env=env)
                stream_errors = []
                for i, request in enumerate(requests):
                    sb.fs.write_text(f'{chan}/req-{i}.json', json.dumps(request, allow_nan=False) + '\n')
                    sb.fs.write_text(f'{chan}/req-{i}.ready', '')
                    while True:
                        if remaining() <= 0:
                            timed_out = True
                            break
                        try:
                            sb.fs.wait_for(f'{chan}/resp-{i}.json', timeout=min(cfg.stream_poll_s, max(remaining(), .01)))
                            break
                        except Exception:
                            tick()
                            if proc.status()[0] >= 0:
                                stream_errors.append(_channel_error(sb, chan) or 'Missing or oversized streaming response')
                                break
                    if timed_out or stream_errors:
                        break
                    line = sb.fs.read_text(f'{chan}/resp-{i}.json')
                    if len(line) > 1_000_000:
                        stream_errors.append('Missing or oversized streaming response')
                        break
                    try:
                        responses.append(json.loads(line))
                    except ValueError as exc:
                        stream_errors.append(str(exc))
                        break
                    tick()
                if not timed_out and not stream_errors:
                    sb.fs.write_text(f'{chan}/eof', '')
                exit_code = _poll(proc, remaining, tick, cfg) if not (timed_out or stream_errors) else None
                if exit_code is None and not stream_errors:
                    timed_out = True
                if not stream_errors and exit_code not in (None, 0):
                    reason = _channel_error(sb, chan)
                    if reason:
                        stream_errors.append(reason)
                if stream_errors:
                    error = '; '.join(stream_errors)
            else:
                proc = sb.process.exec('python', '-I', helper, 'run', fsize, cpu_s, '--', *command,
                                       cwd=cfg.workdir, env=env)
                exit_code = _poll(proc, remaining, tick, cfg)
                timed_out = exit_code is None
            if proc is not None and (timed_out or error):
                try:
                    proc.kill()
                except Exception:
                    pass
            if proc is not None:
                stdout, stderr = _read_stream(proc.stdout), _read_stream(proc.stderr)
            if streaming:
                # Replay roots are throwaway trusted-side temp dirs: nothing flows back.
                raise _Done
            # Sync back (bind-mount parity): also after timeouts, like docker.
            sb.fs.write_text(ctl + '/before.json', json.dumps(local_hashes))
            summary = json.loads(run_helper('collect', ctl + '/before.json', ctl + '/out.tar.gz', cfg.workdir,
                                            str(cfg.max_sync_bytes)).strip().splitlines()[-1])
            if 'error' in summary:
                raise ValueError(summary['error'])
            sb.fs.download_file(ctl + '/out.tar.gz', str(tmp/'out.tar.gz'))
            _unpack_back(tmp/'out.tar.gz', summary['deleted'], root, cfg.max_sync_bytes)
        except _Done:
            pass
        except TimeoutError:
            timed_out = True
        except Exception as exc:
            error = error or f'beam_{exc.__class__.__name__}: {exc}'[:2000]
        finally:
            if sb is not None:
                try:
                    if sb.terminate() is False:
                        cleanup_error = 'terminate_returned_false'
                except Exception as exc:
                    cleanup_error = f'terminate_failed: {exc.__class__.__name__}'
    valid = exit_code == 0 and not timed_out and error is None
    return {'execution_valid': valid,
            'failure_reason': error or ('wall_budget_exhausted' if timed_out else None if exit_code == 0 else 'worker_failed'),
            'peak_rss_bytes_sampled': None, 'exit_code': exit_code, 'wall_s': time.monotonic()-start,
            'stdout': stdout, 'stderr': stderr, 'responses': responses, 'backend': 'beam',
            'scored_resource_isolation': cfg.resource_isolation_verified,
            'public_mount_read_only': cfg.public_mount_read_only,
            'sandbox_id': sandbox_id, 'image': image_id, 'cleanup_error': cleanup_error,
            'limits': {'cpus': cpus, 'memory_gb': memory_gb, 'timeout_s': timeout_s}}


class _Done(Exception):
    pass


def _poll(proc, remaining, tick, cfg):
    """Exit code, or None when the wall budget expires first."""
    while True:
        code = proc.status()[0]
        if code >= 0:
            return code
        if remaining() <= 0:
            return None
        tick()
        time.sleep(min(0.25, max(remaining(), 0)))


def _channel_error(sb, chan):
    try:
        return sb.fs.read_text(f'{chan}/error').strip() or None
    except Exception:
        return None
