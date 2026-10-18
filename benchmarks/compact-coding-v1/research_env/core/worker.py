"""Offline bounded command execution. Scored runs require the container backend."""
from __future__ import annotations
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import json
import re
from functools import lru_cache
import time
from research_env.core.common import tree_hashes


@lru_cache(maxsize=1)
def _docker_allocation():
    values = subprocess.check_output(['docker', 'info', '--format', '{{.NCPU}} {{.MemTotal}}'],
                                     timeout=15, text=True).split()
    return int(values[0]), int(values[1])/1024**3


def _quote(path):
    return '"' + str(path).replace('\\', '\\\\').replace('"', '\\"') + '"'


def execute(workspace, command, *, timeout_s, cpus=8, memory_gb=32,
            backend='docker', image=None, on_tick=None, requests=None):
    """workspace contains ONLY permitted data/source/artifacts. No private mounts.

    Child logs are file bounded; process groups are killed on timeout. An isolated
    Docker cgroup accounts descendants. Local mode is a development aid: macOS
    memory limits are reported but cannot be enforced reliably.
    """
    root = Path(workspace).resolve()
    tree_hashes(root)  # Reject symlinks, including indirect private-file access.
    if not command or any(not isinstance(x, str) for x in command):
        raise ValueError('Command must be a nonempty argv list')
    actual_cpus = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
    actual_memory = 32.
    if sys.platform == 'darwin':
        actual_memory = int(subprocess.check_output(['/usr/sbin/sysctl', '-n', 'hw.memsize']))/1024**3
    elif Path('/sys/fs/cgroup/memory.max').exists():
        cap = Path('/sys/fs/cgroup/memory.max').read_text().strip()
        if cap != 'max':
            actual_memory = int(cap)/1024**3
    if backend == 'docker':
        docker_cpus, docker_memory = _docker_allocation()
        actual_cpus, actual_memory = min(actual_cpus, docker_cpus), min(actual_memory, docker_memory)
    cpus = min(cpus, actual_cpus, 8)
    memory_gb = min(memory_gb, 32, actual_memory)
    if timeout_s <= 0 or cpus < 1 or memory_gb <= 0:
        raise ValueError('Invalid limits')
    env = {'PATH': '/usr/local/bin:/usr/bin:/bin', 'HOME': '/work', 'TMPDIR': '/tmp',
           'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1',
           **{k: str(cpus) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}}
    start = time.monotonic()
    container_name = 'physiology-' + os.urandom(12).hex()
    with tempfile.TemporaryDirectory(prefix='research-worker-') as tmp:
        tmp = Path(tmp).resolve()
        if backend == 'docker':
            if not image or not re.fullmatch(r'(?:[A-Za-z0-9._:/-]+@)?sha256:[0-9a-f]{64}', image):
                raise ValueError('Explicit digest-pinned image required')
            cmd = [shutil.which('docker') or 'docker', 'run', '--rm', '--pull=never', '--name', container_name,
                   '--network=none', '--read-only', '--cap-drop=ALL',
                   '--security-opt=no-new-privileges', '--pids-limit=128',
                   '--cpus', str(cpus), '--memory', str(int(memory_gb*1024**3)), '--memory-swap', str(int(memory_gb*1024**3)),
                   '--user', f'{os.getuid()}:{os.getgid()}', '--tmpfs', '/tmp:rw,noexec,nosuid,size=256m',
                   '--mount', f'type=bind,source={root},target=/work', '-w', '/work']
            if requests is not None:
                cmd.insert(2, '-i')
            for key, value in env.items():
                cmd.extend(['-e', f'{key}={value}'])
            cmd.extend([image, *command])
        elif backend == 'macos-development':
            if sys.platform != 'darwin' or not shutil.which('sandbox-exec'):
                raise RuntimeError('macOS sandbox unavailable')
            # Allow only interpreter/library locations and this staged workspace.
            # No broad /Users or home-directory read permission.
            allowed = {Path(sys.base_prefix).resolve(), Path(sys.prefix).resolve(),
                       Path(sys.executable).resolve().parent.parent, root, tmp}
            profile = '(version 1)\n(deny default)\n(allow process*)\n(deny process-fork)\n(allow sysctl-read)\n(allow mach-lookup)\n(allow file-map-executable)\n(allow file-read-metadata)\n(allow file-read* (literal "/"))\n'
            for path in [*allowed, Path('/System'), Path('/usr'), Path('/bin'), Path('/dev'), Path('/Library/Apple')]:
                profile += f'(allow file-read* (subpath {_quote(path)}))\n'
            profile += f'(allow file-write* (subpath {_quote(root)}) (subpath {_quote(tmp)}) (literal "/dev/null"))\n'
            profile += '(deny network*)\n'
            profile_path = tmp/'worker.sb'
            profile_path.write_text(profile)
            cmd = ['sandbox-exec', '-f', str(profile_path), *command]
            env.update(PATH=str(Path(sys.executable).parent)+':/usr/bin:/bin', HOME=str(tmp), TMPDIR=str(tmp))
        else:
            raise ValueError('Unknown worker backend')

        def limits():
            import resource
            resource.setrlimit(resource.RLIMIT_CPU, (max(1, int(timeout_s*cpus)),)*2)
            resource.setrlimit(resource.RLIMIT_FSIZE, (16*1024*1024,)*2)

        peak_rss_bytes = 0
        last_sample = 0.
        timed_out = False
        error = None
        responses, stream_errors = [], []
        streaming = requests is not None
        with (tmp/'stdout').open('wb') as stdout, (tmp/'stderr').open('wb') as stderr:
            try:
                proc = subprocess.Popen(cmd, cwd=root, env=None if backend == 'docker' else env, stdin=subprocess.PIPE if streaming else subprocess.DEVNULL,
                                        stdout=subprocess.PIPE if streaming else stdout, stderr=stderr, start_new_session=True,
                                        preexec_fn=limits)
                def exchange():
                    try:
                        for request in requests:
                            proc.stdin.write(json.dumps(request, allow_nan=False).encode()+b'\n')
                            proc.stdin.flush()
                            line = proc.stdout.readline(1_000_001)
                            if not line or len(line) > 1_000_000:
                                raise ValueError('Missing or oversized streaming response')
                            responses.append(json.loads(line))
                        proc.stdin.close()
                        if proc.stdout.read(1):
                            raise ValueError('Extra streaming output')
                    except Exception as exc:
                        stream_errors.append(str(exc))
                        if proc.poll() is None:
                            os.killpg(proc.pid, signal.SIGKILL)
                if streaming:
                    exchange_thread = threading.Thread(target=exchange, daemon=True)
                    exchange_thread.start()
                try:
                    while proc.poll() is None:
                        if backend == 'macos-development' and time.monotonic()-last_sample >= .25:
                            sample = subprocess.run(['/bin/ps', '-o', 'rss=', '-p', str(proc.pid)],
                                                    capture_output=True, text=True, timeout=2)
                            if sample.stdout.strip().isdigit():
                                peak_rss_bytes = max(peak_rss_bytes, int(sample.stdout.strip())*1024)
                            last_sample = time.monotonic()
                        if on_tick:
                            on_tick()
                        if time.monotonic()-start >= timeout_s:
                            timed_out = True
                            break
                        time.sleep(.05)
                finally:
                    if proc.poll() is None:
                        os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                if streaming:
                    exchange_thread.join(timeout=2)
                    if exchange_thread.is_alive():
                        stream_errors.append('Streaming protocol did not finish')
                if streaming:
                    proc.stdin.close()
                    proc.stdout.close()
                exit_code = proc.returncode
                if stream_errors:
                    error = '; '.join(stream_errors)
            except OSError as exc:
                exit_code, error = None, str(exc)
            finally:
                if backend == 'docker':
                    subprocess.run(['docker', 'rm', '-f', container_name], stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, timeout=15, check=False)
        stdout_text = (tmp/'stdout').read_text(errors='replace')
        stderr_text = (tmp/'stderr').read_text(errors='replace')
        return {'execution_valid': exit_code == 0 and not timed_out and error is None,
                'failure_reason': error or ('wall_budget_exhausted' if timed_out else None if exit_code == 0 else 'worker_failed'),
                'peak_rss_bytes_sampled': peak_rss_bytes if backend == 'macos-development' else None,
                'exit_code': exit_code, 'wall_s': time.monotonic()-start,
                'stdout': stdout_text[-65536:],
                'stderr': stderr_text[-65536:],
                'stdout_truncated': len(stdout_text) > 65536,
                'stderr_truncated': len(stderr_text) > 65536,
                'log_tail_character_limit': 65536,
                'responses': responses, 'backend': backend, 'scored_resource_isolation': backend == 'docker',
                'limits': {'cpus': cpus, 'memory_gb': memory_gb, 'timeout_s': timeout_s}}
