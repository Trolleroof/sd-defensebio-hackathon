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


# seccomp: deny internet/packet sockets and io_uring without privileges (works inside unprivileged
# containers such as RunPod pods, where a network namespace cannot be created).
_SECCOMP_ARCH = {'x86_64': (0xC000003E, 41, 425, True), 'aarch64': (0xC00000B7, 198, 425, False)}
_DENIED_FAMILIES = (2, 10, 17)  # AF_INET, AF_INET6, AF_PACKET


def _socket_filter_program(machine):
    """Classic BPF for seccomp: kill on foreign arch (and x32), EACCES for denied socket families/io_uring."""
    audit_arch, socket_nr, uring_nr, x32 = _SECCOMP_ARCH[machine]
    LD, JEQ, JGE, RET = 0x20, 0x15, 0x35, 0x06
    prog = [('LD', 4), ('JEQ', audit_arch, None, 'KILL'), ('LD', 0)]
    if x32:
        prog.append(('JGE', 0x40000000, 'KILL', None))
    prog += [('JEQ', socket_nr, None, 'URING'), ('LD', 16)]
    prog += [('JEQ', family, 'DENY', None) for family in _DENIED_FAMILIES]
    prog += [('RET', 0x7FFF0000), ('label', 'URING'), ('LD', 0), ('JEQ', uring_nr, 'DENY', 'ALLOW'),
             ('label', 'ALLOW'), ('RET', 0x7FFF0000), ('label', 'DENY'), ('RET', 0x00050000 | 13),
             ('label', 'KILL'), ('RET', 0x80000000)]
    labels, ops = {}, []
    for op in prog:
        if op[0] == 'label':
            labels[op[1]] = len(ops)
        else:
            ops.append(op)
    out = []
    for i, op in enumerate(ops):
        if op[0] in ('LD', 'RET'):
            out.append(({'LD': LD, 'RET': RET}[op[0]], 0, 0, op[1]))
        else:
            jump = lambda target: 0 if target is None else labels[target]-(i+1)
            out.append(({'JEQ': JEQ, 'JGE': JGE}[op[0]], jump(op[2]), jump(op[3]), op[1]))
    return out


def _install_socket_filter():
    import ctypes
    import platform

    class SockFilter(ctypes.Structure):
        _fields_ = [('code', ctypes.c_uint16), ('jt', ctypes.c_uint8), ('jf', ctypes.c_uint8), ('k', ctypes.c_uint32)]

    class SockFprog(ctypes.Structure):
        _fields_ = [('len', ctypes.c_uint16), ('filter', ctypes.POINTER(SockFilter))]

    program = _socket_filter_program(platform.machine())
    filters = (SockFilter*len(program))(*[SockFilter(*ins) for ins in program])
    libc = ctypes.CDLL(None, use_errno=True)
    PR_SET_NO_NEW_PRIVS, PR_SET_SECCOMP, SECCOMP_MODE_FILTER = 38, 22, 2
    if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'PR_SET_NO_NEW_PRIVS failed')
    fprog = SockFprog(len(program), filters)
    if libc.prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, ctypes.byref(fprog), 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'seccomp filter install failed')


_PROBE_SECCOMP = (
    'import socket, sys\nsys.path.insert(0, {repo!r})\nfrom research_env.core.worker import _install_socket_filter\n'
    '_install_socket_filter()\ntry:\n    socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n'
    'except PermissionError:\n    raise SystemExit(0)\nraise SystemExit(3)\n')


@lru_cache(maxsize=1)
def linux_pod_capabilities():
    """Probe once what this Linux container allows. Fails closed: no network isolation, no run."""
    caps = {'linux': sys.platform.startswith('linux'), 'uid_drop': False, 'network': None}
    if not caps['linux']:
        return caps
    caps['uid_drop'] = os.geteuid() == 0
    import platform
    repo = str(Path(__file__).resolve().parents[2])
    # Probe with this (controller) interpreter: it is the one that installs the filter in preexec.
    probes = [('netns', 'import os; os.unshare(os.CLONE_NEWNET)')]
    if platform.machine() in _SECCOMP_ARCH:
        probes.append(('seccomp', _PROBE_SECCOMP.format(repo=repo)))
    for name, code in probes:
        if subprocess.run([sys.executable, '-I', '-c', code], capture_output=True, timeout=30).returncode == 0:
            caps['network'] = name
            break
    if caps['network'] is None and os.environ.get('PHYSIOLOGY_ALLOW_NETWORK') != '1':
        raise RuntimeError('linux-pod: cannot block network (no netns, no seccomp); refusing to run agent code. '
                           'Set PHYSIOLOGY_ALLOW_NETWORK=1 only for non-scored debugging.')
    return caps


def sandbox_ids(root):
    """Distinct uid/gid per workspace or replay root, so concurrent episodes cannot read each other."""
    import hashlib
    base = int(os.environ.get('PHYSIOLOGY_SANDBOX_UID_BASE', 200_000))
    uid = base + int(hashlib.sha256(str(Path(root).resolve()).encode()).hexdigest(), 16) % 100_000
    return uid, uid


def sandbox_limits(cpus, memory_gb):
    """Pod-wide caps (env-wide, never per episode) so concurrent sandboxes fit the pod."""
    return (min(cpus, int(os.environ.get('PHYSIOLOGY_SANDBOX_MAX_CPUS', cpus))),
            min(memory_gb, float(os.environ.get('PHYSIOLOGY_SANDBOX_MAX_MEMORY_GB', memory_gb))))


def linux_pod_preexec(pod, memory_gb, ids):
    """Runs in the child before exec: network block, memory/process caps, drop to the per-root uid."""
    import resource
    memory = int(memory_gb*1024**3)
    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    if pod['network'] == 'netns':
        os.unshare(os.CLONE_NEWNET)
    if pod['uid_drop']:
        resource.setrlimit(resource.RLIMIT_NPROC, (128, 128))  # per-uid, so only meaningful after the drop
        os.setgroups([])
        os.setgid(ids[1])
        os.setuid(ids[0])
    if pod['network'] == 'seccomp':
        _install_socket_filter()


def worker_python(backend):
    """Interpreter for replay entry scripts. linux-pod may use a separate env venv (trainer venv differs)."""
    if backend == 'linux-pod':
        return os.environ.get('PHYSIOLOGY_SANDBOX_PYTHON', sys.executable)
    return sys.executable if backend == 'macos-development' else 'python'


def _chown_tree(root, uid, gid):
    for path in [root, *root.rglob('*')]:
        os.chown(path, uid, gid, follow_symlinks=False)


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
    if backend == 'beam':
        # Remote Beam sandbox (network blocked, public mount only); limits are not
        # capped by this host. See research_env/beam/sandbox.py.
        from research_env.beam.sandbox import execute_beam
        return execute_beam(root, command, timeout_s=timeout_s, cpus=cpus, memory_gb=memory_gb,
                            image=image, on_tick=on_tick, requests=requests)
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
    if backend == 'linux-pod':
        cpus, memory_gb = sandbox_limits(cpus, memory_gb)
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
        elif backend == 'linux-pod':
            # RunPod-style containers cannot run nested docker. Training-rollout isolation only:
            # dedicated uid (private files must be 0700/root-owned), rlimits, and a fresh network
            # namespace when the container permits it. Never scored-isolated.
            pod = linux_pod_capabilities()
            if not pod['linux']:
                raise RuntimeError('linux-pod backend requires Linux')
            ids = sandbox_ids(root)
            if pod['uid_drop']:
                os.chown(tmp, *ids)
                _chown_tree(root, *ids)
                root.chmod(0o700)  # sibling episodes (other uids) cannot even list it
            cmd = list(command)
            env.update(PATH=str(Path(worker_python(backend)).parent)+':/usr/local/bin:/usr/bin:/bin',
                       HOME=str(tmp), TMPDIR=str(tmp))
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
            if backend == 'linux-pod':
                linux_pod_preexec(pod, memory_gb, ids)

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
        if backend == 'linux-pod':
            import resource
            peak_rss_bytes = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*1024  # cumulative max
        result_extra = ({'network_isolation': pod['network'], 'uid_dropped': pod['uid_drop'],
                         'sandbox_uid': ids[0] if pod['uid_drop'] else None}
                        if backend == 'linux-pod' else {})
        return {'execution_valid': exit_code == 0 and not timed_out and error is None,
                'failure_reason': error or ('wall_budget_exhausted' if timed_out else None if exit_code == 0 else 'worker_failed'),
                **result_extra,
                'peak_rss_bytes_sampled': peak_rss_bytes if backend in ('macos-development', 'linux-pod') else None,
                'exit_code': exit_code, 'wall_s': time.monotonic()-start,
                'stdout': (tmp/'stdout').read_text(errors='replace')[-65536:],
                'stderr': (tmp/'stderr').read_text(errors='replace')[-65536:],
                'responses': responses, 'backend': backend, 'scored_resource_isolation': backend == 'docker',
                'limits': {'cpus': cpus, 'memory_gb': memory_gb, 'timeout_s': timeout_s}}
