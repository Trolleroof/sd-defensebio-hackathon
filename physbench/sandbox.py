"""Process isolation for agent code on a single Linux host (e.g. a RunPod container).

What is enforced:
- every episode runs as its own unprivileged Unix user; its workspace is mode 700;
- the repository, processed data, hidden labels and API keys live under root-only paths;
- environment is stripped (no API keys), CPU/memory/file-size limits, wall-clock timeouts.

What is best-effort (the container cannot create network namespaces):
- network: proxy env vars point at a dead port, curl/wget are not executable for agents,
  and a .pth startup hook in the agent Python blocks non-loopback sockets. A determined
  process could bypass this, so commands are also audited (see agent.audit_command).
"""

from __future__ import annotations

import os
import pwd
import resource
import shutil
import signal
import subprocess
import sys
from pathlib import Path

AGENT_ENV_DIR = Path("/opt/pbenv")
AGENT_PY = AGENT_ENV_DIR / "bin" / "python"
SANDBOX_ROOT = Path("/srv/pb")
AGENT_PACKAGES = ["numpy", "scipy", "pandas", "scikit-learn", "statsmodels", "matplotlib"]

NETGUARD = '''
import socket as _s
_orig = _s.socket.connect
def _guard(self, addr, *a, **k):
    host = addr[0] if isinstance(addr, tuple) else addr
    if isinstance(host, str) and (host.startswith("127.") or host in ("localhost", "::1") or host.startswith("/")):
        return _orig(self, addr, *a, **k)
    raise PermissionError("network access is disabled in this environment")
_s.socket.connect = _guard
'''


def setup_host(repo_root: Path) -> None:
    """Idempotent host preparation. Must run as root."""
    if os.geteuid() != 0:
        raise SystemExit("setup must run as root")
    if not AGENT_PY.exists():
        subprocess.run(["uv", "venv", "-q", "--python", "/usr/bin/python3", str(AGENT_ENV_DIR)], check=True)
        subprocess.run(["uv", "pip", "install", "-q", "--python", str(AGENT_PY), *AGENT_PACKAGES], check=True)
    site = next(AGENT_ENV_DIR.glob("lib/python*/site-packages"))
    # A .pth import line runs at every interpreter start (Ubuntu's own sitecustomize would shadow ours).
    (site / "_pb_netguard.py").write_text(NETGUARD)
    (site / "00_pb_netguard.pth").write_text("import _pb_netguard\n")
    (site / "sitecustomize.py").unlink(missing_ok=True)
    subprocess.run(["chmod", "-R", "go-w", str(AGENT_ENV_DIR)], check=True)
    subprocess.run(["chmod", "-R", "a+rX", str(AGENT_ENV_DIR)], check=True)
    for tool in ("curl", "wget"):
        p = shutil.which(tool)
        if p:
            os.chmod(p, 0o750)
    # Root-only repository (hidden labels, raw data, API key).
    os.chmod(repo_root, 0o700)
    if repo_root.parent != Path("/"):
        os.chmod(repo_root.parent, 0o711)
    SANDBOX_ROOT.mkdir(parents=True, exist_ok=True)
    os.chmod(SANDBOX_ROOT, 0o711)


def ensure_user(name: str) -> tuple[int, int]:
    try:
        pw = pwd.getpwnam(name)
    except KeyError:
        subprocess.run(["useradd", "--no-create-home", "--shell", "/bin/bash", name], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        pw = pwd.getpwnam(name)
    return pw.pw_uid, pw.pw_gid


def chown_tree(path: Path, uid: int, gid: int) -> None:
    for root, dirs, files in os.walk(path):
        os.chown(root, uid, gid)
        for f in files:
            fp = os.path.join(root, f)
            if not os.path.islink(fp):
                os.chown(fp, uid, gid)


def _limits(mem_gb: float, cpu_s: int):
    def fn():
        os.setsid()
        mem = int(mem_gb * 1024 ** 3)
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
        resource.setrlimit(resource.RLIMIT_FSIZE, (512 * 1024 ** 2, 512 * 1024 ** 2))
        resource.setrlimit(resource.RLIMIT_NPROC, (512, 512))
    return fn


def agent_env(home: Path, threads: int = 4) -> dict:
    dead = "http://127.0.0.1:9"
    return {
        "PATH": f"{AGENT_ENV_DIR}/bin:/usr/local/bin:/usr/bin:/bin",
        "HOME": str(home),
        "LANG": "C.UTF-8",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "MPLBACKEND": "Agg",
        "OMP_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "http_proxy": dead, "https_proxy": dead, "HTTP_PROXY": dead, "HTTPS_PROXY": dead,
        "no_proxy": "", "PIP_NO_INDEX": "1",
    }


def run_as(user: str, command: list[str] | str, cwd: Path, timeout: float, mem_gb: float = 16,
           max_chars: int = 6000) -> tuple[int, str, float]:
    """Run as the episode user. Returns (returncode, combined output truncated, seconds)."""
    import time
    uid, gid = ensure_user(user)
    args = ["bash", "-c", command] if isinstance(command, str) else command
    t0 = time.time()
    proc = subprocess.Popen(args, cwd=cwd, env=agent_env(cwd), user=uid, group=gid, extra_groups=[],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            preexec_fn=_limits(mem_gb, int(timeout * 4) + 10))
    try:
        out, _ = proc.communicate(timeout=timeout)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        out, _ = proc.communicate()
        rc = -9
        out = (out or b"") + f"\n[killed: exceeded {timeout:.0f}s time limit]".encode()
    text = (out or b"").decode("utf-8", "replace")
    if len(text) > max_chars:
        half = max_chars // 2
        text = text[:half] + f"\n... [{len(text) - max_chars} chars truncated] ...\n" + text[-half:]
    return rc, text, time.time() - t0


def is_linux_root() -> bool:
    return sys.platform.startswith("linux") and os.geteuid() == 0
