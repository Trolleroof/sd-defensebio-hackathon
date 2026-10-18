"""INTERIM isolated worker for heat artifacts (replace with Set 1 research_env/core).

Runs participant ``train`` / ``predict`` in a fresh Python process under macOS
``sandbox-exec``: reads of /Users, /tmp, /var/folders and the controller's
private root are denied except the worker's own temp directory; network and
forks are denied; CPU, data-segment and open-file limits are applied. The
controller process never imports or unpickles participant code or artifacts:
it exchanges JSON only. Execution fails closed when sandbox-exec is absent.
Like benchmark/executor.py this is a local prototype, not a hardened
multi-tenant sandbox: Python startup needs read access to system paths.
"""
from __future__ import annotations

import json
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

LIMITS = {
    "train_wall_s": 900,
    "predict_wall_s": 300,
    "cpu_threads": 8,
    "memory_bytes": 16 * 1024 ** 3,        # min(32 GB brief default, host 24 GB) less headroom
    "max_output_bytes": 20_000_000,
    "max_artifact_bytes": 200_000_000,
    "max_source_bytes": 5_000_000,
    "network": "denied",
    "gpu": "not available; CPU only",
}
TEMPLATE_DIR = Path(__file__).with_name("workspace_template")

_CHILD = r'''
import importlib.util, json, sys
root, mode = sys.argv[1], sys.argv[2]
sys.path.insert(0, root + "/src")
spec = importlib.util.spec_from_file_location("submitted_model", root + "/src/model.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
if mode == "train":
    out = mod.train(root + "/train_manifest.json", None, root + "/artifact", int(sys.argv[3]))
    json.dump(out if isinstance(out, dict) else {}, open(root + "/train_result.json", "w"))
else:
    reqs = json.load(open(root + "/requests.json"))
    preds = {}
    for group in reqs:          # one group per person: a fresh module instance resets state
        spec2 = importlib.util.spec_from_file_location("submitted_model_%d" % len(preds), root + "/src/model.py")
        fresh = importlib.util.module_from_spec(spec2)
        spec2.loader.exec_module(fresh)
        for r in group:
            preds[r["request_id"]] = fresh.predict(root + "/artifact", r)
    json.dump(preds, open(root + "/predictions.json", "w"), allow_nan=False)
'''


def _q(p: str) -> str:
    return '"' + p.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _runtime_roots(denied) -> list[str]:
    """Interpreter prefixes the worker must read (e.g. a venv under /Users).

    A prefix that contains an explicitly denied root is refused rather than
    allowed, so the runtime exception can never re-open controller data.
    """
    roots = sorted({str(Path(p).resolve()) for p in (sys.prefix, sys.base_prefix)})
    for r in roots:
        for d in denied:
            if Path(d).resolve() == Path(r) or Path(r) in Path(d).resolve().parents:
                raise RuntimeError(f"interpreter prefix {r} contains denied root {d}")
    return roots


def _profile(tmp: Path, denied) -> str:
    deny = {"/Users", "/tmp", "/private/tmp", "/var/folders", *map(str, denied)}
    rules = "\n".join(f"(deny file-read* (subpath {_q(d)}))" for d in sorted(deny))
    runtime = "\n".join(f"(allow file-read* (subpath {_q(r)}))" for r in _runtime_roots(denied))
    redeny = "\n".join(f"(deny file-read* (subpath {_q(str(d))}))" for d in sorted(map(str, denied)))
    return f"""(version 1)
(deny default)
(allow process*)
(deny process-fork)
(allow sysctl-read)
(allow mach-lookup)
(allow ipc-posix-shm)
(allow file-read* (subpath "/"))
{rules}
{runtime}
{redeny}
(allow file-read* (subpath {_q(str(tmp))}))
(allow file-write* (subpath {_q(str(tmp))}))
(deny network*)
"""


def _limits():
    resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))
    for lim in (resource.RLIMIT_DATA, resource.RLIMIT_RSS):
        try:
            resource.setrlimit(lim, (LIMITS["memory_bytes"], LIMITS["memory_bytes"]))
        except (OSError, ValueError):
            pass


def _tree_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


def available() -> bool:
    return sys.platform == "darwin" and shutil.which("sandbox-exec") is not None


def run(mode: str, source_dir: Path, artifact_dir: Path, *, train_manifest: Path | None = None,
        request_groups: list | None = None, seed: int = 0, denied=()) -> dict:
    """Run one worker call; return {ok, error, duration_s, usage, stdout, stderr, ...}."""
    start = time.monotonic()
    result = {"ok": False, "mode": mode, "limits": LIMITS, "executor": "macos-sandbox-exec"}
    if not available():
        return {**result, "executor": "unavailable", "error": "no supported isolation backend"}
    source_dir = Path(source_dir)
    if not (source_dir / "model.py").is_file():
        return {**result, "error": "candidate source must contain model.py"}
    if _tree_bytes(source_dir) > LIMITS["max_source_bytes"]:
        return {**result, "error": "candidate source exceeds size limit"}
    with tempfile.TemporaryDirectory(prefix="heat-worker-") as tmp_s:
        tmp = Path(tmp_s).resolve()
        shutil.copytree(source_dir, tmp / "src", ignore=shutil.ignore_patterns("__pycache__"))
        if not (tmp / "src" / "heatlib.py").exists():
            shutil.copy(TEMPLATE_DIR / "heatlib.py", tmp / "src" / "heatlib.py")
        (tmp / "artifact").mkdir()
        if mode == "train":
            shutil.copy(train_manifest, tmp / "train_manifest.json")
            args = ["train", str(seed)]
            wall = LIMITS["train_wall_s"]
        else:
            shutil.copytree(artifact_dir, tmp / "artifact", dirs_exist_ok=True)
            (tmp / "requests.json").write_text(json.dumps(request_groups, allow_nan=False))
            args = ["predict"]
            wall = LIMITS["predict_wall_s"]
        (tmp / "profile.sb").write_text(_profile(tmp, denied))
        threads = str(LIMITS["cpu_threads"])
        env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp), "TMPDIR": str(tmp), "LANG": "C",
               "LC_ALL": "C", "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
               "OMP_NUM_THREADS": threads, "OPENBLAS_NUM_THREADS": threads,
               "VECLIB_MAXIMUM_THREADS": threads, "MKL_NUM_THREADS": threads}
        cmd = ["sandbox-exec", "-f", str(tmp / "profile.sb"), sys.executable, "-I", "-c", _CHILD,
               str(tmp), *args]
        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        try:
            proc = subprocess.run(cmd, cwd=tmp, env=env, stdin=subprocess.DEVNULL, capture_output=True,
                                  timeout=wall, preexec_fn=_limits)
        except subprocess.TimeoutExpired as exc:
            return {**result, "error": f"{mode} exceeded wall limit {wall}s",
                    "duration_s": round(time.monotonic() - start, 3),
                    "stderr": (exc.stderr or b"")[-4000:].decode(errors="replace")}
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        usage = {"wall_s": round(time.monotonic() - start, 3),
                 "cpu_user_s": round(after.ru_utime - before.ru_utime, 3),
                 "cpu_sys_s": round(after.ru_stime - before.ru_stime, 3),
                 "max_rss_bytes_children_peak": after.ru_maxrss}   # bytes on macOS; peak over all children
        out = {**result, "duration_s": usage["wall_s"], "usage": usage, "returncode": proc.returncode,
               "stdout": proc.stdout[-4000:].decode(errors="replace"),
               "stderr": proc.stderr[-4000:].decode(errors="replace")}
        if proc.returncode != 0:
            return {**out, "error": f"{mode} failed inside worker (exit {proc.returncode})"}
        if mode == "train":
            if _tree_bytes(tmp / "artifact") > LIMITS["max_artifact_bytes"]:
                return {**out, "error": "artifact exceeds size limit"}
            artifact_dir = Path(artifact_dir)
            if artifact_dir.exists():
                shutil.rmtree(artifact_dir)
            shutil.copytree(tmp / "artifact", artifact_dir)
            return {**out, "ok": True, "train_result": json.loads((tmp / "train_result.json").read_text())}
        pred_path = tmp / "predictions.json"
        if not pred_path.is_file() or pred_path.stat().st_size > LIMITS["max_output_bytes"]:
            return {**out, "error": "predictions missing or oversized"}
        try:
            preds = json.loads(pred_path.read_text())
        except ValueError:
            return {**out, "error": "predictions are not valid JSON"}
        return {**out, "ok": True, "predictions": preds}
