"""Interim isolated worker for training and inference of load-task artifacts.

INTERIM: Set 1 owns the shared runner (``research_env.core``). This module
exists so the load task is runnable end to end before that lands; replace the
calls in ``runner.py`` with the shared runner once its interface is frozen.

Isolation (macOS only, fails closed elsewhere): each call copies only the
permitted inputs into a fresh temporary root and runs ``python -I`` under the
Seatbelt profile from ``benchmark.executor`` — network denied, process fork
denied, reads under /Users, /tmp and /var/folders denied except the call's own
root. Private labels are never copied into that root. Inference runs in a new
process per person so in-memory state cannot carry across people. Not a
hardened multi-tenant service.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from benchmark.executor import _profile

LIMITS = {
    "train": {"wall_s": 900.0, "cpu_s": 900, "memory_bytes": 8 * 1024**3, "output_bytes": 64 * 1024**2},
    "predict": {"wall_s": 60.0, "cpu_s": 60, "memory_bytes": 4 * 1024**3, "output_bytes": 4 * 1024**2},
}
CPU_THREADS = "8"

_TRAIN_CHILD = r'''import importlib.util, json, sys
root = sys.argv[1]
spec = importlib.util.spec_from_file_location("submitted_train", root + "/src/train.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
job = json.load(open(root + "/job.json"))
tm = dict(job["train_manifest"], records_path=root + "/data/train.jsonl")
manifest = mod.train(tm, None, root + "/artifact", job["seed"])
json.dump(manifest, open(root + "/train_result.json", "w"), allow_nan=False)
'''

_PREDICT_CHILD = r'''import importlib.util, json, sys
root = sys.argv[1]
spec = importlib.util.spec_from_file_location("submitted_predict", root + "/artifact/predict.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
requests = json.load(open(root + "/requests.json"))
out = {}
for req in requests:
    try:
        out[req["request_id"]] = mod.predict(root + "/artifact", req)
    except Exception as exc:
        out[req["request_id"]] = {"error": type(exc).__name__ + ": " + str(exc)[:200]}
json.dump(out, open(root + "/predictions.json", "w"), allow_nan=False)
'''


def _limits_fn(limits):
    def apply():
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (limits["cpu_s"], limits["cpu_s"]))
        for lim in (resource.RLIMIT_DATA, resource.RLIMIT_RSS):
            try:
                resource.setrlimit(lim, (limits["memory_bytes"], limits["memory_bytes"]))
            except (OSError, ValueError):
                pass
        resource.setrlimit(resource.RLIMIT_FSIZE, (limits["output_bytes"], limits["output_bytes"]))
    return apply


def _run(root: Path, child: str, limits: dict, forbidden_paths) -> dict:
    sandbox = shutil.which("sandbox-exec") if sys.platform == "darwin" else None
    if not sandbox:
        return {"ok": False, "error": "no supported isolation backend (macOS sandbox-exec required)"}
    profile = root / "profile.sb"
    profile.write_text(_profile(root, forbidden_paths))
    env = {"PATH": "/usr/bin:/bin", "HOME": str(root), "TMPDIR": str(root), "LANG": "C", "LC_ALL": "C",
           "PYTHONNOUSERSITE": "1", "PYTHONHASHSEED": "0", "OMP_NUM_THREADS": CPU_THREADS,
           "OPENBLAS_NUM_THREADS": CPU_THREADS, "MKL_NUM_THREADS": CPU_THREADS,
           "VECLIB_MAXIMUM_THREADS": CPU_THREADS}
    start = time.monotonic()
    import resource
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    try:
        with open(root / "stdout.log", "wb") as out, open(root / "stderr.log", "wb") as err:
            proc = subprocess.run([sandbox, "-f", str(profile), sys.executable, "-I", "-c", child, str(root)],
                                  cwd=root, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                  timeout=limits["wall_s"], preexec_fn=_limits_fn(limits), check=False)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "wall time limit exceeded", "wall_s": limits["wall_s"]}
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    usage = {"wall_s": round(time.monotonic() - start, 3),
             "cpu_s": round((after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime), 3),
             "peak_rss_bytes_upper_bound": after.ru_maxrss}  # macOS reports bytes; max over all children
    tail = lambda name: (root / name).read_bytes()[-4000:].decode("utf-8", "replace")
    return {"ok": proc.returncode == 0, "returncode": proc.returncode, "usage": usage,
            "stdout_tail": tail("stdout.log"), "stderr_tail": tail("stderr.log"),
            "error": None if proc.returncode == 0 else "child exited non-zero"}


def train_in_worker(source_dir, train_rows_path, train_manifest: dict, seed: int, artifact_out,
                    forbidden_paths=()) -> dict:
    """Run ``source_dir/train.py``'s train() on permitted rows; copy the artifact to artifact_out."""
    with tempfile.TemporaryDirectory(prefix="load-train-") as tmp:
        root = Path(tmp).resolve()
        shutil.copytree(source_dir, root / "src", ignore=shutil.ignore_patterns("__pycache__"))
        (root / "data").mkdir()
        shutil.copyfile(train_rows_path, root / "data" / "train.jsonl")
        (root / "job.json").write_text(json.dumps({"train_manifest": train_manifest, "seed": seed}))
        result = _run(root, _TRAIN_CHILD, LIMITS["train"], forbidden_paths)
        if result["ok"]:
            try:
                result["artifact_manifest"] = json.loads((root / "train_result.json").read_text())
                if not (root / "artifact" / "predict.py").is_file():
                    raise ValueError("artifact has no predict.py")
            except (OSError, ValueError) as exc:
                result.update(ok=False, error=f"invalid training output: {exc}")
        if result["ok"]:
            if Path(artifact_out).exists():
                shutil.rmtree(artifact_out)
            shutil.copytree(root / "artifact", artifact_out, symlinks=False,
                            ignore=shutil.ignore_patterns("__pycache__"))
        return result


def predict_in_worker(artifact_dir, requests_by_person: dict[str, list[dict]], forbidden_paths=()) -> dict:
    """Fresh copy of the artifact and a fresh process per person. Returns {request_id: prediction}."""
    predictions, runs = {}, []
    for group, requests in requests_by_person.items():
        with tempfile.TemporaryDirectory(prefix="load-predict-") as tmp:
            root = Path(tmp).resolve()
            shutil.copytree(artifact_dir, root / "artifact", symlinks=False,
                            ignore=shutil.ignore_patterns("__pycache__"))
            (root / "requests.json").write_text(json.dumps(requests, allow_nan=False))
            result = _run(root, _PREDICT_CHILD, LIMITS["predict"], forbidden_paths)
            run = {"group": group, "ok": result["ok"], "usage": result.get("usage"), "error": result.get("error"),
                   "stderr_tail": result.get("stderr_tail", "")[-1000:]}
            if result["ok"]:
                try:
                    got = json.loads((root / "predictions.json").read_text())
                    for req in requests:
                        predictions[req["request_id"]] = got.get(req["request_id"])
                except (OSError, ValueError) as exc:
                    run.update(ok=False, error=f"invalid prediction output: {exc}")
            runs.append(run)
    return {"predictions": predictions, "runs": runs, "ok": all(r["ok"] for r in runs)}
