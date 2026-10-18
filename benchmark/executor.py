"""Run frozen predictor source in a constrained local process.

The current backend is macOS Seatbelt (sandbox-exec). This synthetic-benchmark
prototype denies reads under /Users, /tmp, /var/folders, and controller-supplied
private roots, writes outside its temporary directory, network access, and
process forks. Python startup requires broad reads elsewhere, so this is not a
hardened multi-tenant service. Execution fails closed if sandbox-exec is absent.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


TIMEOUT_S = 5.0
MAX_OUTPUT_BYTES = 1_000_000
MAX_INPUT_BYTES = 2_000_000
MEMORY_LIMIT_BYTES = 512 * 1024 * 1024
CPU_LIMIT_S = 10


_CHILD = r'''import json, sys, types
root = sys.argv[1]
with open(root + "/input.json", "r", encoding="utf-8") as f:
    data = json.load(f)
module = types.ModuleType("submitted_predictor")
exec(compile(data["code"], "<submitted predictor>", "exec"), module.__dict__)
if not callable(getattr(module, "predict", None)):
    raise TypeError("predict(history, forecast_times_s) is required")
fit = getattr(module, "fit", None)
if fit is not None:
    if not callable(fit):
        raise TypeError("fit must be callable")
    fit(data["support"])
predictions = {}
for request in data["requests"]:
    predictions[request["id"]] = module.predict(
        request["history"], request["forecast_times_s"])
encoded = json.dumps(predictions, allow_nan=False, separators=(",", ":"))
if len(encoded.encode("utf-8")) > 1000000:
    raise ValueError("prediction output exceeds limit")
with open(root + "/output.json", "w", encoding="utf-8") as f:
    f.write(encoded)
'''


def _sbpl_string(value: str) -> str:
    """Quote a path as an SBPL string literal."""
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _profile(temp_root: Path, forbidden_paths=()) -> str:
    denied_paths = {"/Users", "/tmp", "/private/tmp", "/var/folders"}
    for value in forbidden_paths:
        path = Path(value)
        if not path.is_absolute():
            raise ValueError("forbidden paths must be absolute")
        denied_paths.add(str(path))
        denied_paths.add(str(path.resolve()))
    deny_reads = "\n".join(
        f"(deny file-read* (subpath {_sbpl_string(path)}))"
        for path in sorted(denied_paths)
    )
    return f"""(version 1)
(deny default)
(allow process*)
(deny process-fork)
(allow sysctl-read)
(allow file-read* (subpath "/"))
{deny_reads}
(allow file-read* (subpath {_sbpl_string(str(temp_root))}))
(allow file-write* (subpath {_sbpl_string(str(temp_root))}))
(deny network*)
"""


def _set_limits():
    """Apply limits in the forked subprocess immediately before exec."""
    import resource

    resource.setrlimit(resource.RLIMIT_CPU, (CPU_LIMIT_S, CPU_LIMIT_S))
    # macOS reports an unlimited RLIMIT_AS and rejects lowering it after the
    # interpreter's address space has been reserved. Bound heap allocations and
    # resident set with the limits macOS accepts instead.
    for limit in (resource.RLIMIT_DATA, resource.RLIMIT_RSS):
        try:
            resource.setrlimit(limit, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))
        except (OSError, ValueError):
            pass
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_OUTPUT_BYTES, MAX_OUTPUT_BYTES))
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))


def _base_result(executor: str, start: float) -> dict:
    return {"predictions": {}, "execution": {"valid": False, "executor": executor,
                                               "duration_s": round(time.monotonic() - start, 3)}}


def execute(code: str, support: list, requests: list, forbidden_paths=()) -> dict:
    """Execute predictor code and return predictions plus execution metadata.

    A predictor module defines ``predict(history, forecast_times_s)`` and may
    define ``fit(support)``. The latter receives only the supplied support list.
    Request objects are expected to contain ``id``, ``history`` and
    ``forecast_times_s``. No controller path, environment credentials, or hidden
    target rows are supplied to the child. Controllers should pass private
    storage roots in ``forbidden_paths`` to deny direct access to those files.
    """
    start = time.monotonic()
    sandbox = shutil.which("sandbox-exec") if sys.platform == "darwin" else None
    if not sandbox:
        result = _base_result("unavailable", start)
        result["execution"]["error"] = "No supported isolation backend is available"
        return result
    if not isinstance(code, str) or not isinstance(support, list) or not isinstance(requests, list):
        result = _base_result("macos-sandbox-exec", start)
        result["execution"]["error"] = "Invalid executor input"
        return result
    ids = [r.get("id") for r in requests if isinstance(r, dict)]
    if len(ids) != len(requests) or any(not isinstance(i, str) for i in ids) or len(ids) != len(set(ids)):
        result = _base_result("macos-sandbox-exec", start)
        result["execution"]["error"] = "Requests need unique string IDs"
        return result

    with tempfile.TemporaryDirectory(prefix="predictor-sandbox-") as tmp:
        root = Path(tmp).resolve()
        input_path = root / "input.json"
        output_path = root / "output.json"
        profile_path = root / "profile.sb"
        try:
            payload = json.dumps({"code": code, "support": support, "requests": requests},
                                 allow_nan=False)
            if len(payload.encode("utf-8")) > MAX_INPUT_BYTES:
                result = _base_result("macos-sandbox-exec", start)
                result["execution"]["error"] = "Executor input exceeds limit"
                return result
            input_path.write_text(payload, encoding="utf-8")
            profile_path.write_text(_profile(root, forbidden_paths), encoding="utf-8")
            env = {"PATH": "/usr/bin:/bin", "HOME": str(root), "TMPDIR": str(root),
                   "LANG": "C", "LC_ALL": "C", "PYTHONNOUSERSITE": "1",
                   "PYTHONSAFEPATH": "1"}
            command = [sandbox, "-f", str(profile_path), sys.executable,
                       "-I", "-S", "-c", _CHILD, str(root)]
            stdout_path, stderr_path = root / "stdout.log", root / "stderr.log"
            with stdout_path.open("wb") as stdout_file, stderr_path.open("wb") as stderr_file:
                proc = subprocess.run(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                      stdout=stdout_file, stderr=stderr_file,
                                      timeout=TIMEOUT_S, check=False, preexec_fn=_set_limits)
        except subprocess.TimeoutExpired:
            result = _base_result("macos-sandbox-exec", start)
            result["execution"].update({"timed_out": True, "error": "Predictor exceeded time limit"})
            return result
        except (OSError, ValueError, TypeError) as exc:
            result = _base_result("macos-sandbox-exec", start)
            result["execution"]["error"] = f"Executor setup failed: {type(exc).__name__}"
            return result

        if stdout_path.stat().st_size + stderr_path.stat().st_size > MAX_OUTPUT_BYTES:
            result = _base_result("macos-sandbox-exec", start)
            result["execution"]["error"] = "Predictor output exceeded limit"
            return result
        if proc.returncode != 0 or not output_path.is_file() or output_path.is_symlink():
            result = _base_result("macos-sandbox-exec", start)
            result["execution"]["error"] = "Predictor failed inside sandbox"
            return result
        try:
            if output_path.stat().st_size > MAX_OUTPUT_BYTES:
                raise ValueError("output limit")
            predictions = json.loads(output_path.read_text(encoding="utf-8"))
            if not isinstance(predictions, dict) or set(predictions) != set(ids):
                raise ValueError("prediction coverage mismatch")
            # Validate that the payload can safely cross the controller boundary.
            encoded = json.dumps(predictions, allow_nan=False, separators=(",", ":"))
            if len(encoded.encode("utf-8")) > MAX_OUTPUT_BYTES:
                raise ValueError("output limit")
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            result = _base_result("macos-sandbox-exec", start)
            result["execution"]["error"] = "Predictor returned invalid or oversized output"
            return result
        return {"predictions": predictions,
                "execution": {"valid": True, "executor": "macos-sandbox-exec",
                              "python_runtime": sys.executable,
                              "duration_s": round(time.monotonic() - start, 3),
                              "limits": {"wall_timeout_s": TIMEOUT_S,
                                         "cpu_limit_s": CPU_LIMIT_S,
                                         "memory_limit_bytes": MEMORY_LIMIT_BYTES,
                                         "memory_limit_enforcement": "best effort via macOS RLIMIT_DATA/RLIMIT_RSS",
                                         "input_bytes": MAX_INPUT_BYTES,
                                         "output_bytes": MAX_OUTPUT_BYTES,
                                         "network": "denied",
                                         "host_read_policy": "host-readable except denied roots",
                                         "denied_roots": sorted({"/Users", "/tmp", "/private/tmp",
                                                                 "/var/folders", *(str(p) for p in forbidden_paths)}),
                                         "process_fork": "denied",
                                         "process_exec": "allowed within sandbox"}}}
