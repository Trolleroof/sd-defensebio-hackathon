"""INTERIM live-agent pilot: one unmodified Claude Code agent in a heat workspace.

The participant runs headless (``claude -p``) under macOS sandbox-exec with
file reads denied for the session's private root, the prepared dataset, the
repository checkout and the raw evidence directory. It reaches the evaluator
only through ``workspace/.controller/inbox``, which ``serve`` (this process,
outside the sandbox) answers. Controller paths are never written into the
workspace.
"""
from __future__ import annotations

import json
import subprocess
import threading
import time
from pathlib import Path

from .session import Session

REPO = Path(__file__).resolve().parents[3]
EVIDENCE = Path("/Users/nikhi/Documents/Army_Physiology_Research_20261017")

PROMPT = """You are a research agent. Your working directory is a research workspace.
Read TASK.md, DATA_DICTIONARY.md and task_manifest.json first. Then develop, evaluate and submit a model as TASK.md describes.
Use `python3 tools/heat.py` for every evaluator interaction (dev-eval, declare, status, submit). Each dev-eval
needs --hypothesis and --expected. Put each candidate in its own directory under candidates/.
This is a reduced smoke-test budget: {dev_calls} development calls and {wall_min} minutes of wall time.
Finish by writing REPORT.md and running `python3 tools/heat.py submit <candidate_id> --report REPORT.md`.
Files outside this workspace are not part of the task."""


def _symlinks(path: Path) -> bool:
    return any(p.is_symlink() for p in [path, *path.rglob("*")])


def handle(session: Session, req: dict) -> dict:
    ws = session.public.resolve()
    try:
        if req["op"] == "status":
            st = session.state
            return {"ok": True, "dev_calls_used": st["dev_calls_used"], "budget": st["budget"],
                    "elapsed_s": round(session.elapsed_s(), 1), "incumbents": st["incumbents"],
                    "submitted": st["submitted"]}
        if req["op"] == "dev-eval":
            cand = (ws / req["candidate"]).resolve()
            if ws not in cand.parents or not cand.is_dir() or _symlinks(cand):
                return {"ok": False, "error": "candidate must be a directory inside the workspace without symlinks"}
            return {"ok": True, **session.dev_eval(cand, req["decision"], int(req.get("seed", 0)))}
        if req["op"] == "declare":
            return {"ok": True, "event_id": session.declare(req["candidate_id"], req["reason"])["event_id"]}
        if req["op"] == "submit":
            return {"ok": True, "event_id": session.submit(req["candidate_id"], req["report"])["event_id"],
                    "note": "feedback closed; final scoring happens privately"}
        return {"ok": False, "error": "unknown op"}
    except (PermissionError, ValueError, KeyError) as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def serve(session: Session, stop: threading.Event):
    inbox = session.public / ".controller" / "inbox"
    outbox = session.public / ".controller" / "outbox"
    inbox.mkdir(parents=True, exist_ok=True); outbox.mkdir(parents=True, exist_ok=True)
    while not stop.is_set():
        for f in sorted(inbox.glob("*.json")):
            try:
                req = json.loads(f.read_text())
            except ValueError:
                req = {"op": "invalid"}
            f.unlink()
            with (session.private / "controller_log.jsonl").open("a") as log:
                log.write(json.dumps({"t": round(session.elapsed_s(), 2), "request": req}) + "\n")
            reply = handle(session, req)
            tmp = outbox / (f.stem + ".tmp")
            tmp.write_text(json.dumps(reply, allow_nan=False))
            tmp.rename(outbox / f.name)
        time.sleep(0.3)


def _agent_profile(denied) -> str:
    rules = "\n".join(f'(deny file-read* (subpath "{d}"))' for d in sorted(set(denied)))
    return f"(version 1)\n(allow default)\n{rules}\n"


def run_pilot(prepared: Path, root: Path, model: str = "sonnet", dev_calls: int = 5, wall_min: int = 30,
              max_turns: int = 120) -> dict:
    root = Path(root).resolve()
    session = Session.create(prepared, root, "wearable", budget={"dev_calls": dev_calls, "wall_s": wall_min * 60})
    (session.public / "candidates").mkdir()
    denied = [str(session.private.resolve()), str(Path(prepared).resolve()), str(REPO), str(EVIDENCE)]
    profile = root / "agent_profile.sb"
    profile.write_text(_agent_profile(denied))
    stop = threading.Event()
    th = threading.Thread(target=serve, args=(session, stop), daemon=True)
    th.start()
    cmd = ["sandbox-exec", "-f", str(profile), "claude", "-p",
           PROMPT.format(dev_calls=dev_calls, wall_min=wall_min),
           "--model", model, "--permission-mode", "bypassPermissions", "--output-format", "stream-json",
           "--verbose", "--max-turns", str(max_turns), "--strict-mcp-config", "--setting-sources", "",
           "--disable-slash-commands", "--no-session-persistence"]
    t0 = time.time()
    with (root / "agent_transcript.jsonl").open("w") as out, (root / "agent_stderr.log").open("w") as err:
        try:
            proc = subprocess.run(cmd, cwd=session.public, stdout=out, stderr=err, stdin=subprocess.DEVNULL,
                                  timeout=wall_min * 60 + 300)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    time.sleep(1.0)
    stop.set(); th.join()
    summary = {"model": model, "returncode": rc, "agent_wall_s": round(time.time() - t0, 1),
               "state": {k: session.state[k] for k in ("dev_calls_used", "incumbents", "submitted")},
               "profile_denied_reads": denied}
    (root / "pilot_summary.json").write_text(json.dumps(summary, indent=1))
    return summary
