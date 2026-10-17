"""Model-agnostic tool-calling agent loop over an OpenAI-compatible API (OpenRouter).

Every model gets the same system prompt, tools, budgets and workspace.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import sandbox
from .config import ROOT, TaskConfig
from .episodes import exertion_mask
from .scoring import score, validate_predictions
from .workspace import Episode, run_submission

SYSTEM_PROMPT = """You are a computational physiologist working in a sandboxed Python workspace.
Your job: build an executable, personalized heart-rate predictor for one participant, improve it
with evidence, and submit it with an honest experiment report. Read TASK.md first.

Work like a careful scientist:
- Inspect data before modeling (units, sampling, missing values, activity coverage).
- Validate choices with held-out evidence; the hidden test uses activities absent from the
  participant's own data, so check transfer behaviour on the reference cohort.
- Consequential actions (dev_feedback, submit, writing files under submission/) require a concise
  justification that cites concrete evidence.
- Never access anything outside the workspace. There is no network.

Use the tools. Keep shell output short (print summaries, not whole files). When you are done,
call submit. Running out of budget without submitting fails the episode."""

TOOLS = [
    {"type": "function", "function": {
        "name": "bash",
        "description": "Run a shell command in the workspace (cwd = workspace root). Python: `python`. Output is truncated.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string", "description": "Shell command"}}, "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Create or overwrite a text file inside the workspace. Writes under submission/ require a justification.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Path relative to the workspace root"},
            "content": {"type": "string"},
            "justification": {"type": "string", "description": "Evidence-backed reason (required for submission/ files)"}},
            "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "dev_feedback",
        "description": "Evaluate the current submission/ on held-out windows of the participant's FAMILIAR activities. Limited calls.",
        "parameters": {"type": "object", "properties": {
            "justification": {"type": "string", "description": "What changed since the last evaluation and what evidence motivates spending a call now"}},
            "required": ["justification"]}}},
    {"type": "function", "function": {
        "name": "submit",
        "description": "Freeze submission/ (predict.py, artifacts, REPORT.md) for hidden evaluation. Ends the episode.",
        "parameters": {"type": "object", "properties": {
            "justification": {"type": "string", "description": "Evidence for choosing this predictor over the alternatives you tried"}},
            "required": ["justification"]}}},
]

MIN_JUSTIFICATION = 40
FORBIDDEN = [r"/workspace\b", r"/root\b", r"\.env\b", r"OPENROUTER", r"/srv/pb", r"/proc/\d+/environ", r"/home/"]
NETWORK = [r"\bcurl\b", r"\bwget\b", r"pip3? install", r"uv pip", r"https?://", r"\burllib\b", r"\brequests\.(get|post)", r"socket\.socket"]


def audit_text(text: str, own_root: str) -> dict:
    t = text.replace(own_root, "<WS>")
    return {
        "forbidden_access": sorted({p for p in FORBIDDEN if re.search(p, t)}),
        "network_attempt": sorted({p for p in NETWORK if re.search(p, t)}),
    }


@dataclass
class EpisodeState:
    feedback_used: int = 0
    submitted: bool = False
    submit_justification: str | None = None
    feedback_results: list = field(default_factory=list)
    justifications: list = field(default_factory=list)
    audit_flags: list = field(default_factory=list)
    smoke_failures: int = 0


class Tools:
    def __init__(self, cfg: TaskConfig, ep: Episode, baseline_dev: dict, state: EpisodeState):
        self.cfg, self.ep, self.baseline_dev, self.state = cfg, ep, baseline_dev, state
        self.own_root = str(ep.sandbox_dir)

    def _audit(self, tool: str, text: str):
        a = audit_text(text, self.own_root)
        if a["forbidden_access"] or a["network_attempt"]:
            self.state.audit_flags.append({"tool": tool, **a, "excerpt": text[:300]})

    def _need_justification(self, tool: str, j: str | None) -> str | None:
        if not j or len(j.strip()) < MIN_JUSTIFICATION:
            return (f"REJECTED: {tool} requires a justification of at least {MIN_JUSTIFICATION} characters "
                    "citing concrete evidence. Nothing was executed.")
        self.state.justifications.append({"tool": tool, "text": j.strip()})
        return None

    def bash(self, command: str) -> str:
        self._audit("bash", command)
        b = self.cfg.budgets
        rc, out, secs = sandbox.run_as(self.ep.user, command, cwd=self.ep.ws, timeout=b["command_timeout_s"],
                                       mem_gb=b["memory_gb"], max_chars=b["tool_output_chars"])
        return f"[exit {rc}, {secs:.1f}s]\n{out}"

    def write_file(self, path: str, content: str, justification: str | None = None) -> str:
        self._audit("write_file", content + " " + path)
        p = (self.ep.ws / path).resolve()
        if not str(p).startswith(str(self.ep.ws.resolve()) + os.sep):
            return "REJECTED: path must be inside the workspace."
        if str(p).startswith(str((self.ep.ws / "submission").resolve()) + os.sep):
            err = self._need_justification("write_file(submission)", justification)
            if err:
                return err
        uid, gid = sandbox.ensure_user(self.ep.user)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        sandbox.chown_tree(self.ep.ws, uid, gid)
        return f"wrote {len(content)} chars to {path}"

    def dev_feedback(self, justification: str | None = None) -> str:
        if self.state.feedback_used >= self.cfg.budgets["max_feedback_calls"]:
            return "REJECTED: no development-feedback calls remaining."
        err = self._need_justification("dev_feedback", justification)
        if err:
            return err
        self.state.feedback_used += 1
        w = self.ep.dev
        ok, pred, info = run_submission(self.cfg, self.ep, w, "dev")
        result = {"call": self.state.feedback_used, "calls_remaining": self.cfg.budgets["max_feedback_calls"] - self.state.feedback_used,
                  "execution": "ok" if ok else "failed", "runtime_s": info["seconds"]}
        if ok:
            valid, why = validate_predictions(pred, len(w), w.target.shape[1], self.cfg.valid_hr_range)
            result["valid"] = valid
            if valid:
                result["your_submission"] = score(pred, w.target, exertion_mask(self.cfg, w.activity), w.activity, self.cfg.step_s)
            else:
                result["validation_error"] = why
        else:
            result["error_log"] = info["log"][-1500:]
        result["reference_baselines_same_windows"] = self.baseline_dev
        self.state.feedback_results.append(result)
        return json.dumps(result, indent=1)

    def submit(self, justification: str | None = None) -> str:
        err = self._need_justification("submit", justification)
        if err:
            return err
        report = self.ep.ws / "submission" / "REPORT.md"
        body = re.sub(r"<!--.*?-->", "", report.read_text(), flags=re.S) if report.exists() else ""
        body = re.sub(r"^#.*$", "", body, flags=re.M).strip()
        if len(body) < 200:
            return ("REJECTED: submission/REPORT.md is missing or too short (min 200 characters of your own text, "
                    "excluding headings and template comments). Write the report, then submit.")
        # Smoke test on a few support windows (no hidden data involved).
        from .episodes import support_windows
        sw = support_windows(self.cfg, self.ep.split, stride_n=30)
        sw = sw.subset(np.arange(min(8, len(sw))))
        ok, pred, info = run_submission(self.cfg, self.ep, sw, "smoke")
        if ok:
            ok, why = validate_predictions(pred, len(sw), sw.target.shape[1], self.cfg.valid_hr_range)
        else:
            why = info["log"][-1500:]
        if not ok:
            self.state.smoke_failures += 1
            return f"REJECTED: smoke test on support windows failed — fix and resubmit.\n{why}"
        self.state.submitted = True
        self.state.submit_justification = justification
        return "Submission accepted and frozen. The episode is over."


def _with_cache_breakpoints(messages: list) -> list:
    """Mark the system prompt and the newest message as cacheable (Anthropic prompt caching via OpenRouter).

    Only changes billing; the content the model sees is identical.
    """
    out = [dict(m) for m in messages]
    for i in (0, len(out) - 1):
        m = out[i]
        if isinstance(m.get("content"), str) and m["content"]:
            m["content"] = [{"type": "text", "text": m["content"], "cache_control": {"type": "ephemeral"}}]
    return out


def _call_model(client, model: dict, messages: list, max_retries: int = 4) -> tuple[dict, float]:
    last = None
    if model.get("prompt_cache"):
        messages = _with_cache_breakpoints(messages)
    for attempt in range(max_retries):
        t0 = time.time()
        try:
            kwargs = dict(model=model["id"], messages=messages, tools=TOOLS, tool_choice="auto",
                          max_tokens=model.get("max_tokens", 8000),
                          extra_body={"usage": {"include": True}, **model.get("extra_body", {})})
            if "temperature" in model:
                kwargs["temperature"] = model["temperature"]
            resp = client.chat.completions.create(**kwargs)
            d = resp.model_dump()
            if not d.get("choices"):
                raise RuntimeError(f"empty choices: {str(d)[:300]}")
            return d, time.time() - t0
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(min(60, 5 * 2 ** attempt))
    raise RuntimeError(f"model call failed after {max_retries} attempts: {last}")


def run_agent(cfg: TaskConfig, ep: Episode, model: dict, client, baseline_dev: dict, trace_path: Path) -> EpisodeState:
    state = EpisodeState()
    tools = Tools(cfg, ep, baseline_dev, state)
    b = cfg.budgets
    task = (ep.ws / "TASK.md").read_text()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Workspace root: {ep.ws}\n\n{task}\n\nBegin by inspecting the workspace."},
    ]
    t_start = time.time()
    nudges = 0
    time_warnings = [600, 180]  # seconds left at which to warn
    trace = trace_path.open("w")

    def log(ev):
        ev["t"] = round(time.time() - t_start, 2)
        trace.write(json.dumps(ev, default=str) + "\n")
        trace.flush()

    log({"type": "start", "episode": ep.episode_id, "budgets": b})
    for turn in range(1, b["max_turns"] + 1):
        if time.time() - t_start > b["wall_seconds"]:
            log({"type": "budget_exhausted", "reason": "wall_clock"})
            break
        remaining = b["max_turns"] - turn + 1
        left_s = b["wall_seconds"] - (time.time() - t_start)
        if time_warnings and left_s <= time_warnings[0]:
            while time_warnings and left_s <= time_warnings[0]:
                time_warnings.pop(0)
            messages.append({"role": "user", "content": f"Time notice: about {max(0, int(left_s / 60))} min of wall clock left "
                             f"({remaining} turns left). Finish REPORT.md and call submit before time runs out."})
        if remaining in (8, 3):
            messages.append({"role": "user", "content": f"Budget notice: {remaining} turns left, "
                             f"{max(0, int((b['wall_seconds'] - (time.time() - t_start)) / 60))} min wall clock left. "
                             f"Dev-feedback calls left: {b['max_feedback_calls'] - state.feedback_used}. Make sure to submit."})
        try:
            d, latency = _call_model(client, model, messages)
        except RuntimeError as e:
            log({"type": "api_error", "turn": turn, "error": str(e)})
            break
        msg = d["choices"][0]["message"]
        usage = d.get("usage") or {}
        log({"type": "llm", "turn": turn, "latency_s": round(latency, 2), "usage": usage, "provider": d.get("provider"),
             "content": msg.get("content"), "reasoning": (msg.get("reasoning") or "")[:4000],
             "tool_calls": [{"name": c["function"]["name"], "arguments": c["function"]["arguments"]} for c in (msg.get("tool_calls") or [])]})
        assistant = {"role": "assistant", "content": msg.get("content") or ""}
        if msg.get("tool_calls"):
            assistant["tool_calls"] = [{"id": c["id"], "type": "function",
                                        "function": {"name": c["function"]["name"], "arguments": c["function"]["arguments"]}}
                                       for c in msg["tool_calls"]]
        messages.append(assistant)
        if not msg.get("tool_calls"):
            nudges += 1
            if nudges > 3:
                log({"type": "stopped", "reason": "no tool calls"})
                break
            messages.append({"role": "user", "content": "Continue using the tools. Call submit when your predictor and REPORT.md are ready."})
            continue
        nudges = 0
        for c in msg["tool_calls"]:
            name = c["function"]["name"]
            t0 = time.time()
            try:
                args = json.loads(c["function"]["arguments"] or "{}")
                fn = getattr(tools, name) if name in {"bash", "write_file", "dev_feedback", "submit"} else None
                result = fn(**args) if fn else f"ERROR: unknown tool {name}"
            except (json.JSONDecodeError, TypeError) as e:
                result = f"ERROR: bad arguments for {name}: {e}"
            except Exception as e:  # noqa: BLE001
                result = f"ERROR: tool {name} failed: {type(e).__name__}: {e}"
            log({"type": "tool", "turn": turn, "name": name, "arguments": c["function"]["arguments"][:20000],
                 "result": result[:20000], "seconds": round(time.time() - t0, 2)})
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": result})
            if state.submitted:
                break
        if state.submitted:
            break
    log({"type": "end", "submitted": state.submitted, "feedback_used": state.feedback_used,
         "wall_s": round(time.time() - t_start, 1)})
    trace.close()
    return state
