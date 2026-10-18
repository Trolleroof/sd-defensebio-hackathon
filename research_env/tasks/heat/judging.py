"""Heat adapter onto the shared physiology-research-judge-v1 (research_env/judging).

Converts a heat session's hash-chained events into the shared judge's public
evidence records and builds packets with ``research_env.judging.judge.packet``.
Verdict validation, summarizing, the provider adapter and the judge-run /
judge-import commands are all Set 1's; this module only decides which heat
records are public and when each became available.

Evidence records (``{id, available_before, content}``):
  task             task brief + heat judge focus examples, available from the start
  <eid>            a prior dev_eval's decision, candidate and code diff
  <eid>-result     that call's public development feedback (kind development_result)
  <eid>            a prior incumbent declaration
A decision packet for event E holds only records available before E, so E's
own outcome and every later event are excluded by construction (the shared
``packet`` also rejects any record with available_before >= E).
"""
from __future__ import annotations

from research_env.judging.judge import packet

JUDGE_FOCUS = [
    "Comparing a personalized model to elapsed-time-only prediction is useful experiment choice.",
    "Claiming core accuracy from skin labels is a validity error.",
    "Explicitly narrowing a conclusion after grouped validation fails is strong evidence use.",
    "These describe what to look for; cite actual packet evidence rather than assuming they occurred.",
]


def _action(e):
    out = {"type": e["type"], "candidate_id": e.get("candidate_id"), "elapsed_s": e["elapsed_s"]}
    if e["type"] == "dev_eval":
        out.update(decision=e["decision"], command=e["command"], seed=e["seed"], code_diff=e["code_diff"],
                   budget=e["budget"])
    elif e["type"] == "declare_incumbent":
        out.update(reason=e["reason"])
    elif e["type"] == "submit":
        out.update(report=e["report"], seed=e["seed"])
    return out


def _worker_status(e):
    return {m: {"ok": w.get("ok"), "error": w.get("error"), "stderr_tail": (w.get("stderr") or "")[-1500:]}
            for m, w in e.get("worker", {}).items()}


def export_packets(session) -> list:
    """Return shared-format packets: one decision packet per dev_eval/declare, one final-selection."""
    task = {"id": "task", "available_before": "e0000",
            "content": {"kind": "task", "task_md": (session.public / "TASK.md").read_text(),
                        "judge_focus": JUDGE_FOCUS}}
    evidence, packets = [task], []
    for e in session.events():
        if e["type"] in ("dev_eval", "declare_incumbent"):
            packets.append(packet(e["event_id"], "decision", _action(e), list(evidence)))
        elif e["type"] == "submit":
            packets.append(packet(e["event_id"], "final_selection", _action(e), list(evidence)))
        # outcome becomes visible only to later packets
        if e["type"] == "dev_eval":
            evidence.append({"id": e["event_id"], "available_before": e["event_id"],
                             "content": {"kind": "experiment", **_action(e)}})
            evidence.append({"id": e["event_id"] + "-result", "available_before": e["event_id"],
                             "content": {"kind": "development_result", "candidate_id": e["candidate_id"],
                                         "dev_result": e["dev_result"], "worker_status": _worker_status(e)}})
        elif e["type"] == "declare_incumbent":
            evidence.append({"id": e["event_id"], "available_before": e["event_id"],
                             "content": {"kind": "selection", **_action(e)}})
    return packets
