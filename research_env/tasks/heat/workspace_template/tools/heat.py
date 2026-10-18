"""Participant tool: talk to the evaluator through the workspace inbox.

    python3 tools/heat.py dev-eval CANDIDATE_DIR --hypothesis "..." --expected "..." [--observation "..."] [--evidence e0003 ...] [--seed 0]
    python3 tools/heat.py declare CANDIDATE_ID --reason "..."
    python3 tools/heat.py submit CANDIDATE_ID --report REPORT.md
    python3 tools/heat.py status

The evaluator runs in a separate process; this script only writes a request
file and waits for the reply. Development feedback is also saved in feedback/.
"""
import argparse
import json
import time
import uuid
from pathlib import Path

WS = Path(__file__).resolve().parent.parent
INBOX, OUTBOX = WS / ".controller" / "inbox", WS / ".controller" / "outbox"


def call(payload, timeout=1800):
    INBOX.mkdir(parents=True, exist_ok=True)
    rid = uuid.uuid4().hex
    tmp = INBOX / f"{rid}.tmp"
    tmp.write_text(json.dumps(payload))
    tmp.rename(INBOX / f"{rid}.json")
    reply = OUTBOX / f"{rid}.json"
    end = time.time() + timeout
    while time.time() < end:
        if reply.exists():
            return json.loads(reply.read_text())
        time.sleep(0.5)
    return {"ok": False, "error": "evaluator did not reply in time"}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="op", required=True)
    p = sub.add_parser("dev-eval"); p.add_argument("candidate")
    p.add_argument("--hypothesis", required=True); p.add_argument("--expected", required=True)
    p.add_argument("--observation", default=""); p.add_argument("--evidence", nargs="*", default=[])
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("declare"); p.add_argument("candidate_id"); p.add_argument("--reason", required=True)
    p = sub.add_parser("submit"); p.add_argument("candidate_id"); p.add_argument("--report", required=True)
    sub.add_parser("status")
    a = ap.parse_args()
    if a.op == "dev-eval":
        cand = Path(a.candidate).resolve()
        payload = {"op": "dev-eval", "candidate": str(cand.relative_to(WS)), "seed": a.seed,
                   "decision": {"hypothesis": a.hypothesis, "expected_result": a.expected,
                                "observation": a.observation, "evidence_ids": a.evidence}}
    elif a.op == "declare":
        payload = {"op": "declare", "candidate_id": a.candidate_id, "reason": a.reason}
    elif a.op == "submit":
        payload = {"op": "submit", "candidate_id": a.candidate_id, "report": Path(a.report).read_text()}
    else:
        payload = {"op": "status"}
    print(json.dumps(call(payload), indent=1))


if __name__ == "__main__":
    main()
