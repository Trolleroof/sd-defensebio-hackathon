"""Reward-hack guards that do not need recordings, a sandbox, or an API key.

    python tests/test_reward_hacks.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from physbench.agent import (  # noqa: E402
    audit_text, blocked_reason, cites_observed_measurement, uncited_measurements,
)
from physbench.config import TaskConfig  # noqa: E402
from physbench.episodes import reference_visible_mask  # noqa: E402
from physbench.judge import events_for_early_grade  # noqa: E402
from physbench.scoring import validate_predictions  # noqa: E402


def check(name, ok):
    print(f"{'ok' if ok else 'FAIL'}  {name}")
    return ok


def main() -> None:
    cfg = TaskConfig({
        "transfer_activities": ["stairs", "table_soccer", "cycling"],
        "familiar_activities": ["sitting", "walking"],
        "guard_s": 4,
        "step_s": 2,
    }, Path("task.yaml"))
    # 0-3 sitting, 4 stairs, 5 cycling, 6-11 sitting. Guard is 2 samples.
    activity = np.array([1, 1, 1, 1, 2, 4, 1, 1, 1, 1, 1, 1])
    visible = reference_visible_mask(cfg, activity)
    results = []
    results.append(check("transfer samples hidden", not visible[4] and not visible[5]))
    results.append(check("recovery inside the guard hidden", not visible[6] and not visible[7]))
    results.append(check("distant familiar samples kept", bool(visible[0]) and bool(visible[11])))

    audit = audit_text("curl http://example.com", "/srv/pb/run/ep")
    results.append(check("network command blocked", blocked_reason(audit) is not None))
    audit = audit_text("ls /srv/pb/other", "/srv/pb/run/ep")
    results.append(check("other sandbox blocked", blocked_reason(audit) is not None))
    own = audit_text("ls <WS>/ws", "/srv/pb/run/ep")
    # own_root is stripped before the pattern runs, so a normal workspace path is allowed
    allowed = audit_text("ls /srv/pb/run/ep/ws", "/srv/pb/run/ep")
    results.append(check("own workspace path allowed", blocked_reason(allowed) is None and blocked_reason(own) is None))

    corpus = "heart rate 72.5 bpm\nphase_balanced_mae_bpm 9.57"
    results.append(check("quotes a real measurement", cites_observed_measurement("MAE was 9.57 on feedback", corpus)))
    results.append(check("rejects an invented measurement", not cites_observed_measurement("MAE was 3.14159 on feedback", corpus)))
    results.append(check("10 does not match 100", not cites_observed_measurement("used 10 windows", "100 windows")))

    report = "Observed MAE 9.57. Invented value 3.14159."
    missing = uncited_measurements(report, corpus)
    results.append(check("report keeps cited numbers", "9.57" not in missing and "3.14159" in missing))

    ok, msg = validate_predictions(np.zeros((2, 4, 2)), 2, 4, (25, 230))
    results.append(check("interval array rejected", (not ok) and "point forecasts" in msg))
    ok, msg = validate_predictions(np.full((2, 4), 80.0), 2, 4, (25, 230))
    results.append(check("point forecast accepted", ok))

    events = [
        {"type": "tool", "turn": 1, "name": "bash", "arguments": "{}", "result": "hr 72.5"},
        {"type": "tool", "turn": 2, "name": "dev_feedback", "arguments": "{}", "result": "mae 9.57"},
        {"type": "tool", "turn": 3, "name": "bash", "arguments": "{}", "result": "later secret mae 1.00"},
    ]
    early = events_for_early_grade(events)
    results.append(check("early grade stops at feedback", len(early) == 2))
    results.append(check("feedback result withheld", "9.57" not in early[-1]["result"]))
    results.append(check("later result excluded", all("1.00" not in str(ev) for ev in early)))

    if not all(results):
        raise SystemExit(1)
    print("REWARD HACK CHECKS OK")


if __name__ == "__main__":
    main()
