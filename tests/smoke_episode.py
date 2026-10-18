"""End-to-end plumbing check without an LLM (run as root on the Linux host).

    python tests/smoke_episode.py

Builds one dev episode, drives the tools like an agent would, probes isolation,
and scores the unchanged baseline on the hidden transfer windows.
"""

import json
import shutil

from physbench.agent import EpisodeState, Tools
from physbench.config import load_task
from physbench.runner import _brief, baseline_scores
from physbench.scoring import score
from physbench.episodes import exertion_mask
from physbench.workspace import build_episode, run_submission

cfg = load_task("configs/task_hr_transfer.yaml")
subject = cfg.dev_subjects[0]
bl = baseline_scores(cfg, subject)
ep = build_episode(cfg, subject, "smoke", "smoke-ep")
state = EpisodeState()
tools = Tools(cfg, ep, {k: _brief(v) for k, v in bl["dev"].items()}, state)

checks = {}
ref = "\n".join(p.read_text() for p in (ep.ws / "data" / "reference").glob("R*.csv"))
for name in cfg.transfer_activities:
    checks[f"reference_hides_{name}"] = name not in ref
checks["reference_keeps_sitting"] = "sitting" in ref
hr = (ep.ws / "data" / "target" / "support.csv").read_text().splitlines()[1].split(",")[2]
why = (f"Support file data/target/support.csv lists heart rate {hr} on the first kept row. "
       "Checking the unchanged baseline before editing.")
out = tools.bash("ls -R | head -40 && head -3 data/target/support.csv && whoami")
print(out)
checks["fit_baseline"] = "in-sample MAE" in tools.bash("cd baseline && python fit_baseline.py")
checks["check_submission"] = "OK:" in tools.bash("python tools/check_submission.py")
checks["cannot_read_repo"] = tools.bash("cat /workspace/physbench/.env").startswith("REJECTED")
checks["cannot_read_private"] = tools.bash(f"ls {ep.private}").startswith("REJECTED")
checks["urllib_rejected"] = tools.bash("python -c \"import urllib.request\"").startswith("REJECTED")
net = tools.bash("python -c \"import socket; socket.create_connection(('1.1.1.1', 80), timeout=5)\" 2>&1 | tail -1")
checks["python_raw_socket_blocked"] = "network access is disabled" in net
checks["curl_blocked"] = tools.bash("curl -sS http://example.com").startswith("REJECTED")
checks["no_api_key_in_env"] = "OPENROUTER" not in tools.bash("env")
checks["justification_enforced"] = tools.dev_feedback("too short").startswith("REJECTED")
checks["justification_needs_number"] = tools.dev_feedback("Baseline as provided, checking held-out familiar-activity error before changing anything.").startswith("REJECTED")
fb = tools.dev_feedback(why)
print(fb[:1500])
checks["dev_feedback_ok"] = '"execution": "ok"' in fb
checks["submit_needs_report"] = tools.submit(why).startswith("REJECTED")
tools.write_file("submission/REPORT.md", ("The hidden MAE was 3.14159 bpm and that number was not measured. " * 6),
                 justification=why)
checks["fake_report_number_rejected"] = "3.14159" in tools.submit(why)
tools.write_file("submission/REPORT.md", "# Report\n\nUnchanged conventional kinetic baseline submitted as a plumbing check. " * 5,
                 justification=why)
checks["submit_ok"] = tools.submit(why).startswith("Submission accepted")
ok, pred, info = run_submission(cfg, ep, ep.transfer, "hidden")
checks["hidden_exec"] = ok
s = score(pred, ep.transfer.target, exertion_mask(cfg, ep.transfer.activity), ep.transfer.activity, cfg.step_s)
print("hidden transfer (workspace baseline):", round(s["phase_balanced_mae_bpm"], 2),
      "evaluator kinetic_personal:", round(bl["transfer"]["kinetic_personal"]["phase_balanced_mae_bpm"], 2))
checks["baseline_matches_evaluator"] = abs(s["phase_balanced_mae_bpm"] - bl["transfer"]["kinetic_personal"]["phase_balanced_mae_bpm"]) < 0.05
print("audit flags:", json.dumps(state.audit_flags)[:600])
print(json.dumps(checks, indent=1))
shutil.rmtree(ep.sandbox_dir, ignore_errors=True)
assert all(checks.values()), [k for k, v in checks.items() if not v]
print("SMOKE OK")
