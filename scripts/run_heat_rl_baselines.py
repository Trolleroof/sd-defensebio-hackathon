"""Replay the four heat baselines through the RL task family (shared worker, heat replay).

Usage:
    python scripts/run_heat_rl_baselines.py .research-data/heat/prepared .heat-runs/rl-baselines \
        [--backend macos-development|docker] [--image IMAGE@sha256:...] [--seed 0] [--training-tasks 4]

For every agent_development and agent_final_evaluation task (both tiers), and the
first ``--training-tasks`` agent_training tasks, it stages the participant
workspace, builds each baseline with the workspace's own tools/build_artifact.py,
and scores the artifact on reward people with HeatFamily.score. The frozen
reward baseline (elapsed_time, in-process) is reported next to its worker replay.
"""
import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_env.tasks.heat.rl_pool import FAMILY  # noqa: E402

BASELINES = ("elapsed_time", "ridge_lag", "gradient_boosting", "thermal_state")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prepared", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--backend", default="macos-development")
    ap.add_argument("--image")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--training-tasks", type=int, default=4)
    a = ap.parse_args()
    pool = FAMILY.build_pool(a.prepared, a.seed)
    chosen = [t for t in pool if t.outer_split != "agent_training"]
    for tier in FAMILY.tiers:
        chosen += [t for t in pool if t.outer_split == "agent_training" and t.options["tier"] == tier][:a.training_tasks]
    a.out.mkdir(parents=True, exist_ok=True)
    report = {"seed": a.seed, "backend": a.backend, "image": a.image, "tasks": []}
    start = time.monotonic()
    for spec in chosen:
        row = {"task_id": spec.task_id, "outer_split": spec.outer_split, "tier": spec.options["tier"],
               "profile": spec.profile, "people": [len(spec.fit_people), len(spec.feedback_people), len(spec.reward_people)],
               "frozen_baseline_loss": FAMILY.baseline_loss(spec, a.prepared), "replay": {}}
        with tempfile.TemporaryDirectory(prefix="heat-rl-baselines-") as tmp:
            ws, priv = Path(tmp) / "workspace", Path(tmp) / "private"
            FAMILY.stage_workspace(spec, a.prepared, ws, prebuild=False)
            manifest = FAMILY.private_manifest(spec, a.prepared, "reward", priv)
            for name in BASELINES:
                built = subprocess.run([sys.executable, "tools/build_artifact.py", f"baselines/{name}.py",
                                        "--out", f"art_{name}"], cwd=ws, capture_output=True, text=True)
                if built.returncode:
                    row["replay"][name] = {"execution_valid": False, "failure_reason": built.stderr[-500:]}
                    continue
                r = FAMILY.score(ws / f"art_{name}", manifest, backend=a.backend, image=a.image,
                                 wall_s=1800, profile=spec.profile)
                m = r["metrics"]
                row["replay"][name] = {"execution_valid": r["execution_valid"], "primary_loss": r["primary_loss"],
                                       "failure_reason": r["failure_reason"],
                                       "by_horizon_mae": {h: v["mae_c"] for h, v in m.get("by_horizon", {}).items()},
                                       "is90": m.get("uncertainty", {}).get("is90"),
                                       "coverage90": m.get("uncertainty", {}).get("coverage90"),
                                       "wall_s": r["resource_use"].get("wall_s")}
        report["tasks"].append(row)
        cells = "  ".join(f"{n}={v['primary_loss']:.3f}" if v.get("execution_valid") else f"{n}=INVALID"
                          for n, v in row["replay"].items())
        print(f"{spec.outer_split:23s} {row['tier']:10s} {spec.profile:9s} frozen={row['frozen_baseline_loss']:.3f}  {cells}",
              flush=True)
    report["wall_s"] = time.monotonic() - start
    (a.out / "rl_baseline_report.json").write_text(json.dumps(report, indent=1))
    print(f"wrote {a.out / 'rl_baseline_report.json'} in {report['wall_s']:.1f}s")


if __name__ == "__main__":
    main()
