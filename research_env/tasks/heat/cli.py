"""Command line for the heat task: prepare, baselines, session ops, replay, judging."""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from . import judging, scoring, splits
from .importer import prepare
from .session import Session, build_split, evaluate, TASK_MANIFEST, TEMPLATE, build_train_manifest, _write

BASELINES = ("elapsed_time", "ridge_lag", "gradient_boosting", "thermal_state")
BASELINE_SECRET = "heat-baselines-v1"     # fixed so baseline runs are reproducible; not a session secret


def baseline_source(name: str, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy(TEMPLATE / "baselines" / f"{name}.py", dest / "model.py")
    shutil.copy(TEMPLATE / "heatlib.py", dest / "heatlib.py")
    return dest


def run_baselines(prepared: Path, out: Path, seed: int = 0, freeze: bool = False) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    task = json.loads(TASK_MANIFEST.read_text())
    report = {"seed": seed, "secret": BASELINE_SECRET, "results": {}}
    for tier in ("wearable", "calibrated"):
        train = build_train_manifest(prepared, tier, BASELINE_SECRET)
        train_path = out / f"train_{tier}.json"
        _write(train_path, train)
        splits_ = {g: build_split(prepared, g, tier, BASELINE_SECRET) for g in ("development", "private_test")}
        for name in BASELINES:
            src = baseline_source(name, out / "sources" / name)
            art = out / "artifacts" / tier / name
            row = {}
            for i, (group, (reqs, targets)) in enumerate(splits_.items()):
                res = evaluate(src, art, train_path, reqs, targets, task, tier, seed,
                               denied=[str(prepared.resolve())], retrain=(i == 0), split=group)
                row[group] = res["score"]
                row.setdefault("worker", {}).update({group: res["logs"]})
            report["results"].setdefault(tier, {})[name] = row
            print(f"{tier:10s} {name:18s} dev={_fmt(row['development'])}  test={_fmt(row['private_test'])}")
    for tier, rows in report["results"].items():
        a, b = rows["elapsed_time"], rows["ridge_lag"]
        if a["private_test"].get("per_person") and b["private_test"].get("per_person"):
            report.setdefault("comparisons", {})[f"{tier}: ridge_lag - elapsed_time (test)"] = \
                scoring.paired_bootstrap(b["private_test"]["per_person"], a["private_test"]["per_person"], seed=seed)
    if freeze:
        task["frozen_baselines"] = {
            tier: {"id": "elapsed_time",
                   "loss_c": {g: report["results"][tier]["elapsed_time"][g]["primary_loss_c"]
                              for g in ("development", "private_test")},
                   "frozen_from": "cli baselines --freeze, seed %d" % seed}
            for tier in report["results"]}
        TASK_MANIFEST.write_text(json.dumps(task, indent=1) + "\n")
    _write(out / "baseline_report.json", report)
    return report


def _fmt(s):
    if not s.get("execution_valid"):
        return f"INVALID({s.get('failure_reason')})"
    h = s["by_horizon"]
    return (f"{s['primary_loss_c']:.3f}C [5:{h['5']['mae_c']:.3f} 10:{h['10']['mae_c']:.3f} "
            f"20:{h['20']['mae_c']:.3f}] cov90={s['uncertainty']['coverage90']:.2f} "
            f"hot_under={s['hottest_period']['underprediction_c']:.3f}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m research_env.tasks.heat")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare"); p.add_argument("source_root"); p.add_argument("output_root")
    p.add_argument("--split", default=str(splits.SPLIT_PATH)); p.add_argument("--offline", action="store_true")
    p = sub.add_parser("baselines"); p.add_argument("prepared"); p.add_argument("out")
    p.add_argument("--seed", type=int, default=0); p.add_argument("--freeze", action="store_true")
    p = sub.add_parser("create"); p.add_argument("prepared"); p.add_argument("session")
    p.add_argument("--tier", default="wearable", choices=("wearable", "calibrated"))
    p = sub.add_parser("dev-eval"); p.add_argument("session"); p.add_argument("candidate")
    p.add_argument("--decision", required=True, help="JSON with hypothesis, expected_result, observation, evidence_ids")
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("declare"); p.add_argument("session"); p.add_argument("candidate_id"); p.add_argument("--reason", default="")
    p = sub.add_parser("submit"); p.add_argument("session"); p.add_argument("candidate_id")
    p.add_argument("--report", required=True, help="path to the participant evidence report")
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("replay"); p.add_argument("session")
    p = sub.add_parser("pilot"); p.add_argument("prepared"); p.add_argument("session")
    p.add_argument("--model", default="sonnet"); p.add_argument("--dev-calls", type=int, default=5)
    p.add_argument("--wall-min", type=int, default=30)
    p = sub.add_parser("judge-export"); p.add_argument("session"); p.add_argument("out")
    p = sub.add_parser("judge-import"); p.add_argument("packets"); p.add_argument("verdicts"); p.add_argument("out")
    p = sub.add_parser("judge-run"); p.add_argument("packets"); p.add_argument("config"); p.add_argument("out")
    a = ap.parse_args(argv)
    if a.cmd == "prepare":
        m = prepare(a.source_root, a.output_root, a.split, allow_network=not a.offline)
        print(json.dumps(m["counts"], indent=1))
    elif a.cmd == "baselines":
        run_baselines(Path(a.prepared), Path(a.out), a.seed, a.freeze)
    elif a.cmd == "create":
        s = Session.create(Path(a.prepared), Path(a.session), a.tier)
        print(f"workspace: {s.public}\nprivate: {s.private}")
    elif a.cmd == "dev-eval":
        r = Session(Path(a.session)).dev_eval(Path(a.candidate), json.loads(a.decision), a.seed)
        print(json.dumps(r, indent=1))
    elif a.cmd == "declare":
        print(json.dumps(Session(Path(a.session)).declare(a.candidate_id, a.reason)["event_id"]))
    elif a.cmd == "submit":
        e = Session(Path(a.session)).submit(a.candidate_id, Path(a.report).read_text(), a.seed)
        print(e["event_id"])
    elif a.cmd == "replay":
        r = Session(Path(a.session)).replay_private()
        print(json.dumps({"final": {k: r["final"][k] for k in ("candidate_id", "dev_replay_matches_recorded")},
                          "final_test_primary_loss_c": r["final"]["test"].get("primary_loss_c")}, indent=1))
    elif a.cmd == "pilot":
        from .pilot import run_pilot
        print(json.dumps(run_pilot(Path(a.prepared), Path(a.session), a.model, a.dev_calls, a.wall_min), indent=1))
    elif a.cmd == "judge-export":
        packets = judging.export_packets(Session(Path(a.session)))
        Path(a.out).write_text(json.dumps(packets, indent=1))
        print(f"{len(packets['packets'])} packets; llm_judge_status={packets['llm_judge_status']}")
    elif a.cmd == "judge-import":
        rep = judging.import_verdicts(json.loads(Path(a.packets).read_text()),
                                      json.loads(Path(a.verdicts).read_text()))
        Path(a.out).write_text(json.dumps(rep, indent=1)); print(json.dumps(rep["summary"], indent=1))
    elif a.cmd == "judge-run":
        rep = judging.run_provider(json.loads(Path(a.packets).read_text()), json.loads(Path(a.config).read_text()))
        Path(a.out).write_text(json.dumps(rep, indent=1)); print(json.dumps(rep["summary"], indent=1))


if __name__ == "__main__":
    main()
