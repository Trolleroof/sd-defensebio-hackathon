"""CLI: python -m research_env.tasks.load <command> ...

    prepare      parse and split the source workbook (trusted)
    workspace    build a participant workspace for one track
    baselines    train every baseline in the worker; score development and test; write a report
    search       equal-budget conventional search (30 development calls)
    dev-eval     one public development call for a participant submission (counts against budget)
    final        train a submission and score it privately on test (no feedback to the participant)
    replay       retrain a submission from source with the same seed and compare artifacts/predictions
    judge-export export blind decision/final-selection packets from an experiment event log
    judge-import validate verdict attempts against exported packets
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from . import judging, runner, search
from .importer import prepare, read_jsonl
from .schema import TRACKS
from .scoring import paired_person_bootstrap
from .workspace import build_workspace

PKG = Path(__file__).resolve().parent
DEFAULT_SOURCE = PKG.parents[2] / "sources" / "evobreath-2015"
DEFAULT_SPLIT = PKG / "config" / "split_v1.json"
BASELINES = {
    "primary": ["mean_by_load", "ridge", "gbm", "pandolf", "pandolf_recal"],
    "calibration": ["mean_by_load", "ridge", "pandolf_recal", "calibration_offset"],
    "holdout15": ["mean_by_load", "ridge", "gbm", "pandolf", "pandolf_recal"],
}


def _dump(obj):
    print(json.dumps(obj, indent=2, sort_keys=True, default=str))


def cmd_baselines(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    src = out / "baseline_src"
    src.mkdir(exist_ok=True)
    shutil.copyfile(PKG / "baseline_models.py", src / "train.py")
    results = {}
    for track in args.tracks:
        results[track] = {}
        for model in BASELINES[track]:
            art = out / "artifacts" / track / model
            tr = runner.train_candidate(args.prepared, src, track, args.seed, art, {"model": model})
            entry = {"train": {k: tr.get(k) for k in ("ok", "error", "usage", "artifact_manifest")}}
            if tr["ok"]:
                entry["artifact_sha256"] = runner.artifact_hash(art)
                for split in ("development", "test"):
                    entry[split] = runner.evaluate(args.prepared, art, track, split)
            results[track][model] = entry
        ref = results[track].get("ridge", {})
        for model, entry in results[track].items():
            if model != "ridge" and entry.get("test", {}).get("execution_valid") and ref.get("test", {}).get("execution_valid"):
                entry["test_vs_ridge_bootstrap"] = paired_person_bootstrap(
                    entry["test"]["per_person"], ref["test"]["per_person"], seed=args.seed)
    (out / "baseline_results.json").write_text(json.dumps(results, indent=1, sort_keys=True, default=str))
    (out / "baseline_report.md").write_text(render_report(results))
    print((out / "baseline_report.md").read_text())


def _f(x, nd=1):
    return "—" if x is None else f"{x:.{nd}f}"


def render_report(results: dict) -> str:
    lines = ["# Load task baseline results (executed)", "",
             "Person-balanced metrics. Development = 10 public-feedback people; test = 10 private people. "
             "Intervals are split-conformal from grouped CV on training people. "
             "`Δ vs ridge` is the paired person bootstrap of test MAE differences (negative favors the model).", ""]
    for track, models in results.items():
        lines += [f"## {track}", "",
                  "| baseline | valid | dev MAE W | test MAE W | test W/kg | test bias W | worst load W | ordering | IS90 | cov90 | width W | Δ vs ridge (95% CI) |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for model, e in models.items():
            d, t = e.get("development", {}), e.get("test", {})
            if not t.get("execution_valid"):
                lines.append(f"| {model} | no | — | — | — | — | — | — | — | — | — | {t.get('failure_reason') or e['train'].get('error')} |")
                continue
            s, u = t["secondary"], t["uncertainty"]
            bs = e.get("test_vs_ridge_bootstrap")
            delta = f"{bs['mean_diff']:+.1f} [{bs['ci95'][0]:+.1f}, {bs['ci95'][1]:+.1f}]" if bs else "ref"
            lines.append(f"| {model} | yes | {_f(d['primary']['value'])} | {_f(t['primary']['value'])} | "
                         f"{_f(s['mae_w_per_kg'], 3)} | {_f(s['bias_w'])} | {_f(s['worst_load_mae_w'])} | "
                         f"{_f(s['ordering_concordance'], 2)} | {_f(u['is90'])} | {_f(u['coverage90'], 2)} | "
                         f"{_f(u['width90_w'])} | {delta} |")
        lines.append("")
        lines.append("Per-load test MAE (W):")
        lines.append("")
        loads = sorted({k for e in models.values() for k in (e.get("test", {}).get("per_load") or {})}, key=float)
        lines.append("| baseline | " + " | ".join(f"{l} kg" for l in loads) + " |")
        lines.append("|---|" + "---|" * len(loads))
        for model, e in models.items():
            pl = e.get("test", {}).get("per_load") or {}
            lines.append(f"| {model} | " + " | ".join(_f((pl.get(l) or {}).get("mae_w")) for l in loads) + " |")
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m research_env.tasks.load")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prepare")
    a.add_argument("--source", default=str(DEFAULT_SOURCE))
    a.add_argument("--split", default=str(DEFAULT_SPLIT))
    a.add_argument("--out", required=True)
    a = sub.add_parser("workspace")
    a.add_argument("--prepared", required=True)
    a.add_argument("--track", choices=list(TRACKS), default="primary")
    a.add_argument("--out", required=True)
    a = sub.add_parser("baselines")
    a.add_argument("--prepared", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--tracks", nargs="+", default=list(TRACKS))
    a.add_argument("--seed", type=int, default=0)
    a = sub.add_parser("search")
    a.add_argument("--prepared", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--track", choices=list(TRACKS), default="primary")
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--budget", type=int, default=None)
    for name in ("dev-eval", "final", "replay"):
        a = sub.add_parser(name)
        a.add_argument("--prepared", required=True)
        a.add_argument("--submission", required=True, help="directory containing train.py")
        a.add_argument("--run-dir", required=True)
        a.add_argument("--track", choices=list(TRACKS), default="primary")
        a.add_argument("--seed", type=int, default=0)
        a.add_argument("--candidate-id", default="final")
        a.add_argument("--train-manifest-json", default="{}", help="extra train_manifest keys, e.g. baseline model")
    a = sub.add_parser("judge-export")
    a.add_argument("--events", required=True)
    a.add_argument("--out", required=True)
    a = sub.add_parser("judge-import")
    a.add_argument("--packets", required=True)
    a.add_argument("--verdicts", required=True, help="JSON: {packet_sha256: [attempt, ...]}")
    args = p.parse_args(argv)

    if args.cmd == "prepare":
        _dump(prepare(args.source, args.out, args.split))
    elif args.cmd == "workspace":
        _dump(build_workspace(args.prepared, args.out, args.track))
    elif args.cmd == "baselines":
        cmd_baselines(args)
    elif args.cmd == "search":
        res = search.run_search(args.prepared, args.out, args.track, args.seed, args.budget)
        if res["selected_artifact"]:
            res["test"] = runner.evaluate(args.prepared, res["selected_artifact"], args.track, "test")
        Path(args.out, "search_result.json").write_text(json.dumps(res, indent=1, default=str))
        _dump({k: res.get(k) for k in ("track", "dev_calls_used", "selected")} |
              {"test_primary": (res.get("test") or {}).get("primary")})
    elif args.cmd in ("dev-eval", "final", "replay"):
        extra = json.loads(args.train_manifest_json)
        run = Path(args.run_dir)
        art = run / "artifacts" / args.candidate_id
        if args.cmd == "replay":
            _dump(runner.replay(args.prepared, args.submission, args.track, args.seed, art, run / "replay", extra))
            return
        tr = runner.train_candidate(args.prepared, args.submission, args.track, args.seed, art, extra)
        if not tr["ok"]:
            if args.cmd == "dev-eval":  # a failed attempt still spends a development call
                _dump(runner.DevBudget(run).call(args.prepared, args.submission, args.track, args.candidate_id))
            else:
                _dump({"execution_valid": False, "failure_reason": tr.get("error"), "stderr_tail": tr.get("stderr_tail")})
            sys.exit(1)
        if args.cmd == "dev-eval":
            _dump(runner.DevBudget(run).call(args.prepared, art, args.track, args.candidate_id))
        else:
            res = runner.evaluate(args.prepared, art, args.track, "test")
            (run / "private_final.json").write_text(json.dumps(res, indent=1, default=str))
            _dump({"private_result_written": str(run / "private_final.json"), "execution_valid": res["execution_valid"]})
    elif args.cmd == "judge-export":
        _dump(judging.export_packets(read_jsonl(Path(args.events)), args.out))
    elif args.cmd == "judge-import":
        res = judging.import_verdicts(read_jsonl(Path(args.packets)), json.loads(Path(args.verdicts).read_text()))
        _dump({"completeness": res["completeness"], "log": res["log"],
               "process": judging.process_score(list(res["accepted"].values())) if res["accepted"] else None})


if __name__ == "__main__":
    main()
