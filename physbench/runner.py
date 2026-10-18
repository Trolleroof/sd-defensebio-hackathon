"""Episode orchestration, pass criteria, aggregation and reporting."""

from __future__ import annotations

import datetime as dt
import json
import shutil
import subprocess
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml

from . import judge as judge_mod
from .agent import run_agent
from .baselines import fit_all
from .config import ROOT, TaskConfig, file_sha256, load_yaml
from .episodes import exertion_mask, split_subject, transfer_windows, dev_windows
from .scoring import (ACC_TAU, ACC_WEIGHT, JUDGE_TAU, bootstrap_ci, elo_ratings, elo_win_prob, improvement, score,
                      validate_predictions, wilson_ci)
from .workspace import build_episode, run_submission

CRITERIA_DIR = ROOT / "configs" / "frozen"
_BUILD_LOCK = threading.Lock()


def openrouter_client():
    from openai import OpenAI
    key = None
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("OPENROUTER_API_KEY="):
                key = line.split("=", 1)[1].strip()
    import os
    key = key or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY missing (.env or environment)")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key, timeout=600, max_retries=0)


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


# ---------------------------------------------------------------- baselines

def baseline_scores(cfg: TaskConfig, subject: str) -> dict:
    """Score every baseline on the subject's dev and transfer windows (evaluator side)."""
    sp = split_subject(cfg, subject)
    B = fit_all(cfg, sp)
    out = {}
    for label, w in (("dev", dev_windows(cfg, sp)), ("transfer", transfer_windows(cfg, sp))):
        ex = exertion_mask(cfg, w.activity)
        out[label] = {name: score(f(w), w.target, ex, w.activity, cfg.step_s) for name, f in B.items()}
    return out


def _brief(s: dict) -> dict:
    return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in s.items()
            if k in ("phase_balanced_mae_bpm", "mae_exertion", "mae_rest", "mae_all", "bias")}


# ---------------------------------------------------------------- criteria

def load_criteria(version: str) -> tuple[dict, str] | tuple[None, None]:
    p = CRITERIA_DIR / f"pass_criteria_{version}.yaml"
    if not p.exists():
        return None, None
    return yaml.safe_load(p.read_text()), file_sha256(p)


PROVISIONAL = {"anchor_baseline": "kinetic_pop_calibrated", "improvement_margin": 0.10,
               "judge_report_consistency_min": 1, "provisional": True}


def evaluate_pass(crit: dict, r: dict) -> dict:
    integ = r["programmatic_integrity"]
    j = r.get("judge") or {}
    flags, scores = j.get("flags") or {}, j.get("scores") or {}
    comp = {
        "valid_execution": bool(r["valid_execution"]),
        "beats_anchor_by_margin": r.get("improvement_vs_anchor") is not None and r["improvement_vs_anchor"] >= crit["improvement_margin"],
        "explicit_submit_with_report": integ["explicit_submit_with_report"],
        "justifications_present": integ["justifications_present"],
        "no_forbidden_access": not integ["forbidden_access_flags"],
        "no_network_attempt": not integ["network_attempt_flags"],
        "judge_no_leakage_or_gaming": flags.get("leakage_or_gaming") is False,
        "judge_no_fabricated_claims": flags.get("fabricated_claims") is False,
        "judge_report_consistent": scores.get("report_consistency", -1) >= crit["judge_report_consistency_min"],
    }
    comp["integrity"] = all(comp[k] for k in comp if k not in ("valid_execution", "beats_anchor_by_margin"))
    comp["pass"] = comp["valid_execution"] and comp["beats_anchor_by_margin"] and comp["integrity"]
    return comp


# ---------------------------------------------------------------- one episode

def run_episode(cfg: TaskConfig, model: dict, judge_model: dict, subject: str, run_id: str, repeat: int,
                baselines: dict, criteria: dict, client) -> dict:
    slug = model["name"]
    episode_id = f"{slug}-{subject}-r{repeat}"
    out_dir = ROOT / "runs" / run_id / episode_id
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with _BUILD_LOCK:
        ep = build_episode(cfg, subject, run_id, episode_id, criteria)
    task_md = (ep.ws / "TASK.md").read_text()
    baseline_dev = {k: _brief(v) for k, v in baselines["dev"].items()}
    state = run_agent(cfg, ep, model, client, baseline_dev, out_dir / "trace.jsonl")
    agent_wall = time.time() - t0
    # Stop anything the agent left running (background jobs) before hidden evaluation.
    subprocess.run(["pkill", "-KILL", "-u", ep.user], capture_output=True)

    # Snapshot the final submission (code + report) for inspection.
    snap = out_dir / "submission"
    if snap.exists():
        shutil.rmtree(snap)
    shutil.copytree(ep.ws / "submission", snap, symlinks=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.npy", "*.npz", "*.pkl", "*.joblib"))
    report = (ep.ws / "submission" / "REPORT.md").read_text() if (ep.ws / "submission" / "REPORT.md").exists() else ""

    # Hidden transfer evaluation (only for explicit submissions).
    w = ep.transfer
    r = {"episode_id": episode_id, "run_id": run_id, "model": slug, "model_id": model["id"], "subject": subject,
         "repeat": repeat, "submitted": state.submitted, "feedback_used": state.feedback_used,
         "smoke_failures": state.smoke_failures, "infra_error": state.infra_error}
    agent_metrics, exec_info = None, None
    if state.submitted:
        ok, pred, exec_info = run_submission(cfg, ep, w, "hidden")
        valid, why = (validate_predictions(pred, len(w), w.target.shape[1], cfg.valid_hr_range) if ok else (False, "execution failed"))
        r["valid_execution"] = bool(ok and valid)
        r["validation_message"] = why
        if r["valid_execution"]:
            agent_metrics = score(pred, w.target, exertion_mask(cfg, w.activity), w.activity, cfg.step_s)
            np.save(out_dir / "private" / "hidden_predictions.npy", pred)
    else:
        r["valid_execution"] = False
        r["validation_message"] = "no explicit submission"
        # Informational only: score whatever was left in submission/ (never counts toward Pass).
        ok, pred, info = run_submission(cfg, ep, w, "fallback")
        if ok and validate_predictions(pred, len(w), w.target.shape[1], cfg.valid_hr_range)[0]:
            fb = score(pred, w.target, exertion_mask(cfg, w.activity), w.activity, cfg.step_s)
            r["unsubmitted_fallback_mae"] = fb["phase_balanced_mae_bpm"]
    r["hidden_exec"] = exec_info
    r["transfer_metrics"] = agent_metrics
    r["baselines_transfer"] = {k: _brief(v) for k, v in baselines["transfer"].items()}
    anchor = baselines["transfer"][criteria["anchor_baseline"]]["phase_balanced_mae_bpm"]
    r["anchor_mae"] = anchor
    r["agent_mae"] = agent_metrics["phase_balanced_mae_bpm"] if agent_metrics else None
    r["improvement_vs_anchor"] = improvement(r["agent_mae"], anchor) if agent_metrics else None
    r["dev_feedback"] = state.feedback_results

    # Usage, cost, runtime.
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0, "turns": 0, "llm_latency_s": 0.0}
    for line in (out_dir / "trace.jsonl").read_text().splitlines():
        ev = json.loads(line)
        if ev["type"] == "llm":
            u = ev.get("usage") or {}
            usage["prompt_tokens"] += u.get("prompt_tokens") or 0
            usage["completion_tokens"] += u.get("completion_tokens") or 0
            usage["cost_usd"] += float(u.get("cost") or 0)
            usage["turns"] += 1
            usage["llm_latency_s"] += ev.get("latency_s") or 0
    usage["agent_wall_s"] = round(agent_wall, 1)
    r["usage"] = usage

    # Programmatic integrity.
    r["programmatic_integrity"] = {
        "explicit_submit_with_report": state.submitted,
        "justifications_present": state.submitted and bool(state.submit_justification)
        and all(len(j["text"]) >= 40 for j in state.justifications),
        "forbidden_access_flags": [f for f in state.audit_flags if f["forbidden_access"]],
        "network_attempt_flags": [f for f in state.audit_flags if f["network_attempt"]],
        "justifications": state.justifications,
    }

    # Judge (blind to model identity and hidden accuracy).
    programmatic = {"submitted": state.submitted, "dev_feedback_calls_used": state.feedback_used,
                    "harness_audit_flags": state.audit_flags, "smoke_test_failures": state.smoke_failures}
    r["judge"] = judge_mod.judge_episode(client, judge_model, task_md, out_dir / "trace.jsonl", report, programmatic)
    r["judge_cost_usd"] = float(((r["judge"].get("usage") or {}).get("cost")) or 0)
    r["criteria"] = evaluate_pass(criteria, r)
    r["pass"] = r["criteria"]["pass"]
    r["total_wall_s"] = round(time.time() - t0, 1)
    (out_dir / "result.json").write_text(json.dumps(r, indent=1, default=str))
    (out_dir / "judge.json").write_text(json.dumps(r["judge"], indent=1, default=str))
    subprocess.run(["pkill", "-KILL", "-u", ep.user], capture_output=True)
    shutil.rmtree(ep.sandbox_dir, ignore_errors=True)
    return r


# ---------------------------------------------------------------- batches

def run_batch(cfg: TaskConfig, pilot: dict, split: str, run_id: str, workers: int, models_filter=None,
              subjects_filter=None, allow_provisional: bool = False) -> list[dict]:
    from .sandbox import assert_repo_private
    assert_repo_private(ROOT)
    criteria, crit_sha = load_criteria(cfg.benchmark_version)
    if split == "heldout" and criteria is None:
        raise SystemExit("Held-out runs need frozen pass criteria. Run `physbench freeze` on dev results first.")
    if criteria is None:
        if not allow_provisional:
            raise SystemExit("No frozen criteria; pass --provisional for development runs.")
        criteria = PROVISIONAL
    subjects = list(cfg.dev_subjects if split == "dev" else cfg.heldout_subjects)
    if subjects_filter:
        subjects = [s for s in subjects if s in subjects_filter]
    models = [m for m in pilot["models"] if not models_filter or m["name"] in models_filter]
    repeats = int(pilot.get("repeats", 1))
    run_dir = ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {"run_id": run_id, "split": split, "subjects": subjects, "models": models, "judge": pilot["judge"],
            "repeats": repeats, "task_config": str(cfg.path.relative_to(ROOT)), "task_config_sha256": cfg.sha256,
            "criteria": criteria, "criteria_sha256": crit_sha, "git_commit": git_commit(),
            "started": dt.datetime.now(dt.timezone.utc).isoformat()}
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=1))

    print(f"[{run_id}] fitting baselines for {subjects}", flush=True)
    with ThreadPoolExecutor(max_workers=len(subjects)) as ex:
        bl = dict(zip(subjects, ex.map(lambda s: baseline_scores(cfg, s), subjects)))
    (run_dir / "baselines.json").write_text(json.dumps(bl, indent=1))

    client = openrouter_client()
    jobs = [(m, s, k) for k in range(repeats) for s in subjects for m in models]
    results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_episode, cfg, m, pilot["judge"], s, run_id, k, bl[s], criteria, client): (m["name"], s, k)
                for m, s, k in jobs}
        for f in as_completed(futs):
            name, s, k = futs[f]
            try:
                r = f.result()
                results.append(r)
                print(f"[{run_id}] {name} {s} r{k}: submitted={r['submitted']} valid={r['valid_execution']} "
                      f"mae={r['agent_mae']} anchor={r['anchor_mae']:.2f} judge={r['judge'].get('judge_score')} "
                      f"pass={r['pass']} cost=${r['usage']['cost_usd']:.3f}", flush=True)
            except Exception:  # noqa: BLE001
                err = traceback.format_exc()
                print(f"[{run_id}] {name} {s} r{k}: HARNESS ERROR\n{err}", flush=True)
                results.append({"model": name, "subject": s, "repeat": k, "harness_error": err, "pass": False})
    (run_dir / "results.json").write_text(json.dumps(results, indent=1, default=str))
    write_report(run_id)
    return results


# ---------------------------------------------------------------- freezing

def freeze(cfg: TaskConfig, dev_run_id: str, margin: float, anchor: str, note: str, force: bool = False) -> Path:
    CRITERIA_DIR.mkdir(parents=True, exist_ok=True)
    p = CRITERIA_DIR / f"pass_criteria_{cfg.benchmark_version}.yaml"
    if p.exists() and not force:
        raise SystemExit(f"{p} already frozen; refusing to overwrite.")
    dev = json.loads((ROOT / "runs" / dev_run_id / "baselines.json").read_text())
    res_path = ROOT / "runs" / dev_run_id / "results.json"
    dev_results = json.loads(res_path.read_text()) if res_path.exists() else [
        json.loads(p.read_text()) for p in sorted((ROOT / "runs" / dev_run_id).glob("*/result.json"))]
    calib = {
        "dev_run_id": dev_run_id,
        "dev_baselines_transfer_phase_balanced_mae": {s: {k: round(v["phase_balanced_mae_bpm"], 2) for k, v in d["transfer"].items()} for s, d in dev.items()},
        "dev_agent_improvements": [{"model": r.get("model"), "subject": r.get("subject"), "improvement": r.get("improvement_vs_anchor")} for r in dev_results],
    }
    crit = {
        "benchmark_version": cfg.benchmark_version,
        "frozen_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "task_config": str(cfg.path.relative_to(ROOT)),
        "task_config_sha256": cfg.sha256,
        "primary_metric": "phase_balanced_mae_bpm on hidden transfer windows, per episode",
        "anchor_baseline": anchor,
        "improvement_margin": margin,
        "valid_execution": "all hidden windows predicted within the time limit; finite; inside valid_hr_range",
        "integrity_required": [
            "explicit submit with REPORT.md (>= 200 chars)",
            "justifications (>= 40 chars) quote a number present in the data files or dev_feedback",
            "REPORT.md measurements are a subset of the task, the data files, or dev_feedback",
            "point forecasts only; interval outputs are rejected",
            "no forbidden-access attempts in the harness audit",
            "no network attempts in the harness audit",
            "judge flag leakage_or_gaming == false",
            "judge flag fabricated_claims == false",
            "judge report_consistency >= judge_report_consistency_min",
        ],
        "judge_report_consistency_min": 1,
        "pass": "valid_execution AND improvement_vs_anchor >= improvement_margin AND all integrity criteria",
        "rationale": note,
        "calibration_evidence": calib,
    }
    p.write_text(yaml.safe_dump(crit, sort_keys=False, width=110))
    return p


# ---------------------------------------------------------------- reporting

def _fmt(ci, pct=False, nd=2):
    if ci is None or ci[0] is None:
        return "–"
    f = (lambda x: f"{x * 100:.0f}%") if pct else (lambda x: f"{x:.{nd}f}")
    return f"{f(ci[0])} [{f(ci[1])}, {f(ci[2])}]"


def write_report(run_id: str) -> Path:
    run_dir = ROOT / "runs" / run_id
    meta = json.loads((run_dir / "run_meta.json").read_text())
    results = [r for r in json.loads((run_dir / "results.json").read_text())]
    models = [m["name"] for m in meta["models"]]
    rows, per_model = [], {}
    for m in models:
        rs = [r for r in results if r.get("model") == m]
        ok = [r for r in rs if "harness_error" not in r]
        k = sum(bool(r.get("pass")) for r in rs)
        maes = [r["agent_mae"] for r in ok if r.get("agent_mae") is not None]
        imps = [r["improvement_vs_anchor"] for r in ok if r.get("improvement_vs_anchor") is not None]
        js = [r["judge"]["judge_score"] for r in ok if (r.get("judge") or {}).get("judge_score") is not None]
        per_model[m] = {
            "n": len(rs),
            "pass_at_1": wilson_ci(k, len(rs)),
            "valid_rate": wilson_ci(sum(bool(r.get("valid_execution")) for r in rs), len(rs)),
            "integrity_rate": wilson_ci(sum(bool((r.get("criteria") or {}).get("integrity")) for r in rs), len(rs)),
            "agent_mae": bootstrap_ci(maes),
            "improvement": bootstrap_ci(imps),
            "judge": bootstrap_ci(js),
            "runtime_min": bootstrap_ci([r["usage"]["agent_wall_s"] / 60 for r in ok]),
            "cost_usd_mean": float(np.mean([r["usage"]["cost_usd"] for r in ok])) if ok else None,
            "cost_usd_total": float(sum(r["usage"]["cost_usd"] for r in ok)),
            "judge_cost_usd_total": float(sum(r.get("judge_cost_usd", 0) for r in ok)),
            "harness_errors": len(rs) - len(ok),
        }
        for r in rs:
            rows.append(r)

    anchor_name = meta["criteria"]["anchor_baseline"]
    subjects = meta["subjects"]
    bl = json.loads((run_dir / "baselines.json").read_text())
    base_names = list(next(iter(bl.values()))["transfer"].keys())
    base_mean = {b: bootstrap_ci([bl[s]["transfer"][b]["phase_balanced_mae_bpm"] for s in subjects]) for b in base_names}

    # Paired comparisons between models (same subjects).
    pairs = []
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            diffs = []
            for s in subjects:
                ra = [r for r in results if r.get("model") == a and r.get("subject") == s and r.get("agent_mae") is not None]
                rb = [r for r in results if r.get("model") == b and r.get("subject") == s and r.get("agent_mae") is not None]
                if ra and rb:
                    diffs.append(np.mean([r["agent_mae"] for r in ra]) - np.mean([r["agent_mae"] for r in rb]))
            pairs.append((a, b, len(diffs), bootstrap_ci(diffs)))

    # Margin-aware Bradley-Terry Elo from head-to-head episodes on the same subject.
    elo = elo_ratings(results, models, subjects)

    # Predeclared diagnosis rules.
    p1 = {m: per_model[m]["pass_at_1"][0] or 0 for m in models}
    diag = []
    if all(v >= 0.8 for v in p1.values()):
        diag.append("TOO EASY: every model passes on >= 80% of cases. Add harder supported slices (less support data, longer horizon, held-out dynamics).")
    if all(v <= 0.1 for v in p1.values()):
        best_imp = max((per_model[m]["improvement"][0] or -9) for m in models)
        diag.append("POSSIBLY TOO HARD: no model passes on > 10% of cases" + (
            " and no model beats the anchor on average — check information limits before raising difficulty." if best_imp <= 0 else
            "; some models improve on the anchor on average, so the margin may be strict rather than the task impossible."))
    sep_mae = [f"{a} vs {b}" for a, b, n, ci in pairs if ci[0] is not None and (ci[1] > 0 or ci[2] < 0)]
    sep_pass = [f"{a} vs {b}" for i, a in enumerate(models) for b in models[i + 1:]
                if per_model[a]["pass_at_1"][1] is not None and (per_model[a]["pass_at_1"][1] > per_model[b]["pass_at_1"][2] or per_model[b]["pass_at_1"][1] > per_model[a]["pass_at_1"][2])]
    if sep_mae or sep_pass:
        diag.append("DISTINGUISHES MODELS: " + "; ".join(
            ([f"transfer-MAE paired CI excludes 0 for {', '.join(sep_mae)}"] if sep_mae else []) +
            ([f"non-overlapping Pass@1 intervals for {', '.join(sep_pass)}"] if sep_pass else [])))
    else:
        diag.append(f"NOT YET DISTINGUISHING at n={len(subjects)} cases per model: no paired MAE difference or Pass@1 gap is resolved by the 95% intervals.")

    L = [f"# SoliderPhys-Bench results — {run_id}", "",
         f"- Split: **{meta['split']}** — subjects {', '.join(subjects)} ({meta['repeats']} run per model per subject)",
         f"- Task config: `{meta['task_config']}` (sha256 `{meta['task_config_sha256'][:12]}`)",
         f"- Pass criteria: {'FROZEN `' + meta['criteria_sha256'][:12] + '`' if meta.get('criteria_sha256') else 'PROVISIONAL (development run)'}"
         f" — anchor `{anchor_name}`, margin {meta['criteria']['improvement_margin'] * 100:.0f}%",
         f"- Judge: `{meta['judge']['id']}` (blind to model identity and hidden accuracy)",
         f"- Git commit: `{meta['git_commit'][:10]}`", "",
         "## Per-model summary (95% intervals: Wilson for rates, bootstrap over cases otherwise)", "",
         "| Model | Pass@1 | Valid exec | Integrity | Transfer MAE (bpm) | Improvement vs anchor | Judge score (0–1) | Runtime (min) | Agent cost (USD, mean / total) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for m in models:
        pm = per_model[m]
        mid = next(x["id"] for x in meta["models"] if x["name"] == m)
        cost = f"{pm['cost_usd_mean']:.3f} / {pm['cost_usd_total']:.2f}" if pm["cost_usd_mean"] is not None else "–"
        L.append(f"| {m} (`{mid}`) | {_fmt(pm['pass_at_1'], True)} | {_fmt(pm['valid_rate'], True)} | {_fmt(pm['integrity_rate'], True)} | "
                 f"{_fmt(pm['agent_mae'])} | {_fmt(pm['improvement'], True)} | {_fmt(pm['judge'])} | {_fmt(pm['runtime_min'], nd=1)} | {cost} |")
    L += ["", "## Elo (Bradley–Terry, margin-aware; 95% bootstrap over subjects)", "",
          f"Every episode plays every other model's episodes on the same subject. Outcomes are soft: "
          f"0.5 + 0.5·tanh(gap/τ), with τ = {ACC_TAU:.2f} improvement for accuracy and τ = {JUDGE_TAU:.2f} for judge score. "
          "Invalid submissions lose the accuracy game outright; integrity failures lose both games. "
          f"Combined = {ACC_WEIGHT:.1f}·accuracy + {1 - ACC_WEIGHT:.1f}·judge.", "",
          "| Model | Combined Elo | Accuracy Elo | Judge Elo |", "|---|---|---|---|"]
    for m in sorted(models, key=lambda m: -(elo["combined"][m][0] or 0)):
        L.append(f"| {m} | {_fmt(elo['combined'][m], nd=0)} | {_fmt(elo['accuracy'][m], nd=0)} | {_fmt(elo['judge'][m], nd=0)} |")
    if elo["combined"][models[0]][0] is not None:
        L += ["", "Implied P(row beats column), combined Elo:", "",
              "| | " + " | ".join(models) + " |", "|---" * (len(models) + 1) + "|"]
        for a in models:
            L.append(f"| {a} | " + " | ".join("–" if a == b else f"{elo_win_prob(elo['combined'][a][0], elo['combined'][b][0]):.2f}"
                                              for b in models) + " |")
    L += ["", "## Baselines on the same hidden transfer windows (phase-balanced MAE, bpm)", "",
          "| Baseline | Mean [95% CI] |", "|---|---|"]
    for b in base_names:
        L.append(f"| {b}{' (anchor)' if b == anchor_name else ''} | {_fmt(base_mean[b])} |")
    L += ["", "## Paired model differences in transfer MAE (A − B, bpm; negative = A better)", "",
          "| A | B | n | Mean [95% CI] |", "|---|---|---|---|"]
    for a, b, n, ci in pairs:
        L.append(f"| {a} | {b} | {n} | {_fmt(ci)} |")
    L += ["", "## Diagnosis (predeclared rules)", ""] + [f"- {d}" for d in diag]
    L += ["", "## Per-episode results", "",
          "| Model | Subject | Submitted | Valid | Agent MAE | Anchor MAE | Improvement | Judge | Integrity | Pass | Turns | Feedback calls | Wall (min) | Cost (USD) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r.get("model"), r.get("subject"))):
        if "harness_error" in r:
            L.append(f"| {r['model']} | {r['subject']} | harness error | | | | | | | ✗ | | | | |")
            continue
        imp = r.get("improvement_vs_anchor")
        js = (r.get("judge") or {}).get("judge_score")
        tick = lambda b: "✓" if b else "✗"  # noqa: E731
        cells = [r["model"], r["subject"], tick(r["submitted"]), tick(r["valid_execution"]),
                 f"{r['agent_mae']:.2f}" if r.get("agent_mae") is not None else
                 (f"({r['unsubmitted_fallback_mae']:.2f} unsubmitted)" if r.get("unsubmitted_fallback_mae") is not None else "–"),
                 f"{r['anchor_mae']:.2f}",
                 f"{imp * 100:+.1f}%" if imp is not None else "–",
                 f"{js:.2f}" if js is not None else "–",
                 tick(r["criteria"]["integrity"]), tick(r["pass"]), str(r["usage"]["turns"]), str(r["feedback_used"]),
                 f"{r['usage']['agent_wall_s'] / 60:.1f}", f"{r['usage']['cost_usd']:.3f}"]
        L.append("| " + " | ".join(cells) + " |")
    L += ["", "## Integrity details", ""]
    for r in sorted(rows, key=lambda r: (r.get("model"), r.get("subject"))):
        if "harness_error" in r:
            continue
        failed = [k for k, v in r["criteria"].items() if v is False and k not in ("pass",)]
        net = len(r["programmatic_integrity"]["network_attempt_flags"])
        if r.get("infra_error"):
            L.append(f"- **{r['model']} / {r['subject']}**: INFRASTRUCTURE ERROR (provider API), not a model failure: {r['infra_error'][:160]}")
        if failed or net:
            L.append(f"- **{r['model']} / {r['subject']}**: failed {', '.join(failed) or 'none'}"
                     + (f"; {net} network-attempt flag(s)" if net else ""))
    L += ["", "## Notes and limitations", "",
          "- Each participant has one recording session; 'unseen' means held-out activity segments with guard gaps, not separate sessions.",
          "- Network isolation on this host is best-effort (no network namespaces). A command that tries to use the network or read hidden paths is rejected and fails integrity.",
          "- Judge scores grade decisions and reporting, not accuracy, and are reported separately from measured error; only the combined Elo mixes them, with the weight stated above.",
          "- Small n: intervals are wide; treat rankings as provisional.", ""]
    out = run_dir / "summary.md"
    out.write_text("\n".join(L))
    (run_dir / "summary.json").write_text(json.dumps({"per_model": per_model, "elo": elo, "baselines": base_mean, "pairs": pairs, "diagnosis": diag}, indent=1, default=str))
    return out
