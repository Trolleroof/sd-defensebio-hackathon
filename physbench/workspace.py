"""Build one episode: agent-visible workspace + evaluator-private files."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import sandbox
from .baselines import kinetic
from .config import ROOT, TaskConfig
from .data.dalia import ACTIVITY_NAMES, CHANNELS, load_recording
from .episodes import Split, Windows, dev_windows, split_subject, support_windows, transfer_windows

TEMPLATE = Path(__file__).parent / "workspace_template"


@dataclass
class Episode:
    episode_id: str
    run_id: str
    subject: str
    user: str
    ws: Path  # agent-visible workspace
    private: Path  # evaluator-only directory
    split: Split
    dev: Windows
    transfer: Windows

    @property
    def sandbox_dir(self) -> Path:
        return self.ws.parent


def _frame(cfg: TaskConfig, rec: dict, mask: np.ndarray, segment: np.ndarray | None) -> pd.DataFrame:
    cols = {"t": rec["t"][mask]}
    if segment is not None:
        cols["segment"] = segment[mask]
    cols[cfg.target_channel] = rec[cfg.target_channel][mask]
    for c in cfg.input_channels:
        cols[c] = rec[c][mask]
    df = pd.DataFrame(cols)
    if "activity" in df:
        df["activity"] = df["activity"].astype(int)
        df["activity_name"] = df["activity"].map(ACTIVITY_NAMES)
    return df


def _channels_doc(cfg: TaskConfig) -> dict:
    return {
        "input_channels": list(cfg.input_channels),
        "target_channel": cfg.target_channel,
        "target_provenance": cfg.target_provenance,
        "sample_period_s": cfg.step_s,
        "sample_time_convention": "t is seconds since recording start; each value uses only data at or before t",
        "channels": {c: dict(zip(["unit", "source", "description"], CHANNELS[c]))
                     for c in [cfg.target_channel, *cfg.input_channels]},
        "activity_names": {str(k): v for k, v in ACTIVITY_NAMES.items()},
    }


def _render_task(cfg: TaskConfig, n_dev: int, n_transfer: int, criteria: dict | None) -> str:
    text = (TEMPLATE / "TASK.md").read_text()
    margin = criteria["improvement_margin"] if criteria else 0.10
    b = cfg.budgets
    repl = {
        "{{HISTORY_S}}": str(cfg.history_s), "{{HORIZON_S}}": str(cfg.horizon_s),
        "{{FAMILIAR}}": ", ".join(cfg.familiar_activities), "{{TRANSFER}}": ", ".join(cfg.transfer_activities),
        "{{EXERTION}}": ", ".join(cfg.exertion_activities), "{{N_DEV}}": str(n_dev), "{{N_TRANSFER}}": str(n_transfer),
        "{{MAX_FEEDBACK}}": str(b["max_feedback_calls"]), "{{MAX_TURNS}}": str(b["max_turns"]),
        "{{WALL_MIN}}": str(int(b["wall_seconds"] // 60)), "{{CMD_TIMEOUT}}": str(b["command_timeout_s"]),
        "{{PREDICT_TIMEOUT}}": str(b["predict_timeout_s"]), "{{MARGIN_PCT}}": f"{margin * 100:.0f}",
        "{{N_REF}}": str(len(cfg.reference_subjects)), "{{INPUTS}}": ", ".join(cfg.input_channels),
    }
    for k, v in repl.items():
        text = text.replace(k, v)
    return text


def build_episode(cfg: TaskConfig, subject: str, run_id: str, episode_id: str, criteria: dict | None = None) -> Episode:
    split = split_subject(cfg, subject)
    dev, transfer = dev_windows(cfg, split), transfer_windows(cfg, split)

    user = "pb_" + episode_id.replace("-", "")[-24:].lower()
    root = sandbox.SANDBOX_ROOT / run_id / episode_id
    if root.exists():
        shutil.rmtree(root)
    ws = root / "ws"
    private = ROOT / "runs" / run_id / episode_id / "private"
    private.mkdir(parents=True, exist_ok=True)

    shutil.copytree(TEMPLATE, ws, ignore=shutil.ignore_patterns("TASK.md", "REPORT_TEMPLATE.md", "eval_runner.py", "__pycache__"))
    (ws / "TASK.md").write_text(_render_task(cfg, len(dev), len(transfer), criteria))
    (ws / "docs").mkdir(exist_ok=True)
    (ws / "docs" / "channels.json").write_text(json.dumps(_channels_doc(cfg), indent=2))

    # Target participant: support data only.
    tdir = ws / "data" / "target"
    tdir.mkdir(parents=True)
    _frame(cfg, split.rec, split.support_mask, split.segment).to_csv(tdir / "support.csv", index=False, float_format="%.5g")
    meta = json.loads((ROOT / cfg.processed_dir / f"{subject}.json").read_text())
    pub = {k: meta[k] for k in ("age", "gender", "height_cm", "weight_kg", "sport_level")}
    (tdir / "meta.json").write_text(json.dumps({"participant": "P", **pub}, indent=2))

    # Reference cohort: complete recordings, pseudonymised.
    rdir = ws / "data" / "reference"
    rdir.mkdir(parents=True)
    rng = np.random.default_rng(int(hashlib.sha256(episode_id.encode()).hexdigest()[:8], 16))
    for i, s in enumerate(rng.permutation(cfg.reference_subjects)):
        rec = load_recording(ROOT / cfg.processed_dir / f"{s}.npz")
        _frame(cfg, rec, np.ones(len(rec["t"]), bool), np.zeros(len(rec["t"]), int)).to_csv(
            rdir / f"R{i + 1}.csv", index=False, float_format="%.5g")
        m = json.loads((ROOT / cfg.processed_dir / f"{s}.json").read_text())
        (rdir / f"R{i + 1}_meta.json").write_text(json.dumps({k: m[k] for k in pub}, indent=2))

    # Working baseline, pre-fitted on support data; submission starts as a copy.
    sup = support_windows(cfg, split)
    x = np.stack([sup.fut[f] for f in kinetic.FEATURES], -1)
    params = kinetic.fit(sup.hist[cfg.target_channel][:, -1], x, sup.target)
    (ws / "baseline" / "params.json").write_text(json.dumps({"params": params.tolist(), "features": kinetic.FEATURES}))
    sub = ws / "submission"
    sub.mkdir()
    for f in ("predict.py", "kinetic.py", "params.json"):
        shutil.copy(ws / "baseline" / f, sub / f)
    (sub / "REPORT.md").write_text((TEMPLATE / "REPORT_TEMPLATE.md").read_text())
    (ws / "notes").mkdir()

    # Hidden windows stay root-only.
    for name, w in (("dev", dev), ("transfer", transfer)):
        np.savez_compressed(private / f"{name}_windows.npz", **w.inputs_npz(), target=w.target, activity=w.activity,
                            start_idx=w.start_idx)
    (private / "episode.json").write_text(json.dumps({"subject": subject, "episode_id": episode_id, "run_id": run_id}))
    os.chmod(private, 0o700)

    uid, gid = sandbox.ensure_user(user)
    os.chmod(root.parent, 0o711)
    os.chown(root, uid, gid)
    os.chmod(root, 0o700)
    sandbox.chown_tree(ws, uid, gid)
    return Episode(episode_id, run_id, subject, user, ws, private, split, dev, transfer)


def run_submission(cfg: TaskConfig, ep: Episode, windows: Windows, label: str) -> tuple[bool, np.ndarray | None, dict]:
    """Execute the current submission on ``windows`` as the episode user."""
    d = ep.sandbox_dir / f"eval_{label}"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir()
    shutil.copytree(ep.ws / "submission", d / "submission", symlinks=True)
    np.savez(d / "windows.npz", **windows.inputs_npz())
    shutil.copy(TEMPLATE / "eval_runner.py", d / "eval_runner.py")
    uid, gid = sandbox.ensure_user(ep.user)
    sandbox.chown_tree(d, uid, gid)
    rc, out, secs = sandbox.run_as(ep.user, [str(sandbox.AGENT_PY), "eval_runner.py", "submission", "windows.npz", "preds.npy"],
                                   cwd=d, timeout=cfg.budgets["predict_timeout_s"], mem_gb=cfg.budgets["memory_gb"], max_chars=4000)
    info = {"returncode": rc, "seconds": round(secs, 2), "log": out[-2000:]}
    preds = None
    if rc == 0 and (d / "preds.npy").exists():
        try:
            preds = np.load(d / "preds.npy", allow_pickle=False)
        except Exception as e:  # noqa: BLE001
            info["log"] += f"\ncould not read predictions: {e}"
    shutil.rmtree(d, ignore_errors=True)
    return preds is not None, preds, info
