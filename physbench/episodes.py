"""Split one participant into support / development / hidden-transfer data.

Within a target participant:

- transfer block: from the first transfer-activity sample to the first
  familiar-activity sample after the last transfer-activity sample (so the
  recovery after the transfer activities stays hidden too);
- development chunks: contiguous chunks of familiar time held out for
  feedback;
- support: the remaining familiar time, minus guard gaps around every
  held-out region so no support sample sits next to a scored window.

Windows are history (observed target + covariates) followed by a horizon
(covariates only); the predictor returns the target over the horizon.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import ROOT, TaskConfig
from .data.dalia import ACTIVITY_NAMES, load_recording

NAME_TO_ID = {v: k for k, v in ACTIVITY_NAMES.items()}


@dataclass
class Windows:
    """Fixed-length windows. ``hist[ch]`` is (N, H); ``fut[ch]`` is (N, F)."""

    hist: dict[str, np.ndarray]
    fut: dict[str, np.ndarray]
    target: np.ndarray  # (N, F) hidden horizon target
    activity: np.ndarray  # (N, F) activity ids over the horizon (for scoring breakdowns)
    start_idx: np.ndarray = field(default_factory=lambda: np.zeros(0, int))

    def __len__(self):
        return len(self.target)

    def inputs_npz(self) -> dict[str, np.ndarray]:
        out = {f"hist_{k}": v for k, v in self.hist.items()}
        out.update({f"fut_{k}": v for k, v in self.fut.items()})
        return out

    def subset(self, idx) -> "Windows":
        return Windows(
            {k: v[idx] for k, v in self.hist.items()},
            {k: v[idx] for k, v in self.fut.items()},
            self.target[idx],
            self.activity[idx],
            self.start_idx[idx],
        )

    @staticmethod
    def concat(ws: list["Windows"]) -> "Windows":
        ws = [w for w in ws if len(w)]
        return Windows(
            {k: np.concatenate([w.hist[k] for w in ws]) for k in ws[0].hist},
            {k: np.concatenate([w.fut[k] for w in ws]) for k in ws[0].fut},
            np.concatenate([w.target for w in ws]),
            np.concatenate([w.activity for w in ws]),
            np.concatenate([w.start_idx for w in ws]),
        )


@dataclass
class Split:
    subject: str
    rec: dict[str, np.ndarray]
    support_mask: np.ndarray
    dev_mask: np.ndarray
    transfer_mask: np.ndarray
    segment: np.ndarray  # contiguous support segment id, -1 outside support


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index runs where mask is True."""
    m = np.r_[False, mask, False].astype(int)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def transfer_block_mask(cfg: TaskConfig, activity: np.ndarray) -> np.ndarray:
    """Hidden transfer block: first transfer activity through the sample before familiar time resumes."""
    act = np.asarray(activity).astype(int)
    n = len(act)
    transfer_ids = [NAME_TO_ID[a] for a in cfg.transfer_activities]
    familiar_ids = [NAME_TO_ID[a] for a in cfg.familiar_activities]
    is_tr = np.isin(act, transfer_ids)
    is_fam = np.isin(act, familiar_ids)
    transfer = np.zeros(n, bool)
    if not is_tr.any():
        return transfer
    t0 = int(np.flatnonzero(is_tr)[0])
    last_tr = int(np.flatnonzero(is_tr)[-1])
    after = np.flatnonzero(is_fam & (np.arange(n) > last_tr))
    t1 = int(after[0]) if len(after) else n
    transfer[t0:t1] = True
    return transfer


def _guarded(mask: np.ndarray, guard: int) -> np.ndarray:
    near = np.zeros(len(mask), bool)
    for a, b in _runs(mask):
        near[max(0, a - guard):min(len(mask), b + guard)] = True
    return near


def reference_visible_mask(cfg: TaskConfig, activity: np.ndarray) -> np.ndarray:
    """Reference samples the agent may see.

    The transfer block and the guard around it are removed, so heart rate during
    the hidden activities is not in the workspace and cannot be memorized from
    other people.
    """
    guard = int(cfg.guard_s // cfg.step_s)
    return ~_guarded(transfer_block_mask(cfg, activity), guard)


def split_subject(cfg: TaskConfig, subject: str) -> Split:
    rec = load_recording(ROOT / cfg.processed_dir / f"{subject}.npz")
    act = rec["activity"].astype(int)
    n = len(act)
    guard = int(cfg.guard_s // cfg.step_s)
    transfer = transfer_block_mask(cfg, act)

    # Development chunks: every k-th fixed-size chunk of familiar time outside the transfer block.
    chunk = int(cfg.dev_chunk_s // cfg.step_s)
    every = max(2, int(round(1 / cfg.dev_fraction)))
    dev = np.zeros(n, bool)
    eligible = ~_guarded(transfer, guard)
    for a, b in _runs(eligible):
        starts = list(range(a + guard, b - guard - chunk + 1, chunk))
        for j, s in enumerate(starts):
            if j % every == every - 1 or (len(starts) < every and j == len(starts) // 2 and b - a > 3 * chunk):
                dev[s:s + chunk] = True

    blocked = transfer | dev
    near = _guarded(blocked, guard)
    support = ~near
    seg = np.full(n, -1)
    for i, (a, b) in enumerate(_runs(support)):
        seg[a:b] = i
    return Split(subject, rec, support, dev, transfer, seg)


def make_windows(cfg: TaskConfig, rec: dict, region: np.ndarray, *, history_inside: bool, stride_n: int | None = None) -> Windows:
    """Windows whose horizon lies inside ``region``.

    ``history_inside``: require the history to lie inside ``region`` too
    (used for dev and support windows). Transfer windows may draw history
    from just before the block — that is observed input, never scored.
    """
    H, F = cfg.hist_n, cfg.hor_n
    stride = stride_n or cfg.stride_n
    chans = list(cfg.input_channels)
    tgt = cfg.target_channel
    starts = []
    for a, b in _runs(region):
        h0 = a + H if history_inside else max(a, H)
        while h0 + F <= b:
            s = h0 - H
            if np.all(np.isfinite(rec[tgt][s:h0 + F])):
                starts.append(s)
            h0 += stride
    starts = np.array(starts, int)
    idx_h = starts[:, None] + np.arange(H)[None] if len(starts) else np.zeros((0, H), int)
    idx_f = starts[:, None] + H + np.arange(F)[None] if len(starts) else np.zeros((0, F), int)
    hist = {"t": rec["t"][idx_h], tgt: rec[tgt][idx_h]}
    hist.update({c: rec[c][idx_h] for c in chans})
    fut = {"t": rec["t"][idx_f]}
    fut.update({c: rec[c][idx_f] for c in chans})
    return Windows(hist, fut, rec[tgt][idx_f], rec["activity"][idx_f].astype(int), starts)


def support_windows(cfg: TaskConfig, split: Split, stride_n: int = 5) -> Windows:
    return make_windows(cfg, split.rec, split.support_mask, history_inside=True, stride_n=stride_n)


def dev_windows(cfg: TaskConfig, split: Split) -> Windows:
    return make_windows(cfg, split.rec, split.dev_mask, history_inside=True)


def transfer_windows(cfg: TaskConfig, split: Split) -> Windows:
    return make_windows(cfg, split.rec, split.transfer_mask, history_inside=False)


def reference_visible_windows(cfg: TaskConfig, subject: str, stride_n: int = 5) -> Windows:
    """Reference-cohort windows on the same samples the agent is allowed to read."""
    rec = load_recording(ROOT / cfg.processed_dir / f"{subject}.npz")
    mask = reference_visible_mask(cfg, rec["activity"])
    return make_windows(cfg, rec, mask, history_inside=True, stride_n=stride_n)


def full_windows(cfg: TaskConfig, subject: str, stride_n: int = 5) -> Windows:
    """All windows of a participant (used for reference-cohort fitting)."""
    sp = split_subject(cfg, subject)
    return make_windows(cfg, sp.rec, np.ones(len(sp.rec["t"]), bool), history_inside=True, stride_n=stride_n)


def exertion_mask(cfg: TaskConfig, activity: np.ndarray) -> np.ndarray:
    return np.isin(activity, [NAME_TO_ID[a] for a in cfg.exertion_activities])
