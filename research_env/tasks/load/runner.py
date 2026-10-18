"""Trusted controller for the load task: permitted-data materialization, scoring calls,
public development budget and artifact replay.

The controller holds private records. Workers receive only the allow-listed
training file or allow-listed requests built by ``schema.build_request``.
Request IDs are opaque; the mapping back to record IDs never leaves here.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from . import worker
from .importer import read_jsonl
from .schema import LOADS_KG, TRACKS, build_request
from .scoring import score

PACKAGE_DIR = Path(__file__).resolve().parent
TASK_MANIFEST = PACKAGE_DIR / "config" / "task_manifest.json"
TRAIN_FIELDS = ("person_key", "sex", "body_mass_kg", "stature_cm", "age_years", "load_kg",
                "speed_m_s", "grade_fraction", "reference_w")


def task_manifest() -> dict:
    return json.loads(TASK_MANIFEST.read_text())


def _opaque(kind: str, value: str) -> str:
    return kind + hashlib.sha256(f"load-evobreath-v1/{kind}/{value}".encode()).hexdigest()[:10]


def training_rows(records: list[dict], track: str) -> list[dict]:
    """Eligible training-split rows at the track's training loads, allow-listed fields only."""
    loads = set(TRACKS[track]["train_loads"])
    rows = []
    for r in records:
        if r["split"] == "train" and r["eligible"] and r["load_kg"] in loads:
            row = {k: r[k] for k in TRAIN_FIELDS if k != "person_key"}
            row["person_key"] = _opaque("p", r["source_person_id"])
            rows.append(row)
    return rows


def requests_for(records: list[dict], track: str, split: str):
    """Returns (requests_by_group, request_to_record, targets). Groups are opaque per person."""
    if split == "train":
        raise ValueError("requests are built for development/test people only")
    score_loads = set(TRACKS[track]["score_loads"])
    by_person: dict[str, list[dict]] = {}
    for r in records:
        if r["split"] == split:
            by_person.setdefault(r["source_person_id"], []).append(r)
    groups, mapping, targets = {}, {}, []
    for pid, prs in sorted(by_person.items()):
        reqs = []
        for r in sorted(prs, key=lambda x: x["load_kg"]):
            if not (r["eligible"] and r["load_kg"] in score_loads):
                continue
            if TRACKS[track]["calibration"] and not any(
                    x["load_kg"] == 0.0 and x["eligible"] for x in prs):
                continue
            rid = _opaque("r", f"{split}/{r['record_id']}")
            reqs.append(build_request(r, track, prs, request_id=rid))
            mapping[rid] = r["record_id"]
            targets.append(r)
        if reqs:
            groups[_opaque("g", f"{split}/{pid}")] = reqs
    return groups, mapping, targets


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def scoring_config(track: str, split: str) -> dict:
    tm = task_manifest()
    frozen = tm["skill"]["frozen_baseline_loss_w"].get(track) or {}
    return {"score_loads": list(TRACKS[track]["score_loads"]), "epsilon_w": tm["skill"]["epsilon_w"],
            "frozen_baseline_loss": frozen.get(split)}


def train_candidate(prepared_dir, source_dir, track: str, seed: int, artifact_out,
                    train_manifest_extra: dict | None = None) -> dict:
    records = read_jsonl(Path(prepared_dir) / "records.jsonl")
    rows_path = Path(artifact_out).parent / f"_train_rows_{track}.jsonl"
    write_jsonl(rows_path, training_rows(records, track))
    tm = {"track": track, **(train_manifest_extra or {})}
    try:
        return worker.train_in_worker(source_dir, rows_path, tm, seed, artifact_out,
                                      forbidden_paths=[str(Path(prepared_dir).resolve())])
    finally:
        rows_path.unlink(missing_ok=True)


def evaluate(prepared_dir, artifact_dir, track: str, split: str) -> dict:
    records = read_jsonl(Path(prepared_dir) / "records.jsonl")
    groups, mapping, targets = requests_for(records, track, split)
    run = worker.predict_in_worker(artifact_dir, groups, forbidden_paths=[str(Path(prepared_dir).resolve())])
    predictions = {mapping[rid]: pred for rid, pred in run["predictions"].items()}
    result = score(targets, predictions, scoring_config(track, split))
    if not run["ok"] and result["execution_valid"]:  # defensive: never valid if a worker failed
        result = {**result, "execution_valid": False, "failure_reason": "worker failure"}
    result["worker"] = [{k: v for k, v in r.items() if k != "group"} for r in run["runs"]]
    result["track"], result["split"] = track, split
    result["predictions_sha256"] = hashlib.sha256(
        json.dumps(predictions, sort_keys=True).encode()).hexdigest()
    return result


class DevBudget:
    """Counts public development calls (failed calls included) and logs immutable events."""

    def __init__(self, run_dir, max_calls: int | None = None):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "dev_calls.jsonl"
        self.max_calls = max_calls if max_calls is not None else task_manifest()["budget"]["dev_calls"]

    def used(self) -> int:
        return len(read_jsonl(self.path)) if self.path.exists() else 0

    def call(self, prepared_dir, artifact_dir, track: str, candidate_id: str) -> dict:
        if self.used() >= self.max_calls:
            return {"execution_valid": False, "failure_reason": "development call budget exhausted",
                    "budget": {"used": self.used(), "max": self.max_calls}}
        result = evaluate(prepared_dir, artifact_dir, track, "development")
        # Public feedback: aggregate and per-load only; no per-person rows or predictions.
        public = {k: result.get(k) for k in ("execution_valid", "failure_reason", "primary", "skill",
                                              "secondary", "uncertainty", "per_load", "n_people")}
        if public.get("secondary"):
            public["secondary"] = {k: v for k, v in public["secondary"].items() if k != "mae_w_by_sex"}
        event = {"t": time.time(), "candidate_id": candidate_id, "track": track,
                 "artifact_sha256": artifact_hash(artifact_dir), "public_result": public}
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, sort_keys=True) + "\n")
        public["budget"] = {"used": self.used(), "max": self.max_calls}
        return public


def artifact_hash(artifact_dir) -> str:
    h = hashlib.sha256()
    for p in sorted(Path(artifact_dir).rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            h.update(str(p.relative_to(artifact_dir)).encode())
            h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def replay(prepared_dir, source_dir, track: str, seed: int, reference_artifact, work_dir,
           train_manifest_extra=None) -> dict:
    """Retrain from source with the same seed and compare artifact bytes and test predictions."""
    out = Path(work_dir) / "replayed_artifact"
    res = train_candidate(prepared_dir, source_dir, track, seed, out, train_manifest_extra)
    if not res["ok"]:
        return {"replay_ok": False, "error": res.get("error"), "stderr_tail": res.get("stderr_tail")}
    a, b = artifact_hash(reference_artifact), artifact_hash(out)
    ea = evaluate(prepared_dir, reference_artifact, track, "test")
    eb = evaluate(prepared_dir, out, track, "test")
    return {"replay_ok": a == b and ea["predictions_sha256"] == eb["predictions_sha256"],
            "artifact_sha256": [a, b], "predictions_sha256": [ea["predictions_sha256"], eb["predictions_sha256"]]}


__all__ = ["training_rows", "requests_for", "train_candidate", "evaluate", "DevBudget", "replay",
           "artifact_hash", "LOADS_KG"]
