"""INTERIM trusted controller for one heat research session.

Holds private requests/targets outside the participant workspace, enforces the
wall and development-call budgets, records immutable experiment events, and
replays the final selection privately. Replace with Set 1's shared runner when
``research_env/core`` exists; the task-specific pieces it calls (requests,
targets, scoring, worker JSON protocol) are designed to move over unchanged.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import secrets
import shutil
import time
from pathlib import Path

from . import requests as rq
from . import scoring, worker
from .importer import sha256_json

TASK_MANIFEST = Path(__file__).with_name("task_manifest.json")
TEMPLATE = Path(__file__).with_name("workspace_template")
BUDGET = {"wall_s": 7200, "dev_calls": 30, "cpu_cores": 8,
          "memory_bytes": worker.LIMITS["memory_bytes"]}
CHECKPOINTS_MIN = (15, 30, 60, 120)


def _rows(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l]


def _write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1, sort_keys=True, allow_nan=False))


def tree_hash(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(Path(root).rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            h.update(str(p.relative_to(root)).encode() + b"\0" + hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def _person_bundle(subject, alias, inputs, refs, context):
    return {"person": alias,
            "context": {f: context[str(subject)][f]["value"] for f in rq.CONTEXT_FIELDS},
            "inputs": [{"minute": r["minute"], "field": r["field"], "value": r["value"]}
                       for r in inputs if r["subject"] == subject],
            "reference": [{"minute": r["minute"], "rectal_c": r["rectal_c"], "eligible": r["eligible"]}
                          for r in refs if r["subject"] == subject]}


def build_split(prepared: Path, group: str, tier: str, secret: str):
    """Return (request_groups, targets) for one held-out group; targets are private."""
    manifest = json.loads((prepared / "dataset_manifest.json").read_text())
    inputs, refs = _rows(prepared / "inputs.jsonl"), _rows(prepared / "references.jsonl")
    context = json.loads((prepared / "context.json").read_text())
    groups, targets = [], []
    for subject in manifest["split_manifest"]["groups"][group]:
        person = rq.opaque_id(secret, "person", subject)
        p_inputs = [r for r in inputs if r["subject"] == subject]
        p_refs = [r for r in refs if r["subject"] == subject]
        reqs = []
        for c in rq.CUTOFFS_MIN:
            rid = rq.opaque_id(secret, "request", subject, tier, c)
            reqs.append(rq.build_request(rid, c, p_inputs, context[str(subject)], tier, p_refs))
            targets.extend({**t, "person": person} for t in rq.build_targets(rid, c, p_refs))
        groups.append(reqs)
    return groups, targets


def build_train_manifest(prepared: Path, tier: str, secret: str) -> dict:
    manifest = json.loads((prepared / "dataset_manifest.json").read_text())
    inputs, refs = _rows(prepared / "inputs.jsonl"), _rows(prepared / "references.jsonl")
    context = json.loads((prepared / "context.json").read_text())
    people = [_person_bundle(s, "train-" + rq.opaque_id(secret, "train", s)[:8], inputs, refs, context)
              for s in manifest["split_manifest"]["groups"]["train"]]
    people.sort(key=lambda p: p["person"])
    return {"task": "heat-core-forecast-v1", "tier": tier, "people": people,
            "fields": {"inputs": list(rq.HISTORY_FIELDS), "context": list(rq.CONTEXT_FIELDS)},
            "note": "rectal_c is the reference target; it never appears in inference requests "
                    "except the minute-1 value in the calibrated tier"}


def evaluate(source_dir: Path, artifact_dir: Path, train_manifest: Path, groups, targets,
             task: dict, tier: str, seed: int, denied=(), retrain=True, split="development") -> dict:
    """Train (optionally) then predict and score one split. Returns score + worker logs."""
    logs = {}
    if retrain:
        tr = worker.run("train", source_dir, artifact_dir, train_manifest=train_manifest, seed=seed,
                        denied=denied)
        logs["train"] = {k: tr.get(k) for k in ("ok", "error", "usage", "stdout", "stderr", "returncode")}
        if not tr["ok"]:
            return {"score": {"execution_valid": False, "primary_loss_c": None,
                              "skill_vs_baseline": None, "failure_reason": tr["error"]}, "logs": logs}
    pr = worker.run("predict", source_dir, artifact_dir, request_groups=groups, denied=denied)
    logs["predict"] = {k: pr.get(k) for k in ("ok", "error", "usage", "stdout", "stderr", "returncode")}
    if not pr["ok"]:
        return {"score": {"execution_valid": False, "primary_loss_c": None, "skill_vs_baseline": None,
                          "failure_reason": pr["error"]}, "logs": logs}
    reqs = {r["request_id"]: r for g in groups for r in g}
    frozen = task["frozen_baselines"].get(tier, {})
    cfg = {"requests": reqs, "baseline_loss_c": frozen.get("loss_c", {}).get(split),
           "baseline_id": frozen.get("id")}
    return {"score": scoring.score(targets, pr["predictions"], cfg), "logs": logs,
            "predictions_sha256": sha256_json(pr["predictions"])}


class Session:
    def __init__(self, root: Path, clock=time.time):
        self.root = Path(root)
        self.public = self.root / "workspace"
        self.private = self.root / "private"
        self.clock = clock

    # ---- creation ---------------------------------------------------------
    @classmethod
    def create(cls, prepared: Path, root: Path, tier: str = "wearable", clock=time.time,
               secret: str | None = None, budget: dict | None = None) -> "Session":
        root = Path(root)
        if root.exists():
            raise FileExistsError(root)
        s = cls(root, clock)
        prepared = Path(prepared).resolve()
        secret = secret or secrets.token_hex(16)
        shutil.copytree(TEMPLATE, s.public, ignore=shutil.ignore_patterns("__pycache__"))
        s.private.mkdir(parents=True)
        (s.private / "candidates").mkdir()
        train = build_train_manifest(prepared, tier, secret)
        (s.public / "data").mkdir()
        _write(s.public / "data" / "train_manifest.json", train)
        dev_groups, dev_targets = build_split(prepared, "development", tier, secret)
        test_groups, test_targets = build_split(prepared, "private_test", tier, secret)
        _write(s.public / "data" / "example_request.json", dev_groups[0][10])
        _write(s.private / "dev_requests.json", dev_groups)
        _write(s.private / "dev_targets.json", dev_targets)
        _write(s.private / "test_requests.json", test_groups)
        _write(s.private / "test_targets.json", test_targets)
        task = json.loads(TASK_MANIFEST.read_text())
        _write(s.private / "task_manifest.json", task)
        public_task = {k: task[k] for k in task["participant_visible_keys"]}
        public_task["budget"] = {**task["budget"], **(budget or {})}
        # private-test baseline losses stay private; only development values are published
        public_task["frozen_baselines"] = {
            t: {"id": v["id"], "loss_c": {"development": v["loss_c"]["development"]}}
            for t, v in task["frozen_baselines"].items()}
        _write(s.public / "task_manifest.json", public_task)
        state = {"tier": tier, "secret": secret, "created_at": clock(), "prepared": str(prepared),
                 "dataset_manifest_sha256": hashlib.sha256((prepared / "dataset_manifest.json").read_bytes()).hexdigest(),
                 "train_manifest_sha256": sha256_json(train), "budget": {**BUDGET, **(budget or {})},
                 "dev_calls_used": 0, "incumbents": [], "submitted": None, "last_event_sha256": None,
                 "events": 0}
        _write(s.private / "state.json", state)
        s._event("session_created", {"tier": tier, "workspace_sha256": tree_hash(s.public)})
        return s

    # ---- state / events ---------------------------------------------------
    @property
    def state(self):
        return json.loads((self.private / "state.json").read_text())

    def _save(self, state):
        _write(self.private / "state.json", state)

    def elapsed_s(self):
        return self.clock() - self.state["created_at"]

    def _event(self, kind, payload):
        state = self.state
        body = {"event_id": f"e{state['events'] + 1:04d}", "type": kind,
                "elapsed_s": round(self.clock() - state["created_at"], 3),
                "prev_sha256": state["last_event_sha256"], **payload}
        body["event_sha256"] = sha256_json(body)
        with (self.private / "events.jsonl").open("a") as f:
            f.write(json.dumps(body, sort_keys=True, allow_nan=False) + "\n")
        state["events"] += 1
        state["last_event_sha256"] = body["event_sha256"]
        self._save(state)
        return body

    def events(self):
        return _rows(self.private / "events.jsonl")

    def _denied(self):
        return [str(self.private.resolve())]

    # ---- participant operations ------------------------------------------
    def register(self, source_dir: Path) -> str:
        source_dir = Path(source_dir)
        cid = "c-" + tree_hash(source_dir)[:16]
        dest = self.private / "candidates" / cid / "src"
        if not dest.exists():
            shutil.copytree(source_dir, dest, ignore=shutil.ignore_patterns("__pycache__"))
        return cid

    def _diff(self, cid):
        prev = [e["candidate_id"] for e in self.events() if e["type"] == "dev_eval"]
        if not prev:
            return None
        a, b = self.private / "candidates" / prev[-1] / "src", self.private / "candidates" / cid / "src"
        out = []
        for rel in sorted({p.relative_to(a) for p in a.rglob("*.py")} | {p.relative_to(b) for p in b.rglob("*.py")}):
            la = (a / rel).read_text().splitlines(True) if (a / rel).exists() else []
            lb = (b / rel).read_text().splitlines(True) if (b / rel).exists() else []
            out.extend(difflib.unified_diff(la, lb, f"{prev[-1]}/{rel}", f"{cid}/{rel}"))
        return "".join(out)[:20000]

    def dev_eval(self, source_dir: Path, decision: dict, seed: int = 0) -> dict:
        """One public development call. Counts against the budget even if it fails."""
        state = self.state
        if state["submitted"]:
            raise PermissionError("session already submitted; feedback closed")
        if self.elapsed_s() > state["budget"]["wall_s"]:
            raise PermissionError("wall budget exhausted")
        if state["dev_calls_used"] >= state["budget"]["dev_calls"]:
            raise PermissionError("development call budget exhausted")
        for key in ("hypothesis", "expected_result"):
            if not isinstance(decision.get(key), str) or not decision[key].strip():
                raise ValueError(f"decision.{key} is required")
        cid = self.register(source_dir)
        state["dev_calls_used"] += 1
        self._save(state)
        task = json.loads((self.private / "task_manifest.json").read_text())
        src = self.private / "candidates" / cid / "src"
        art = self.private / "candidates" / cid / f"artifact-seed{seed}"
        res = evaluate(src, art, self.public / "data" / "train_manifest.json",
                       json.loads((self.private / "dev_requests.json").read_text()),
                       json.loads((self.private / "dev_targets.json").read_text()),
                       task, state["tier"], seed, denied=self._denied())
        feedback = public_feedback(res["score"])
        event = self._event("dev_eval", {
            "candidate_id": cid, "source_sha256": tree_hash(src), "seed": seed,
            "command": f"dev_eval(candidate={cid}, seed={seed})",
            "decision": {k: decision.get(k) for k in ("observation", "evidence_ids", "hypothesis", "expected_result")},
            "code_diff": self._diff(cid),
            "artifact_sha256": tree_hash(art) if art.exists() else None,
            "worker": res["logs"], "dev_result": feedback,
            "budget": {"dev_calls_used": state["dev_calls_used"], "elapsed_s": round(self.elapsed_s(), 1)}})
        (self.public / "feedback").mkdir(exist_ok=True)
        _write(self.public / "feedback" / f"{event['event_id']}.json", feedback)
        return {"event_id": event["event_id"], "candidate_id": cid, "feedback": feedback}

    def declare(self, cid: str, reason: str):
        if not (self.private / "candidates" / cid).exists():
            raise ValueError("unknown candidate")
        state = self.state
        if state["submitted"]:
            raise PermissionError("already submitted")
        state["incumbents"].append({"candidate_id": cid, "elapsed_s": round(self.elapsed_s(), 3)})
        self._save(state)
        return self._event("declare_incumbent", {"candidate_id": cid, "reason": reason})

    def submit(self, cid: str, report: str, seed: int = 0):
        state = self.state
        if state["submitted"]:
            raise PermissionError("already submitted")
        if not any(e["type"] == "dev_eval" and e["candidate_id"] == cid for e in self.events()):
            raise ValueError("final candidate must have a recorded development evaluation")
        state["submitted"] = {"candidate_id": cid, "seed": seed, "elapsed_s": round(self.elapsed_s(), 3)}
        self._save(state)
        return self._event("submit", {"candidate_id": cid, "seed": seed, "report": report[:20000],
                                      "reproduction": f"python -m research_env.tasks.heat replay {self.root}"})

    # ---- private evaluation (no participant feedback) --------------------
    def checkpoint_incumbents(self):
        state = self.state
        out = {}
        for m in CHECKPOINTS_MIN:
            before = [i for i in state["incumbents"] if i["elapsed_s"] <= m * 60]
            out[str(m)] = before[-1]["candidate_id"] if before else None
        return out

    def replay_private(self) -> dict:
        state = self.state
        if not state["submitted"]:
            raise PermissionError("no final submission")
        task = json.loads((self.private / "task_manifest.json").read_text())
        groups = json.loads((self.private / "test_requests.json").read_text())
        targets = json.loads((self.private / "test_targets.json").read_text())
        dev_groups = json.loads((self.private / "dev_requests.json").read_text())
        dev_targets = json.loads((self.private / "dev_targets.json").read_text())
        train = self.public / "data" / "train_manifest.json"

        def clean(cid, seed):
            src = self.private / "candidates" / cid / "src"
            art = self.private / "replay" / cid / f"seed{seed}"
            art.parent.mkdir(parents=True, exist_ok=True)
            test = evaluate(src, art, train, groups, targets, task, state["tier"], seed, self._denied(),
                            split="private_test")
            dev = evaluate(src, art, train, dev_groups, dev_targets, task, state["tier"], seed,
                           self._denied(), retrain=False)
            recorded = [e for e in self.events() if e["type"] == "dev_eval" and e["candidate_id"] == cid
                        and e["seed"] == seed]
            same = bool(recorded) and recorded[-1]["dev_result"] == public_feedback(dev["score"])
            return {"candidate_id": cid, "seed": seed, "source_sha256": tree_hash(src),
                    "artifact_sha256": tree_hash(art) if art.exists() else None,
                    "test": test["score"], "dev_replay_matches_recorded": same, "logs": test["logs"]}

        final = state["submitted"]
        out = {"final": clean(final["candidate_id"], final["seed"]),
               "checkpoints": {m: (clean(c, 0)["test"] if c else None)
                               for m, c in self.checkpoint_incumbents().items()},
               "selection_policy": "final score uses the participant's own selection; checkpoint "
                                   "scores are reported only and never used to choose a winner"}
        _write(self.private / "final_score.json", out)
        return out


def public_feedback(score: dict) -> dict:
    """Development feedback: aggregate and per-dev-person metrics, no target values."""
    keep = ("metric_version", "execution_valid", "failure_reason", "primary_loss_c", "skill_vs_baseline",
            "skill_baseline_id", "people", "by_horizon", "uncertainty", "hottest_period",
            "requested_targets", "eligible_targets")
    fb = {k: score.get(k) for k in keep if k in score}
    if score.get("per_person"):
        fb["per_person_loss_c"] = {f"dev-person-{i + 1}": row["loss_c"]
                                   for i, (_, row) in enumerate(sorted(score["per_person"].items()))}
    return fb
