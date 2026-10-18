"""Heat task family for GRPO post-training (RL_ENV_CONTRACT v1).

Pool design (documented in docs/SET2_HEAT_ENVIRONMENT.md):

* ``agent_training``: every person comes from the frozen dataset training
  group (14 people). Each task draws a seeded 8 fit / 3 feedback / 3 reward
  partition of those 14.
* ``agent_development``: 11 training people fit, 3 training people feedback,
  the 5 dataset development people as reward (checkpoint selection).
* ``agent_final_evaluation``: same 11/3 fit/feedback, the 5 private-test
  people as reward. Private-test people appear in no other outer split.

Frozen reward baseline: ``elapsed_time`` (mandatory protocol-shortcut
reference, see workspace_template/baselines/elapsed_time.py), fitted on
``fit_people`` and scored on ``reward_people`` with the same requests,
targets and profile cutoffs as the candidate. It is trusted builder code, so
it runs in-process; candidate artifacts only ever run through
``research_env.core.worker.execute``.

Replay (``score``) is heat-specific because ``research_env/core/evaluation.py``
imports pulse data and scoring. It reuses the shared worker: one fresh worker
per person (one bout per person in this dataset), requests streamed in cutoff
order, each request prefix-only.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from research_env.core.common import canonical, digest, tree_hashes
from research_env.core.worker import execute, worker_python

from . import requests as rq
from . import scoring

try:  # rl-core owns the shared API; fall back to a structurally identical local copy.
    from research_env.rl.api import OUTER_SPLITS, TaskSpec  # type: ignore
    try:
        from research_env.rl.api import register as _register  # type: ignore
    except ImportError:  # pragma: no cover
        _register = None
except ImportError:  # TODO: import from research_env.rl.api once rl-core lands it.
    _register = None
    OUTER_SPLITS = ("agent_training", "agent_development", "agent_final_evaluation")

    @dataclass(frozen=True)
    class TaskSpec:  # mirror of research_env.rl.api.TaskSpec (contract v1)
        task_family: str
        task_id: str
        outer_split: str
        fit_people: tuple[str, ...]
        feedback_people: tuple[str, ...]
        reward_people: tuple[str, ...]
        profile: str
        options: dict = field(default_factory=dict)


HERE = Path(__file__).parent
TEMPLATE = HERE / "workspace_template"
RL_TEMPLATE = HERE / "rl_workspace"
SPLIT_PATH = HERE / "split_manifest.json"
POOL_VERSION = "heat-rl-pool-v1"
MANIFEST_SCHEMA = "heat-rl-private-manifest-v1"
WORKSPACE_MARKER = ".heat-rl-workspace"
INNER_PARTITION = {"fit": 8, "feedback": 3, "reward": 3}       # of the 14 training people
HELDOUT_PARTITION = {"fit": 11, "feedback": 3}                 # dev/final: reward = held-out group
REWARD_BASELINE_ID = "elapsed_time"
PROFILES = {
    "fast": {"wall_s": 1200, "development_calls": 8, "cutoffs_min": tuple(range(5, 56, 5))},
    "benchmark": {"wall_s": 7200, "development_calls": 30, "cutoffs_min": rq.CUTOFFS_MIN},
}
TIER_NOTES = {
    "wearable": "Inputs: HR, CORE-device skin temperature, work rate, measured ambient temperature/RH and demographics. No rectal value is ever an input.",
    "calibrated": "Wearable inputs plus one rectal reference at minute 1 (`request.calibration`), which needs a rectal probe before exercise. Scored separately from the wearable tier.",
}
RESPONSE_KEYS = ("horizon_min", "mean_c", "lower_c", "upper_c")


# ---------------------------------------------------------------- data access

def _rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def _load(prepared_root):
    prepared = Path(prepared_root)
    manifest = json.loads((prepared / "dataset_manifest.json").read_text())
    groups = {g: [str(s) for s in v] for g, v in manifest["split_manifest"]["groups"].items()}
    frozen = {g: [str(s) for s in v] for g, v in json.loads(SPLIT_PATH.read_text())["groups"].items()}
    if groups != frozen:
        raise ValueError("prepared split differs from the frozen heat-split-v1 manifest")
    # Allowlist: only permitted history fields ever reach a request or the
    # workspace. Rectal history and vendor core estimates are rejected here even
    # if a future importer wrote them into inputs.jsonl.
    inputs = [r for r in _rows(prepared / "inputs.jsonl") if r["field"] in rq.HISTORY_FIELDS]
    refs = _rows(prepared / "references.jsonl")
    context = json.loads((prepared / "context.json").read_text())
    return groups, inputs, refs, context


def _alias(secret, subject):
    return "p-" + rq.opaque_id(secret, "person", str(subject))[:10]


def _secret(spec: TaskSpec, role: str) -> str:
    return hashlib.sha256(canonical([POOL_VERSION, spec.task_id, role])).hexdigest()


def _cutoffs(profile):
    return list(PROFILES[profile]["cutoffs_min"])


def _person_requests(subject, inputs, refs, context, tier, secret, cutoffs):
    p_inputs = [r for r in inputs if str(r["subject"]) == str(subject)]
    p_refs = [r for r in refs if str(r["subject"]) == str(subject)]
    reqs, targets = [], []
    person = _alias(secret, subject)
    for c in cutoffs:
        rid = rq.opaque_id(secret, "request", str(subject), tier, c)
        reqs.append(rq.build_request(rid, c, p_inputs, context[str(subject)], tier, p_refs))
        targets.extend({**t, "person": person} for t in rq.build_targets(rid, c, p_refs))
    return reqs, targets


def _train_bundle(subjects, inputs, refs, context, tier, secret):
    people = []
    for s in subjects:
        people.append({
            "person": _alias(secret, s),
            "context": {f: context[str(s)][f]["value"] for f in rq.CONTEXT_FIELDS},
            "inputs": [{"minute": r["minute"], "field": r["field"], "value": r["value"]}
                       for r in inputs if str(r["subject"]) == str(s)],
            "reference": [{"minute": r["minute"], "rectal_c": r["rectal_c"], "eligible": r["eligible"]}
                          for r in refs if str(r["subject"]) == str(s)]})
    people.sort(key=lambda p: p["person"])
    return {"task": "heat-core-forecast-v1", "tier": tier, "people": people,
            "fields": {"inputs": list(rq.HISTORY_FIELDS), "context": list(rq.CONTEXT_FIELDS)},
            "note": "rectal_c is the reference target; it never appears in inference requests "
                    "except the minute-1 value in the calibrated tier"}


# ---------------------------------------------------------------- pool

def task_id(outer_split, seed, fit, feedback, reward, profile, options) -> str:
    body = {"family": "heat", "version": POOL_VERSION, "outer_split": outer_split, "seed": seed,
            "fit": list(fit), "feedback": list(feedback), "reward": list(reward),
            "profile": profile, "options": options}
    return "heat-" + digest(body)[:16]


def _spec(outer_split, seed, fit, feedback, reward, profile, options):
    fit, feedback, reward = (tuple(sorted(x, key=int)) for x in (fit, feedback, reward))
    if set(fit) & set(feedback) or set(fit) & set(reward) or set(feedback) & set(reward):
        raise ValueError("fit/feedback/reward people must be disjoint")
    return TaskSpec(task_family="heat", task_id=task_id(outer_split, seed, fit, feedback, reward, profile, options),
                    outer_split=outer_split, fit_people=fit, feedback_people=feedback, reward_people=reward,
                    profile=profile, options=dict(options))


class HeatFamily:
    name = "heat"
    primary_unit = "degC"

    def __init__(self, training_partitions: int = 8, tiers=("wearable", "calibrated"),
                 heldout_profile: dict | None = None):
        self.training_partitions = training_partitions
        self.tiers = tuple(tiers)
        self.heldout_profile = heldout_profile or {"agent_development": "fast",
                                                   "agent_final_evaluation": "benchmark"}
        self._baseline_cache: dict = {}

    # -- pool --------------------------------------------------------------
    def build_pool(self, prepared_root: Path, seed: int) -> list:
        groups, *_ = _load(prepared_root)
        train = sorted(groups["train"], key=int)
        n = INNER_PARTITION
        if len(train) < sum(n.values()):
            raise ValueError("not enough training people for the inner partition")
        pool = []
        for tier in self.tiers:
            options = {"tier": tier, "horizons_min": list(rq.HORIZONS_MIN)}
            for i in range(self.training_partitions):
                people = list(train)
                random.Random(f"{POOL_VERSION}:{seed}:{tier}:{i}").shuffle(people)
                fit = people[:n["fit"]]
                fb = people[n["fit"]:n["fit"] + n["feedback"]]
                rw = people[n["fit"] + n["feedback"]:n["fit"] + n["feedback"] + n["reward"]]
                pool.append(_spec("agent_training", seed, fit, fb, rw, "fast", {**options, "partition": i}))
            people = list(train)
            random.Random(f"{POOL_VERSION}:{seed}:{tier}:heldout").shuffle(people)
            fit, fb = people[:HELDOUT_PARTITION["fit"]], people[HELDOUT_PARTITION["fit"]:]
            pool.append(_spec("agent_development", seed, fit, fb, groups["development"],
                              self.heldout_profile["agent_development"], options))
            pool.append(_spec("agent_final_evaluation", seed, fit, fb, groups["private_test"],
                              self.heldout_profile["agent_final_evaluation"], options))
        return pool

    # -- workspace ---------------------------------------------------------
    def stage_workspace(self, spec, prepared_root: Path, workspace: Path, *, prebuild: bool = True) -> None:
        _, inputs, refs, context = _load(prepared_root)
        ws = Path(workspace)
        ws.mkdir(parents=True, exist_ok=True)
        tier = spec.options["tier"]
        prof = PROFILES[spec.profile]
        secret = _secret(spec, "fit")
        (ws / "data").mkdir(exist_ok=True)
        train = _train_bundle(spec.fit_people, inputs, refs, context, tier, secret)
        (ws / "data" / "train_manifest.json").write_text(json.dumps(train, allow_nan=False))
        first = spec.fit_people[0]
        example, _ = _person_requests(first, inputs, refs, context, tier, secret, [30])
        (ws / "data" / "example_request.json").write_text(json.dumps(example[0], indent=1))
        shutil.copyfile(TEMPLATE / "DATA_DICTIONARY.md", ws / "DATA_DICTIONARY.md")
        shutil.copyfile(TEMPLATE / "heatlib.py", ws / "heatlib.py")
        shutil.copytree(TEMPLATE / "baselines", ws / "baselines", dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__"))
        (ws / "tools").mkdir(exist_ok=True)
        shutil.copyfile(RL_TEMPLATE / "build_artifact.py", ws / "tools" / "build_artifact.py")
        (ws / "TASK.md").write_text((RL_TEMPLATE / "TASK.md").read_text().format(
            tier=tier, tier_note=TIER_NOTES[tier], fit_n=len(spec.fit_people),
            feedback_n=len(spec.feedback_people), wall_min=prof["wall_s"] // 60,
            dev_calls=prof["development_calls"]))
        (ws / "task_manifest.json").write_text(json.dumps({
            "task": "heat-core-forecast-v1", "task_id": spec.task_id, "tier": tier,
            "horizons_min": list(rq.HORIZONS_MIN),
            "cutoffs_min": "scored cutoffs: " + ("every 5th minute 5..55" if spec.profile == "fast" else "every minute 5..55")
                           + "; horizons requested only where cutoff + horizon <= 60",
            "profile": spec.profile, "budget": {"wall_s": prof["wall_s"],
                                                "development_calls": prof["development_calls"],
                                                "cpu_cores": 8, "gpu": "none"},
            "people": {"fit": len(spec.fit_people), "feedback": len(spec.feedback_people)},
            "metric": "person-balanced mean of equal 5/10/20-minute MAE, degrees C; IS90 reported separately",
            "frozen_reference_baseline": REWARD_BASELINE_ID}, indent=1))
        (ws / WORKSPACE_MARKER).write_text(spec.task_id + "\n")
        if prebuild:
            self._build_in_process(TEMPLATE / "baselines" / f"{REWARD_BASELINE_ID}.py",
                                   ws / "data" / "train_manifest.json", ws / "artifact", ws / "heatlib.py")

    @staticmethod
    def _build_in_process(model_py, train, out, heatlib):
        spec = importlib.util.spec_from_file_location("heat_build_artifact", RL_TEMPLATE / "build_artifact.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        saved = sys.modules.pop("heatlib", None), sys.modules.pop("model", None)
        try:
            return mod.build(model_py, out, train, 0, heatlib)
        finally:
            sys.modules.pop("heatlib", None)
            sys.modules.pop("model", None)
            for name, m in zip(("heatlib", "model"), saved):
                if m is not None:
                    sys.modules[name] = m

    # -- private manifests ---------------------------------------------------
    def private_manifest(self, spec, prepared_root: Path, people: str, out_dir: Path) -> Path:
        if people not in ("feedback", "reward"):
            raise ValueError("people must be 'feedback' or 'reward'")
        out_dir = Path(out_dir).resolve()
        for parent in (out_dir, *out_dir.parents):
            if (parent / WORKSPACE_MARKER).exists():
                raise ValueError("private manifest must be written outside any participant workspace")
        _, inputs, refs, context = _load(prepared_root)
        subjects = spec.feedback_people if people == "feedback" else spec.reward_people
        tier = spec.options["tier"]
        secret = _secret(spec, people)
        groups, targets = [], []
        for s in subjects:
            reqs, tg = _person_requests(s, inputs, refs, context, tier, secret, _cutoffs(spec.profile))
            groups.append(reqs)
            targets.extend(tg)
        body = {"schema": MANIFEST_SCHEMA, "task_id": spec.task_id, "people_role": people, "tier": tier,
                "profile": spec.profile, "cutoffs_min": _cutoffs(spec.profile), "groups": groups,
                "targets": targets}
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{spec.task_id}-{people}.json"
        path.write_text(json.dumps(body, allow_nan=False))
        return path

    # -- scoring -----------------------------------------------------------
    def score(self, artifact_dir: Path, manifest: Path, *, backend: str, image: str | None,
              wall_s: float, profile: str) -> dict:
        return replay(artifact_dir, manifest, backend=backend, image=image, wall_s=wall_s, profile=profile)

    def baseline_loss(self, spec, prepared_root: Path) -> float:
        key = (spec.task_id, str(Path(prepared_root).resolve()))
        if key not in self._baseline_cache:
            self._baseline_cache[key] = in_process_baseline(spec, prepared_root, REWARD_BASELINE_ID)["primary_loss"]
        return self._baseline_cache[key]


# ---------------------------------------------------------------- replay

def _invalid(reason, resource_use=None, metrics=None):
    return {"execution_valid": False, "primary_loss": None, "failure_reason": reason,
            "metrics": metrics or {}, "resource_use": resource_use or {}, "unit": "degC"}


def _normalize(response, request):
    """Strict response shape: list of exactly the requested horizons with the four keys."""
    if isinstance(response, dict) and set(response) <= {"forecasts", "state"} and "forecasts" in response:
        response = response["forecasts"]
    if not isinstance(response, list) or any(not isinstance(r, dict) for r in response):
        raise ValueError(f"{request['request_id']}: response must be a list of records")
    for r in response:
        if set(r) != set(RESPONSE_KEYS):
            raise ValueError(f"{request['request_id']}: record keys must be exactly {RESPONSE_KEYS}")
    return [{k: r[k] for k in RESPONSE_KEYS} for r in response]


def replay(artifact_dir, manifest, *, backend="docker", image=None, wall_s=7200.0, profile="benchmark",
           on_tick=None) -> dict:
    """Trusted heat replay. Candidate code runs only inside research_env.core.worker.execute."""
    artifact_dir = Path(artifact_dir)
    start = time.monotonic()
    try:
        hashes = tree_hashes(artifact_dir)
    except (OSError, ValueError) as exc:
        return _invalid(f"artifact_rejected: {exc}")
    if "inference.py" not in hashes:
        return _invalid("artifact must include inference.py with predict(artifact_dir, request)")
    body = json.loads(Path(manifest).read_text())
    if body.get("schema") != MANIFEST_SCHEMA:
        return _invalid("unknown private manifest schema")
    keep = set(_cutoffs(profile)) & set(body["cutoffs_min"])
    groups = [[r for r in g if r["cutoff_min"] in keep] for g in body["groups"]]
    rids = {r["request_id"] for g in groups for r in g}
    targets = [t for t in body["targets"] if t["request_id"] in rids]
    predictions, workers = {}, []
    python = worker_python(backend)
    for group in groups:
        remaining = wall_s - (time.monotonic() - start)
        if remaining <= 0:
            return _invalid("replay_wall_budget", {"wall_s": time.monotonic() - start, "workers": workers})
        with tempfile.TemporaryDirectory(prefix="heat-replay-") as tmp:
            root = Path(tmp)
            shutil.copytree(artifact_dir, root / "artifact", ignore=shutil.ignore_patterns("__pycache__"))
            shutil.copyfile(RL_TEMPLATE / "replay_entry.py", root / "entry.py")
            try:
                run = execute(root, [python, "-I", "entry.py"], timeout_s=remaining, backend=backend,
                              image=image, on_tick=on_tick, requests=iter(group))
            except (OSError, RuntimeError, ValueError) as exc:
                return _invalid(f"worker_error: {type(exc).__name__}: {exc}",
                                {"wall_s": time.monotonic() - start, "workers": workers})
        workers.append({k: v for k, v in run.items() if k not in ("responses", "stdout", "stderr")})
        resource_use = {"wall_s": time.monotonic() - start, "workers": workers, "backend": backend,
                        "scored_resource_isolation": backend == "docker"}
        if not run["execution_valid"] or len(run["responses"]) != len(group):
            return _invalid(run["failure_reason"] or "incomplete_replay", resource_use,
                            {"stderr_tail": run.get("stderr", "")[-2000:]})
        try:
            for req, resp in zip(group, run["responses"]):
                predictions[req["request_id"]] = _normalize(resp, req)
        except ValueError as exc:
            return _invalid(str(exc), resource_use)
    requests_by_id = {r["request_id"]: r for g in groups for r in g}
    result = scoring.score(targets, predictions, {"requests": requests_by_id, "baseline_loss_c": None,
                                                  "baseline_id": None})
    resource_use = {"wall_s": time.monotonic() - start, "workers": workers, "backend": backend,
                    "scored_resource_isolation": backend == "docker"}
    metrics = {k: v for k, v in result.items() if k not in ("execution_valid", "failure_reason")}
    metrics.update(profile=profile, tier=body["tier"], people_role=body["people_role"],
                   artifact_sha256=digest(hashes))
    return {"execution_valid": bool(result["execution_valid"]),
            "primary_loss": result["primary_loss_c"] if result["execution_valid"] else None,
            "failure_reason": result["failure_reason"], "metrics": metrics,
            "resource_use": resource_use, "unit": "degC"}


# ---------------------------------------------------------------- trusted baselines

def _load_baseline(name):
    """Import a builder baseline in-process (trusted code) with the template heatlib."""
    saved = sys.modules.get("heatlib")
    hl_spec = importlib.util.spec_from_file_location("heatlib", TEMPLATE / "heatlib.py")
    heatlib = importlib.util.module_from_spec(hl_spec)
    hl_spec.loader.exec_module(heatlib)
    sys.modules["heatlib"] = heatlib
    try:
        spec = importlib.util.spec_from_file_location(f"heat_baseline_{name}", TEMPLATE / "baselines" / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if saved is None:
            sys.modules.pop("heatlib", None)
        else:
            sys.modules["heatlib"] = saved
    return module


def in_process_baseline(spec, prepared_root, name=REWARD_BASELINE_ID, people="reward", seed=0) -> dict:
    """Fit a builder baseline on fit_people; score on feedback or reward people (same requests as replay)."""
    _, inputs, refs, context = _load(prepared_root)
    tier = spec.options["tier"]
    module = _load_baseline(name)
    train = _train_bundle(spec.fit_people, inputs, refs, context, tier, _secret(spec, "fit"))
    subjects = spec.reward_people if people == "reward" else spec.feedback_people
    secret = _secret(spec, people)
    with tempfile.TemporaryDirectory(prefix=f"heat-baseline-{name}-") as tmp:
        tmp = Path(tmp)
        (tmp / "train.json").write_text(json.dumps(train))
        art = tmp / "artifact"
        art.mkdir()
        module.train(str(tmp / "train.json"), None, str(art), seed)
        reqs, targets, preds = {}, [], {}
        for s in subjects:
            rs, tg = _person_requests(s, inputs, refs, context, tier, secret, _cutoffs(spec.profile))
            targets.extend(tg)
            for r in rs:
                reqs[r["request_id"]] = r
                preds[r["request_id"]] = module.predict(str(art), r)
    result = scoring.score(targets, preds, {"requests": reqs, "baseline_loss_c": None, "baseline_id": None})
    if not result["execution_valid"]:
        raise RuntimeError(f"frozen baseline {name} invalid: {result['failure_reason']}")
    return {"primary_loss": result["primary_loss_c"], "metrics": result, "baseline_id": name}


FAMILY = HeatFamily()
if _register is not None:  # pragma: no cover - depends on rl-core landing
    try:
        _register(FAMILY)
    except Exception:  # duplicate registration on re-import
        pass
