# RL environment contract v1 (GRPO post-training)

Status: interface agreement for parallel builders, October 17, 2026. Not a claim of implemented behavior.

Decisions already made (PLAN.md §2.11): post-training method is **GRPO**. GPU trainer and rollout inference run on **RunPod**. Agent code executes in **Beam** sandboxes (network blocked), with a separate trusted Beam scorer that holds private labels. Base model, compute budget and hyperparameters are open: keep them as config values, never hard-code a choice.

Existing foundation: `research_env/core/` (Session ledger, ResearchEnvironment actions edit/command/evaluate/select/submit, worker backends `docker` / `macos-development`, causal streaming replay), `research_env/tasks/pulse/`, `research_env/judging/`. Read `docs/SET1_PULSE_ENVIRONMENT.md` and `docs/AGENT_BUILD_INSTRUCTIONS.md` (common contract A–E plus your Set) before coding. Do not rewrite shared infrastructure outside your ownership; extend through the hooks below.

## Ownership (one writer per path)

| Builder | Owns |
| --- | --- |
| rl-core | `research_env/rl/` (new), `research_env/core/evaluation.py` and `research_env/core/agent.py` changes needed for task adapters and the fast profile, `research_env/tasks/pulse/rl_pool.py`, `tests/test_rl_env.py` |
| heat | `research_env/tasks/heat/`, `tests/test_heat_task.py`, heat scripts/docs |
| load | `research_env/tasks/load/`, `tests/test_load_rl_task.py`, load scripts/docs |
| beam | `research_env/core/worker.py` (add `backend='beam'`), `research_env/beam/` (new), `tests/test_beam_backend.py`, Beam docs |
| gpu-trainer | `training/` (new), `tests/test_training_rollout.py`, training docs |

If you need a change in a path you do not own, do not make it: describe it precisely in your handoff.

## Task family interface (`research_env/rl/api.py`, written by rl-core)

```python
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

OUTER_SPLITS = ('agent_training', 'agent_development', 'agent_final_evaluation')

@dataclass(frozen=True)
class TaskSpec:
    task_family: str               # 'pulse' | 'heat' | 'load'
    task_id: str                   # deterministic from (family, seed, people, options)
    outer_split: str               # one of OUTER_SPLITS
    fit_people: tuple[str, ...]    # public training data the agent may fit on
    feedback_people: tuple[str, ...]  # scored by the agent's `evaluate` calls (visible feedback)
    reward_people: tuple[str, ...]    # hidden; trainer reward only; disjoint from the two above
    profile: str                   # 'fast' | 'benchmark'
    options: dict = field(default_factory=dict)  # family-specific: bouts, channels, horizons, loads, tier

class TaskFamily(Protocol):
    name: str
    primary_unit: str              # 'bpm' | 'degC' | 'W'
    def build_pool(self, prepared_root: Path, seed: int) -> list[TaskSpec]: ...
    def stage_workspace(self, spec: TaskSpec, prepared_root: Path, workspace: Path) -> None: ...
    def private_manifest(self, spec: TaskSpec, prepared_root: Path, people: str, out_dir: Path) -> Path: ...
        # people in {'feedback', 'reward'}; writes outside any workspace
    def score(self, artifact_dir: Path, manifest: Path, *, backend: str, image: str | None,
              wall_s: float, profile: str) -> dict: ...
    def baseline_loss(self, spec: TaskSpec, prepared_root: Path) -> float: ...
        # frozen reference baseline (pulse: ridge) fitted on fit_people, scored on reward_people

def register(family: TaskFamily) -> None: ...
def get_family(name: str) -> TaskFamily: ...
```

`score` returns at least: `execution_valid` (bool), `primary_loss` (float | None, physical units), `failure_reason` (str | None), `metrics` (dict), `resource_use` (dict). Invalid never becomes zero loss.

Splits: person-level, before windowing. Agent-training tasks draw all three people sets **only from each dataset's inner training people**. Agent-development tasks use the dataset's development people as reward_people (for checkpoint selection). Dataset private-test people are never used by any TaskSpec in `agent_training` or `agent_development`; they are reserved for `agent_final_evaluation`. Tests must assert this.

Suggested inner partitions (adjust to cohort size, document choice): pulse 13 training people → 8 fit / 2 feedback / 3 reward; heat 14 → 8/3/3; load 28 → 18/5/5.

## Environment API (`research_env/rl/env.py`, written by rl-core)

```python
env = PhysiologyResearchEnv(prepared_roots: dict[str, Path], run_root: Path, *,
                            backend='docker', image=None, reward_version='physiology-reward-v1',
                            max_turns=40, history_chars=60_000)
obs: dict = env.reset(task: TaskSpec, seed: int)
obs, reward, done, info = env.step(action: dict)   # action = existing ResearchEnvironment operation dict
env.tool_schemas() -> list[dict]                   # OpenAI-style function tools, one per operation
env.render_observation(obs) -> str                 # text the model sees; truncated history
env.close()
```

- Malformed/illegal actions return an error observation and count toward `info['invalid_actions']`; they never raise to the caller.
- Reward is **terminal only** (0.0 on non-terminal steps). Episode ends on submit, wall/turn budget exhaustion, or `max_turns`.
- Every episode in a GRPO group uses identical budgets (turns, development calls, wall time).
- `reset` creates a fresh controller directory outside the workspace; private manifests never enter the workspace.

## Reward v1 (`research_env/rl/reward.py`, frozen, versioned `physiology-reward-v1`)

Evaluated once at termination on `reward_people` with the submitted candidate:

| Outcome | Reward |
| --- | --- |
| Valid submission | `clip(1 - loss / max(baseline_loss, eps), -1, 1)` |
| Submitted, but invalid on reward people | `-1.25` |
| No submission, at least one valid `evaluate` | `-1.5` |
| No valid evaluation at all | `-2.0` |

`info['reward_components']` reports tier, loss, baseline_loss, eps, skill, invalid_actions, turns, development_calls_used, reward_version. No LLM-judge component in v1. Any change to the numbers requires a new version string.

## Profiles

| Profile | Wall | Dev calls | Replay |
| --- | --- | --- | --- |
| `fast` (training rollouts) | 20 min | 8 | subsampled bouts/records, one worker per person with in-process state reset per bout |
| `benchmark` (final eval) | 2 h | 30 | existing strict one-fresh-worker-per-bout replay |

## Trajectory record (`research_env/rl/trajectory.py`; consumed by `training/`)

One JSON object per episode, JSONL:

```json
{"schema": "physiology-rl-trajectory-v1", "task_id": "...", "task_family": "pulse",
 "outer_split": "agent_training", "seed": 0, "group_id": "...", "reward": 0.21,
 "reward_version": "physiology-reward-v1", "reward_components": {},
 "messages": [{"role": "system|user|assistant|tool", "content": "...", "tool_calls": [], "tool_call_id": null}],
 "loss_mask": [0, 0, 1, 0],
 "ledger_head_sha256": "...", "budgets": {}}
```

`loss_mask[i] = 1` only for assistant messages (model-written tokens). Tool outputs, file contents and scores are masked out. Export refuses `agent_final_evaluation` tasks as training data.

## Rules for every builder

- CPU-only environments. No paid compute: do not create RunPod pods, Beam deployments/sandboxes, or call paid model APIs. Do not read credential files (e.g. `~/.runpod/config.toml`) or print secrets.
- Open datasets named in `AGENT_BUILD_INSTRUCTIONS.md` may be fetched into git-ignored directories (`.research-data/`), verifying checksums and license.
- Do not commit, push or switch branches. Leave changes uncommitted in your worktree.
- Tests use independent hand-computed examples, not re-implementations. Mock cloud SDKs where needed.
- Handoff: files changed, exact commands, checks executed (with output), what is pending, blockers, and any change requested in a path you don't own.
