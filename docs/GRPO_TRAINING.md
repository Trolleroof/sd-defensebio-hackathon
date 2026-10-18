# GRPO training on RunPod (physiology RL env, contract v1)

Status, October 17, 2026: the glue code exists and its CPU tests pass. Nothing here has run on a GPU, and no pod has been created. The base model, compute budget and every hyperparameter are still open, owned by Nikhil (PLAN.md §2.11). The env API (`research_env/rl/`) is being built in parallel by rl-core. This code targets the signatures in `RL_ENV_CONTRACT.md`, and `training/env_client.py` is the only file that has to change if the real API differs.

## Framework choice: verl v0.9.1

Releases and APIs were checked against official repos and docs on 2026-10-17.

| Framework | Latest release | Multi-turn tool use with an external env | Why / why not |
| --- | --- | --- | --- |
| **verl** ([repo](https://github.com/volcengine/verl), [agent loop doc](https://verl.readthedocs.io/en/latest/advance/agent_loop.html), [multi-turn doc](https://verl.readthedocs.io/en/latest/sglang_multiturn/multiturn.html), [GRPO doc](https://verl.readthedocs.io/en/latest/algo/grpo.html)) | v0.9.1 (2026-09-20) | You subclass `AgentLoopBase` and register it with `@register(name)` plus an `agent_loop_config_path` YAML. `run()` controls the whole episode and returns `AgentLoopOutput(prompt_ids, response_ids, response_mask, reward_score, …)`. Generation goes through `server_manager.generate` (async vLLM server). | **Chosen.** The loop is fully ours, so each sample can be a minutes-long Beam episode that runs asynchronously. Token-exact `response_mask` comes from the Continuous Token builder; the verl docs say retokenizing chat history broke PPO convergence. The env reward passes straight through as `reward_score`. GRPO is built in (`algorithm.adv_estimator=grpo`), as are FSDP, checkpoint/resume, and hybrid-engine weight sync to vLLM. Downsides: it is a heavy stack (Ray), and the API lives in the `experimental` namespace. |
| TRL ([repo](https://github.com/huggingface/trl), [GRPOTrainer doc](https://huggingface.co/docs/trl/main/en/grpo_trainer)) | v1.16.0 (2026-10-15) | `GRPOTrainer(environment_factory=...)` builds one env per rollout and exposes its public methods as tools. It has a `reset` hook and optional `get_reward`, and vLLM runs in `colocate` or `server` mode. Requires `transformers>=5.2.0`. | A good fallback. Our env takes action dicts through `step(action)`, so we would need a wrapper with one method per operation. The trainer owns the conversation loop, which gives us less control over long multi-turn episodes and over cutting them off when the context runs out. |
| OpenRLHF ([repo](https://github.com/OpenRLHF/OpenRLHF)) | v0.11.2 (2026-09-14) | Ray + vLLM, with agent-style RL. | Its agent API was **not checked in detail**, so it was not chosen on that basis. |

Pinned stack, taken from verl v0.9.1's own `pyproject.toml`: torch 2.11.0, **vllm 0.24.0**, transformers 5.9.0, CUDA 13.0 base `nvidia/cuda:13.0.2-devel-ubuntu24.04` ([install doc](https://verl.readthedocs.io/en/latest/start/install.html)). The latest vLLM is v0.31.0, but verl v0.9.1 locks 0.24.0, and the Dockerfile follows verl's lock.

## Architecture

```
RunPod GPU pod (training/runpod/Dockerfile)
┌─────────────────────────────────────────────────────────────────────────┐
│ verl.trainer.main_ppo (Ray)                                             │
│  FSDP actor + ref ──weights (hybrid engine, every step)──► async vLLM    │
│        ▲ GRPO update: adv = (r - mean_g)/(std_g + eps), token mask      │
│        │                                                ▲ generate()    │
│  AgentLoopWorkers × N: PhysiologyEnvAgentLoop (training/verl_agent_loop) │
│     one episode per sample; G samples per TaskSpec row (rollout.n)      │
│        │ env_client (only research_env.rl importer)                     │
│        ▼                                                                │
│  PhysiologyResearchEnv (env orchestrator, in-process, CPU)              │
└────────┼────────────────────────────────────────────────────────────────┘
         │ backend='beam' (beam builder)
         ▼
  Beam sandboxes (agent code, network blocked)   Beam trusted scorer (private labels)
         └──── workspace commands / evaluate ────────────► terminal reward (reward v1)
```

* **Group construction**: `training/build_dataset.py` writes one parquet row per TaskSpec (`agent_name=physiology_env_agent`, `extra_info.task_spec`, `extra_info.env_seed`). verl repeats each row `rollout.n = grpo.group_size` times under one uid. Every sample in a group gets the same TaskSpec, the same `EnvSettings` (max_turns, history_chars, backend, reward_version; the profile fixes wall time and dev calls) and the same env seed. Only the scratch `run_root` differs. The standalone path (`training/rollout.py::run_group`) also checks that the env-reported budgets are identical and raises if they are not.
* **Tool calls**: tool name equals the ResearchEnvironment operation. Bad JSON, an unknown tool, non-object arguments, or no tool call at all becomes an explicit `__malformed_tool_call__` action. The env returns an error observation for it and counts it in `info['invalid_actions']`. If the env raises anyway, the adapter turns the exception into an error observation.
* **Reward and advantage**: the env's terminal reward (frozen `physiology-reward-v1`) becomes `AgentLoopOutput.reward_score`. verl places it on the last response token (`rm_scores`) and skips its own reward function. The GRPO estimator then computes the advantage per uid with `torch.std`, which is Bessel-corrected. `training/advantage.py` mirrors this exactly for logging and tests: rewards [1, 0, -1, 0] give advantages [1.2247, 0, -1.2247, 0].
* **Masking**: `response_mask` is 1 only on tokens the model generated and 0 on tool and observation tokens (verl Continuous Token builder). Trajectory JSONL records carry a message-level `loss_mask` that is 1 only on assistant messages.
* **Checkpointing**: `trainer.save_freq`, `trainer.default_local_dir=<checkpoint_dir>/<project>/<experiment>` and `trainer.resume_mode=auto`. Put `checkpoint_dir` on a network volume so checkpoints survive pod stops.
* **Context exhaustion**: if the response-token budget runs out, the episode is ended through `env.terminate(reason)` when it exists. Otherwise the adapter sends malformed no-op actions until the env's own turn budget ends the episode. Either way, the env alone decides the terminal reward.
* **Records**: each episode is appended as `physiology-rl-trajectory-v1` to `rollout_client.output_dir/verl-<pid>.jsonl`. `agent_final_evaluation` tasks are refused by the config validator, the dataset builder, `run_group`, `env_client.reset`, `build_record` and `validate_record`.

## Files

| Path | Role |
| --- | --- |
| `training/env_client.py` | The only `research_env.rl` importer: TaskSpec, make_env/reset/step/terminate, malformed actions, budgets |
| `training/rollout.py` | `run_episode` / `run_group` / `run_batch`, `OpenAIChatPolicy` (vLLM OpenAI server), CLI |
| `training/advantage.py` | Group-relative advantage (verl-identical) |
| `training/trajectory_io.py` | Trajectory record build/validate/JSONL |
| `training/config.py`, `grpo_config.yaml`, `grpo_config.smoke.yaml` | Config + loader (`extends`, `PHYS_GRPO__section__key` env overrides) |
| `training/verl_agent_loop.py`, `agent_loop.yaml` | verl agent loop (GPU image only) |
| `training/verl_overrides.py` | Config → verl Hydra overrides; execs `python -m verl.trainer.main_ppo` |
| `training/build_dataset.py` | TaskSpec pool → verl parquet |
| `training/runpod/{Dockerfile,entrypoint.sh,launch.sh,env.example}` | Image, pod entrypoint, `runpodctl` launcher (dry run by default) |
| `tests/test_training_rollout.py` | CPU tests with a mock env and a scripted policy |

## Launch steps (run these yourself; each one costs money from step 4 on)

1. **Choose the base model**: `export PHYS_GRPO__model__base_model=<hf-id>`. Any Hugging Face id works. Its chat template must support tools, and `model.tool_call_format` must match it (hermes, qwen3_coder, gpt-oss, … from verl's `ToolParser` registry). Real training refuses to start while the value is still the placeholder.
2. **Build and push the image** from the repo root:
   `docker build -f training/runpod/Dockerfile -t <registry>/physiology-grpo:<tag> . && docker push <registry>/physiology-grpo:<tag>`
3. **Create RunPod secrets** in the console (Secrets): `huggingface_token`, and `beam_token` once the beam backend lands. They are referenced as `{{ RUNPOD_SECRET_<name> }}` ([secrets doc](https://docs.runpod.io/pods/templates/secrets)). For a private registry, add `runpodctl registry create`, then set `PHYS_GRPO_REGISTRY_AUTH_ID`.
4. **Configure and check**: `export RUNPOD_API_KEY=…` (runpodctl reads it from the environment). Then `cp training/runpod/env.example training/runpod/.env.local`, edit it, run `set -a; source training/runpod/.env.local; set +a`, and run `training/runpod/launch.sh` to print the exact `runpodctl pod create …` command without running it. Choose the GPU with `runpodctl gpu list`.
5. **Smoke run**: `PHYS_GRPO_MODE=smoke training/runpod/launch.sh --yes`, then `runpodctl pod logs <pod-id> --follow`. This uses the tiny-model config (G=2, 1 task, 2 steps). Prepared task data must exist at `env.prepared_roots` on the volume first. Stage it with `runpodctl send` / `receive`, or build it in a `PHYS_GRPO_MODE=shell` pod.
6. **Training run**: `PHYS_GRPO_MODE=train`, with the overrides from §Open set.
7. **Stop billing**: `runpodctl pod stop <pod-id>`, then `runpodctl pod delete <pod-id>`.

CLI syntax was checked against [runpodctl docs/runpodctl_pod_create.md](https://github.com/runpod/runpodctl/blob/main/docs/runpodctl_pod_create.md) (runpodctl v3.0.1, 2026-10-15).

You can also collect trajectories or evaluate without training against any OpenAI-compatible vLLM server: `vllm serve <model> --enable-auto-tool-choice --tool-call-parser <fmt>`, then `python -m training.rollout --config training/grpo_config.yaml --tasks tasks.jsonl --out traj.jsonl`.

## Open (owned by Nikhil)

* Base model, including license and the U.S. open-weight requirement (PLAN.md §2.11). The smoke model is only for testing the wiring.
* Compute budget, GPU type and count, and number of nodes.
* `group_size`, `tasks_per_batch`, learning rate, KL settings, clip ratio, `max_turns`, token limits, total steps, and checkpoint selection on `agent_development`.

## Not verified yet

* Whether verl passes the dataset's `extra_info` column to `AgentLoopBase.run(**kwargs)` under that name. `ToolAgentLoop` reads dataset fields from kwargs; confirm with the smoke run.
* Whether `{{ RUNPOD_SECRET_* }}` substitution works for `runpodctl pod create --env`. It is documented for templates. If it does not work, put the env in a template and pass `--template-id`.
* The Docker image has not been built here.
* The real `PhysiologyResearchEnv` signatures and behavior.

## Cost estimation method (no numbers are assumed here)

GPU cost ≈ `n_pods × gpus_per_pod × hourly_rate × wall_hours`, plus storage, plus Beam sandbox and scorer cost.

* `hourly_rate`: read it for the chosen GPU and cloud type (Secure or Community) from [RunPod pricing](https://www.runpod.io/pricing) or `runpodctl gpu list`, which shows on-demand pricing. Add volume and network-volume storage from the same page. `runpodctl billing pods` reports what was actually spent.
* `wall_hours ≈ total_training_steps × step_seconds / 3600`. Measure `step_seconds` in the smoke run (verl logs timing per step), then scale it. Rollout time is roughly bounded below by the slowest episode in the batch. Under the `fast` profile an episode may take up to 20 minutes of wall time (contract), so this time will likely dominate the GPU update time. Episodes per step = `group_size × tasks_per_batch`.
* Beam cost: episodes per step × steps × (sandbox seconds + scorer seconds) × Beam's published rates.
