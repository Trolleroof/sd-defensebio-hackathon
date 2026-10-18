"""Translate ``grpo_config.yaml`` into verl v0.9.1 Hydra overrides and launch ``verl.trainer.main_ppo``.

Every key used here appears in verl v0.9.1's ``verl/trainer/config/ppo_trainer.yaml``,
``rollout/rollout.yaml``, ``docs/algo/grpo.md`` or ``examples/grpo_trainer/run_qwen3_8b_fsdp.sh``.
Importing this module does not import verl, so it is CPU-testable.

    python -m training.verl_overrides --config training/grpo_config.yaml --data /workspace/data/rl --print
    python -m training.verl_overrides --config training/grpo_config.yaml --data /workspace/data/rl  # runs
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Sequence

AGENT_NAME = 'physiology_env_agent'
AGENT_LOOP_YAML = Path(__file__).with_name('agent_loop.yaml')


def _b(x: bool) -> str:
    return 'True' if x else 'False'


def overrides(cfg: dict, data_dir: str | Path) -> list[str]:
    m, g, gen, o, t, e = (cfg[k] for k in ('model', 'grpo', 'generation', 'optim', 'trainer', 'env'))
    data_dir = Path(data_dir)
    return [
        # algorithm: GRPO with group-relative advantage on the env's terminal reward
        'algorithm.adv_estimator=grpo',
        f'algorithm.norm_adv_by_std_in_grpo={_b(g["norm_adv_by_std"])}',
        'algorithm.use_kl_in_reward=False',
        # data: one row per TaskSpec; batch = tasks per step
        f"data.train_files=['{data_dir / 'train.parquet'}']",
        f"data.val_files=['{data_dir / 'dev.parquet'}']",
        f'data.train_batch_size={int(g["tasks_per_batch"])}',
        f'data.max_prompt_length={int(gen["max_prompt_tokens"])}',
        f'data.max_response_length={int(gen["max_response_tokens"])}',
        'data.filter_overlong_prompts=True',
        "data.truncation='error'",
        # model (base model is config; any HF id)
        f'actor_rollout_ref.model.path={m["base_model"]}',
        f'actor_rollout_ref.model.trust_remote_code={_b(m.get("trust_remote_code", False))}',
        'actor_rollout_ref.model.use_remove_padding=True',
        'actor_rollout_ref.model.enable_gradient_checkpointing=True',
        # actor / GRPO loss
        f'actor_rollout_ref.actor.optim.lr={float(o["learning_rate"])}',
        f'actor_rollout_ref.actor.ppo_mini_batch_size={int(o["ppo_mini_batch_size"])}',
        f'actor_rollout_ref.actor.ppo_epochs={int(o["ppo_epochs"])}',
        f'actor_rollout_ref.actor.clip_ratio={float(o["clip_ratio"])}',
        f'actor_rollout_ref.actor.use_kl_loss={_b(o["use_kl_loss"])}',
        f'actor_rollout_ref.actor.kl_loss_coef={float(o["kl_loss_coef"])}',
        f'actor_rollout_ref.actor.kl_loss_type={o["kl_loss_type"]}',
        f'actor_rollout_ref.actor.entropy_coeff={float(o["entropy_coeff"])}',
        f'actor_rollout_ref.actor.loss_agg_mode={o["loss_agg_mode"]}',
        # rollout: vLLM async server + our agent loop; n = G episodes per task
        'actor_rollout_ref.rollout.name=vllm',
        'actor_rollout_ref.rollout.mode=async',
        f'actor_rollout_ref.rollout.n={int(g["group_size"])}',
        f'actor_rollout_ref.rollout.temperature={float(gen["temperature"])}',
        f'actor_rollout_ref.rollout.top_p={float(gen["top_p"])}',
        f'actor_rollout_ref.rollout.tensor_model_parallel_size={int(t["rollout_tensor_parallel"])}',
        f'actor_rollout_ref.rollout.gpu_memory_utilization={float(t["rollout_gpu_memory_utilization"])}',
        'actor_rollout_ref.rollout.multi_turn.enable=True',
        f'actor_rollout_ref.rollout.multi_turn.format={m["tool_call_format"]}',
        f'actor_rollout_ref.rollout.multi_turn.max_assistant_turns={int(e["max_turns"])}',
        f'actor_rollout_ref.rollout.agent.agent_loop_config_path={AGENT_LOOP_YAML}',
        f'actor_rollout_ref.rollout.agent.default_agent_loop={AGENT_NAME}',
        f'actor_rollout_ref.rollout.agent.num_workers={int(t["agent_loop_workers"])}',
        # trainer: checkpointing + resume
        f'trainer.total_training_steps={int(t["total_training_steps"])}',
        f'trainer.save_freq={int(t["save_freq"])}',
        f'trainer.test_freq={int(t["test_freq"])}',
        f'trainer.resume_mode={t["resume_mode"]}',
        f'trainer.default_local_dir={Path(t["checkpoint_dir"]) / t["project_name"] / t["experiment_name"]}',
        f'trainer.project_name={t["project_name"]}',
        f'trainer.experiment_name={t["experiment_name"]}',
        f'trainer.logger={json.dumps(list(t["logger"]))}',
        f'trainer.nnodes={int(t["nnodes"])}',
        f'trainer.n_gpus_per_node={int(t["n_gpus_per_node"])}',
    ]


def command(cfg: dict, data_dir: str | Path) -> list[str]:
    return [sys.executable, '-m', 'verl.trainer.main_ppo', *overrides(cfg, data_dir)]


def main(argv: Sequence[str] | None = None) -> int:
    from training.config import base_model_is_placeholder, load_config
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=str(Path(__file__).with_name('grpo_config.yaml')))
    ap.add_argument('--data', required=True)
    ap.add_argument('--print', action='store_true', help='print the command and exit')
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    cmd = command(cfg, args.data)
    if args.print:
        print(shlex.join(cmd))
        return 0
    if base_model_is_placeholder(cfg):
        raise SystemExit('model.base_model is still the placeholder; set PHYS_GRPO__model__base_model')
    os.environ.setdefault('PHYS_GRPO_CONFIG', str(Path(args.config).resolve()))  # read by the agent loop
    os.execv(cmd[0], cmd)
    return 0  # unreachable


if __name__ == '__main__':
    raise SystemExit(main())
