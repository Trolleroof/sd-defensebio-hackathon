#!/usr/bin/env bash
# Container entrypoint on the RunPod pod. All configuration comes from environment variables
# (set by launch.sh via `runpodctl pod create --env`). Secrets are never echoed.
#   PHYS_GRPO_MODE    train | smoke | dataset | shell        (default: shell)
#   PHYS_GRPO_CONFIG  path to grpo config yaml inside the image (default: training/grpo_config.yaml)
#   PHYS_GRPO_DATA    dataset dir on the volume              (default: /workspace/data/rl)
#   PHYS_GRPO__<section>__<key>  config overrides (see training/config.py), e.g. PHYS_GRPO__model__base_model
set -euo pipefail
cd /workspace/physbench
MODE="${PHYS_GRPO_MODE:-shell}"
DATA="${PHYS_GRPO_DATA:-/workspace/data/rl}"

case "$MODE" in
  smoke)
    export PHYS_GRPO_CONFIG=/workspace/physbench/training/grpo_config.smoke.yaml
    python -m training.build_dataset --config "$PHYS_GRPO_CONFIG" --out "$DATA"
    exec python -m training.verl_overrides --config "$PHYS_GRPO_CONFIG" --data "$DATA" ;;
  dataset)
    exec python -m training.build_dataset --config "$PHYS_GRPO_CONFIG" --out "$DATA" ;;
  train)
    [ -f "$DATA/train.parquet" ] || python -m training.build_dataset --config "$PHYS_GRPO_CONFIG" --out "$DATA"
    exec python -m training.verl_overrides --config "$PHYS_GRPO_CONFIG" --data "$DATA" ;;
  shell)
    # Keep the pod alive for interactive use (ssh / web terminal); run modes by hand.
    if [ -x /usr/sbin/sshd ]; then mkdir -p /run/sshd && /usr/sbin/sshd || true; fi
    exec sleep infinity ;;
  *)
    echo "unknown PHYS_GRPO_MODE=$MODE" >&2; exit 2 ;;
esac
