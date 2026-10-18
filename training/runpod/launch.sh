#!/usr/bin/env bash
# Create the GRPO trainer pod on RunPod with runpodctl (CLI syntax verified against
# runpodctl docs/runpodctl_pod_create.md, v3.0.x: `runpodctl pod create --image --gpu-id ...`).
#
# THIS CREATES PAID COMPUTE. Default is a dry run that only prints the command.
#   training/runpod/launch.sh            # dry run: print the exact runpodctl command
#   training/runpod/launch.sh --yes      # actually create the pod
#
# Configuration: environment variables (see training/runpod/env.example). Optionally
#   set -a; source training/runpod/.env.local; set +a
# Auth: runpodctl reads RUNPOD_API_KEY from the environment. This script never reads
# ~/.runpod/config.toml and never prints the key.
set -euo pipefail

CONFIRM=0
[ "${1:-}" = "--yes" ] && CONFIRM=1

: "${PHYS_GRPO_IMAGE:?set PHYS_GRPO_IMAGE (image built from training/runpod/Dockerfile)}"
: "${PHYS_GRPO_GPU_ID:?set PHYS_GRPO_GPU_ID (see: runpodctl gpu list)}"
POD_NAME="${PHYS_GRPO_POD_NAME:-physiology-grpo}"
GPU_COUNT="${PHYS_GRPO_GPU_COUNT:-1}"
CLOUD_TYPE="${PHYS_GRPO_CLOUD_TYPE:-SECURE}"
DISK_GB="${PHYS_GRPO_CONTAINER_DISK_GB:-100}"
VOLUME_GB="${PHYS_GRPO_VOLUME_GB:-200}"
MODE="${PHYS_GRPO_MODE:-smoke}"

if [ "$MODE" = "train" ] && [[ "${PHYS_GRPO__model__base_model:-CHOOSE_BASE_MODEL}" == CHOOSE_BASE_MODEL* ]]; then
  echo "refusing: PHYS_GRPO__model__base_model is still the placeholder (base model not chosen)" >&2
  exit 2
fi

# Pod environment as a JSON object (runpodctl --env takes JSON). Forward PHYS_GRPO_MODE and every
# PHYS_GRPO__* config override; secrets are passed as RunPod secret references, not values.
ENV_JSON="$(python3 - <<'PY'
import json, os
env = {k: v for k, v in os.environ.items() if k.startswith('PHYS_GRPO__')}
env['PHYS_GRPO_MODE'] = os.environ.get('PHYS_GRPO_MODE', 'smoke')
hf = os.environ.get('PHYS_GRPO_HF_SECRET_NAME')
beam = os.environ.get('PHYS_GRPO_BEAM_SECRET_NAME')
if hf:
    env['HF_TOKEN'] = '{{ RUNPOD_SECRET_%s }}' % hf
if beam:
    env['BEAM_TOKEN'] = '{{ RUNPOD_SECRET_%s }}' % beam
print(json.dumps(env, sort_keys=True))
PY
)"

CMD=(runpodctl pod create
  --name "$POD_NAME"
  --image "$PHYS_GRPO_IMAGE"
  --gpu-id "$PHYS_GRPO_GPU_ID"
  --gpu-count "$GPU_COUNT"
  --cloud-type "$CLOUD_TYPE"
  --container-disk-in-gb "$DISK_GB"
  --ports "22/tcp"
  --env "$ENV_JSON")
if [ -n "${PHYS_GRPO_NETWORK_VOLUME_ID:-}" ]; then
  CMD+=(--network-volume-id "$PHYS_GRPO_NETWORK_VOLUME_ID")
else
  CMD+=(--volume-in-gb "$VOLUME_GB" --volume-mount-path /workspace)
fi
[ -n "${PHYS_GRPO_REGISTRY_AUTH_ID:-}" ] && CMD+=(--registry-auth-id "$PHYS_GRPO_REGISTRY_AUTH_ID")

printf 'runpodctl command:\n  '; printf '%q ' "${CMD[@]}"; echo
if [ "$CONFIRM" -ne 1 ]; then
  echo "dry run only. Re-run with --yes to create the pod (billed per RunPod pricing)."
  exit 0
fi
command -v runpodctl >/dev/null || { echo "runpodctl not installed" >&2; exit 1; }
[ -n "${RUNPOD_API_KEY:-}" ] || echo "note: RUNPOD_API_KEY not set; runpodctl will use its own configured key" >&2
"${CMD[@]}"
echo "follow logs:  runpodctl pod logs <pod-id> --follow"
echo "stop billing: runpodctl pod stop <pod-id>   (delete with: runpodctl pod delete <pod-id>)"
