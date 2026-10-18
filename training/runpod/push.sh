#!/usr/bin/env bash
# Run LOCALLY: copy the tracked repository files and prepared heat/load data to a running pod
# over its SSH endpoint (from `runpodctl ssh info`), then run bootstrap.sh there.
#   training/runpod/push.sh <pod-id>             # copy + full bootstrap (trainer included)
#   SKIP_TRAINER=1 training/runpod/push.sh <id>  # copy + env/sandbox bootstrap only
set -euo pipefail
POD_ID="${1:?usage: push.sh <pod-id>}"
cd "$(git rev-parse --show-toplevel)"

SSH_CMD="$(runpodctl ssh info "$POD_ID" | python3 -c '
import json, sys
info = json.load(sys.stdin)
info = info[0] if isinstance(info, list) else info
cmd = info.get("command") or info.get("sshCommand") or info.get("ssh_command")
if not cmd:
    sys.exit("no ssh command in runpodctl ssh info output: keys " + ",".join(info))
print(cmd)')"
read -r -a SSH <<<"$SSH_CMD"
SSH+=(-o StrictHostKeyChecking=accept-new)
echo "pod ssh: ${SSH[*]}"

# Tracked + new (non-ignored) files only: no venvs, run outputs or local secrets.
git ls-files -co --exclude-standard -z | tar --null -T - -czf - \
  | "${SSH[@]}" 'mkdir -p /workspace/physbench && tar -xzf - -C /workspace/physbench'
for family in heat load pulse; do
  src=".research-data/$family/prepared"
  [ -d "$src" ] || { echo "skip $family: $src missing"; continue; }
  tar -czf - -C "$src" . | "${SSH[@]}" "mkdir -p /workspace/data/prepared/$family && tar -xzf - -C /workspace/data/prepared/$family"
done
"${SSH[@]}" "SKIP_TRAINER=${SKIP_TRAINER:-0} bash /workspace/physbench/training/runpod/bootstrap.sh"
