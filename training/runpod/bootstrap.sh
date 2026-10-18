#!/usr/bin/env bash
# Runs ON the RunPod pod as root, after training/runpod/push.sh copied the repository to
# /workspace/physbench and prepared data to /workspace/data/prepared/<family>.
# Uses the stock RunPod PyTorch image, so no custom image or registry is needed.
#
#   SKIP_TRAINER=1 bash training/runpod/bootstrap.sh   # env + sandbox only (CPU check)
#   bash training/runpod/bootstrap.sh                  # also installs verl v0.9.1 + vLLM
set -euo pipefail
REPO=/workspace/physbench
DATA=/workspace/data
VERL_REF="${VERL_REF:-v0.9.1}"
# uv-managed interpreters must live outside /root (0700) so the sandbox uid can execute them.
export UV_PYTHON_INSTALL_DIR=/opt/uv-python UV_LINK_MODE=copy UV_HTTP_TIMEOUT=500
cd "$REPO"

command -v uv >/dev/null || python3 -m pip install --quiet --break-system-packages uv==0.11.16 \
  || python3 -m pip install --quiet uv==0.11.16

# 1. Sandbox/env venv: the pinned numeric stack agent code and replay entries run with.
uv venv --quiet --python 3.13 /opt/env-venv
uv pip install --quiet --python /opt/env-venv/bin/python -r research_env/requirements.lock xlrd==2.0.1 pyyaml

# 2. Trainer venv: verl from its release tag with its committed lock (exact pins).
if [ "${SKIP_TRAINER:-0}" != 1 ]; then
  apt-get update -qq && apt-get install -y -qq git >/dev/null
  [ -d /opt/verl ] || git clone --quiet --depth 1 --branch "$VERL_REF" https://github.com/verl-project/verl.git /opt/verl
  (cd /opt/verl && uv sync --quiet --frozen --python 3.12 --extra vllm --extra fsdp)
  uv pip install --quiet --python /opt/verl/.venv/bin/python -e "$REPO" pyyaml
  /opt/verl/.venv/bin/python -c "import verl, vllm, torch; print('verl', verl.__version__, 'vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"
fi

# 3. Permissions for the linux-pod sandbox (uid 65534): private labels, controller code and
#    the repository are root-only; run directories are traversable; interpreters readable.
mkdir -p /workspace/runs
chmod 711 /workspace /workspace/runs
chmod -R go-rwx "$DATA" "$REPO"
chmod -R a+rX /opt/env-venv /opt/uv-python

cat > /workspace/pod-env.sh <<'EOF'
export PHYSIOLOGY_SANDBOX_PYTHON=/opt/env-venv/bin/python
# ssh shells may not inherit the pod's env: keep sandbox caps explicit (16 episodes x 4 GiB = 64 GiB).
export PHYSIOLOGY_SANDBOX_MAX_CPUS="${PHYSIOLOGY_SANDBOX_MAX_CPUS:-2}"
export PHYSIOLOGY_SANDBOX_MAX_MEMORY_GB="${PHYSIOLOGY_SANDBOX_MAX_MEMORY_GB:-4}"
export PYTHONPATH=/workspace/physbench
export PHYS_GRPO_CONFIG=/workspace/physbench/training/grpo_config.yaml
if [ -d /opt/verl/.venv ]; then export VIRTUAL_ENV=/opt/verl/.venv PATH=/opt/verl/.venv/bin:$PATH; fi
EOF

echo "pod: $(nproc) CPUs, $(awk '/MemTotal/ {printf "%.0f", $2/1048576}' /proc/meminfo) GiB RAM; size grpo.max_concurrent_episodes x PHYSIOLOGY_SANDBOX_MAX_MEMORY_GB to fit"

# 4. Self-check: sandbox gates on this exact container (fails closed if the network cannot be blocked).
. /workspace/pod-env.sh
/opt/env-venv/bin/python -m unittest tests.test_linux_pod_backend -v 2>&1 | tail -12
/opt/env-venv/bin/python -c "from research_env.core.worker import linux_pod_capabilities as c; print('sandbox capabilities:', c())"
echo "bootstrap done. In a shell on the pod: . /workspace/pod-env.sh"
