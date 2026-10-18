# GRPO trainer image for RunPod: verl v0.9.1 (FSDP actor + async vLLM rollout) + this repo.
# Pins follow verl v0.9.1's own pyproject.toml / uv.lock: torch 2.11.0, vllm 0.24.0,
# transformers 5.9.0, CUDA 13.0 base (verl docs/start/install.rst "Pick a base image").
# Build context = repository root:
#   docker build -f training/runpod/Dockerfile -t <registry>/physiology-grpo:<tag> .
ARG CUDA_BASE=nvidia/cuda:13.0.2-devel-ubuntu24.04
FROM ${CUDA_BASE}

ARG VERL_REPO=https://github.com/verl-project/verl.git
ARG VERL_REF=v0.9.1
ARG PYTHON_VERSION=3.12
ENV DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy \
    UV_HTTP_TIMEOUT=500 \
    VLLM_ENABLE_CUDA_COMPATIBILITY=0 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates curl build-essential openssh-server tini && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.16 /uv /uvx /bin/

# verl from its release tag, resolved from verl's committed uv.lock (--frozen = exact pins).
RUN git clone --depth 1 --branch ${VERL_REF} ${VERL_REPO} /opt/verl && \
    cd /opt/verl && uv python install ${PYTHON_VERSION} && \
    uv sync --frozen --python ${PYTHON_VERSION} --extra vllm --extra fsdp
ENV VIRTUAL_ENV=/opt/verl/.venv \
    PATH=/opt/verl/.venv/bin:${PATH}

# This repository (env + training glue) into the same venv.
WORKDIR /workspace/physbench
COPY pyproject.toml README.md ./
COPY research_env ./research_env
COPY physbench ./physbench
COPY benchmark ./benchmark
COPY training ./training
RUN uv pip install --python /opt/verl/.venv/bin/python -e . && \
    python -c "import verl, vllm, torch; print('verl', verl.__version__, 'vllm', vllm.__version__, 'torch', torch.__version__)"

ENV PYTHONPATH=/workspace/physbench \
    PHYS_GRPO_CONFIG=/workspace/physbench/training/grpo_config.yaml
COPY training/runpod/entrypoint.sh /usr/local/bin/physiology-grpo-entrypoint
RUN chmod +x /usr/local/bin/physiology-grpo-entrypoint
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/physiology-grpo-entrypoint"]
