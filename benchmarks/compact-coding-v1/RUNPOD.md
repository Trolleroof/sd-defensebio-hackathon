# Evaluate a fine-tuned coding agent served on RunPod

## Recommended topology

Run **agent inference on the RunPod GPU** and keep the trusted CPU controller on the host with the original prepared data, frozen suite and ARM64 Docker image. Only public participant prompts/code responses go to the model server. Offline predictor training and hidden replay stay on the controller.

The active evaluator image is Linux/ARM64, `sha256:7751f6262d2768889bebc8b567c771af0dae369db54ef79b208db155ebae9314`. A native AMD64 rebuild on a RunPod GPU host has a different image/allocation. That can evaluate a base/fine-tuned pair as a separately frozen replica pool, but its receipts cannot join the original Elo pool. A standard containerized Pod also should not be assumed to have a usable nested Docker daemon.

## 1. Prepare the GPU model server

Use a RunPod GPU template with a pinned vLLM image/version appropriate for the checkpoint. Determine memory needs from the actual base/checkpoint; this release does not select a GPU or rent a Pod. Keep the model weights, tokenizer and chat template immutable, and record their file hashes. Preserve the serving image digest and command in your deployment record.

Inside the Pod, after mounting the checkpoint:

```sh
# Set these to the actual checkpoint and the model-family-specific tool parser.
export CHECKPOINT=/workspace/checkpoints/YOUR_CHECKPOINT
export SERVED_MODEL=YOUR_IMMUTABLE_CHECKPOINT_LABEL
export TOOL_PARSER=YOUR_MODEL_FAMILY_TOOL_PARSER
# CHECKPOINT_API_KEY must be supplied through the Pod's secret environment.
vllm serve "$CHECKPOINT" \
  --served-model-name "$SERVED_MODEL" \
  --host 0.0.0.0 --port 8000 \
  --api-key "$CHECKPOINT_API_KEY" \
  --generation-config vllm \
  --enable-auto-tool-choice --tool-call-parser "$TOOL_PARSER"
```

The tool parser and chat template must match the actual checkpoint family. This is a template, not a tested deployment of your as-yet-unspecified weights. vLLM documents its [OpenAI-compatible tool calling and parsers](https://docs.vllm.ai/en/latest/features/tool_calling/). Its dependencies/version belong to the GPU serving environment, independently of the pinned CPU numerical environment in this release.

Prefer an SSH tunnel to port 8000 using the Pod's current SSH host/port from its Connect panel:

```sh
ssh -N -L 8000:127.0.0.1:8000 \
  -p "$RUNPOD_SSH_PORT" "$RUNPOD_SSH_USER@$RUNPOD_SSH_HOST"
```

Alternatively expose HTTP port 8000 and use `https://POD_ID-8000.proxy.runpod.net/v1/chat/completions` as the endpoint. The HTTP proxy has a 100-second timeout; slow non-streaming generation may need the SSH/direct TCP route. Keep API authentication enabled. See [RunPod port exposure](https://docs.runpod.io/pods/configuration/expose-ports).

## 2. Prepare the trusted controller

Clone the repository's `main`, then work from this versioned directory:

```sh
cd benchmarks/compact-coding-v1
python3.13 -m venv .venv
.venv/bin/pip install -r research_env/requirements.lock
.venv/bin/python -m scripts.verify_release
```

Obtain the **existing prepared** `.leaderboard-data/{pulse,heat,load}` and the original controller suite from the benchmark owner. They are intentionally absent from Git. Preserve the frozen eligible-window masks, participant splits and all bytes. Original data attributions/URLs and tree hashes are in `reference.json`; re-downloading a paper's workbook alone does not recreate a verified prepared bundle. Never put private evaluation labels in the GPU serving container or its training corpus.

On the original controller host, its suite points to the existing dataset paths and local image. Use an absolute suite path when running from this release directory:

```sh
export REFERENCE_SUITE=/absolute/path/to/original/.leaderboard-runs/compact-eight-suite.json
.venv/bin/python -m scripts.verify_release --suite "$REFERENCE_SUITE"
```

This must verify all source/data hashes, the suite seal, policy and CPU/memory allocation before scoring. Do not overwrite the original suite or alter its frozen source tree. This release's copied implementation is byte-identical, while its checkpoint adapter lives outside the source-hashed package.

## 3. Fill a checkpoint configuration

Copy `configs/checkpoint.example.json` to a private working config. Fill the exact served model name, competitor label, unique slug, endpoint and immutable checkpoint provenance. The runner rejects placeholders and missing checkpoint manifests. For a multi-file checkpoint, `checkpoint.manifest_sha256` is the SHA-256 of your immutable file-name/SHA-256 manifest, not a claimed remote verification of the served weights. Preserve the matching server deployment record.

Keep `decoding.max_tokens` at 1,536 and use the same native `respond` tool contract. Automatic tool choice must actually work. Record actual dtype, quantization, tokenizer and chat-template versions. If this model lacks native Medium/High control, explicitly record `effort_support: "unsupported"`; do not label its effort Medium by analogy. Supported reasoning fields may be added to `decoding` only if the server honors them, and then must be identical for base/fine-tuned comparisons. Extra decoding fields that the compact transport would drop are rejected.

The example gives self-hosted token prices as zero because no per-token OpenRouter participant charge applies. **RunPod GPU time/storage charges are separate** and are not controlled by the API-spend ledger. The required `--max-usd` below caps accounted API costs such as the external judge; it does not rent or stop a Pod. Stop rented infrastructure through your normal RunPod workflow when finished.

## 4. Run the nine episodes

Supply `CHECKPOINT_API_KEY` and `OPENROUTER_API_KEY` in the controller's environment; key values are never saved to the run files. Use a fresh output directory and an explicit judge API budget:

```sh
.venv/bin/python -m scripts.evaluate_checkpoint \
  --suite "$REFERENCE_SUITE" \
  --config /absolute/path/to/YOUR_FILLED_CHECKPOINT_CONFIG.json \
  --output .leaderboard-runs/YOUR_CHECKPOINT_SLUG \
  --max-usd 1.00
```

The example $1 is a future **judge/API budget**, independently selected for this checkpoint run; publishing this release does not authorize or start it. The runner first probes the native tool protocol, then runs three tasks × three agent seeds with the exact short protocol. It writes `checkpoint-status.json`, receipts, traces, artifacts, individual reviews and final `leaderboard.json` / `LEADERBOARD.md`. Resume with unchanged config/suite/output/cap; charges and attempted responses persist. A single checkpoint gets numerical losses and reviews; an informative Elo comparison requires opponents.

For a paired base/fine-tuned test, pass `--config` twice with distinct model/slug/competitor identities and matching deployment conditions. That produces 18 episodes. Both endpoints must accept the configured checkpoint API credential, and the same judge config applies to both.

To compare against the active eight-model pool, use the **exact original suite**, preserve and verify all nine reference episodes for every included competitor, and assemble a separate comparison directory with explicit copies of those complete run directories plus checkpoint runs. Load exact configs and use `research_env.compact.board(root, configs, judge, [0,1,2])`. Its receipt/configuration/hash checks reject a changed suite or identity. Keep the existing active sweep directory intact. Recompute Elo over the enlarged opponent pool; do not append a made-up Elo to an existing table.

## All-Linux replica option

On a Linux controller with a working Docker daemon, transfer the identical prepared data and run `scripts.verify_release --data-root PATH`. Build a native numerical image from the pinned requirements, resolve/pin an architecture-compatible Python base image, then obtain the immutable built image ID. Freeze a **new** suite using the provided `research_env.leaderboard freeze` CLI and that image. Verify with `--suite NEW_SUITE --replica`, and pass `--replica` to `scripts.evaluate_checkpoint`. Evaluate both unchanged base and fine-tuned checkpoints in this new pool. An architecture change and a new allocation are explicitly different conditions, even with identical task/data hashes.

No GPU deployment, checkpoint-specific tool parser, Linux Docker isolation, or live fine-tuned inference has been verified by the publication tests. Those are the next operational gates once the actual checkpoint and serving environment are supplied.
