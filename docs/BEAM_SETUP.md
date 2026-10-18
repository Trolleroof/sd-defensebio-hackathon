# Beam backend setup (participant sandboxes + trusted scorer)

Status, October 17, 2026: code and mocked-SDK tests only. Nobody has created a Beam sandbox, volume, image or function for this repository, and no Beam spend has been authorized. Every command below that talks to Beam is a one-time user step. Verified against `beam-client` 0.2.233 and the docs at https://docs.beam.cloud/llms.txt.

## What runs where

| Workload | Code | Mounts | Network | Secrets |
| --- | --- | --- | --- | --- |
| Participant sandbox (agent `command`s, and every candidate replay bout) | `research_env/beam/sandbox.py` via `worker.execute(..., backend='beam')` | public data only (`/public`) plus the synced workspace at `/work` | `block_network=True`, `ports=[]` | none (`secrets=[]`); sandboxes do not inherit the workspace token |
| Trusted scorer (holds private labels) | `research_env/beam/scorer_app.py` (`beam.function`) | private volume at `/private` | default | `BEAM_TOKEN`, bound by name so it can launch participant sandboxes |

The scorer receives a frozen artifact (tar bytes plus content digest), checks the digest, resolves a manifest *name* on its private volume, and runs the existing trusted streaming replay with `backend='beam'`. Candidate code never runs in the scorer container. Each bout gets a fresh network-blocked participant sandbox with no private mount, and the scorer returns only the score dict (`predictions` are stripped). Participant sandboxes get no Beam token, no secrets and no outbound network, and they expose no ports, so they cannot invoke the scorer.

Streaming protocol: Beam process stdin is uploaded once and then closed with EOF; it is not interactive (see the Processes doc). To keep the causal one-request-then-one-response order, `remote_helper.py bridge` runs inside the sandbox. It relays file-channel requests (`req-N.json` + `req-N.ready`) to the candidate's stdin, one line at a time, and publishes `resp-N.json` atomically. Request N+1 does not exist in the sandbox until the controller has read response N. Extra output, missing or oversized responses, and nonfinite JSON make the attempt invalid, the same as with docker.

## One-time setup (user runs; billable steps marked $)

```sh
# 1. SDK/CLI (package name per https://docs.beam.cloud/v2/getting-started/installation.md)
uv venv --python 3.13 .venv-beam
uv pip install --python .venv-beam/bin/python beam-client -r research_env/requirements.lock
. .venv-beam/bin/activate        # add .venv-beam/ to .gitignore first (see requested changes)

# 2. Sign in (stores ~/.beam/config.ini)
beam login                       # or: beam login --browserless

# 3. Scorer token secret (plaintext env var inside the scorer only; never bound to sandboxes)
beam secret create BEAM_TOKEN "<api token from platform.beam.cloud/settings/api-keys>"

# 4. Prepare data locally, then dry-run and upload public vs private to separate volumes
python -m research_env.beam.upload_data .research-runs/new-pulse --dataset pulse \
    --private-extra RUN/private/task-manifests          # dry run: prints beam volume create / beam cp
python -m research_env.beam.upload_data .research-runs/new-pulse --dataset pulse --execute   # $ storage

# 5. Build the participant image (python:3.13.3-slim-bookworm@sha256:56a1... + pip 26.2.1 + requirements.lock)
python -m research_env.beam.image                      # $ build; prints the image id
export PHYSIOLOGY_BEAM_PARTICIPANT_IMAGE_ID=<printed id>

# 6. Optional per-task reset image for fast GRPO episode resets
python -c "from research_env.beam.snapshots import create_reset_image as c; print(c('WORKSPACE', '$PHYSIOLOGY_BEAM_PARTICIPANT_IMAGE_ID'))"   # $
```

`upload_data` refuses a public tree that contains symlinks, any `private` path component, or JSON with `reference*` keys. It also refuses a private extra that overlaps the public tree. Public data lands at `beam://physiology-public/<dataset>/` and private data at `beam://physiology-private/<dataset>/`.

Configuration lives in `research_env/beam/config.py` (`BeamConfig`). Override any field with `PHYSIOLOGY_BEAM_<FIELD>`, for example `PHYSIOLOGY_BEAM_PUBLIC_MOUNT_MODE=cloud_bucket` with `PHYSIOLOGY_BEAM_PUBLIC_BUCKET=<bucket>`. In that mode the bucket keys are Beam secret *names* (`PHYSIOLOGY_PUBLIC_BUCKET_ACCESS_KEY` / `..._SECRET_KEY`). Leave `PHYSIOLOGY_BEAM_RESOURCE_ISOLATION_VERIFIED` false until the gates below pass on Beam.

Before scored use, re-run the container gates on Beam: numerical fit, unreadable private files, network denied, wall timeout, and a forced memory-limit failure (`scripts/check_pulse_container.py` is docker-specific). Each gate starts a sandbox, so each is billable.

## Verified Beam API facts

- `Sandbox(cpu, memory, image, keep_warm_seconds, volumes, secrets, env, sync_local_dir, block_network, allow_list, docker_enabled, ports, ...)`. The defaults are 1 CPU, 128 MiB and a 600 s lifetime, and `keep_warm_seconds` is a TTL, not an idle timer. https://docs.beam.cloud/v2/sandbox/configuration.md
- `block_network=True` blocks outbound traffic while exposed ports stay reachable. Do not combine it with `allow_list`. https://docs.beam.cloud/v2/sandbox/networking.md
- `process.exec(*args, cwd, env, stdin)`: stdin is upload-then-EOF. `wait(timeout)` does not kill the process; `kill()` does. `status()` returns -1 while the process runs. https://docs.beam.cloud/v2/sandbox/processes.md
- `fs.upload_file`, `download_file`, `write_text`, `read_text`, `wait_for` and `create_directory`. https://docs.beam.cloud/v2/sandbox/filesystem.md
- `create_image_from_filesystem()` returns an image id for `Image.from_id`. `snapshot_memory()` is a separate checkpoint, and this backend does not use it. https://docs.beam.cloud/v2/sandbox/snapshots.md
- `Volume(name, mount_path)` has no read-only option. `CloudBucketConfig(read_only=True, access_key=<secret name>, ...)` does. Volumes are workspace-scoped, and `beam volume create` and `beam cp` manage them. https://docs.beam.cloud/v2/data/volume.md
- Function and endpoint secrets arrive as plaintext env vars. Sandboxes get brokered placeholders and no workspace token. https://docs.beam.cloud/v2/environment/secrets.md
- `@function(cpu, memory, image, volumes, secrets, env, timeout, retries)` and `.remote()`. https://docs.beam.cloud/v2/function/running-functions.md
- `Image.from_registry`, `add_commands`, `add_python_packages(<requirements path>)` and `build()`. The SDK `PythonVersion` enum stops at 3.12, so the image uses the base image's own Python 3.13.3. https://docs.beam.cloud/v2/environment/custom-images.md

## Cost estimate (rates from https://www.beam.cloud/pricing, read 2026-10-17; recheck before spending)

| Item | Listed rate |
| --- | --- |
| Sandbox CPU | $0.0000375 / s per physical core (2 vCPU equivalent) |
| Sandbox RAM | $0.0000064 / s per GiB |
| Serverless CPU-only (the scorer function) | $0.0000125 / s per core, $0.0000021 / s per GiB |
| Volume storage | free up to 1 TB, then $0.021 / GB-month (https://docs.beam.cloud/v2/resources/pricing-and-billing.md) |

Billing covers the time a container runs. Machine start and image pull are not billed (pricing-and-billing page).

Per sandbox-second at the contract cap (32 GiB), the cost depends on what `cpu=8` buys. The pricing page lists physical cores, and the SDK does not say whether `cpu` counts cores or vCPUs (unverified):
- if `cpu=8` means 8 physical cores: 8 x 0.0000375 + 32 x 0.0000064 = $0.000505/s, about $1.82/h;
- if it means 8 vCPU (4 cores): 4 x 0.0000375 + 32 x 0.0000064 = $0.000355/s, about $1.28/h.

Only sandbox running time is billed. Per GRPO group: cost ≈ episodes x (sum of command seconds + replay seconds) x rate. Benchmark replay starts one sandbox per bout, so cold starts add running time per bout. Measure seconds on a small pilot before budgeting; this document does not assume episode durations.

## Known gaps and differences from docker

- **Public mount read-only is not enforced in `volume` mode.** `Volume` has no RO flag, and results report `public_mount_read_only: false`. A participant could modify the shared public volume while other episodes read it. Use `cloud_bucket` mode (RO enforced) or `none` (stage public data into the workspace or the reset image) for training.
- **Hardening flags.** The SDK exposes nothing comparable to docker's `--read-only` root, `--cap-drop=ALL`, `no-new-privileges`, `--pids-limit` or fixed `--user`. The helper applies `RLIMIT_FSIZE` (16 MiB) and `RLIMIT_CPU`. Each call gets a fresh sandbox, and the TTL (timeout + 120 s) is a backstop if the controller dies.
- **Memory peak** is not reported (`peak_rss_bytes_sampled: None`). Memory-limit and OOM behavior is unverified. `scored_resource_isolation` stays false until `resource_isolation_verified` is set after the gates pass.
- **Kill semantics.** `process.kill()` kills the main process only, so background children survive until `terminate()`, which runs on every path.
- **Latency.** Each streamed request costs about 3 filesystem RPCs, and the workspace is synced as a hash delta both ways (2 GiB cap). Replay roots are not synced back.
- **Trusted-side token.** The scorer holds a plaintext `BEAM_TOKEN` (documented Beam behavior for functions). Anyone with workspace access can read either volume, so keep the Beam workspace restricted to trusted operators.
- `.remote()` on a function syncs the local working directory unless `.beamignore` excludes it (functions doc).

## Requested changes outside beam ownership

- Repo root `.gitignore` and a new `.beamignore`: add `.venv-beam/`, `.research-data/`, `.research-runs/` and `.venv*/` so the scorer upload never ships local data or venvs.
- `research_env/core/evaluation.py` (rl-core): `scored_resource_isolation=backend == 'docker'` reports Beam as non-isolated even after verification. Suggestion: `all(r.get('scored_resource_isolation') for r in resources)`.
- `research_env/__main__.py` (Set 1 shared CLI): add `'beam'` to the `--backend` choices for `replay`, `run` and `agent`.
