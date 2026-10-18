# Active compact physiology coding benchmark v1

This is the versioned implementation of the **currently running eight-model, 72-episode comparison**. It is separate from the repository's older `physbench`, `benchmark`, RL and task-specific research environments. Those have different protocols and budgets. Run commands **from this directory** so Python imports this release's `research_env` package.

- [Exact protocol and portability rules](SPEC.md)
- [RunPod fine-tuned checkpoint instructions](RUNPOD.md)
- [Machine-readable running reference](reference.json), [wire protocol](protocol.json)
- [Exact eight-model plan/configurations](leaderboard/sweeps/compact-eight/plan.json)
- [Operational sweep history and amendments](ACTIVE_SWEEP.md)

The vendored `research_env` files and `scripts/search_tasks.py` match all 42 hashes of the running sweep. `scripts/verify_release.py` checks that identity without making an API call. Prepared data, private test labels, live traces, credentials and checkpoints are not stored in this release. The controller needs the existing prepared dataset bundle; exact dataset hashes, participant splits and source provenance are recorded in `reference.json`.

```sh
python3.13 -m venv .venv
.venv/bin/pip install -r research_env/requirements.lock
.venv/bin/python -m scripts.verify_release
.venv/bin/python -m unittest discover -s tests -v
```

No checkpoint was evaluated on RunPod as part of publishing this release. A model checkpoint and tool-capable inference server must be supplied. The reference numerical evaluator uses a pinned ARM64 image. Serving model inference on a RunPod GPU while retaining the original evaluator preserves that context; moving numerical replay to Linux/AMD64 creates a separately frozen comparison pool.
