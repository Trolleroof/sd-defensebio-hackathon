"""Set 3: load-carriage metabolic-cost modeling task (EVOBREATH 2015).

Shared interface (see docs/AGENT_BUILD_INSTRUCTIONS.md, common contract A):
    prepare  -> importer.prepare
    train    -> baseline_models.train (baselines) / participant submission/train.py
    predict  -> artifact predict.py, run by worker.predict_in_worker
    score    -> scoring.score
"""
TASK_ID = "load-evobreath-v1"
