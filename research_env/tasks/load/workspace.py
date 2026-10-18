"""Build a participant workspace: instructions, permitted training data, examples, baseline.

The workspace contains no development/test labels, no laboratory-only fields and
no source person IDs. Development feedback is obtained through the controller
(``python -m research_env.tasks.load dev-eval``), not by reading files.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from . import runner
from .importer import read_jsonl

PKG = Path(__file__).resolve().parent
TRACK_NOTES = {
    "primary": "Track: **primary**. Scored on 0, 5, 10 and 15 kg for unseen people. No measured metabolic input.",
    "calibration": ("Track: **calibration** (scored separately from primary). Each request includes the new person's "
                    "measured 0 kg power; you are scored on 5, 10 and 15 kg. This needs a laboratory measurement and "
                    "must be compared with the no-calibration baseline while disclosing the extra input."),
    "holdout15": ("Track: **holdout15** (scored separately). Training rows contain 0, 5 and 10 kg only; you are scored "
                  "on 15 kg for unseen people, so this tests load extrapolation plus person transfer."),
}


def build_workspace(prepared_dir, out_dir, track: str) -> dict:
    out = Path(out_dir)
    if out.exists():
        raise FileExistsError(f"{out} exists; refusing to overwrite a workspace")
    records = read_jsonl(Path(prepared_dir) / "records.jsonl")
    tm = runner.task_manifest()
    b = tm["budget"]
    budget = (f"- Wall clock: {b['wall_clock_s'] // 60} minutes; {b['dev_calls']} development calls\n"
              f"- CPU only, up to {b['cpu_cores']} cores and {b['ram_gb']} GB RAM (capped by the host)\n"
              f"- Incumbent snapshots at {', '.join(map(str, tm['checkpoints_min']))} minutes; declare your incumbent.")
    task = (PKG / "workspace_template" / "TASK.md").read_text().format(
        track=track, track_note=TRACK_NOTES[track], dev_calls=b["dev_calls"], budget=budget,
        cal_model=", `calibration_offset`" if track == "calibration" else "")
    (out / "examples").mkdir(parents=True)
    (out / "data").mkdir()
    (out / "baseline").mkdir()
    (out / "submission").mkdir()
    (out / "TASK.md").write_text(task)
    shutil.copyfile(PKG / "workspace_template" / "DATA_DICTIONARY.md", out / "DATA_DICTIONARY.md")
    rows = runner.training_rows(records, track)
    runner.write_jsonl(out / "data" / "train.jsonl", rows)
    # Example request: a training person's inputs (labels for training people are already public here).
    sample = next(r for r in records if r["split"] == "train" and r["eligible"] and r["load_kg"] == 10.0)
    prs = [r for r in records if r["source_person_id"] == sample["source_person_id"]]
    from .schema import build_request
    req = build_request(sample, track if track != "holdout15" else "primary", prs, request_id="r_example")
    req["track"] = track
    (out / "examples" / "request.json").write_text(json.dumps(req, indent=2))
    (out / "examples" / "prediction.json").write_text(json.dumps(
        {"mean_w": 250.0, "lower_w": 200.0, "upper_w": 300.0}, indent=2))
    shutil.copyfile(PKG / "baseline_models.py", out / "baseline" / "train.py")
    (out / "baseline" / "README.md").write_text(
        "Baseline: `train(train_manifest, None, out_dir, seed)` with `train_manifest = {\"records_path\": "
        "\"data/train.jsonl\", \"model\": \"ridge\"}`. The artifact's `predict.py` is a copy of this file.\n")
    return {"workspace": str(out), "track": track, "n_training_rows": len(rows),
            "n_training_people": len({r["person_key"] for r in rows})}
