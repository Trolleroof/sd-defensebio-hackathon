"""physbench command line.

    physbench prepare                                   # raw PPG-DaLiA -> processed recordings
    physbench setup-sandbox                             # host isolation setup (root, Linux)
    physbench baselines --split dev                     # baseline table on dev subjects
    physbench run --split dev --run-id dev-0 --provisional
    physbench freeze --dev-run dev-0 --margin 0.10 --note "..."
    physbench run --split heldout --run-id pilot-0      # needs frozen criteria
    physbench report --run-id pilot-0
"""

from __future__ import annotations

import os

# Episodes run in threads; multi-threaded BLAS called from several threads can deadlock.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402

from .config import ROOT, load_task, load_yaml  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(prog="physbench")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--raw", default="data/raw/x/PPG_FieldStudy")
    p.add_argument("--out", default="data/processed/dalia")
    sub.add_parser("setup-sandbox")
    p = sub.add_parser("baselines")
    p.add_argument("--task", default="configs/task_hr_transfer.yaml")
    p.add_argument("--split", choices=["dev"], default="dev", help="held-out baselines are only computed inside a frozen pilot run")
    p = sub.add_parser("run")
    p.add_argument("--pilot", default="configs/pilot-0.yaml")
    p.add_argument("--split", choices=["dev", "heldout"], required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--models", nargs="*")
    p.add_argument("--subjects", nargs="*")
    p.add_argument("--provisional", action="store_true", help="allow dev runs before criteria are frozen")
    p = sub.add_parser("freeze")
    p.add_argument("--task", default="configs/task_hr_transfer.yaml")
    p.add_argument("--dev-run", required=True)
    p.add_argument("--margin", type=float, required=True)
    p.add_argument("--anchor", default="kinetic_pop_calibrated")
    p.add_argument("--note", required=True)
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("report")
    p.add_argument("--run-id", required=True)
    a = ap.parse_args(argv)

    if a.cmd == "prepare":
        from .data.dalia import convert_all
        for m in convert_all(ROOT / a.raw, ROOT / a.out):
            print(m)
    elif a.cmd == "setup-sandbox":
        from .sandbox import setup_host
        setup_host(ROOT)
        print("sandbox ready")
    elif a.cmd == "baselines":
        from .runner import baseline_scores
        cfg = load_task(a.task)
        for s in cfg.dev_subjects:
            b = baseline_scores(cfg, s)
            print(s, json.dumps({k: {n: round(v["phase_balanced_mae_bpm"], 2) for n, v in d.items()} for k, d in b.items()}))
    elif a.cmd == "run":
        from .runner import run_batch, write_report
        pilot = load_yaml(a.pilot)
        cfg = load_task(pilot["task_config"])
        run_batch(cfg, pilot, a.split, a.run_id, a.workers, a.models, a.subjects, a.provisional)
        print(f"report: {ROOT / 'runs' / a.run_id / 'summary.md'}")
    elif a.cmd == "freeze":
        from .runner import freeze
        cfg = load_task(a.task)
        print(f"frozen: {freeze(cfg, a.dev_run, a.margin, a.anchor, a.note, a.force)}")
    elif a.cmd == "report":
        from .runner import write_report
        print(write_report(a.run_id))


if __name__ == "__main__":
    main()
