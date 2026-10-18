"""Trusted builder profiling of load baselines: fit on 28 train people, select on the
10 development people, then report private test. Not a participant score."""
import argparse
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_env.core.common import write
from research_env.tasks.load.baseline_model import KINDS, predict
from research_env.tasks.load.baselines import train_rows
from research_env.tasks.load.data import load_records, load_splits
from research_env.tasks.load.evaluation import evaluate, replay_in_process
from research_env.tasks.load.rl_pool import CONFIGS
from research_env.tasks.load.tasks import fit_rows, write_manifest

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('prepared')
parser.add_argument('output')
parser.add_argument('--worker-backend', default=None, help='also replay via worker (e.g. macos-development)')
args = parser.parse_args()
root, out = Path(args.prepared), Path(args.output)
out.mkdir(parents=True, exist_ok=False)
records, splits = load_records(root), load_splits(root)
salt = 'builder-baselines-v1'
report = {}
for tier, variant in CONFIGS:
    config = f'{tier}-{variant}'
    rows = fit_rows(records, splits['train'], variant, salt)
    manifests = {}
    for role in ('development', 'test'):
        manifests[role] = out/'private'/f'{config}-{role}.json'
        write_manifest(manifests[role], records, splits[role], tier=tier, variant=variant, role=role, salt=salt)
    dev, test = {}, {}
    for kind in KINDS:
        artifact = out/'artifacts'/config/kind
        start = time.monotonic()
        train_rows(rows, artifact, 0, kind, tier)
        dev[kind], _ = replay_in_process(artifact, manifests['development'], predict)
        dev[kind]['training_wall_s'] = time.monotonic()-start
        if args.worker_backend:
            worker, _ = evaluate(artifact, manifests['development'], backend=args.worker_backend, wall_s=600)
            dev[kind]['worker_replay_primary_loss'] = worker['primary_loss']
            dev[kind]['worker_replay_valid'] = worker['execution_valid']
    selection = min(dev, key=lambda k: dev[k]['primary_loss'])  # development only
    write(out/f'{config}-selection.json', {'selected_on_development': selection})
    for kind in KINDS:
        test[kind], _ = replay_in_process(out/'artifacts'/config/kind, manifests['test'], predict)
    report[config] = {'development': dev, 'private_test': test, 'selected_on_development': selection}
    for kind in KINDS:
        print(f'{config:22s} {kind:13s} dev {dev[kind]["primary_loss"]:7.2f} W  test {test[kind]["primary_loss"]:7.2f} W'
              + (f'  worker {dev[kind].get("worker_replay_primary_loss")}' if args.worker_backend else ''))
    print(f'{config:22s} selected on development: {selection}')
write(out/'report.json', {'baselines': report, 'execution_context': 'trusted_builder_only',
                          'live_agent': 'pending', 'live_judge': 'pending', 'strict_container_replay': 'pending'})
