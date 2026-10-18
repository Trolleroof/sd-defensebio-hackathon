"""Frozen conventional search through the same participant command/evaluation budget."""
import argparse
from pathlib import Path
import shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_env.core.agent import ResearchEnvironment
from research_env.core.common import read, write


def stage(public_root, workspace):
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=False)
    shutil.copytree(public_root, workspace/'data')
    package = Path(__file__).resolve().parents[1]/'research_env'
    # Copy permitted training implementation only, not trusted runner/judging.
    for name in ('__init__.py', '__main__.py', 'requirements.lock', 'core/__init__.py', 'core/common.py',
                 'tasks/__init__.py', 'tasks/pulse/__init__.py', 'tasks/pulse/data.py',
                 'tasks/pulse/baselines.py', 'tasks/pulse/TASK.md', 'tasks/pulse/DATA_DICTIONARY.md',
                 'tasks/pulse/example-request.json'):
        target = workspace/'research_env'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(package/name, target)
    return workspace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset')
    parser.add_argument('run_root')
    parser.add_argument('--backend', choices=('docker', 'macos-development'), default='docker')
    parser.add_argument('--image')
    parser.add_argument('--calls', type=int, default=30)
    args = parser.parse_args()
    if not 1 <= args.calls <= 30:
        raise ValueError('Calls must be 1..30')
    root = Path(args.run_root)
    workspace = stage(Path(args.dataset)/'public', root/'workspace')
    environment = ResearchEnvironment(root/'private', workspace, Path(args.dataset)/'private/development/manifest.json',
                                      backend=args.backend, image=args.image)
    # Predeclared grid; no private test query. Same max wall/calls as participant.
    grid = [('median', 10., .5), ('spectral', 10., .5)]
    grid += [('motion', 10., weight) for weight in (0., .25, .5, .75, 1.)]
    grid += [('ridge', 10**(-4+i*.4), .5) for i in range(23)]
    best, evidence, results = None, [], []
    for kind, alpha, motion in grid[:args.calls]:
        if environment.session.remaining() <= 0:
            break
        decision = {'hypothesis': f'Frozen grid candidate {kind} alpha={alpha} motion_weight={motion}',
                    'expected_result': 'Measure development MAE using identical people and eligible windows',
                    'evidence_ids': list(evidence), 'code_diff': 'Set only baseline kind and declared scalar hyperparameters'}
        python = sys.executable if args.backend == 'macos-development' else 'python'
        code = ('import shutil;from research_env.tasks.pulse.baselines import train;'
                'shutil.rmtree("artifact",ignore_errors=True);'
                f'train("data/train/manifest.json",None,"artifact",0,{kind!r},{alpha!r},{motion!r})')
        trained = environment.act({'operation': 'command', 'argv': [python, '-c', code], 'decision': decision, 'seed': 0})
        result = environment.act({'operation': 'evaluate', 'decision': decision}) if trained['execution_valid'] else {'execution_valid': False, 'primary_loss': None}
        results.append(result)
        if result['execution_valid'] and (best is None or result['primary_loss'] < best['primary_loss']):
            best = result
            environment.act({'operation': 'select', 'candidate_id': best['candidate_id'], 'reason': 'Lowest public development MAE so far', 'evidence_ids': evidence})
        evidence = [e['event_id'] for e in environment.session.events() if e['kind'] == 'development_result']
        write(root/'search-results.json', {'results': results, 'selected': best,
              'validity_rate': sum(r['execution_valid'] for r in results)/len(results),
              'strict_resource_comparison': args.backend == 'docker'})
    if best is not None and environment.session.remaining() > 0:
        environment.act({'operation': 'submit', 'candidate_id': best['candidate_id'],
                         'reason': 'Minimum public development loss in frozen grid', 'evidence_ids': evidence})
    environment.session.export(root/'trajectory.jsonl', 'agent_development')

if __name__ == '__main__':
    main()
