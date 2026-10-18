"""Only public data and reference training tools enter a participant workspace."""
from pathlib import Path
import shutil
from research_env.tasks.registry import identify


def stage(public_root, workspace):
    public_root, workspace = Path(public_root), Path(workspace)
    task = identify(public_root/'train/manifest.json')
    workspace.mkdir(parents=True, exist_ok=False)
    shutil.copytree(public_root, workspace/'data')
    package = Path(__file__).resolve().parent
    names = ['__init__.py','__main__.py','requirements.lock','core/__init__.py','core/common.py','tasks/__init__.py',
             f'tasks/{task}/__init__.py', f'tasks/{task}/TASK.md', f'tasks/{task}/DATA_DICTIONARY.md']
    if task == 'pulse':
        names += ['tasks/pulse/data.py','tasks/pulse/baselines.py','tasks/pulse/example-request.json']
    else:
        names += ['tasks/baselines.py']
    for name in names:
        target = workspace/'research_env'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(package/name, target)
    return workspace
