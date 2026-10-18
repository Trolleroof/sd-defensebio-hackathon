"""Explicit task dispatch; never silently apply pulse scoring to another target."""
from research_env.core.common import read, validate_prediction

TASKS = ('pulse', 'heat', 'load')


def identify(manifest):
    m = read(manifest)
    task = m.get('task')
    if task is None and m.get('version') == 'galaxy-pulse-causal-v1':
        task = 'pulse'
    if task not in TASKS:
        raise ValueError('Unknown task manifest')
    return task


def iter_windows(manifest, with_reference=False):
    if identify(manifest) == 'pulse':
        from research_env.tasks.pulse.data import iter_windows as iterator
    else:
        from research_env.tasks.tabular import iter_windows as iterator
    return iterator(manifest, with_reference)


def validate(prediction, task):
    if task == 'pulse':
        return validate_prediction(prediction)
    from research_env.tasks.scoring import validate as tabular_validate
    return tabular_validate(prediction, task)


def score(references, predictions, task):
    if task == 'pulse':
        from research_env.tasks.pulse.scoring import score as pulse_score
        return pulse_score(references, predictions)
    from research_env.tasks.scoring import score as tabular_score
    return tabular_score(references, predictions, task)
