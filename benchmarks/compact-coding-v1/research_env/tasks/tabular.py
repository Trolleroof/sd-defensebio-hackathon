"""Shared measured-record transport for heat and load; labels stay controller-side."""
from pathlib import Path
from research_env.core.common import digest, read, write, validate_splits


def export_dataset(output, task, records, people, sources, audit, splits=None):
    root = Path(output)
    if root.exists():
        raise ValueError('Refuse to overwrite a prepared dataset')
    people = sorted(people, key=lambda p: digest([task+'-split-v1', p]))
    n = 5 if task == 'heat' else 10
    groups = validate_splits(read(splits) if splits else {
        'train': people[:-2*n], 'development': people[-2*n:-n], 'test': people[-n:]})
    if {p for group in groups.values() for p in group} != set(people):
        raise ValueError('Split must cover exactly the audited people')
    for role, members in groups.items():
        selected = [r for r in records if r['person'] in members]
        if set(members) != {r['person'] for r in selected}:
            raise ValueError('Every person needs eligible records')
        manifest = {'version': f'{task}-v1', 'task': task, 'role': role,
                    'sources': sources, 'records': selected}
        write(root/'private'/role/'manifest.json', manifest)
        if role != 'test':
            public = {**manifest, 'records': selected if role == 'train' else [
                {k: r[k] for k in ('id', 'person', 'session', 'request')} for r in selected]}
            write(root/'public'/role/'manifest.json', public)
    write(root/'private/splits.json', groups)
    write(root/'private/audit.json', audit)
    result = {'version': f'{task}-v1', 'task': task, 'splits': groups, 'sources': sources,
              'n_people': len(people), 'n_records': len(records),
              'reference_status': 'measured; retrospective builder access',
              'tier': 'primary-no-reference-calibration'}
    write(root/'dataset.json', result)
    return result


def iter_windows(manifest_path, with_reference=False):
    manifest = read(manifest_path)
    for r in manifest['records']:
        if with_reference and 'targets' not in r:
            raise ValueError('Private reference labels required')
        yield r['request'], r if with_reference else {k: r[k] for k in ('id', 'person', 'session')}
