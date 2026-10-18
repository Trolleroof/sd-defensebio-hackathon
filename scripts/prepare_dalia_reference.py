"""Reuse SoliderPhys-Bench's restricted loader and adapter for allowed PPG-DaLiA files.

Does not unpickle or convert any final-test participant. Supply the existing
SoliderPhys-Bench source checkout rather than duplicating its source adapter here.
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from zipfile import ZipFile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('archive'); p.add_argument('output'); p.add_argument('--physbench-source', required=True)
    args = p.parse_args()
    source = Path(args.physbench_source).resolve(); root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(source))
    from physbench.data.dalia import convert_subject, CHANNELS, ACTIVITY_NAMES
    # Reference cohort fits the generator; development cohort only audits fidelity.
    allowed = {'S1', 'S4', 'S8', 'S12', 'S2', 'S3', 'S7', 'S10'}
    archive = Path(args.archive)
    with ZipFile(archive) as outer:
        nested = [x for x in outer.infolist() if Path(x.filename).name == 'data.zip']
        if nested:
            inner_path = root/'data.zip'
            with outer.open(nested[0]) as src, inner_path.open('wb') as dst:
                shutil.copyfileobj(src, dst)
            z = ZipFile(inner_path)
        else:
            z = outer
        try:
            members = {Path(m.filename).stem: m for m in z.infolist()
                       if m.filename.endswith('.pkl') and Path(m.filename).stem in allowed}
            if set(members) != allowed:
                raise ValueError(f'Missing allowed participant files: {sorted(allowed-set(members))}')
            metas = []
            for sid in sorted(allowed, key=lambda s: int(s[1:])):
                raw = root/f'{sid}.pkl'
                with z.open(members[sid]) as src, raw.open('wb') as dst:
                    shutil.copyfileobj(src, dst)
                meta = convert_subject(raw, root/'processed')
                if str(meta['subject']) != sid:
                    raise ValueError('Source participant identity mismatch')
                metas.append(meta)
                raw.unlink()
                print(json.dumps({'converted': sid, 'samples': meta['n_samples']}), flush=True)
        finally:
            if z is not outer:
                z.close(); inner_path.unlink()
    (root/'processed/manifest.json').write_text(json.dumps({
        'dataset': 'PPG-DaLiA', 'source_url': 'https://doi.org/10.24432/C53890',
        'license': 'CC BY 4.0', 'authors': 'Reiss, Indlekofer, Schmidt (2019)',
        'adapter_sha256': hashlib.sha256((source/'physbench/data/dalia.py').read_bytes()).hexdigest(),
        'subjects': metas, 'grid_step_s': 2, 'activity_names': ACTIVITY_NAMES,
        'channels': {k: {'unit': u, 'source': s, 'description': d} for k, (u, s, d) in CHANNELS.items()},
        'final_test_participants_converted': False}, indent=2)+'\n')


if __name__ == '__main__':
    main()
