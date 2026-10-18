"""Reproduce per-window exclusions against a frozen prepared dataset."""
import argparse
from collections import Counter
import csv
import io
from pathlib import Path
import sys
import zipfile
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_env.core.common import file_hash, read, write
from research_env.tasks.pulse.data import make_request, _segment, reference_hr

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('source_root')
parser.add_argument('dataset')
parser.add_argument('output')
a = parser.parse_args()
root = Path(a.dataset)
audit = read(root/'private/audit.json')
records = []
with zipfile.ZipFile(Path(a.source_root)/'GalaxyPPG.zip') as archive:
    for person, expected in audit['people'].items():
        with np.load(root/'private'/f'{person}.npz', allow_pickle=False) as data:
            ppg, acc, ecg = data['ppg'], data['acceleration'], data['ecg']
        with archive.open(f'Dataset/{person}/Event.csv') as stream:
            events = list(csv.DictReader(io.TextIOWrapper(stream)))
        counts, valid = Counter(), 0
        for index, event in enumerate(events):
            if event['status'].upper() != 'ENTER':
                continue
            exits = [e for e in events[index+1:] if e['session'] == event['session'] and e['status'].upper() == 'EXIT']
            if not exits:
                continue
            for cutoff in np.arange(float(event['timestamp'])/1000+8, float(exits[0]['timestamp'])/1000+1e-6, 2):
                try:
                    make_request(ppg, acc, cutoff)
                    reference_hr(_segment(ecg, cutoff-8, cutoff, .05))
                    valid += 1
                except ValueError as exc:
                    reason = str(exc)
                    counts[reason] += 1
                    records.append({'person': person, 'session': f'b{index:03}', 'cutoff_s': float(cutoff), 'reason': reason})
        if dict(counts) != expected['excluded_windows'] or valid != expected['eligible_windows']:
            raise ValueError(f'{person}: frozen eligibility no longer reproduces')
write(a.output, {'frozen_audit_sha256': file_hash(root/'private/audit.json'),
                 'all_masks_reproduced': True, 'excluded_windows': records,
                 'script_sha256': file_hash(__file__)})
print('Masks reproduced; excluded windows:', len(records))
