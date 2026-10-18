"""GalaxyPPG v1 importer. Raw bytes immutable; references never enter requests."""
from __future__ import annotations
from collections import Counter
import csv
import hashlib
from pathlib import Path
import shutil
import zipfile
import numpy as np
from scipy import signal
from research_env.core.common import digest, file_hash, read, validate_splits, write

VERSION = 'galaxy-pulse-causal-v1'
RULES = {'window_s': 8, 'step_s': 2, 'sample_rate_hz': 25,
         'boundary': 'start <= timestamp <= cutoff, 201 resampled points at 25 Hz',
         'reference': '60 / mean RR between detected ECG R peaks within closed 8-second window',
         'ecg_clock': '(phoneTimestamp - 32400000) / 1000, matching original authors',
         'input_clock': 'Galaxy timestamp / 1000; sensor acquisition time, not gateway receipt',
         'quality': 'finite; coverage within 0.1 s; max PPG/ACC gap <=0.2 s; ECG <=0.05 s; '
                    '>=4 peaks; RR 0.28..1.5 s; RR CV <=0.25; two prominence estimates within 5 bpm',
         'reference_status': 'provisional ECG detector; human reference audit required before benchmark freeze',
         'acceleration_units': 'm/s^2 as source Galaxy accelerometer; not normalized by g',
         'excluded_inference_channels': ['ECG', 'vendor HR', 'activity', 'Empatica', 'person ID', 'source path'],
         'dalia_exposures': {'generator': ['S1', 'S4', 'S8', 'S12'], 'development': ['S2', 'S3', 'S7', 'S10']}}


def fetch(source_root):
    """Explicit original-source download, metadata/license/checksum retention."""
    import urllib.request
    root = Path(source_root)
    root.mkdir(parents=True, exist_ok=True)
    url = 'https://zenodo.org/api/records/14635823'
    with urllib.request.urlopen(url, timeout=60) as response:
        metadata = __import__('json').load(response)
    write(root/'record.json', metadata)
    if metadata['metadata']['license']['id'] != 'cc-by-4.0':
        raise ValueError('License changed: audit before using data')
    source = next(f for f in metadata['files'] if f['key'] == 'GalaxyPPG.zip')
    archive = root/source['key']
    if not archive.exists():
        with urllib.request.urlopen(source['links']['self'], timeout=60) as response, archive.with_suffix('.part').open('wb') as out:
            shutil.copyfileobj(response, out)
        archive.with_suffix('.part').replace(archive)
    validate_archive(root)
    with urllib.request.urlopen('https://creativecommons.org/licenses/by/4.0/legalcode.txt', timeout=60) as response:
        (root/'LICENSE-CC-BY-4.0.txt').write_bytes(response.read())
    return archive


def validate_archive(root):
    root = Path(root)
    metadata = read(root/'record.json')
    source = next(f for f in metadata['files'] if f['key'] == 'GalaxyPPG.zip')
    archive = root/'GalaxyPPG.zip'
    if archive.stat().st_size != source['size']:
        raise ValueError('Archive size mismatch; resume source download')
    h = hashlib.md5()
    with archive.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    if 'md5:'+h.hexdigest() != source['checksum']:
        raise ValueError('Archive checksum mismatch')
    return {'sha256': file_hash(archive), 'md5': h.hexdigest(), 'bytes': archive.stat().st_size,
            'doi': metadata['metadata']['doi'], 'version': metadata['metadata']['version'],
            'license': metadata['metadata']['license']['id']}


def _csv(archive, path, columns):
    with archive.open(path) as stream:
        header = stream.readline().decode('utf-8-sig').strip().split(',')
        indices = [header.index(c) for c in columns]
        values = np.loadtxt(stream, delimiter=',', usecols=indices, ndmin=2)
    return values


def _ordered(values):
    if not np.isfinite(values).all():
        # Keep nonfinite signal values for window-specific exclusion. Bad clock is fatal.
        if not np.isfinite(values[:, 0]).all():
            raise ValueError('Nonfinite timestamp')
    order = np.argsort(values[:, 0], kind='stable')
    values = values[order]
    unique = np.r_[True, np.diff(values[:, 0]) > 0]
    return values[unique], int((~unique).sum())


def _segment(values, start, cutoff, max_gap, tolerance=.1):
    times = values[:, 0]
    left, right = np.searchsorted(times, [start, cutoff], side='left')
    right = np.searchsorted(times, cutoff, side='right')
    part = values[left:right]
    if len(part) < 2 or part[0, 0]-start > tolerance or cutoff-part[-1, 0] > tolerance:
        raise ValueError('boundary_coverage')
    if not np.isfinite(part).all():
        raise ValueError('nonfinite')
    if np.max(np.diff(part[:, 0])) > max_gap:
        raise ValueError('timestamp_gap')
    return part


def reference_hr(ecg):
    """Independent ECG reference; never vendor wrist HR or published labels."""
    t, x = ecg[:, 0], ecg[:, 1]
    rate = 1/np.median(np.diff(t))
    if not 110 <= rate <= 150 or np.std(x) < 1e-6:
        raise ValueError('ecg_rate_or_flat')
    filtered = signal.sosfiltfilt(signal.butter(3, [5, 25], fs=rate, btype='bandpass', output='sos'), x)
    if abs(np.quantile(filtered, .01)) > abs(np.quantile(filtered, .99)):
        filtered = -filtered
    prominence = max(.5*np.std(filtered), .15*(np.quantile(filtered, .99)-np.quantile(filtered, .01)))
    estimates, chosen = [], None
    for multiplier in (1., 1.5):
        peaks, _ = signal.find_peaks(filtered, distance=int(.28*rate), prominence=prominence*multiplier,
                                     height=.4*np.quantile(filtered, .99))
        if len(peaks) < 4:
            raise ValueError('ecg_insufficient_peaks')
        rr = np.diff(t[peaks])
        if np.min(rr) < .28 or np.max(rr) > 1.5 or np.std(rr)/np.mean(rr) > .25:
            raise ValueError('ecg_irregular_peaks')
        estimates.append(60/float(np.mean(rr)))
        chosen = peaks if chosen is None else chosen
    if abs(estimates[0]-estimates[1]) > 5:
        raise ValueError('ecg_detector_disagreement')
    return estimates[0], chosen


def make_request(ppg, acceleration, cutoff, state=None):
    """Closed trailing window, no interpolation support from outside the cutoff."""
    start = cutoff-8
    p = _segment(ppg, start, cutoff, .2)
    a = _segment(acceleration, start, cutoff, .2)
    if np.std(p[:, 1]) < 1e-6:
        raise ValueError('flat_ppg')
    grid = start+np.arange(201)/25
    return {'timestamps_s': (grid-cutoff).tolist(),
            'ppg': np.interp(grid, p[:, 0], p[:, 1]).tolist(),
            'acceleration': np.column_stack([np.interp(grid, a[:, 0], a[:, i]) for i in (1, 2, 3)]).tolist(),
            'sample_rates_hz': {'ppg': 25, 'acceleration': 25}, 'cutoff_s': 0., 'state': state}


def prepare(source_root, output_root, split_manifest=None):
    """Trusted builder operation. output/private is never staged into workers."""
    source_root, out = Path(source_root), Path(output_root)
    provenance = validate_archive(source_root)
    out.mkdir(parents=True, exist_ok=False)
    private = out/'private'
    private.mkdir()
    exclusions, subjects, audits, records = [], [], {}, []
    with zipfile.ZipFile(source_root/'GalaxyPPG.zip') as archive:
        for person in [f'P{i:02}' for i in range(1, 25)]:
            files = {name: f'Dataset/{person}/{name}' for name in
                     ('GalaxyWatch/PPG.csv', 'GalaxyWatch/ACC.csv', 'PolarH10/ECG.csv', 'Event.csv')}
            absent = [p for p in files.values() if p not in archive.namelist()]
            if absent:
                exclusions.append({'person': person, 'reason': 'missing_source_files', 'files': absent})
                continue
            ppg = _csv(archive, files['GalaxyWatch/PPG.csv'], ['timestamp', 'ppg'])
            acc = _csv(archive, files['GalaxyWatch/ACC.csv'], ['timestamp', 'x', 'y', 'z'])
            ecg = _csv(archive, files['PolarH10/ECG.csv'], ['phoneTimestamp', 'ecg'])
            ppg[:, 0] /= 1000
            acc[:, 0] /= 1000
            ecg[:, 0] = (ecg[:, 0]-32400000)/1000
            ppg, pd = _ordered(ppg)
            acc, ad = _ordered(acc)
            ecg, ed = _ordered(ecg)
            with archive.open(files['Event.csv']) as stream:
                events = list(csv.DictReader(__import__('io').TextIOWrapper(stream, encoding='utf-8-sig')))
            event_records, excluded, sample_audit, excluded_records = [], Counter(), [], []
            for index, event in enumerate(events):
                if event['status'].upper() != 'ENTER':
                    continue
                exits = [e for e in events[index+1:] if e['session'] == event['session'] and e['status'].upper() == 'EXIT']
                if not exits:
                    continue
                start, end = float(event['timestamp'])/1000, float(exits[0]['timestamp'])/1000
                session = f'b{index:03}'
                for cutoff in np.arange(start+8, end+1e-6, 2):
                    try:
                        make_request(ppg, acc, cutoff)
                        segment = _segment(ecg, cutoff-8, cutoff, .05)
                        hr, peaks = reference_hr(segment)
                    except ValueError as exc:
                        excluded[str(exc)] += 1
                        excluded_records.append({'session': session, 'cutoff': float(cutoff), 'reason': str(exc)})
                        continue
                    record = {'id': digest([person, session, float(cutoff)])[:24],
                              'person': person, 'session': session, 'activity': event['session'],
                              'cutoff': float(cutoff), 'reference_bpm': hr}
                    event_records.append(record)
                    if len(sample_audit) < 3 or (event['session'] in ('walking', 'jogging', 'running') and not any(s['activity'] == event['session'] for s in sample_audit)):
                        sample_audit.append({'activity': event['session'], 'cutoff': float(cutoff),
                                             'reference_bpm': hr, 'peak_times_s': (segment[peaks, 0]-cutoff).tolist()})
            audits[person] = {'eligible_windows': len(event_records), 'excluded_windows': dict(excluded),
                              'excluded_window_records': excluded_records,
                              'duplicate_timestamps_removed': {'ppg': pd, 'acceleration': ad, 'ecg': ed},
                              'time_ranges_s': {k: [float(v[0, 0]), float(v[-1, 0])] for k, v in [('ppg', ppg), ('acceleration', acc), ('ecg', ecg)]},
                              'median_rates_hz': {k: float(1/np.median(np.diff(v[:, 0]))) for k, v in [('ppg', ppg), ('acceleration', acc), ('ecg', ecg)]},
                              'ecg_inspection_samples': sample_audit,
                              'source_files': {k: hashlib.sha256(archive.read(v)).hexdigest() for k, v in files.items()}}
            if not event_records:
                exclusions.append({'person': person, 'reason': 'no_eligible_windows'})
                continue
            subjects.append(person)
            np.savez_compressed(private/f'{person}.npz', ppg=ppg, acceleration=acc, ecg=ecg)
            records.extend(event_records)
            print(f'{person}: {len(event_records)} eligible windows', flush=True)
    if split_manifest:
        splits = validate_splits(read(split_manifest))
        if set(p for group in splits.values() for p in group) != set(subjects):
            raise ValueError('Split does not match audited cohort')
    else:
        people = sorted(subjects, key=lambda p: digest(['galaxy-pulse-split-v1', p]))
        if len(people) < 11:
            raise ValueError('Insufficient usable subjects for provisional 5+5 held-out split')
        splits = validate_splits({'train': people[:-10], 'development': people[-10:-5], 'test': people[-5:]})
    write(private/'splits.json', splits)
    write(private/'audit.json', {'source': provenance, 'rules': RULES, 'people': audits, 'exclusions': exclusions,
                                 'release': 'retrospective; builder accessed labels; reference review pending'})
    for role, people in splits.items():
        folder = private/role
        folder.mkdir()
        role_records = [r for r in records if r['person'] in people]
        # Each complete recording belongs to one person and one split.
        entries = []
        for person in people:
            opaque = digest(['recording', person])[:16]
            raw = np.load(private/f'{person}.npz', allow_pickle=False)
            np.savez_compressed(folder/f'{opaque}.npz', ppg=raw['ppg'], acceleration=raw['acceleration'])
            entries.append({'person': person, 'signals': f'{opaque}.npz',
                            'sha256': file_hash(folder/f'{opaque}.npz'),
                            'records': [r for r in role_records if r['person'] == person]})
        manifest = {'version': VERSION, 'role': role, 'rules': RULES, 'sources': provenance, 'entries': entries,
                    'reference_audit_status': 'pending'}
        write(folder/'manifest.json', manifest)
        if role != 'test':
            public = out/'public'/role
            shutil.copytree(folder, public)
            if role == 'development':
                for entry in manifest['entries']:
                    for r in entry['records']:
                        r.pop('reference_bpm')
                        r.pop('activity')
                write(public/'manifest.json', manifest)
    result = {'version': VERSION, 'splits': splits, 'source': provenance, 'rules_sha256': digest(RULES),
              'eligible_windows': len(records), 'exclusions': exclusions,
              'private_manifest_paths': {r: f'private/{r}/manifest.json' for r in splits},
              'benchmark_ready': False, 'next_gate': 'Inspect ECG references and clock alignment; freeze reviewed masks'}
    write(out/'dataset.json', result)
    return result


def iter_windows(manifest_path, with_reference=False):
    path = Path(manifest_path)
    manifest = read(path)
    for entry in manifest['entries']:
        signals = path.parent/entry['signals']
        if not signals.resolve().is_relative_to(path.parent.resolve()) or file_hash(signals) != entry['sha256']:
            raise ValueError('Signal path/hash mismatch')
        with np.load(signals, allow_pickle=False) as data:
            ppg, acceleration = data['ppg'], data['acceleration']
            for ref in entry['records']:
                req = make_request(ppg, acceleration, ref['cutoff'])
                if with_reference and 'reference_bpm' not in ref:
                    raise ValueError('Reference labels unavailable in public development data')
                yield req, dict(ref) if with_reference else {k: ref[k] for k in ('id', 'person', 'session')}
