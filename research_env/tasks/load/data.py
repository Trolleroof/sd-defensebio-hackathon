"""EVOBREATH 2015 importer (Mendeley rhmdzmx6kk v1, CC BY 4.0).

Controller-side only. Reads the original BIFF .xls with xlrd (pinned in
``requirements-import.txt``); workers never need xlrd. Every numeric cell is parsed
explicitly from its stored type, and the original cell type/value is kept.
"""
from __future__ import annotations
import hashlib
import re
import shutil
from pathlib import Path
from research_env.core.common import digest, file_hash, read, write

DATASET = {
    'name': 'EVOBREATH 2015: energetic expenditure during walking and load-carriage in humans',
    'repository': 'Mendeley Data', 'record': 'rhmdzmx6kk', 'version': '1',
    'doi': '10.17632/rhmdzmx6kk.1', 'url': 'https://data.mendeley.com/datasets/rhmdzmx6kk/1',
    'license': 'CC BY 4.0', 'license_url': 'https://creativecommons.org/licenses/by/4.0/',
    'attribution': 'Ana Mateos, CENIEH (Burgos, Spain). Dataset from EVOBREATH 2015.',
}
# SHA-256 values published by the Mendeley files API (content_details.sha256_hash).
FILES = {
    'Mateos2020_Dataset_WW2015.xls': {
        'sha256': '51b42232fd63e15f2566f1d1053c92fe2de88477c3c4f14ce95fdc75784c58da', 'size': 38400,
        'url': 'https://data.mendeley.com/public-files/datasets/rhmdzmx6kk/files/79263e68-64fc-4e6a-be33-62504ddb1642/file_downloaded'},
    'Mateos2020_Dataset_WW2015.pdf': {
        'sha256': '5410a21e0ab3bdbd96b7dc564dd8e4f1a7e948d158a70f9f5c673b3e7cac57f5', 'size': 42528,
        'url': 'https://data.mendeley.com/public-files/datasets/rhmdzmx6kk/files/385b82a7-9dbc-4e97-9ef4-7adbc55f80a6/file_downloaded'},
}
WORKBOOK = 'Mateos2020_Dataset_WW2015.xls'
COLUMNS = ('ID', 'Sex', 'Age', 'Height', 'Weight', 'FL', 'BIL', 'FM', 'FFM', 'RMR',
           'EE_WW00', 'EE_WW05', 'EE_WW10', 'EE_WW15')
LOAD_COLUMNS = {0.: 'EE_WW00', 5.: 'EE_WW05', 10.: 'EE_WW10', 15.: 'EE_WW15'}
LOADS = tuple(LOAD_COLUMNS)
SPEED_KM_H = 4.0
SPEED_M_S = SPEED_KM_H/3.6          # 4 km/h = 4000 m / 3600 s
GRADE_PERCENT = 0.0                 # treadmill with no inclination
BOUT_MIN = 11
SPLIT_SALT = 'evobreath-load-split-v1'
SPLIT_SIZES = {'train': 28, 'development': 10, 'test': 10}
_NUMBER = re.compile(r'-?\d+(?:\.\d+)?')
_MISSING = {'-'}


def kcal_per_day_to_w(kcal_per_day):
    """Rate conversion: 1 kcal = 4184 J, 1 day = 86400 s."""
    return kcal_per_day*4184/86400


def fetch(dest):
    """Download originals only if absent; verify published SHA-256. Never overwrite."""
    import urllib.request
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for name, meta in FILES.items():
        path = dest/name
        if not path.exists():
            tmp = path.with_suffix(path.suffix+'.part')
            with urllib.request.urlopen(meta['url'], timeout=60) as response, tmp.open('wb') as out:
                shutil.copyfileobj(response, out)
            if file_hash(tmp) != meta['sha256']:
                tmp.unlink()
                raise ValueError(f'Checksum mismatch for downloaded {name}')
            tmp.replace(path)
    return verify_originals(dest)


def verify_originals(root):
    root = Path(root)
    hashes = {}
    for name, meta in FILES.items():
        path = root/name
        if not path.exists():
            raise FileNotFoundError(f'Missing original {name}; run fetch() or copy verified originals')
        h = file_hash(path)
        if h != meta['sha256'] or path.stat().st_size != meta['size']:
            raise ValueError(f'Original {name} does not match published checksum')
        hashes[name] = h
    return hashes


def _cell_name(row, col):
    return f'{chr(ord("A")+col)}{row+1}'


def parse_number(ctype, value, field):
    """Explicit typed parsing. Returns (float|None, reason|None). Raises on unknown text."""
    # xlrd ctypes: 1 text, 2 number.
    if ctype == 2:
        return float(value), None
    if ctype == 1:
        text = value.strip()
        if _NUMBER.fullmatch(text):
            return float(text), None
        if text in _MISSING:
            return None, 'source_missing_dash'
    raise ValueError(f'Unparseable {field}: ctype={ctype} value={value!r}')


def read_workbook(path):
    import xlrd
    book = xlrd.open_workbook(str(path), on_demand=True)
    if set(book.sheet_names()) != {'DATA', 'Variables'}:
        raise ValueError(f'Unexpected sheets {book.sheet_names()}')
    sheet = book.sheet_by_name('DATA')
    if tuple(sheet.row_values(0)) != COLUMNS:
        raise ValueError('Unexpected DATA header')
    variables = book.sheet_by_name('Variables')
    units = {variables.cell_value(r, 0): variables.cell_value(r, 2) for r in range(1, variables.nrows)}
    for column in LOAD_COLUMNS.values():
        if units.get(column) != 'kcal/day':
            raise ValueError(f'{column} unit is not kcal/day')
    people = []
    for r in range(1, sheet.nrows):
        original, parsed, missing = {}, {}, {}
        for c, field in enumerate(COLUMNS):
            ctype, value = sheet.cell_type(r, c), sheet.cell_value(r, c)
            original[field] = {'cell': _cell_name(r, c), 'ctype': ctype, 'value': value}
            if field in ('ID', 'Sex'):
                if ctype != 1 or not value.strip():
                    raise ValueError(f'Bad {field} at {_cell_name(r, c)}')
                parsed[field] = value.strip()
            else:
                parsed[field], reason = parse_number(ctype, value, f'{field}@{_cell_name(r, c)}')
                if reason:
                    missing[field] = reason
        if parsed['Sex'] not in ('M', 'F'):
            raise ValueError(f'Bad sex at row {r+1}')
        required = set(COLUMNS)-set(LOAD_COLUMNS.values())-{'ID', 'Sex'}
        if any(parsed[f] is None for f in required):
            raise ValueError(f'Missing non-target field at row {r+1}')
        people.append({'row': r+1, 'parsed': parsed, 'original': original, 'missing': missing})
    ids = [p['parsed']['ID'] for p in people]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate person IDs')
    return people, units


def build_records(people, source_sha256):
    """Condition-level rows. Gross target only; net is a separate unscored field."""
    records, exclusions = [], []
    for p in people:
        v, person = p['parsed'], p['parsed']['ID']
        rmr_w = kcal_per_day_to_w(v['RMR'])
        gross = {load: v[col] for load, col in LOAD_COLUMNS.items()}
        present = [l for l in LOADS if gross[l] is not None]
        nonmonotone = any(gross[b] <= gross[a] for a, b in zip(present, present[1:]))
        for load, column in LOAD_COLUMNS.items():
            cell = p['original'][column]
            if gross[load] is None:
                exclusions.append({'person': person, 'load_kg': load, 'cell': cell['cell'],
                                   'original': cell['value'], 'reason': p['missing'][column]})
                continue
            w = kcal_per_day_to_w(gross[load])
            flags = []
            if w <= rmr_w*1.5:
                flags.append('gross_walking_le_1.5x_measured_rmr')
            if nonmonotone:
                flags.append('person_condition_costs_not_strictly_increasing')
            records.append({
                'id': f'{person}-L{int(load):02d}', 'person': person, 'source_row': p['row'],
                'source_cell': cell['cell'], 'source_sha256': source_sha256,
                'source_kcal_per_day': gross[load], 'source_original': cell,
                'sex': v['Sex'], 'age_years': v['Age'], 'stature_cm': v['Height'],
                'body_mass_kg': v['Weight'], 'load_kg': load, 'speed_m_s': SPEED_M_S,
                'grade_percent': GRADE_PERCENT, 'bout_min': BOUT_MIN,
                'gross_w': w, 'target_convention': 'gross',
                'net_w_secondary': w-rmr_w,  # separate track; never the primary target
                # Laboratory measurements: never in primary or calibration requests.
                'excluded_measurements': {'rmr_w': rmr_w, 'femur_length_cm': v['FL'],
                                          'biiliac_breadth_cm': v['BIL'], 'fat_mass_kg': v['FM'],
                                          'fat_free_mass_kg': v['FFM']},
                'quality_flags': flags})
    return records, exclusions


def make_splits(person_ids, sex_by_person=None, sizes=None):
    """Deterministic, sex-stratified hash order; frozen before any model comparison."""
    sizes = sizes or SPLIT_SIZES
    total = sum(sizes.values())
    if len(person_ids) != total:
        raise ValueError(f'Expected {total} people, got {len(person_ids)}')
    key = lambda p: hashlib.sha256(f'{SPLIT_SALT}:{p}'.encode()).hexdigest()
    sex_by_person = sex_by_person or {p: '' for p in person_ids}
    groups = {}
    for p in sorted(person_ids, key=key):
        groups.setdefault(sex_by_person[p], []).append(p)
    # Interleave strata so each prefix is approximately sex balanced.
    order, cursor = [], {g: 0 for g in groups}
    while len(order) < total:
        for g in sorted(groups, key=lambda g: (cursor[g]/len(groups[g]), g)):
            if cursor[g] < len(groups[g]):
                order.append(groups[g][cursor[g]])
                cursor[g] += 1
                break
    splits, start = {}, 0
    for name in ('test', 'development', 'train'):
        splits[name] = sorted(order[start:start+sizes[name]])
        start += sizes[name]
    return splits


def prepare(source_root, output_root, split_manifest=None):
    source_root, out = Path(source_root), Path(output_root)
    hashes = verify_originals(source_root)
    people, units = read_workbook(source_root/WORKBOOK)
    records, exclusions = build_records(people, hashes[WORKBOOK])
    ids = [p['parsed']['ID'] for p in people]
    sex = {p['parsed']['ID']: p['parsed']['Sex'] for p in people}
    splits = read(split_manifest) if split_manifest else make_splits(ids, sex)
    if sorted(p for g in splits.values() for p in g) != sorted(ids):
        raise ValueError('Split must cover exactly the 48 source people once')
    (out/'private').mkdir(parents=True, exist_ok=True)
    originals = out/'private'/'originals'
    originals.mkdir(exist_ok=True)
    for name in FILES:
        if not (originals/name).exists():
            shutil.copyfile(source_root/name, originals/name)
    write(out/'private'/'records.json', {'schema': 'evobreath-load-records-v1', 'records': records})
    write(out/'private'/'splits.json', splits)
    by_person = {}
    for r in records:
        by_person.setdefault(r['person'], []).append(r['load_kg'])
    audit = {
        'source_people': len(ids), 'people_with_any_condition': len(by_person),
        'condition_observations': len(records), 'possible_condition_observations': len(ids)*len(LOADS),
        'independent_people_note': 'Condition rows are repeated measures within 48 people, not independent cases',
        'excluded_conditions': exclusions, 'excluded_people': sorted(set(ids)-set(by_person)),
        'people_missing_any_load': {p: sorted(set(LOADS)-set(l)) for p, l in by_person.items() if len(l) < len(LOADS)},
        'quality_flag_counts': {f: sum(f in r['quality_flags'] for r in records)
                                for f in sorted({f for r in records for f in r['quality_flags']})},
        'string_stored_numeric_cells': sum(1 for p in people for f, c in p['original'].items()
                                           if f not in ('ID', 'Sex') and c['ctype'] == 1),
        'units': units, 'sex_counts': {s: list(sex.values()).count(s) for s in ('F', 'M')},
        'split_counts': {k: len(v) for k, v in splits.items()},
        'split_sex_counts': {k: {s: sum(sex[p] == s for p in v) for s in ('F', 'M')} for k, v in splits.items()},
        'conversion': 'watts = kcal_per_day * 4184 / 86400; speed = 4/3.6 m/s; grade = 0 %',
        'target': 'gross metabolic power (EE_WWxx); net = gross - RMR kept as a separate unscored field',
    }
    write(out/'private'/'audit.json', audit)
    manifest = {'schema': 'load-dataset-manifest-v1', 'dataset': DATASET, 'source_sha256': hashes,
                'records_sha256': file_hash(out/'private'/'records.json'),
                'splits_sha256': file_hash(out/'private'/'splits.json'),
                'split_salt': SPLIT_SALT, 'audit': {k: audit[k] for k in
                ('source_people', 'condition_observations', 'excluded_people', 'split_counts')}}
    write(out/'dataset_manifest.json', manifest)
    return manifest


def load_records(prepared_root):
    return read(Path(prepared_root)/'private'/'records.json')['records']


def load_splits(prepared_root):
    return read(Path(prepared_root)/'private'/'splits.json')
