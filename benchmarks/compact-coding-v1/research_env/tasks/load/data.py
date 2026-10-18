"""EVOBREATH measured gross metabolic rate, no unloaded-reference calibration."""
from pathlib import Path
from research_env.core.common import file_hash, finite
from research_env.tasks.tabular import export_dataset

SOURCE = 'https://doi.org/10.17632/rhmdzmx6kk.1'
SHA256 = '51b42232fd63e15f2566f1d1053c92fe2de88477c3c4f14ce95fdc75784c58da'


def watts(kcal_day):
    return kcal_day * 4184 / 86400


def numeric(v):
    if isinstance(v, bool):
        raise ValueError('Boolean numeric cell')
    result = float(v)
    if not finite(result):
        raise ValueError('Nonfinite numeric cell')
    return result


def prepare(source_root, output_root, split_manifest=None):
    import xlrd
    path = Path(source_root)/'Mateos2020_Dataset_WW2015.xls'
    if file_hash(path) != SHA256:
        raise ValueError('Original Mendeley workbook checksum mismatch')
    sheet = xlrd.open_workbook(str(path)).sheet_by_name('DATA')
    expected = ['ID','Sex','Age','Height','Weight','FL','BIL','FM','FFM','RMR',
                'EE_WW00','EE_WW05','EE_WW10','EE_WW15']
    if sheet.row_values(0) != expected:
        raise ValueError('Workbook schema drift')
    records, people, lineage, exclusions = [], [], [], []
    for row in range(1, sheet.nrows):
        raw = sheet.row_values(row)
        person = raw[0]
        if not isinstance(person, str) or not person or person in people:
            raise ValueError('Invalid/duplicate person')
        age, height, mass = map(numeric, raw[2:5])
        if not (18 <= age <= 100 and 100 <= height <= 230 and 30 <= mass <= 250):
            raise ValueError('Invalid anthropometry')
        people.append(person)
        for col, load in enumerate((0, 5, 10, 15), 10):
            if raw[col] in ('-', ''):
                exclusions.append({'person': person, 'load_kg': load, 'row': row+1,
                                   'column': expected[col], 'original': raw[col], 'reason': 'source target missing'})
                continue
            y = numeric(raw[col])
            if not 0 < y < 30000:
                raise ValueError('Invalid metabolic rate')
            records.append({'id': f'{person}-{load}', 'person': person, 'session': 'walking',
                            'request': {'age_years': age, 'height_cm': height, 'mass_kg': mass,
                                        'load_kg': load, 'speed_m_s': 4/3.6, 'grade': 0, 'state': None},
                            'targets': [{'slice': load, 'value': watts(y)}]})
            lineage.append({'person': person, 'load_kg': load, 'row': row+1, 'column': expected[col],
                            'original_kcal_day': raw[col], 'reference_w': watts(y)})
        lineage.append({'person': person, 'row': row+1,
                        'anthropometry_original': dict(zip(expected[2:5], raw[2:5]))})
    if len(people) != 48:
        raise ValueError('Expected all 48 original people')
    sources = {'url': SOURCE, 'sha256': SHA256, 'license': 'CC BY 4.0', 'filename': path.name}
    return export_dataset(output_root, 'load', records, people, sources,
                          {'sources': sources, 'lineage': lineage, 'exclusions': exclusions,
                           'conversion': 'watts = kcal/day * 4184 / 86400; gross, no RMR subtraction',
                           'excluded_inputs': ['ID','sex','RMR','FL','BIL','FM','FFM','other condition targets'],
                           'speed_m_s': 4/3.6, 'grade': 0}, split_manifest)
