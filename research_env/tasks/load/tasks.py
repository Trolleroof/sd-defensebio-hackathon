"""Requests, tiers, variants and manifests for the load task (controller side)."""
from __future__ import annotations
import hashlib
from pathlib import Path
from research_env.core.common import write

TIERS = ('primary', 'calibration')
VARIANTS = ('person', 'heldout15')
HELDOUT_LOAD = 15.
CALIBRATION_LOAD = 0.
# Exactly the permitted request inputs. Nothing else from a record may enter a request.
REQUEST_FIELDS = ('sex', 'age_years', 'stature_cm', 'body_mass_kg', 'load_kg', 'speed_m_s',
                  'grade_percent', 'calibration')
TRAIN_FIELDS = ('sex', 'age_years', 'stature_cm', 'body_mass_kg', 'load_kg', 'speed_m_s', 'grade_percent')


def pseudonym(salt, person):
    return 'P' + hashlib.sha256(f'{salt}:{person}'.encode()).hexdigest()[:10]


def check_options(tier, variant):
    if tier not in TIERS or variant not in VARIANTS:
        raise ValueError(f'Unknown tier/variant {tier}/{variant}')


def make_request(record, calibration_record=None):
    """Primary request when calibration_record is None. The calibration record must be
    the same person's unloaded condition and is used only in the calibration tier."""
    request = {k: record[k] for k in REQUEST_FIELDS if k != 'calibration'}
    request['calibration'] = None
    if calibration_record is not None:
        if calibration_record['person'] != record['person'] or calibration_record['load_kg'] != CALIBRATION_LOAD:
            raise ValueError('Calibration must be the same person\'s unloaded measurement')
        request['calibration'] = {'load_kg': CALIBRATION_LOAD, 'gross_w': calibration_record['gross_w']}
    return request


def fit_rows(records, people, variant, salt):
    """Public training rows for fit people. Held-out-15 variant drops every 15 kg target."""
    people = set(people)
    rows = []
    for r in records:
        if r['person'] in people and not (variant == 'heldout15' and r['load_kg'] == HELDOUT_LOAD):
            rows.append({'person': pseudonym(salt, r['person']), **{k: r[k] for k in TRAIN_FIELDS},
                         'gross_w': r['gross_w'], 'quality_flags': r['quality_flags']})
    return rows


def scored_records(records, people, tier, variant):
    """Eligible scored conditions, frozen before comparison.

    calibration tier: the unloaded condition is revealed and therefore not scored.
    heldout15 variant: only the 15 kg condition is scored.
    """
    check_options(tier, variant)
    people = set(people)
    unloaded = {r['person']: r for r in records if r['load_kg'] == CALIBRATION_LOAD}
    out = []
    for r in records:
        if r['person'] not in people:
            continue
        if variant == 'heldout15' and r['load_kg'] != HELDOUT_LOAD:
            continue
        if tier == 'calibration':
            if r['load_kg'] == CALIBRATION_LOAD or r['person'] not in unloaded:
                continue
            request = make_request(r, unloaded[r['person']])
        else:
            request = make_request(r)
        out.append((r, request))
    return out


def write_manifest(path, records, people, *, tier, variant, role, salt):
    entries = []
    for r, request in scored_records(records, people, tier, variant):
        entries.append({'id': 'q' + hashlib.sha256(f'{salt}:{r["id"]}'.encode()).hexdigest()[:12],
                        'person': pseudonym(salt, r['person']), 'source_person': r['person'],
                        'load_kg': r['load_kg'], 'sex': r['sex'], 'body_mass_kg': r['body_mass_kg'],
                        'reference_w': r['gross_w'], 'request': request})
    entries.sort(key=lambda e: (e['person'], e['load_kg']))
    if not entries:
        raise ValueError('No eligible scored conditions for these people')
    manifest = {'schema': 'load-scoring-manifest-v1', 'role': role, 'tier': tier, 'variant': variant,
                'units': 'W', 'target': 'gross metabolic power', 'people': sorted({e['person'] for e in entries}),
                'records': entries}
    write(path, manifest)
    return manifest


def public_requests(manifest):
    """Development inputs without labels or source identities."""
    return [{'request_id': e['id'], 'person': e['person'], 'request': e['request']} for e in manifest['records']]
