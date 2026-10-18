"""CPU-only baselines. JSON parameters avoid unsafe deserialization in controller."""
from pathlib import Path
import time
import numpy as np
from scipy import signal
from research_env.core.common import digest, file_hash, read, write

KINDS = ('median', 'spectral', 'motion', 'ridge')


def spectral_features(request, motion_weight=0.):
    ppg = np.asarray(request['ppg'], dtype=float)
    acc = np.asarray(request['acceleration'], dtype=float)
    rate = request['sample_rates_hz']['ppg']
    # Filtering is confined to the already available trailing window; no future data.
    ppg = signal.detrend(ppg)
    freq, power = signal.periodogram(ppg, fs=rate, window='hann', nfft=2048)
    valid = (freq >= .667) & (freq <= 3.5)
    power = power[valid]
    freq = freq[valid]
    power /= max(float(power.max()), 1e-12)
    accel_power = np.zeros_like(power)
    for axis in acc.T:
        _, ap = signal.periodogram(signal.detrend(axis), fs=rate, window='hann', nfft=2048)
        accel_power += ap[valid]
    accel_power /= max(float(accel_power.max()), 1e-12)
    penalized = power-motion_weight*accel_power
    order = np.argsort(penalized)[-3:][::-1]
    peaks = freq[order]*60
    return [*peaks, float(np.std(ppg)), float(np.std(acc, axis=0).mean()),
            float(power.max()), float(np.dot(power, accel_power)/max(power.sum(), 1e-12))]


def predict(artifact_dir, request):
    model = read(Path(artifact_dir)/'model.json')
    kind = model['kind']
    state = request.get('state')
    previous = None if state is None else state.get('previous_bpm')
    if kind == 'median':
        mean = model['median']
    else:
        f = spectral_features(request, model.get('motion_weight', 0.))
        mean = f[0]
        if kind == 'ridge':
            x = (np.asarray(f)-np.asarray(model['center']))/np.asarray(model['scale'])
            mean = float(np.dot(x, model['coef'])+model['intercept'])
        elif kind == 'motion' and previous is not None:
            # A causal continuity prior; state resets for every person/session.
            mean = min(f[:3], key=lambda v: abs(v-previous))
            mean = .6*mean+.4*previous
    mean = float(np.clip(mean, 35, 220))
    width = model['half_width']
    return {'mean_bpm': mean, 'lower_bpm': mean-width, 'upper_bpm': mean+width,
            'state': {'previous_bpm': mean} if kind == 'motion' else None}


def train(train_manifest, calibration_manifest, output_dir, seed, kind='median', alpha=10., motion_weight=.5):
    if kind not in KINDS:
        raise ValueError('Unknown baseline')
    from research_env.tasks.pulse.data import iter_windows
    train_data = read(train_manifest)
    calibration_data = read(calibration_manifest or train_manifest)
    train_people = {e['person'] for e in train_data['entries']}
    calibration_people = {e['person'] for e in calibration_data['entries']}
    if train_data['role'] != 'train' or calibration_data['role'] != 'train' or not calibration_people <= train_people:
        raise ValueError('Calibration must use permitted training people only')
    records = list(iter_windows(train_manifest, with_reference=True))
    if not records:
        raise ValueError('No training windows')
    ys = np.array([ref['reference_bpm'] for _, ref in records])
    model = {'kind': kind, 'seed': seed, 'median': float(np.median(ys)),
             'half_width': float(np.quantile(np.abs(ys-np.median(ys)), .9)),
             'motion_weight': motion_weight if kind == 'motion' else 0., 'alpha': alpha}
    if kind == 'ridge':
        from sklearn.linear_model import Ridge
        x = np.array([spectral_features(req) for req, _ in records])
        center, scale = x.mean(axis=0), x.std(axis=0)
        scale[scale < 1e-12] = 1.
        fitted = Ridge(alpha=alpha).fit((x-center)/scale, ys)
        model.update(center=center.tolist(), scale=scale.tolist(), coef=fitted.coef_.tolist(),
                     intercept=float(fitted.intercept_))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    write(out/'model.json', model)
    # Only explicitly permitted TRAIN calibration data can set intervals.
    calibration = list(iter_windows(calibration_manifest or train_manifest, with_reference=True))
    residuals = []
    previous_session, state = None, None
    for req, ref in calibration:
        key = (ref['person'], ref['session'])
        if key != previous_session:
            state = None
        req['state'] = state
        pred = predict(out, req)
        state, previous_session = pred['state'], key
        residuals.append(abs(pred['mean_bpm']-ref['reference_bpm']))
    model['half_width'] = float(np.quantile(residuals, .9))
    write(out/'model.json', model)
    source = Path(__file__).read_text().split('\ndef train(')[0]
    source = source.replace('from research_env.core.common import digest, file_hash, read, write',
                            'import json\ndef read(path):\n    return json.loads(Path(path).read_text())')
    (out/'inference.py').write_text(source)
    (out/'training_source.py').write_text(Path(__file__).read_text())
    from shutil import copyfile
    copyfile(Path(__file__).parents[2]/'requirements.lock', out/'requirements.lock')
    manifest = {'interface': 'physiology-research-v1', 'candidate_id': digest(model), 'seed': seed,
                'model_sha256': file_hash(out/'model.json'), 'kind': kind,
                'training_manifest_sha256': file_hash(train_manifest),
                'calibration_manifest_sha256': file_hash(calibration_manifest or train_manifest),
                'interval_note': 'Training residual quantile; coverage is empirical, not guaranteed',
                'source_sha256': file_hash(__file__),
                'inference_sha256': file_hash(out/'inference.py'),
                'dependency_lock_sha256': file_hash(out/'requirements.lock')}
    write(out/'artifact.json', manifest)
    return manifest


def replay(artifact_dir, manifest):
    from research_env.tasks.pulse.data import iter_windows
    from research_env.tasks.pulse.scoring import score
    artifact = read(Path(artifact_dir)/'artifact.json')
    if file_hash(Path(artifact_dir)/'model.json') != artifact['model_sha256']:
        raise ValueError('Artifact hash mismatch')
    references, predictions, latency = [], {}, []
    previous_session, state = None, None
    for req, ref in iter_windows(manifest, with_reference=True):
        key = (ref['person'], ref['session'])
        if key != previous_session:
            state = None
        req['state'] = state
        start = time.perf_counter()
        pred = predict(artifact_dir, req)
        latency.append(time.perf_counter()-start)
        predictions[ref['id']] = pred
        references.append(ref)
        previous_session, state = key, pred['state']
    result = score(references, predictions)
    if latency:
        result['resource_use'] = {'streaming_latency_p50_s': float(np.median(latency)),
                                  'streaming_latency_p95_s': float(np.quantile(latency, .95))}
    return result, predictions
