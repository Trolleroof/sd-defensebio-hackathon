"""Activity-conditioned synthetic recordings fitted to processed PPG-DaLiA.

This is a statistical feature simulator, not a waveform or mechanistic body model.
The fit artifact is private training infrastructure, never participant evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import NormalDist

import numpy as np

VERSION = 'dalia-feature-copula-var-v1'
REFERENCE_SUBJECTS = ('S1', 'S4', 'S8', 'S12')
CHANNELS = ('hr_ecg', 'acc_wrist_enmo', 'acc_wrist_std', 'acc_chest_enmo',
            'acc_chest_std', 'eda_wrist', 'temp_wrist', 'resp_chest_std')
KEYS = {'t', 'activity', *CHANNELS}
ACTIVITY_NAMES = {0: 'transient', 1: 'sitting', 2: 'stairs', 3: 'table_soccer',
                  4: 'cycling', 5: 'driving', 6: 'lunch', 7: 'walking', 8: 'working'}
UNITS = {k: ('bpm' if k == 'hr_ecg' else 'uS' if k == 'eda_wrist' else
             'degC' if k == 'temp_wrist' else 'a.u.' if k == 'resp_chest_std' else 'g')
         for k in CHANNELS}
SKIN_TEMPERATURE = {
    'channel': 'temp_wrist', 'quantity': 'wrist_skin_temperature', 'unit': 'degC',
    'reference_source': 'PPG-DaLiA Empatica E4 wrist temperature',
    'synthetic_source': 'Joint feature generator fitted to permitted reference recordings',
    'interpretation': 'Synthetic wrist skin temperature; not a core-temperature estimate.',
}


def load_recording(path):
    with np.load(path, allow_pickle=False) as z:
        if set(z.files) != KEYS:
            raise ValueError('Expected the exact PPG-DaLiA processed channel contract')
        r = {k: z[k] for k in z.files}
    n = len(r['t'])
    if n < 20 or any(v.shape != (n,) for v in r.values()):
        raise ValueError('Channels must be aligned, nonempty one-dimensional arrays')
    if not np.isfinite(r['t']).all() or not np.allclose(np.diff(r['t']), 2):
        raise ValueError('Expected the adapter 2-second time grid')
    if not np.isfinite(r['activity']).all() or not np.isin(r['activity'], range(9)).all():
        raise ValueError('Expected activity IDs 0..8')
    if any(np.isinf(r[k]).any() or not np.isfinite(r[k]).any() for k in CHANNELS):
        raise ValueError('Each channel needs finite observations; infinity is invalid')
    return r


def runs(activity):
    edges = np.r_[0, np.flatnonzero(np.diff(activity) != 0) + 1, len(activity)]
    return [(int(activity[a]), int(b-a)) for a, b in zip(edges[:-1], edges[1:])]


def _matrix(r):
    return np.column_stack([r[k] for k in CHANNELS])


def _encode(x, quantiles, gaussian):
    return np.column_stack([np.interp(np.where(np.isfinite(x[:, j]), x[:, j], quantiles[j, len(gaussian)//2]), quantiles[j], gaussian)
                            for j in range(len(CHANNELS))])


def _fit_pairs(x, y):
    xm, ym = x.mean(0), y.mean(0)
    xc, yc = x-xm, y-ym
    # Fixed small ridge and spectral cap regularize poorly observed dynamics.
    A = np.linalg.solve(xc.T@xc + 1e-3*np.eye(x.shape[1]), xc.T@yc).T
    radius = float(np.max(np.abs(np.linalg.eigvals(A))))
    if radius > .995:
        A *= .995/radius
    center = (xm+ym)/2
    b = center-A@center
    errors = y-(x@A.T+b)
    cov = errors.T@errors/max(len(errors)-1, 1) + 1e-8*np.eye(x.shape[1])
    return {'A': A.tolist(), 'b': b.tolist(), 'cov': cov.tolist(),
            'initial': ym.tolist(), 'transitions_fitted': int(len(y))}


def _var(z, activity):
    models = {}
    for a in np.unique(activity):
        selected = activity[1:] == a
        x, y = z[:-1][selected], z[1:][selected]
        if len(y) >= len(CHANNELS) + 2:
            models[str(int(a))] = _fit_pairs(x, y)
    return models


def fit(processed_dir, subjects):
    if not subjects or len(set(subjects)) != len(subjects) or not set(subjects) <= set(REFERENCE_SUBJECTS):
        raise ValueError('Fit only explicitly selected reference subjects S1/S4/S8/S12; development/test subjects are forbidden')
    root = Path(processed_dir)
    recordings = {s: load_recording(root/f'{s}.npz') for s in subjects}
    joined = np.concatenate([_matrix(r) for r in recordings.values()])
    levels = np.linspace(.001, .999, 257)
    quantiles = np.nanquantile(joined, levels, axis=0).T
    gaussian = np.array([NormalDist().inv_cdf(float(p)) for p in levels])
    encoded = {s: _encode(_matrix(r), quantiles, gaussian) for s, r in recordings.items()}
    # No artificial adjacency across recording boundaries in fitting.
    pooled = {}
    for a in range(9):
        pairs = [(z[:-1][r['activity'][1:] == a], z[1:][r['activity'][1:] == a])
                 for s, r in recordings.items() for z in [encoded[s]]]
        x, y = np.concatenate([p[0] for p in pairs]), np.concatenate([p[1] for p in pairs])
        if len(x) >= len(CHANNELS) + 2:
            pooled[str(a)] = _fit_pairs(x, y)
    lengths = {str(a): [] for a in range(9)}
    transitions = np.zeros((9, 9), dtype=int)
    starts, donors = [], []
    for s, r in recordings.items():
        blocks = runs(r['activity']); starts.append(blocks[0][0])
        for a, n in blocks:
            lengths[str(a)].append(n)
        for (a, _), (b, _) in zip(blocks[:-1], blocks[1:]):
            transitions[a, b] += 1
        donor = _var(encoded[s], r['activity'])
        donors.append({'models': {a: donor.get(a, m) for a, m in pooled.items()},
                       'sample_count': len(r['t']), 'activity_order': [a for a, _ in blocks]})
    if any(v and a not in pooled for a, v in lengths.items()):
        raise ValueError('Insufficient adjacent observations to fit each observed activity')
    missing = {str(a): np.mean(~np.isfinite(joined[np.concatenate([
        r['activity'] == a for r in recordings.values()])]), axis=0).tolist()
        for a in range(9) if lengths[str(a)]}
    return {'version': VERSION, 'origin': 'synthetic-fit-from-public-recordings',
            'skin_temperature': dict(SKIN_TEMPERATURE),
            'source_dataset': 'PPG-DaLiA', 'source_url': 'https://doi.org/10.24432/C53890',
            'source_license': 'CC BY 4.0', 'source_authors': 'Reiss, Indlekofer, Schmidt (2019)',
            'generator_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'numpy_version': np.__version__, 'rng': 'numpy PCG64',
            'fit_subjects': list(subjects), 'fit_files': {s: hashlib.sha256((root/f'{s}.npz').read_bytes()).hexdigest() for s in subjects},
            'reserved_subjects_read': False, 'step_s': 2, 'time_start_s': 8,
            'channels': list(CHANNELS), 'quantile_levels': levels.tolist(),
            'quantiles': quantiles.tolist(), 'gaussian': gaussian.tolist(),
            'durations_samples': lengths, 'transition_counts': transitions.tolist(),
            'start_activities': starts, 'missing_fraction': missing, 'donors': donors,
            'limitations': ['Processed features only; no raw PPG/ECG waveforms.',
                'Gaussian innovations, activity-mean anchoring, and regularized VAR(1) are modeling assumptions.',
                'Marginals limited to fitted 0.1–99.9 percentile range.',
                'Reference activity orders resampled with fitted activity durations; novel protocols are unvalidated.',
                'New random realizations are not independently measured people or repeated real sessions.',
                'No clinical, causal, physiological realism, privacy, or downstream-transfer validation.']}


def generate(model, seed):
    if model.get('version') != VERSION:
        raise ValueError('Unsupported fitted generator version')
    rng = np.random.default_rng(seed)
    donor = model['donors'][int(rng.integers(len(model['donors'])))]
    # Preserve a recorded protocol order; a free Markov walk can interleave all
    # transfer activities with familiar ones and destroy the runner's split.
    activities = np.concatenate([np.full(int(rng.choice(model['durations_samples'][str(a)])), a, dtype=float)
                                 for a in donor['activity_order']])
    n = len(activities)
    matrices = {k: (np.array(v['A']), np.array(v['b']), np.linalg.cholesky(np.array(v['cov'])))
                for k, v in donor['models'].items()}
    state = np.array(donor['models'][str(int(activities[0]))]['initial'])
    z = np.empty((n, len(CHANNELS)))
    capped = 0
    gaussian = np.array(model['gaussian'])
    for i, a in enumerate(activities):
        A, b, L = matrices[str(int(a))]
        state = A@state+b+L@rng.normal(size=len(CHANNELS))
        capped += int(np.sum((state < gaussian[0]) | (state > gaussian[-1])))
        state = np.clip(state, gaussian[0], gaussian[-1]); z[i] = state
    quantiles = np.array(model['quantiles'])
    data = {k: np.interp(z[:, j], gaussian, quantiles[j]) for j, k in enumerate(CHANNELS)}
    for a in np.unique(activities):
        idx = np.flatnonzero(activities == a)
        for j, k in enumerate(CHANNELS):
            data[k][idx[rng.random(len(idx)) < model['missing_fraction'][str(int(a))][j]]] = np.nan
    data.update(t=8+2*np.arange(n, dtype=float), activity=activities)
    return data, {'origin': 'synthetic', 'generator_version': VERSION, 'seed': int(seed),
                  'skin_temperature': dict(SKIN_TEMPERATURE),
                  'n_samples': n, 'duration_s': float(data['t'][-1]),
                  'hr_ecg_provenance': 'Synthetic ECG-reference-shaped HR; no ECG measured or generated.',
                  'age': -1, 'gender': '', 'height_cm': None, 'weight_kg': None, 'sport_level': -1,
                  'demographic_provenance': 'Unavailable; demographics are not simulated.',
                  'latent_cap_fraction': capped/(n*len(CHANNELS))}


def summarize(recordings):
    x = np.concatenate([_matrix(r) for r in recordings])
    channels = {}
    for j, k in enumerate(CHANNELS):
        p, q = np.concatenate([r[k][:-1] for r in recordings]), np.concatenate([r[k][1:] for r in recordings])
        valid = np.isfinite(p) & np.isfinite(q)
        lag = float(np.corrcoef(p[valid], q[valid])[0, 1]) if np.std(p[valid]) > 0 and np.std(q[valid]) > 0 else None
        channels[k] = {'mean': float(np.nanmean(x[:, j])), 'std': float(np.nanstd(x[:, j])),
                       'q05_q50_q95': np.nanquantile(x[:, j], [.05, .5, .95]).tolist(), 'lag1': lag,
                       'missing_fraction': float(np.mean(~np.isfinite(x[:, j])))}
    activities = np.concatenate([r['activity'] for r in recordings])
    per_activity = {str(a): {'fraction': float(np.mean(activities == a)),
                     'hr_mean': float(np.nanmean(x[activities == a, 0])),
                     'hr_std': float(np.nanstd(x[activities == a, 0])),
                     'duration_samples': [n for r in recordings for act, n in runs(r['activity']) if act == a]}
                    for a in np.unique(activities).astype(int)}
    # Pairwise complete observations; preserve missing channels as missing.
    corr = np.eye(len(CHANNELS))
    for j in range(len(CHANNELS)):
        for k in range(j):
            valid = np.isfinite(x[:, j]) & np.isfinite(x[:, k])
            corr[j, k] = corr[k, j] = np.corrcoef(x[valid, j], x[valid, k])[0, 1] if np.std(x[valid, j]) > 0 and np.std(x[valid, k]) > 0 else 0
    transitions = np.zeros((9, 9), dtype=int)
    for r in recordings:
        blocks = runs(r['activity'])
        for (a, _), (b, _) in zip(blocks[:-1], blocks[1:]):
            transitions[a, b] += 1
    return {'recordings': len(recordings), 'samples': len(x), 'channels': channels,
            'transition_counts': transitions.tolist(),
            'activities': per_activity, 'channel_correlation': corr.tolist()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    f = sub.add_parser('fit'); f.add_argument('processed'); f.add_argument('output')
    f.add_argument('--subjects', nargs='+', required=True)
    g = sub.add_parser('generate'); g.add_argument('model'); g.add_argument('output')
    g.add_argument('--count', type=int, default=12); g.add_argument('--seed', type=int, default=0)
    a = sub.add_parser('audit'); a.add_argument('real'); a.add_argument('synthetic'); a.add_argument('output')
    a.add_argument('--subjects', nargs='+', required=True)
    args = p.parse_args()
    if args.command == 'fit':
        result = fit(args.processed, args.subjects)
    elif args.command == 'generate':
        if not 1 <= args.count <= 1000:
            p.error('count must be 1..1000')
        model = json.loads(Path(args.model).read_text()); root = Path(args.output)
        root.mkdir(parents=True, exist_ok=False); metas = []
        for i in range(args.count):
            sid = f'SYN{i+1:04d}'; data, meta = generate(model, args.seed+i)
            np.savez_compressed(root/f'{sid}.npz', **data)
            meta['subject'] = sid; (root/f'{sid}.json').write_text(json.dumps(meta, indent=2)); metas.append(meta)
        result = {'origin': 'synthetic', 'generator_version': VERSION,
                  'skin_temperature': dict(SKIN_TEMPERATURE),
                  'fit_model_sha256': hashlib.sha256(Path(args.model).read_bytes()).hexdigest(),
                  'generator_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'numpy_version': np.__version__,
                  'grid_step_s': 2, 'sample_time_convention': 't[i] is the end of the corresponding synthetic feature window; no raw waveforms exist.',
                  'channels': {**{k: {'unit': UNITS[k], 'source': 'Synthetic generator fitted to reference recordings', 'description': 'Synthetic '+k+' in the processed PPG-DaLiA feature contract'} for k in CHANNELS},
                               'activity': {'unit': 'id', 'source': 'Resampled reference protocol order and durations', 'description': 'Activity ID; see activity_names'}},
                  'activity_names': ACTIVITY_NAMES, 'subjects': metas,
                  'source_dataset': model['source_dataset'], 'source_url': model['source_url'],
                  'source_authors': model['source_authors'], 'source_license': model['source_license'],
                  'limitations': model['limitations']}
        (root/'manifest.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({'output': str(root), 'recordings': len(metas)})); return
    else:
        # Explicit development/reference audit only; never silently inspect final subjects.
        if not set(args.subjects) <= {'S1', 'S4', 'S8', 'S12', 'S2', 'S3', 'S7', 'S10'}:
            p.error('Audits cannot inspect reserved final-test participants')
        real = [load_recording(Path(args.real)/f'{s}.npz') for s in args.subjects]
        synthetic = [load_recording(f) for f in sorted(Path(args.synthetic).glob('SYN*.npz'))]
        if not synthetic:
            p.error('No synthetic recordings')
        result = {'origin': 'synthetic-fidelity-audit', 'comparison_subjects': args.subjects,
                  'real': summarize(real), 'synthetic': summarize(synthetic),
                  'claim': 'Descriptive fidelity only; not physiological realism or prediction-transfer proof.'}
        result['correlation_mean_absolute_gap'] = float(np.mean(np.abs(np.array(result['real']['channel_correlation'])-np.array(result['synthetic']['channel_correlation']))))
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError('Use a new output file')
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'output': str(output), 'version': result.get('version', result.get('origin'))}))


if __name__ == '__main__':
    main()
