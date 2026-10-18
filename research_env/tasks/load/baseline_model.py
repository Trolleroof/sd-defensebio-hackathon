"""Load-carriage baselines: fit + predict in one self-contained module.

This file is copied verbatim into every baseline artifact as ``inference.py`` and
into participant workspaces as ``baseline/baseline_model.py``. It must import only
the standard library and numpy (no research_env), because the worker sees only
the artifact directory. Parameters are JSON, never pickles.
"""
import json
import math
from pathlib import Path
import numpy as np

KINDS = ('mean_by_load', 'ridge', 'tree', 'pandolf')
FIELDS = ('mean_w', 'lower_w', 'upper_w')
RIDGE_FEATURES = ('body_mass_kg', 'stature_cm', 'age_years', 'is_male', 'load_kg',
                  'mass_x_load', 'load_sq')
TREE_FEATURES = ('body_mass_kg', 'stature_cm', 'age_years', 'is_male', 'load_kg', 'total_mass_kg')


def _x(request, names):
    mass, load = float(request['body_mass_kg']), float(request['load_kg'])
    values = {'body_mass_kg': mass, 'stature_cm': float(request['stature_cm']),
              'age_years': float(request['age_years']), 'is_male': float(request.get('sex') == 'M'),
              'load_kg': load, 'mass_x_load': mass*load, 'load_sq': load*load,
              'total_mass_kg': mass+load}
    return [values[n] for n in names]


def pandolf_w(body_mass_kg, load_kg, speed_m_s, grade_percent, terrain=1.0):
    """Pandolf, Givoni & Goldman (1977), J Appl Physiol 43:577-581. Gross watts.

    M = 1.5 W + 2.0 (W + L)(L / W)^2 + eta (W + L)(1.5 V^2 + 0.35 V G)
    W body mass kg, L load kg, V m/s, G grade percent, eta terrain (treadmill 1.0).
    No parameter is fitted to this dataset.
    """
    w, l, v, g = body_mass_kg, load_kg, speed_m_s, grade_percent
    return 1.5*w + 2.0*(w+l)*(l/w)**2 + terrain*(w+l)*(1.5*v*v + 0.35*v*g)


def _tree_fit(x, y, depth, min_leaf):
    if depth == 0 or len(y) < 2*min_leaf:
        return {'value': float(np.mean(y))}
    best = None
    base = float(np.sum((y-y.mean())**2))
    for j in range(x.shape[1]):
        order = np.argsort(x[:, j], kind='stable')
        xs, ys = x[order, j], y[order]
        csum, csq = np.cumsum(ys), np.cumsum(ys*ys)
        total, total_sq, n = csum[-1], csq[-1], len(ys)
        for i in range(min_leaf, n-min_leaf+1):
            if xs[i-1] == xs[i]:
                continue
            left = csq[i-1] - csum[i-1]**2/i
            right = (total_sq-csq[i-1]) - (total-csum[i-1])**2/(n-i)
            sse = left+right
            if best is None or sse < best[0] - 1e-9:
                best = (sse, j, float((xs[i-1]+xs[i])/2))
    if best is None or best[0] >= base - 1e-9:
        return {'value': float(np.mean(y))}
    _, j, t = best
    mask = x[:, j] <= t
    return {'feature': j, 'threshold': t, 'left': _tree_fit(x[mask], y[mask], depth-1, min_leaf),
            'right': _tree_fit(x[~mask], y[~mask], depth-1, min_leaf)}


def _tree_predict(node, x):
    while 'value' not in node:
        node = node['left'] if x[node['feature']] <= node['threshold'] else node['right']
    return node['value']


def raw_predict(model, request):
    """Point prediction without calibration offset."""
    kind = model['kind']
    if kind == 'mean_by_load':
        table = {float(k): v for k, v in model['load_means'].items()}
        load = float(request['load_kg'])
        if load in table:
            return table[load]
        return model['load_line'][0] + model['load_line'][1]*load
    if kind == 'ridge':
        x = (np.asarray(_x(request, RIDGE_FEATURES)) - model['center'])/np.asarray(model['scale'])
        return float(np.dot(x, model['coef']) + model['intercept'])
    if kind == 'tree':
        return _tree_predict(model['tree'], _x(request, TREE_FEATURES))
    if kind == 'pandolf':
        return pandolf_w(float(request['body_mass_kg']), float(request['load_kg']),
                         float(request['speed_m_s']), float(request['grade_percent']))
    raise ValueError('Unknown kind')


def point(model, request):
    mean = raw_predict(model, request)
    cal = request.get('calibration')
    if cal is not None:
        # Calibration tier: shift by this person's own measured unloaded residual.
        reference = dict(request, load_kg=float(cal['load_kg']), calibration=None)
        mean += float(cal['gross_w']) - raw_predict(model, reference)
    return float(mean)


def fit(rows, kind, *, tier='primary', alpha=1.0, depth=3, min_leaf=8, seed=0):
    """rows: dicts with request fields plus 'person' and 'gross_w' (training people only)."""
    if kind not in KINDS:
        raise ValueError('Unknown baseline kind')
    y = np.array([float(r['gross_w']) for r in rows])
    model = {'kind': kind, 'seed': seed, 'tier': tier, 'units': 'W'}
    if kind == 'mean_by_load':
        loads = sorted({float(r['load_kg']) for r in rows})
        means = {str(l): float(np.mean([float(r['gross_w']) for r in rows if float(r['load_kg']) == l])) for l in loads}
        line = np.polyfit(loads, [means[str(l)] for l in loads], 1) if len(loads) > 1 else [0., means[str(loads[0])]]
        model.update(load_means=means, load_line=[float(line[1]), float(line[0])])
    elif kind == 'ridge':
        x = np.array([_x(r, RIDGE_FEATURES) for r in rows])
        center, scale = x.mean(axis=0), x.std(axis=0)
        scale[scale < 1e-12] = 1.
        z = (x-center)/scale
        ym = y.mean()
        coef = np.linalg.solve(z.T@z + alpha*np.eye(z.shape[1]), z.T@(y-ym))
        model.update(alpha=alpha, features=list(RIDGE_FEATURES), center=center.tolist(),
                     scale=scale.tolist(), coef=coef.tolist(), intercept=float(ym))
    elif kind == 'tree':
        x = np.array([_x(r, TREE_FEATURES) for r in rows])
        model.update(features=list(TREE_FEATURES), depth=depth, min_leaf=min_leaf,
                     tree=_tree_fit(x, y, depth, min_leaf))
    else:
        model.update(equation='Pandolf 1977', terrain_factor=1.0, fitted_parameters=[])
    # In-sample absolute residual 90% quantile; coverage on new people is measured, not promised.
    residuals = []
    if tier == 'calibration':
        unloaded = {r['person']: float(r['gross_w']) for r in rows if float(r['load_kg']) == 0.}
        for r in rows:
            if float(r['load_kg']) != 0. and r['person'] in unloaded:
                req = dict(r, calibration={'load_kg': 0., 'gross_w': unloaded[r['person']]})
                residuals.append(abs(point(model, req) - float(r['gross_w'])))
    if not residuals:
        residuals = [abs(point(model, dict(r, calibration=None)) - float(r['gross_w'])) for r in rows]
    model['half_width_w'] = float(np.quantile(residuals, .9))
    return model


def predict(artifact_dir, request):
    model = json.loads((Path(artifact_dir)/'model.json').read_text())
    mean = point(model, request)
    if not math.isfinite(mean):
        raise ValueError('Nonfinite prediction')
    width = model['half_width_w']
    return {'mean_w': mean, 'lower_w': mean-width, 'upper_w': mean+width}


if __name__ == '__main__':
    # Workspace reproduction: python baseline/baseline_model.py data/train.json artifact [kind] [tier]
    import shutil
    import sys
    train_path, out = Path(sys.argv[1]), Path(sys.argv[2])
    kind = sys.argv[3] if len(sys.argv) > 3 else 'ridge'
    tier = sys.argv[4] if len(sys.argv) > 4 else 'primary'
    data = json.loads(train_path.read_text())
    out.mkdir(parents=True, exist_ok=True)
    (out/'model.json').write_text(json.dumps(fit(data['rows'], kind, tier=tier), sort_keys=True))
    shutil.copyfile(__file__, out/'inference.py')
    print(json.dumps({'kind': kind, 'tier': tier, 'artifact': str(out)}))
