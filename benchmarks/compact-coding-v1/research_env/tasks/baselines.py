"""CPU reference predictors for heat/load, fitted on training people only."""
from pathlib import Path
import shutil
import numpy as np
from research_env.core.common import file_hash, read, tree_hashes, write


def features(request, task, horizon=0, kind='ridge'):
    if task == 'load':
        mass, load = request['mass_kg'], request['load_kg']
        return [mass, load, mass*load/100, load**2/100, request['height_cm'], request['age_years']]
    history = request['history']
    first, last = history[0], history[-1]
    t = request['cutoff_min']
    future = t+horizon
    if kind == 'elapsed':
        return [future, future**2/100, future**3/10000]
    if kind == 'thermal':
        return [1-np.exp(-future/20), first['skin_c'], last['skin_c']-first['skin_c'],
                last['hr_bpm']/100, future/60, last['ambient_c']]
    demo = request['demographics']
    return [t, horizon, future, first['skin_c'], last['skin_c'], last['skin_c']-first['skin_c'],
            first['hr_bpm'], last['hr_bpm'], float(np.mean([r['hr_bpm'] for r in history])),
            last['ambient_c'], last['humidity_pct'], demo['age_years'], demo['mass_kg'], demo['height_cm']]


def predict(artifact_dir, request):
    import joblib
    a = Path(artifact_dir)
    config = read(a/'config.json')
    task, kind = config['task'], config['kind']
    model = joblib.load(a/'model.joblib')
    def one(h):
        if kind == 'mean_by_load':
            mean = model[str(request['load_kg'])]
        else:
            mean = float(model.predict([features(request,task,h,kind)])[0])
        radius = config['radius'][str(h)]
        return mean, mean-radius, mean+radius
    if task == 'heat':
        return {'forecasts': [dict(zip(('horizon_min','mean_c','lower_c','upper_c'), (h,*one(h))))
                              for h in (5,10,20)], 'state': None}
    m,lo,hi = one(0)
    return {'mean_w': m, 'lower_w': lo, 'upper_w': hi, 'state': None}


def train(train_manifest, output_dir, kind='ridge', seed=0, alpha=10):
    import joblib
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    manifest = read(train_manifest)
    task = manifest['task']
    if manifest['role'] != 'train' or task not in ('heat','load'):
        raise ValueError('Train only on permitted training people')
    allowed = ('elapsed','ridge','boosting','thermal') if task=='heat' else ('mean_by_load','ridge','forest')
    if kind not in allowed:
        raise ValueError('Baseline unavailable for task')
    root = Path(output_dir)
    root.mkdir(parents=True,exist_ok=False)
    rows = [(r, y) for r in manifest['records'] for y in r['targets']]
    x = np.array([features(r['request'],task,y['slice'] if task=='heat' else 0,kind) for r,y in rows])
    y = np.array([v['value'] for _,v in rows])
    if kind=='mean_by_load':
        model = {str(load): float(np.mean([v['value'] for r,v in rows if r['request']['load_kg']==load])) for load in (0,5,10,15)}
        fitted = np.array([model[str(r['request']['load_kg'])] for r,_ in rows])
    else:
        model = (HistGradientBoostingRegressor(max_iter=80,max_leaf_nodes=7,l2_regularization=alpha,random_state=seed)
                 if kind=='boosting' else RandomForestRegressor(n_estimators=80,max_depth=4,min_samples_leaf=4,random_state=seed,n_jobs=1)
                 if kind=='forest' else make_pipeline(StandardScaler(),Ridge(alpha=alpha)))
        model.fit(x,y)
        fitted = model.predict(x)
    radii = {}
    for h in ((5,10,20) if task=='heat' else (0,)):
        indices = [i for i,(_,v) in enumerate(rows) if task=='load' or v['slice']==h]
        radii[str(h)] = float(np.quantile(abs(y[indices]-fitted[indices]),.9))
    joblib.dump(model,root/'model.joblib')
    write(root/'config.json',{'task':task,'kind':kind,'seed':seed,'alpha':alpha,'radius':radii,
                             'interval_method':'90th percentile training residual; calibration not guaranteed',
                             'train_manifest_sha256':file_hash(train_manifest)})
    # Self-contained artifact: no controller imports or external model deserialization.
    source = Path(__file__).read_text().split('\ndef train(', 1)[0].replace('from research_env.core.common import file_hash, read, tree_hashes, write',
        'import json\ndef read(path):\n    return json.loads(Path(path).read_text())')
    (root/'inference.py').write_text(source)
    package = Path(__file__).resolve().parents[1]
    shutil.copyfile(package/'requirements.lock',root/'requirements.lock')
    return {'artifact_hashes':tree_hashes(root),'task':task,'kind':kind,'seed':seed}
