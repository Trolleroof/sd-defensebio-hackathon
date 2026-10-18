"""Controller-side baseline training: writes self-contained JSON artifacts."""
from pathlib import Path
import shutil
from research_env.core.common import digest, file_hash, read, write
from research_env.tasks.load import baseline_model
from research_env.tasks.load.baseline_model import KINDS, fit, predict  # noqa: F401

MODULE = Path(baseline_model.__file__)
LOCK = Path(__file__).parents[2]/'requirements.lock'


def train(train_manifest, calibration_manifest, output_dir, seed, kind='ridge', tier='primary'):
    """Common v1 signature. train_manifest is a JSON file with {'role': 'train', 'rows': [...]}.
    calibration_manifest, if given, must contain only permitted training people."""
    data = read(train_manifest)
    if data.get('role') != 'train':
        raise ValueError('Training manifest must have role=train')
    if calibration_manifest is not None:
        cal = read(calibration_manifest)
        if cal.get('role') != 'train' or not {r['person'] for r in cal['rows']} <= {r['person'] for r in data['rows']}:
            raise ValueError('Calibration must use permitted training people only')
    return train_rows(data['rows'], output_dir, seed, kind, tier, train_manifest_sha256=file_hash(train_manifest))


def train_rows(rows, output_dir, seed=0, kind='ridge', tier='primary', **provenance):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = fit(rows, kind, tier=tier, seed=seed)
    write(out/'model.json', model)
    shutil.copyfile(MODULE, out/'inference.py')
    shutil.copyfile(LOCK, out/'requirements.lock')
    manifest = {'interface': 'physiology-research-v1', 'task': 'load', 'kind': kind, 'tier': tier,
                'seed': seed, 'candidate_id': digest(model), 'model_sha256': file_hash(out/'model.json'),
                'inference_sha256': file_hash(out/'inference.py'),
                'dependency_lock_sha256': file_hash(out/'requirements.lock'),
                'interval_note': 'In-sample residual 90% quantile; coverage measured, not promised',
                'reproduce': f'python baseline/baseline_model.py data/train.json artifact {kind} {tier}',
                **provenance}
    write(out/'artifact.json', manifest)
    return manifest
