"""Export a compact, source-linked build record from executed local evidence."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_env.core.common import file_hash, read, write
from research_env.tasks.pulse.scoring import paired_person_bootstrap

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('dataset')
parser.add_argument('output')
args = parser.parse_args()
root=Path(args.dataset)
data=read(root/'dataset.json')
baselines=read(root/'baselines-final/report.json')
keep=('primary_loss','units','per_person_mae_bpm','secondary','uncertainty','n_people','n_bouts','n_windows','training_wall_s','resource_use')
summary={'source':data['source'],'splits':data['splits'],'eligible_windows':data['eligible_windows'],
         'exclusions':data['exclusions'],'reference_status':'provisional; builder inspected six excerpts; broader annotation and clock review pending',
         'baseline_context':'CPU trusted-builder; final model choice frozen on development evidence',
         'selected_on_development':baselines['selected_on_development'],
         'baselines':{k:{f:v[f] for f in keep if f in v} for k,v in baselines['baselines'].items()},
         'private_test_baselines':{k:{f:v[f] for f in keep if f in v} for k,v in baselines['private_test_baselines'].items()},
         'live_agent':'pending; no authorized provider invoked','live_llm_judge':'pending; no authorized provider invoked',
         'equal_budget_30_call_search':'implemented, not executed; one-candidate wiring smoke executed',
         'checks':{'new_tests':13,'legacy_tests':60,'legacy_interpreter':'normal Python, not numerical virtualenv'},
         'evidence':{str(p.relative_to(root)):file_hash(p) for p in root.rglob('*')
                     if p.is_file() and (p.parent==root/'evidence') and p.suffix in ('.json','.png','.log')},
         'source_hashes':{str(p):file_hash(p) for p in Path('research_env').rglob('*') if p.is_file() and '__pycache__' not in str(p)}}
summary['paired_person_bootstrap_ridge_minus_median_bpm']=paired_person_bootstrap(
    baselines['private_test_baselines']['ridge']['per_person_mae_bpm'],
    baselines['private_test_baselines']['median']['per_person_mae_bpm'])
for name in ('container-checks.json','docker-replay.json'):
    if (root/'evidence'/name).exists():summary[name]=read(root/'evidence'/name)
write(args.output,summary)
print(args.output)
