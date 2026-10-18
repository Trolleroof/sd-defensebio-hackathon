import json, sys
from pathlib import Path
import argparse
p=argparse.ArgumentParser(description='Exercise existing SoliderPhys-Bench windows and scoring on synthetic recordings')
p.add_argument('processed'); p.add_argument('--physbench-source',required=True); p.add_argument('--output',required=True)
args=p.parse_args()
source=Path(args.physbench_source).resolve()
sys.path.insert(0,str(source))
from physbench.config import load_task, TaskConfig
from physbench.data.dalia import load_recording
from physbench.episodes import split_subject, support_windows, dev_windows, transfer_windows
from physbench.scoring import score, validate_predictions
import numpy as np
root=Path(args.processed).resolve()
base=load_task(source/'configs/task_hr_transfer.yaml')
raw=dict(base.raw)
raw.update(benchmark_version='synthetic-dalia-contract-smoke-v1',dataset='synthetic_dalia',processed_dir=str(root),
 target_provenance='Synthetic HR fitted to reference PPG-DaLiA statistics; no ECG measured.',
 reference_subjects=[f'SYN{i:04d}' for i in range(1,5)],dev_subjects=[f'SYN{i:04d}' for i in range(5,9)],
 heldout_subjects=[f'SYN{i:04d}' for i in range(9,13)],excluded_subjects=[])
config=root/'task-synthetic-contract-smoke.json';config.write_text(json.dumps(raw,indent=2)+'\n')
cfg=TaskConfig(raw,config)
reports=[]
for subject in [f'SYN{i:04d}' for i in range(1,13)]:
 r=load_recording(root/f'{subject}.npz');sp=split_subject(cfg,subject)
 sw,dw,tw=support_windows(cfg,sp),dev_windows(cfg,sp),transfer_windows(cfg,sp)
 assert len(sw)>0 and len(dw)>0 and len(tw)>0,(subject,len(sw),len(dw),len(tw))
 for w in (sw,dw,tw):
  assert w.hist['hr_ecg'].shape==(len(w),30)
  assert w.target.shape==(len(w),60)
  assert 'hr_ecg' not in w.fut
  assert all(x.shape==(len(w),60) for x in w.fut.values())
 pred=np.repeat(tw.hist['hr_ecg'][:,-1:],60,axis=1)
 valid,msg=validate_predictions(pred,len(tw),60,cfg.valid_hr_range)
 assert valid,msg
 ex=np.isin(tw.activity,[2,3,4,7]);metrics=score(pred,tw.target,ex,tw.activity,2)
 reports.append({'subject':subject,'support_windows':len(sw),'development_windows':len(dw),'transfer_windows':len(tw),
                 'persistence_phase_balanced_mae_bpm':metrics['phase_balanced_mae_bpm']})
Path(args.output).write_text(json.dumps({'source':str(source),'config':str(config),'status':'pass','recordings':reports,'claim':'Existing SoliderPhys-Bench loader, splits, windows, validation and numerical scoring executed; no LLM run. All these seeds are inspected smoke cases, not untouched final tests.'},indent=2)+'\n')
print(json.dumps({'compatible_recordings':len(reports),'first_recording':reports[0]},indent=2))
