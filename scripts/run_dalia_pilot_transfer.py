from pathlib import Path
from benchmark.workspace import read,write
from benchmark.executor import execute
from benchmark.campaign import CampaignWorkspace
from benchmark.dalia_campaign import baseline_codes
import argparse
p=argparse.ArgumentParser()
p.add_argument('run_root')
root=Path(p.parse_args().run_root)
rows=[]
for case in range(3):
 syn=read(root/f'synthetic-case-{case}.json')
 syn_support=[s for s in syn['sessions'] if s['role'] in ('support','acquisition')]
 for realcase in range(3):
  ep=read(root/f'real_development-case-{realcase}.json')
  directory=root/'transfer-evaluation'/f'case-{case}-real-{realcase}'
  if not directory.exists():w=CampaignWorkspace.create(ep,directory/'private',directory/'agent')
  else:w=CampaignWorkspace(directory/'private')
  w.state['stage']=3;w._save()
  requests=[w._request(s) for s in ep['sessions'] if s['role']=='test']
  sources=baseline_codes()
  luna=root/f'case-{case}/gpt-6-luna/private/submission.py'
  if luna.exists():sources['gpt-6-luna']=luna.read_text()
  for name,code in sources.items():
   for mode in ('synthetic_calibration','real_calibration'):
    support=syn_support if mode=='synthetic_calibration' else [s for s in ep['sessions'] if s['role'] in ('support','acquisition')]
    result=execute(code,support,requests,forbidden_paths=[str(root.resolve())])
    final=w._score_role(result['predictions'],'test') if result['execution']['valid'] else {'valid':False}
    rows.append({'synthetic_case':case,'real_development_case':realcase,'competitor':name,'calibration':mode,'final':final})
write(root/'real-transfer-report.json',{'runs':rows,'claim':'Development-only source replay; no real-target agent revision; not untouched final evaluation.'})
print('completed transfer evaluations',len(rows))
