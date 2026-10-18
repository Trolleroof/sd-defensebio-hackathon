from pathlib import Path
from benchmark.workspace import read,write
from benchmark.campaign_pilot import run_one
from benchmark.dalia_campaign import baseline_codes
from benchmark.ratings import campaign_ratings
import argparse
p=argparse.ArgumentParser()
p.add_argument('run_root')
root=Path(p.parse_args().run_root)
runs=[]
for case in range(3):
 ep=read(root/f'synthetic-case-{case}.json')
 for name,code in baseline_codes().items():
  result=run_one(ep,root/f'case-{case}'/name,code)
  runs.append({'case':case,'competitor':name,**result})
  print(case,name,result['final'].get('valid'),result['final'].get('mae_bpm'),flush=True)
write(root/'baseline-report.json',{'runs':runs,'ratings':campaign_ratings(runs)})
