"""Recompute reported G from snapshot JSON; no model execution."""
from pathlib import Path
import sys,json
root,out=map(Path,sys.argv[1:3]);rows=[]
for p in sorted((root/'f_lif_pop_v3/forecasting/results/etagrid-113240').glob('*.json')):
 j=json.loads(p.read_text());rows.append({'file':p.name,'recall_mse':j['test']['recall']['mse'],'epochs':j['train']['epochs_run']})
f=next(x['recall_mse'] for x in rows if '_sparse_eta0_' in x['file']);o=next(x['recall_mse'] for x in rows if '_oracle_eta1_' in x['file'])
for x in rows:x['G_common_full_oracle1']=(f-x['recall_mse'])/(f-o)
r={'method':'G=(full_eta0 - condition)/(full_eta0 - oracle_eta1), same etagrid suite','full':f,'oracle1':o,'rows':rows};out.write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
