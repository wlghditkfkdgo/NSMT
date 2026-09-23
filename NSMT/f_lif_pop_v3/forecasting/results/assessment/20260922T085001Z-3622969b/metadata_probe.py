"""Snapshot metadata and checkpoint identities only; no dataset or forward call."""
import sys,json,csv,hashlib
from pathlib import Path
from types import SimpleNamespace
import torch
root,out=map(Path,sys.argv[1:3]);src=root/'f_lif_pop_v3/forecasting';sys.path.insert(0,str(src));torch.set_num_threads(2)
from config import Config
from model import LOAD_MODEL
from utils import parameter_hash
fields=['seed','data_seed','task','n_train','n_val','n_test','n_confirm','epoch','batch_size','test','g11_every','mode','qk_norm','qk_eps','eta_fixed','theta','input_scale','input_norm','key_norm','alpha','tau','g11_bound']
r={'completed':[],'without_completion_json':[]}
complete=set()
for p in sorted((src/'results/seeds-173737').glob('*.json')):
 j=json.loads(p.read_text());seed=j['seed'];complete.add(seed);state=next((src/'log/seeds-173737').glob('**/seed'+str(seed)+'_*/model_state'));a=Config();a.load_args(str(state.parent),SimpleNamespace(cpu=True,num_device=0));a.save_model_state_path=str(state);m=LOAD_MODEL[a.model](a,False);csvp=state.parent/'log/best_log_0.csv';rows=list(csv.DictReader(csvp.open()));pv=j['provenance'];ch=hashlib.sha256((state/'best+model.pt').read_bytes()).hexdigest();vh=min(float(x['val_loss']) for x in rows)
 x={'seed':seed,'config':{k:getattr(a,k,None) for k in fields},'test':j['test'],'test_skipped':j.get('test_skipped'),'epochs_run':j['train']['epochs_run'],'csv_rows':len(rows),'best_val_loss':j['train']['best_val_loss'],'csv_min_val_loss':vh,'csv_rounding_consistent':abs(vh-j['train']['best_val_loss'])<.00000051,'checkpoint_sha256':ch,'checkpoint_hash_matches':ch==pv['checkpoint_sha256'],'evaluated_parameter_hash_matches':parameter_hash(m)==pv['parameter_hash_evaluated'],'source_mismatches':[n for n,h in pv['source_sha256'].items() if hashlib.sha256((src/n).read_bytes()).hexdigest()!=h],'max_recorded_watch_state':max(float(x['train_max_abs_state']) for x in rows),'last_train':j['train']};r['completed'].append(x);print({k:v for k,v in x.items() if k not in ['last_train','config']})
for p in (src/'log/seeds-173737').rglob('config.pt'):
 z=torch.load(p,map_location='cpu');seed=z['seed']
 if seed in complete:continue
 cp=p.parent.parent/'log/best_log_0.csv';rows=list(csv.DictReader(cp.open())) if cp.exists() else [];r['without_completion_json'].append({'seed':seed,'csv_rows_at_snapshot':len(rows),'config':{k:z.get(k) for k in fields},'status':'incomplete artifacts at snapshot; no failure inferred'})
r['missing_registered_completion_seeds']=sorted(set([7,13,21,42,123,256,512,1024])-complete)
r['shared_config_differences']={k:{str(x['seed']):x['config'][k] for x in r['completed']} for k in fields if k!='seed' and len({json.dumps(x['config'][k]) for x in r['completed']})>1}
out.write_text(json.dumps(r,indent=2,allow_nan=False));print('incomplete',r['without_completion_json']);print('missing',r['missing_registered_completion_seeds'],'config differences',r['shared_config_differences'])
