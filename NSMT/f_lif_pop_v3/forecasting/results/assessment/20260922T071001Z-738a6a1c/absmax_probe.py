"""Snapshot reducer audit and checkpoint forward only; no training."""
import sys,json,ast,csv,hashlib,math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import DataLoader
root,out=map(Path,sys.argv[1:3]);src=root/'f_lif_pop_v3/forecasting';sys.path.insert(0,str(src))
from config import Config
from model import LOAD_MODEL
from data_provider.synthetic import Dataset_Recall
from test import evaluate,selection_diagnostics
from utils import parameter_hash

torch.set_num_threads(2);torch.manual_seed(7)
tree=ast.parse((src/'train.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train_one_epoch');reducer=next(n for n in fn.body if isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='rate');code=ast.Module(body=[reducer],type_ignores=[]);ast.fix_missing_locations(code)
env={'np':np,'result':{},'rate':[.1,.2],'peak':1.,'watched':{'eta':[1.,9.]},'selector_grad':{'grad_WQ_absmax':[1.,9.],'grad_WK_absmax':[2.,10.],'grad_absmax_all':[1.,9.],'grad_observations':[1.,1.]},'every_batch':{'grad_total_norm_pre':[1.,9.],'clip_rate':[0.,1.]}}
exec(compile(code,'actual-snapshot-reducer','exec'),env);r={'reducer_two_observation_probe':env['result']}
state=next((src/'log/absmaxchk-160633').glob('**/model_state'));a=Config();a.load_args(str(state.parent),SimpleNamespace(cpu=True,num_device=0));a.save_model_state_path=str(state);p=src/'results/absmaxchk-160633'/(a.run_id+'.json');obs=json.loads(p.read_text());m=LOAD_MODEL[a.model](a,False);ds=Dataset_Recall(a,'test');dl=DataLoader(ds,batch_size=a.batch_size);ev=evaluate(m,dl,a)[0];diag=selection_diagnostics(m,dl,a);pv=obs['provenance'];rows=list(csv.DictReader((state.parent/'log/best_log_0.csv').open()))
r.update({'metrics':ev,'diagnostics':diag,'config':{k:getattr(a,k,None) for k in ['seed','data_seed','epoch','n_train','n_val','n_test','batch_size','g11_every','max_train_batches','max_eval_batches','qk_norm','theta','eta_fixed']},'mse_difference':ev['mse']-obs['test']['mse'],'m_eff_difference':diag['m_eff']-obs['diagnostics']['m_eff'],'checkpoint_hash_match':hashlib.sha256((state/'best+model.pt').read_bytes()).hexdigest()==pv['checkpoint_sha256'],'parameter_hash_match':parameter_hash(m)==pv['parameter_hash_evaluated'],'source_mismatches':[n for n,h in pv['source_sha256'].items() if hashlib.sha256((src/n).read_bytes()).hexdigest()!=h],'csv_rows':rows,'train_json':obs['train'],'observations_present_in_train_json':'grad_observations' in obs['train'],'expected_training_batches':math.ceil(a.n_train/a.batch_size),'expected_watch_count':len(range(0,math.ceil(a.n_train/a.batch_size),max(a.g11_every,1)))})
out.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');print('reducer',r['reducer_two_observation_probe']);print('config',r['config']);print('recall',ev['recall']['mse'],'Meff',diag['m_eff'],'diff',r['mse_difference'],r['m_eff_difference'],'hash',r['checkpoint_hash_match'],r['parameter_hash_match'],'sources',r['source_mismatches']);print('watch_count',r['expected_watch_count'],'json_observations',r['observations_present_in_train_json'])
