"""Evaluate only newly completed snapshot checkpoints on CPU. No training."""
import sys,json,hashlib,csv
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.utils.data import DataLoader
root,out=map(Path,sys.argv[1:3]);src=root/'f_lif_pop_v3/forecasting';sys.path.insert(0,str(src))
from config import Config
from model import LOAD_MODEL
from data_provider.synthetic import Dataset_Recall
from test import evaluate,selection_diagnostics
from utils import parameter_hash

torch.set_num_threads(2);torch.manual_seed(7);r={'runs':[],'incomplete':[]}
for state in sorted((src/'log/qk2-140541').glob('**/model_state')):
 a=Config();a.load_args(str(state.parent),SimpleNamespace(cpu=True,num_device=0));a.save_model_state_path=str(state)
 if a.eta_fixed in (0.,.2):continue
 path=src/'results/qk2-140541'/(a.run_id+'.json')
 if not path.exists():r['incomplete'].append({'run':a.run_id,'csv_epochs':len(list(csv.DictReader((state.parent/'log/best_log_0.csv').open()))),'status':'no completed JSON, not evaluated'});continue
 obs=json.loads(path.read_text());m=LOAD_MODEL[a.model](a,False);ds=Dataset_Recall(a,'test');dl=DataLoader(ds,batch_size=a.batch_size);ev=evaluate(m,dl,a)[0];diag=selection_diagnostics(m,dl,a);pv=obs['provenance'];rows=list(csv.DictReader((state.parent/'log/best_log_0.csv').open()))
 r['runs'].append({'run':a.run_id,'metrics':ev,'diagnostics':diag,'train':obs['train'],'config':{k:getattr(a,k,None) for k in ['seed','data_seed','n_train','n_val','n_test','epoch','patience','batch_size','max_train_batches','max_eval_batches','qk_norm','qk_eps','theta','eta_fixed','g11_every']},'mse_difference':ev['mse']-obs['test']['mse'],'m_eff_difference':diag['m_eff']-obs['diagnostics']['m_eff'],'checkpoint_hash_match':hashlib.sha256((state/'best+model.pt').read_bytes()).hexdigest()==pv['checkpoint_sha256'],'parameter_hash_match':parameter_hash(m)==pv['parameter_hash_evaluated'],'source_mismatches':[n for n,h in pv['source_sha256'].items() if hashlib.sha256((src/n).read_bytes()).hexdigest()!=h],'csv_epochs':len(rows)})
full=.2763386361581949;oracle=.03496568833854037
for x in r['runs']:x['G_vs_prior_oracle1']=(full-x['metrics']['recall']['mse'])/(full-oracle)
out.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');print(json.dumps(r,indent=2))
