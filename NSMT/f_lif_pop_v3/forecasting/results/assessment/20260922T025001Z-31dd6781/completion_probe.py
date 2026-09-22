"""CPU snapshot evaluation and oracle mass bounds; no training."""
import csv,json,sys,hashlib,copy
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.utils.data import DataLoader
root,out=map(lambda p:Path(p).resolve(),sys.argv[1:3]);src=root/'f_lif_pop_v3/forecasting';sys.path.insert(0,str(src))
from config import Config
from model import LOAD_MODEL
from data_provider.synthetic import Dataset_Recall
from test import evaluate,selection_diagnostics
from utils import parameter_hash
from layers import fractional_coefficients,Selector

torch.set_num_threads(2);torch.manual_seed(7);r={'runs':[]}
for state in sorted((src/'log/etagrid-113240').glob('**/model_state')):
 if not ('_oracle_' in str(state) or '_sparse_k3/' in str(state)):continue
 a=Config();a.load_args(str(state.parent),SimpleNamespace(cpu=True,num_device=0));a.save_model_state_path=str(state);result=src/'results/etagrid-113240'/(a.run_id+'.json')
 if not result.exists():r['runs'].append({'run_id':a.run_id,'status':'incomplete, not evaluated'});continue
 obs=json.loads(result.read_text());m=LOAD_MODEL[a.model](a,False);ds=Dataset_Recall(a,'test');dl=DataLoader(ds,batch_size=a.batch_size);ev=evaluate(m,dl,a)[0];d=selection_diagnostics(m,dl,a);pv=obs['provenance'];csvrows=list(csv.DictReader((state.parent/'log/best_log_0.csv').open()))
 r['runs'].append({'run_id':a.run_id,'eta':a.eta_fixed,'actual_eta':m.embedding.neuron.selector.eta.item(),'metrics':ev,'diagnostics':d,'mse_difference':ev['mse']-obs['test']['mse'],'m_eff_difference':d['m_eff']-obs['diagnostics']['m_eff'],'parameter_hash_match':parameter_hash(m)==pv['parameter_hash_evaluated'],'checkpoint_hash_match':hashlib.sha256((state/'best+model.pt').read_bytes()).hexdigest()==pv['checkpoint_sha256'],'config':{k:getattr(a,k,None) for k in ['seed','data_seed','epoch','patience','n_train','n_val','n_test','batch_size','max_train_batches','max_eval_batches','input_scale','theta','input_norm','eta_fixed']},'train':obs['train'],'csv_epochs':len(csvrows),'best_csv_val':min(float(x['val_loss']) for x in csvrows),'source_mismatches':[n for n,h in pv['source_sha256'].items() if hashlib.sha256((src/n).read_bytes()).hexdigest()!=h]})

 # Fingerprint generated split tensors without training or viewing new held-out labels for tuning.
 r['runs'][-1]['split_fingerprints']={}
 for split in ['train','val','test']:
  dataset=Dataset_Recall(a,split);h=hashlib.sha256()
  for tensor in [dataset.x,dataset.y,dataset.truth,dataset.recall]:h.update(tensor.numpy().tobytes())
  r['runs'][-1]['split_fingerprints'][split]={'count':len(dataset),'sha256':h.hexdigest()}
# Point estimates only, no seed confidence interval.
all_results=[json.loads(p.read_text()) for p in (src/'results/etagrid-113240').glob('*.json')]
full=next(o for o in all_results if 'sparse_eta0_' in o['run_id'])['test']['recall']['mse']
oracle1=next(o for o in all_results if 'oracle_eta1_' in o['run_id'])['test']['recall']['mse']
r['exploratory_headroom']=full-oracle1
r['G_relative_to_oracle1']={o['run_id']:(full-o['test']['recall']['mse'])/(full-oracle1) for o in all_results}
truth,kind=ds.truth,ds.recall
per_seq=[]
for t,k in zip(truth,kind):
 vals=[t[n,:n].double().sum().item()/n for n in range(1,len(k)) if k[n]>0 and t[n,:n].any()]
 per_seq.append(sum(vals)/len(vals))
r['uniform_slot_chance_test_query_then_sequence']=sum(per_seq)/len(per_seq)
out.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
for x in r['runs']:print(x['run_id'],x['metrics']['recall'],x['diagnostics']['m_eff'],'differences',x['mse_difference'],x['m_eff_difference'],'hash',x['parameter_hash_match'],x['checkpoint_hash_match'],'sources',x['source_mismatches'])
print('headroom',r['exploratory_headroom'],'chance',r['uniform_slot_chance_test_query_then_sequence'])
