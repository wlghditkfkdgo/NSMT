"""Snapshot-only evaluation and bounded gradient/causality checks. No training."""
import copy,contextlib,csv,hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from torch.utils.data import DataLoader
root,out=map(lambda x:Path(x).resolve(),sys.argv[1:3]);src=root/'f_lif_pop_v3/forecasting';sys.path.insert(0,str(src))
import ours
from config import Config
from model import LOAD_MODEL
from data_provider.synthetic import Dataset_Recall
from data_provider.data_factory import data_provider
from test import evaluate,selection_diagnostics
from utils import parameter_hash

torch.set_num_threads(2);torch.manual_seed(7);r={'runs':[]};original_oracle=ours.truth_to_oracle_p
# Reproduce old oracle semantics only as an explicitly labelled compatibility diagnostic.
def legacy(truth,kind,step,dim):return original_oracle(truth,None,step,dim)
for suite in ['diag3-102747','drive-103547','csvchk-103953','ettchk-103004','notest-103025']:
 for state in sorted((src/'log'/suite).glob('**/model_state')):
  a=Config();a.load_args(str(state.parent),SimpleNamespace(cpu=True,num_device=0));a.save_model_state_path=str(state)
  obs=json.loads((src/'results'/suite/(a.run_id+'.json')).read_text());m=LOAD_MODEL[a.model](a,False);prov=obs.get('provenance',{})
  row={'suite':suite,'run_id':a.run_id,'readout':a.readout,'mode':a.mode,'parameter_hash':parameter_hash(m),'hash_matches':parameter_hash(m)==prov.get('parameter_hash_evaluated',prov.get('parameter_hash')),'checkpoint_hash_matches':None if 'checkpoint_sha256' not in prov else hashlib.sha256((state/'best+model.pt').read_bytes()).hexdigest()==prov['checkpoint_sha256'],'last_differs_evaluated':prov.get('parameter_hash_last_epoch')!=prov.get('parameter_hash_evaluated') if 'parameter_hash_evaluated' in prov else None,'config':{k:getattr(a,k,None) for k in ['seed','data_seed','epoch','n_train','n_val','n_test','batch_size','max_train_batches','max_eval_batches','theta','input_scale','eta_fixed','test']}}
  if obs.get('test_skipped'):
   row['test_skipped']=True;row['auditor_test_access']=False;r['runs'].append(row);continue
  if a.task=='recall':dataset=Dataset_Recall(a,'test');loader=DataLoader(dataset,batch_size=64)
  else:
   a.root_path=str(root/'forecasting/dataset/ETT-small');dataset,loader=data_provider(a,'test')
  ev=evaluate(m,loader,a,baselines=True)[0];row.update(metrics_current_policy=ev,recorded_metrics=obs['test'],mse_current_minus_recorded=ev['mse']-obs['test']['mse'],diagnostics=selection_diagnostics(m,loader,a))
  if a.mode=='oracle':
   with patch.object(ours,'truth_to_oracle_p',side_effect=legacy):old=evaluate(m,loader,a)[0]
   row['metrics_legacy_policy']=old;row['mse_legacy_minus_recorded']=old['mse']-obs['test']['mse']
  if suite=='csvchk-103953':
   rows=list(csv.DictReader((state.parent/'log/best_log_0.csv').open()));row['csv']={'rows':len(rows),'epochs':[x['epoch'] for x in rows],'columns':list(rows[0]),'min_val_loss':min(float(x['val_loss']) for x in rows),'best_val_loss_json':obs['train']['best_val_loss'],'final_result_csv_present':(state.parent/'log/final+result.csv').exists()}
  if a.task!='recall':row['dataset_windows']=len(dataset);row['evaluated_windows']=ev['all']['elements']//(a.pred_len*7)
  r['runs'].append(row)
  if suite=='drive-103547' and a.mode=='sparse':driveargs=a;drivemodel=m;drivedata=dataset
# Drive = learned branch mixture; attached gradient and no future access.
x,y,truth,kind=next(iter(DataLoader(drivedata,batch_size=2)));x=x.float();x.requires_grad_(True)
pred,aux=drivemodel(x,return_aux=True)
expected=(aux['state']*drivemodel.embedding.neuron.soma.weight).sum(-1)
live=aux['analog'];r['drive']={'aux_shapes':{'state':list(aux['state'].shape),'analog':list(live.shape)},'mixture_max_diff':(live-expected).abs().max().item(),'attached':live.requires_grad}
grads=torch.autograd.grad(pred.square().mean(),[x,drivemodel.embedding.neuron.soma.weight,drivemodel.embedding.neuron.selector.query.weight],allow_unused=True)
r['drive']['gradient_norms']=[None if g is None else g.norm().item() for g in grads]
with torch.no_grad():
 x2=x.detach().clone();x2[:,20*driveargs.patch_size:]+=3;p2=drivemodel(x2);r['drive']['causal_prefix_difference']=(pred[:,:20]-p2[:,:20]).abs().max().item()
# Oracle helper policy on all 256 sequences; copy nonuniform must be zero.
a=driveargs;data=DataLoader(drivedata,batch_size=256);xb,yb,tr,k=next(iter(data));count=oldcount=rec=0;masserr=0.
for n in range(1,tr.shape[1]):
 p=original_oracle(tr,k,n,1).squeeze(1);q=legacy(tr,k,n,1).squeeze(1);copy_mask=k[:,n]==0;valid=(k[:,n]>0)&tr[:,n,:n].any(-1)
 count+=int(((p-1/n).abs().max(-1).values[copy_mask]>1e-6).sum());oldcount+=int(((q-1/n).abs().max(-1).values[copy_mask]>1e-6).sum());rec+=int(valid.sum())
 if valid.any():masserr=max(masserr,((p*tr[:,n,:n]).sum(-1)[valid]-1).abs().max().item())
r['oracle_policy']={'copy_nonuniform_current':count,'copy_nonuniform_legacy':oldcount,'recall_queries':rec,'recall_answer_mass_error':masserr}
try:drivemodel(x.detach(),mode='oracle',truth=truth);r['oracle_policy']['missing_kind_rejected']=False
except ValueError:r['oracle_policy']['missing_kind_rejected']=True
out.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
for row in r['runs']:
 print(row['suite'],row['readout'],row['mode'],'delta',row.get('mse_current_minus_recorded'),'legacy_delta',row.get('mse_legacy_minus_recorded'),'recall',row.get('metrics_current_policy',{}).get('recall'),'hash',row['hash_matches'])
print('drive',r['drive']);print('oracle',r['oracle_policy'])
