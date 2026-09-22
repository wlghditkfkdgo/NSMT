"""No training: temporary checkpoint faults, saved checkpoint evaluation, isolated reducers."""
import ast,copy,csv,hashlib,json,sys,tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import DataLoader
root,out=map(lambda p:Path(p).resolve(),sys.argv[1:3]);src=root/'f_lif_pop_v3/forecasting';sys.path.insert(0,str(src))
from config import Config
from model import LOAD_MODEL
from data_provider.synthetic import Dataset_Recall
from test import evaluate
from utils import parameter_hash
import train

torch.set_num_threads(2);torch.manual_seed(7);r={'restore':[]}
state=next((src/'log/gradchk-104646').glob('**/model_state'));a=Config();a.load_args(str(state.parent),SimpleNamespace(cpu=True,num_device=0));a.save_model_state_path=str(state);sd=torch.load(state/'best+model.pt',map_location='cpu')
cases=[('intact',{},[]),('input_frozen_missing',{},['embedding.norm_mean','embedding.norm_std']),('input_none_missing',{'input_norm':'none'},['embedding.norm_mean','embedding.norm_std']),('key_frozen_missing',{'key_norm':'frozen'},['embedding.neuron.selector.key_mean','embedding.neuron.selector.key_std']),('key_none_missing',{'key_norm':'none'},['embedding.neuron.selector.key_mean','embedding.neuron.selector.key_std']),('eta_fixed_missing',{'eta_fixed':.5},['embedding.neuron.selector.eta_value']),('eta_learned_missing',{'eta_fixed':None},['embedding.neuron.selector.eta_value']),('weight_missing',{},['embedding.emb_linear.weight'])]
with tempfile.TemporaryDirectory(prefix='nsmt_restore_audit_') as tmp:
 for name,updates,removed in cases:
  args=copy.copy(a);args.save_model_state_path=tmp
  for k,v in updates.items():setattr(args,k,v)
  altered=dict(sd)
  for k in removed:altered.pop(k)
  torch.save(altered,Path(tmp)/'best+model.pt')
  row={'case':name,'removed':removed}
  try:LOAD_MODEL[args.model](args,False);row['accepted']=True
  except RuntimeError as e:row.update(accepted=False,error=str(e))
  r['restore'].append(row)
model=LOAD_MODEL[a.model](a,False);obs=json.loads(next((src/'results/gradchk-104646').glob('*.json')).read_text());ev=evaluate(model,DataLoader(Dataset_Recall(a,'test'),batch_size=a.batch_size),a)[0]
r['gradchk']={'metrics':ev,'mse_difference':ev['mse']-obs['test']['mse'],'parameter_hash_matches':parameter_hash(model)==obs['provenance']['parameter_hash_evaluated'],'checkpoint_hash_matches':hashlib.sha256((state/'best+model.pt').read_bytes()).hexdigest()==obs['provenance']['checkpoint_sha256'],'calibration':obs['provenance']['calibration'],'config':{k:getattr(a,k,None) for k in ['seed','data_seed','batch_size','epoch','n_train','n_val','n_test','g11_every','max_train_batches','max_eval_batches']},'train_record':obs['train'],'csv_rows':list(csv.DictReader((state.parent/'log/best_log_0.csv').open()))}
# Existing historical checkpoint with missing key buffers must still load and reproduce.
p=next((src/'log/pilot-eta-001742').glob('**/model_state'));old=Config();old.load_args(str(p.parent),SimpleNamespace(cpu=True,num_device=0));old.save_model_state_path=str(p);om=LOAD_MODEL[old.model](old,False);r['legacy_pilot_recall_mse']=evaluate(om,DataLoader(Dataset_Recall(old,'test'),batch_size=128),old)[0]['recall']['mse']
# Execute only the exact existing diagnostic aggregation node, after loss/updates are excluded.
tree=ast.parse((src/'train.py').read_text());fun=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train_one_epoch');node=next(n for n in fun.body if isinstance(n,ast.If) and ast.unparse(n.test)=='rate');ns={'np':np,'result':{},'peak':0.,'rate':[.1],'watched':{},'selector_grad':{'grad_WQ_absmax':[1.,9.],'grad_absmax_all':[2.,10.],'grad_WQ_norm':[2.,4.],'score_std':[float('nan'),1.]}}
exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'<frozen aggregation>','exec'),ns);r['isolated_aggregation']={'inputs':{'grad_WQ_absmax':[1.,9.],'grad_absmax_all':[2.,10.]},'output':ns['result'],'training_invoked':False}
# Capture precisely the result payload assignment to check the nonfinite field survives.
fun=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train');node=next(n for n in fun.body if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=="payload['train']")
ns={'payload':{},'epoch':0,'time':SimpleNamespace(time=lambda:1.),'started':0.,'stopper':SimpleNamespace(val_loss_min=.2),'train_result':ns['result']};exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'<payload assignment>','exec'),ns);r['payload_nonfinite_preserved']=ns['payload']['train'].get('score_std_nonfinite')==1
out.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');print(json.dumps(r,indent=2,allow_nan=False))
