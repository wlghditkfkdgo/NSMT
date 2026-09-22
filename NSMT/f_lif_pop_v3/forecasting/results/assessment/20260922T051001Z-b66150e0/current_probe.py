"""Snapshot forward-only model audit; tiny local Jacobians, no training/optimizer."""
import sys,json,hashlib,copy,ast,importlib.util
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
from layers import Selector,fractional_coefficients
import train

torch.set_num_threads(2);torch.manual_seed(7);r={'runs':[]}
for state in sorted((src/'log/qk2-140541').glob('**/model_state')):
 a=Config();a.load_args(str(state.parent),SimpleNamespace(cpu=True,num_device=0));a.save_model_state_path=str(state);rp=src/'results/qk2-140541'/(a.run_id+'.json')
 if not rp.exists():continue
 obs=json.loads(rp.read_text());model=LOAD_MODEL[a.model](a,False);ds=Dataset_Recall(a,'test');dl=DataLoader(ds,batch_size=a.batch_size)
 ev=evaluate(model,dl,a)[0];diag=selection_diagnostics(model,dl,a);pv=obs['provenance']
 item={'run':a.run_id,'config':{k:getattr(a,k,None) for k in ['seed','data_seed','epoch','patience','n_train','n_val','n_test','batch_size','g11_every','max_train_batches','max_eval_batches','theta','qk_norm','qk_eps','eta_fixed','key_norm']},'metrics':ev,'diagnostics':diag,'train':obs['train'],'mse_difference':ev['mse']-obs['test']['mse'],'m_eff_difference':diag['m_eff']-obs['diagnostics']['m_eff'],'parameter_hash_match':parameter_hash(model)==pv['parameter_hash_evaluated'],'checkpoint_hash_match':hashlib.sha256((state/'best+model.pt').read_bytes()).hexdigest()==pv['checkpoint_sha256']}
 if a.eta_fixed==.2:
  x=ds.x[:2].clone();xx=x.clone();xx[:,168:]=torch.randn_like(xx[:,168:])
  with torch.no_grad():y=model(x);yy=model(xx)
  item['future_perturbation_prefix_maxdiff']=(y[:,:21]-yy[:,:21]).abs().max().item()
 r['runs'].append(item)
# Calibration compatibility check forced to a specific captured artifact, no live file writes.
cal=src/'results/calibration/recall_k3_r2_a0.7_norm-frozen_qknorm_seed7_260922-140541.json';saved=json.loads(cal.read_text());train.glob.glob=lambda pattern:[str(cal)]
r['calibration']={'saved_qk_eps':saved['args'].get('qk_eps'),'saved_key_norm':saved['args'].get('key_norm'),'checks':[]}
for field,value in [('qk_eps',1.0),('key_norm','frozen' if a.key_norm=='none' else 'none'),('qk_norm',False)]:
 aa=copy.copy(a);setattr(aa,field,value)
 try:res=train.load_calibration(aa);status='accepted';detail=res
 except ValueError as e:status='rejected';detail=str(e)
 r['calibration']['checks'].append({'field':field,'requested':value,'status':status,'detail':detail})
# Test actual selector against a separately calculated soft-normalised distance on small input.
sel=Selector(4,theta=.38146987702788376,qk_norm=True,qk_eps=.01,eta_fixed=.2).double()
xi=torch.randn(2,3,5,dtype=torch.float64);hist=torch.randn(2,3,6,5,dtype=torch.float64);b=fractional_coefficients(.7,7,dtype=torch.float64)
with torch.no_grad():
 c,aux=sel(xi,hist,b[1:].flip(0),b[0].item());q=sel.query(xi);k=sel.key(hist);q=q/torch.sqrt(torch.sum(q*q,-1,keepdim=True)+.0001);k=k/torch.sqrt(torch.sum(k*k,-1,keepdim=True)+.0001);expected=-torch.sum((q.unsqueeze(-2)-k)**2,-1)/(4*sel.theta)
 zc,za=sel(torch.zeros_like(xi),torch.zeros_like(hist),b[1:].flip(0),b[0].item())
r['selector']={'distance_maxdiff':(aux['score']-expected).abs().max().item(),'zero_input_finite':bool(torch.isfinite(zc).all() and torch.isfinite(za['score']).all()),'max_soft_q_norm':q.norm(dim=-1).max().item()}
# Hard floor already caps the local derivative; soft norm smooths it, not uniquely bounds it.
r['local_jacobians']=[]
for eps in [1e-6,.01]:
 for label,fun in [('hard',lambda x:x/x.norm().clamp_min(eps)),('soft',lambda x:x/(x.square().sum()+eps**2).sqrt())]:
  j=torch.autograd.functional.jacobian(fun,torch.zeros(4,dtype=torch.float64));r['local_jacobians'].append({'eps':eps,'form':label,'at_zero_operator_norm':torch.linalg.norm(j,ord=2).item()})
# Execute only the actual diagnostic reducer, no loss/backward/optimizer/train loop.
tree=ast.parse((src/'train.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train_one_epoch');node=next(n for n in fn.body if isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='rate');mod=ast.Module(body=[node],type_ignores=[]);ast.fix_missing_locations(mod)
env={'np':np,'result':{},'rate':[.1,.2],'peak':1.,'watched':{},'selector_grad':{'grad_absmax_all':[1.,9.],'grad_total_norm_pre':[.5,10.],'clip_rate':[0.,1.]}}
exec(compile(mod,'snapshot-reducer','exec'),env);r['diagnostic_reducer']=env['result']
# Exact bound correction, using prior independently constructed policy.
spec=importlib.util.spec_from_file_location('reach',root/'f_lif_pop_v3/analysis/meff_reachable.py');reach=importlib.util.module_from_spec(spec);spec.loader.exec_module(reach)
previous=json.loads(Path('f_lif_pop_v3/forecasting/results/assessment/20260922T043001Z-65597c1e/constructive_bound.json').read_text());idx=np.array(previous['answers']);val=reach.exact_bound(previous['eta'],idx)
r['exact_bound']={'actual':float(val),'constructed_previous':previous['constructed_mass'],'difference':float(val-previous['constructed_mass']),'sampled_best_function_exists':hasattr(reach,'sampled_best'),'free_bound_function_exists':hasattr(reach,'free_bound')}
out.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
for x in r['runs']:print(x['run'],x['metrics']['recall']['mse'],x['diagnostics']['m_eff'],'diff',x['mse_difference'],x['m_eff_difference'],'hash',x['parameter_hash_match'],x['checkpoint_hash_match'])
for key in ['calibration','selector','local_jacobians','diagnostic_reducer','exact_bound']:print(key,r[key])
