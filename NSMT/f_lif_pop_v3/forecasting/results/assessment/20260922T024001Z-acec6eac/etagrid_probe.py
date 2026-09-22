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
 a=Config();a.load_args(str(state.parent),SimpleNamespace(cpu=True,num_device=0));a.save_model_state_path=str(state);result=src/'results/etagrid-113240'/(a.run_id+'.json')
 if not result.exists():r['runs'].append({'run_id':a.run_id,'status':'incomplete, not evaluated'});continue
 obs=json.loads(result.read_text());m=LOAD_MODEL[a.model](a,False);ds=Dataset_Recall(a,'test');dl=DataLoader(ds,batch_size=a.batch_size);ev=evaluate(m,dl,a)[0];d=selection_diagnostics(m,dl,a);pv=obs['provenance'];csvrows=list(csv.DictReader((state.parent/'log/best_log_0.csv').open()))
 r['runs'].append({'run_id':a.run_id,'eta':a.eta_fixed,'actual_eta':m.embedding.neuron.selector.eta.item(),'metrics':ev,'diagnostics':d,'mse_difference':ev['mse']-obs['test']['mse'],'m_eff_difference':d['m_eff']-obs['diagnostics']['m_eff'],'parameter_hash_match':parameter_hash(m)==pv['parameter_hash_evaluated'],'checkpoint_hash_match':hashlib.sha256((state/'best+model.pt').read_bytes()).hexdigest()==pv['checkpoint_sha256'],'config':{k:getattr(a,k,None) for k in ['seed','data_seed','epoch','patience','n_train','n_val','n_test','batch_size','max_train_batches','max_eval_batches','input_scale','theta','input_norm','eta_fixed']},'train':obs['train'],'csv_epochs':len(csvrows),'best_csv_val':min(float(x['val_loss']) for x in csvrows),'source_mismatches':[n for n,h in pv['source_sha256'].items() if hashlib.sha256((src/n).read_bytes()).hexdigest()!=h]})
# Exact recurrence-independent coefficient mass, same test sequences and query->sequence aggregation.
_,_,truth,kind=next(iter(DataLoader(ds,batch_size=len(ds))));b=fractional_coefficients(.7,42,dtype=torch.float64);C=b[0];rows=[]
for eta in [0.,.2,.5,.655,.7,.75,.9,1.]:
 totals={k:torch.zeros(len(ds),dtype=torch.float64) for k in ['uniform_oracle','optimal_mass','full_mass']};counts=torch.zeros(len(ds),dtype=torch.float64);qmin=1.;qmax=0.
 for n in range(1,42):
  mask=truth[:,n,:n].double();valid=(kind[:,n]>0)&(mask.sum(-1)>0);bh=b[1:n+1].flip(0);B=bh.sum();BA=(mask*bh).sum(-1);num=mask.sum(-1);p=mask/num.clamp_min(1).unsqueeze(-1);raw=(1-eta)*bh+eta*B*bh*p/(p*bh).sum(-1,keepdim=True).clamp_min(1e-12);c=raw.clamp_max(C);uniform=(c*mask).sum(-1)/c.sum(-1).clamp_min(1e-12)
  # q_i=b_i p_i/sum(b p) is an arbitrary simplex because b_i>0.
  # All adaptive mass can be assigned to answer slots; maximize retained mass under per-slot cap.
  target=torch.minimum((1-eta)*BA+eta*B,num*C);opt=target/(target+(1-eta)*(B-BA)).clamp_min(1e-12)
  for k,x in [('uniform_oracle',uniform),('optimal_mass',opt),('full_mass',BA/B)]:totals[k]+=x*valid
  counts+=valid
  if valid.any():qmin=min(qmin,uniform[valid].min().item());qmax=max(qmax,uniform[valid].max().item())
 rows.append({'eta':eta,**{k:(v[counts>0]/counts[counts>0]).mean().item() for k,v in totals.items()},'uniform_query_min':qmin,'uniform_query_max':qmax,'sequences':int((counts>0).sum()),'queries':int(counts.sum())})
r['same_test_coefficient_bounds']=rows
# Concrete valid p beating uniform oracle on two answer lags, tested by actual Selector.
J=41;bh=b[1:J+1].flip(0);B=bh.sum();eta=.2;inds=[0,40];base=(1-eta)*bh;remaining=eta*B;allocation=torch.zeros(J,dtype=torch.float64)
for idx in inds:
 take=min(remaining.item(),(C-base[idx]).item());allocation[idx]+=take;remaining-=take
if remaining>0:allocation[inds[0]]+=remaining
popt=allocation/bh;popt/=popt.sum();pu=torch.zeros(J,dtype=torch.float64);pu[inds]=.5;sel=Selector(4,eta_fixed=eta).double();xi=torch.zeros(1,1,5,dtype=torch.float64);hist=torch.zeros(1,1,J,5,dtype=torch.float64)
vals={}
for name,p in [('uniform',pu),('cap_aware',popt)]:
 with torch.no_grad():c,aux=sel(xi,hist,bh,C.item(),'oracle',p.reshape(1,1,-1))
 vals[name]={'answer_mass':(c[...,inds].sum(-1)/c.sum(-1)).item(),'policy_on_answers':p[inds].tolist()}
r['uniform_not_upper_bound_example']={'history':41,'answer_lags':[41,1],'eta':eta,**vals}
out.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
for x in r['runs']:print('run',x.get('eta'),x.get('metrics',{}).get('recall'),x.get('diagnostics',{}).get('m_eff'),'diff',x.get('mse_difference'),'hash',x.get('parameter_hash_match'),x.get('checkpoint_hash_match'))
print('bounds',rows);print('counterexample',r['uniform_not_upper_bound_example'])
