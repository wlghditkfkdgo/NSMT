"""CPU validation-only score/policy/coefficient audit; no training."""
from pathlib import Path
from types import SimpleNamespace
import sys,json,math
import torch
from torch.utils.data import DataLoader
root,out=map(lambda p:Path(p).resolve(),sys.argv[1:3]);src=root/'f_lif_pop_v3/forecasting';sys.path.insert(0,str(src))
from config import Config
from model import LOAD_MODEL
from data_provider.synthetic import Dataset_Recall

torch.set_num_threads(2);torch.manual_seed(7)
state=next((src/'log/diag3-102747').glob('**/seed7_flatten_spike_heterogeneous_sparse_k3/model_state'));a=Config();a.load_args(str(state.parent),SimpleNamespace(cpu=True,num_device=0));a.save_model_state_path=str(state);m=LOAD_MODEL[a.model](a,False);m.eval();sel=m.embedding.neuron.selector
x,y,truth,kind=next(iter(DataLoader(Dataset_Recall(a,'val'),batch_size=64,shuffle=False)));records=[]
def hook(module,inputs,output):
 xi,hist,b,b0,*rest=inputs;c,aux=output
 records.append({'score':aux['score'].clone(),'p':aux['p'].clone(),'c':c.detach().clone(),'b':b.clone(),'keys':module.key(hist).detach()})
h=sel.register_forward_hook(hook)
with torch.no_grad():m(x.float())
h.remove();B,D=64,a.embed_dim;tot={k:torch.zeros(B,dtype=torch.float64) for k in ['score_hit','p_hit','c_hit','p_mass','c_mass','chance_hit','kernel_mass']};counts=torch.zeros(B,dtype=torch.float64)
for n,rec in enumerate(records,1):
 ans=truth[:,n,:n];valid=(kind[:,n]>0)&ans.any(-1);mask=ans[:,None,:].double();counts+=valid.double()
 for key,arr in [('score_hit',rec['score']),('p_hit',rec['p']),('c_hit',rec['c'])]:
  hit=ans.gather(1,arr.argmax(-1)).double().mean(-1);tot[key]+=hit*valid
 for key,arr in [('p_mass',rec['p']),('c_mass',rec['c'])]:
  share=((arr.double()*mask).sum(-1)/arr.double().sum(-1).clamp_min(1e-12)).mean(-1);tot[key]+=share*valid
 tot['chance_hit']+=ans.double().mean(-1)*valid
 tot['kernel_mass']+=((rec['b'].double()*ans).sum(-1)/rec['b'].double().sum())*valid
metrics={k:(v[counts>0]/counts[counts>0]).mean().item() for k,v in tot.items()}
# Last available past keys: [B,D,41,4]. Singular values define PR of Gram eigenvalues.
keys=records[-1]['keys'].double();sv=torch.linalg.svdvals(keys);lam=sv.square();pr=lam.sum(-1).square()/lam.square().sum(-1).clamp_min(1e-30)
# Ideal policy one-hot at each lag; threshold at which its coefficient exceeds recent.
b=m.embedding.neuron.b.double();J=41;bh=b[1:J+1].flip(0);mass=bh.sum();r=[]
for lag in [2,5,10,20,41]:
 bt=b[lag];recent=b[1];crit=(recent-bt)/(mass+recent-bt);target=J-lag
 row={'lag':lag,'eta_for_target_to_beat_recent_before_cap':crit.item(),'etas':[]}
 for eta in [.026,.2,.5,1.]:
  p=torch.zeros(J,dtype=torch.float64);p[target]=1.;raw=(1-eta)*bh+eta*mass*p;c=raw.clamp_max(b[0]);row['etas'].append({'eta':eta,'target_argmax':bool(c.argmax()==target),'answer_mass':(c[target]/c.sum()).item(),'capped':bool((raw>b[0]).any())})
 r.append(row)
# Concrete counterexample: low eta can change allocation even if argmax stays recent.
p1=torch.zeros(J,dtype=torch.float64);p1[0]=1.;p2=torch.zeros_like(p1);p2[-1]=1.
c1=((1-.026)*bh+.026*mass*p1).clamp_max(b[0]);c2=((1-.026)*bh+.026*mass*p2).clamp_max(b[0])
report={'source_root':str(root),'checkpoint':str(state),'split':'val','sequences':B,'recall_queries':int(counts.sum()),'aggregation':'embedding-unit mean then recall-query mean per sequence then sequence mean; argmax uses first-index tie break','eta':sel.eta.item(),'same_sample_metrics':metrics,'projected_key_shape':list(keys.shape),'participation_ratio_uncentered_feature_gram':{'mean':pr.mean().item(),'min':pr.min().item(),'max':pr.max().item(),'ceiling':keys.shape[-1]},'fractional_kernel':{'alpha':a.alpha,'b0':b[0].item(),'b1':b[1].item(),'B41':mass.item()},'onehot_policy_bound_examples':r,'low_eta_coefficient_change_L1':(c1-c2).abs().sum().item(),'training':'not run'}
out.write_text(json.dumps(report,indent=2)+'\n');print(out.read_text())
