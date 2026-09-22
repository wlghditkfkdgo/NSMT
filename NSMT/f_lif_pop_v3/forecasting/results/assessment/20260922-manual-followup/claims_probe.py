"""Small CPU algebra probes; no checkpoint/training/GPU access."""
import json,sys
from pathlib import Path
import torch
root,out=map(lambda x:Path(x).resolve(),sys.argv[1:3]);sys.path.insert(0,str(root/'f_lif_pop_v3/forecasting'))
from layers import fractional_coefficients,Selector

torch.set_num_threads(2);torch.manual_seed(7)
b=fractional_coefficients(.7,42,dtype=torch.float64);bh=b[1:42].flip(0);B=bh.sum();rows=[]
for lag in [2,5,10,20,41]:
 target=41-lag;p=torch.zeros(1,1,41,dtype=torch.float64);p[...,target]=1.
 for eta in [.2,.5,.9,.92,.93,.95,.99,1.]:
  s=Selector(4,eta_fixed=eta).double();xi=torch.zeros(1,1,5,dtype=torch.float64);hist=torch.zeros(1,1,41,5,dtype=torch.float64)
  with torch.no_grad():c,aux=s(xi,hist,bh,b[0].item(),mode='oracle',oracle_p=p)
  direct=(b[0]/(b[0]+(1-eta)*(B-b[lag]))).item()
  rows.append({'lag':lag,'eta':eta,'postcap_answer_mass':(c[...,target]/c.sum(-1)).item(),'target_capped':bool(c[...,target]>=b[0]-1e-10),'closed_form_when_capped':direct,'eta_boundary_for_mass_half_when_capped':(1-b[0]/(B-b[lag])).item()})
# Each member has rank <= 4; averaging token Grams can exceed four.
z=torch.randn(64,42,4,dtype=torch.float64);z=z/z.norm(dim=-1,keepdim=True);g=z@z.transpose(-1,-2)
def pr(m):
 e=torch.linalg.eigvalsh(m).clamp_min(0);return e.sum(-1).square()/e.square().sum(-1)
report={'torch':torch.__version__,'seed':7,'alpha':.7,'history':41,'b0':b[0].item(),'B':B.item(),'onehot_policy_rows':rows,'random_geometry_example':{'shape':list(z.shape),'normalization':'unit L2 per token','individual_token_rank_max':int(torch.linalg.matrix_rank(g).max()),'mean_individual_token_PR':pr(g).mean().item(),'averaged_token_rank':int(torch.linalg.matrix_rank(g.mean(0))),'PR_of_averaged_token_Gram':pr(g.mean(0)).item(),'not_a_reproduction_of_agent_35_67':True},'training':'not run'}
out.write_text(json.dumps(report,indent=2)+'\n');print(out.read_text())
