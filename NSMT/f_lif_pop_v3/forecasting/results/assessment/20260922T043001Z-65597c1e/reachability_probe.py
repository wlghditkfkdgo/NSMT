"""Bounded CPU check of snapshot analysis functions; no model training."""
import importlib.util,sys,json,hashlib
from pathlib import Path
import numpy as np
root,out=map(Path,sys.argv[1:3]);p=root/'f_lif_pop_v3/analysis/meff_reachable.py'
spec=importlib.util.spec_from_file_location('reachability_snapshot',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def exact(eta,idx):
 ba=m.PAST[idx].sum();s=min((1-eta)*ba+eta*m.MASS,len(idx)*m.B0)
 return float(s/(s+(1-eta)*(m.MASS-ba)))
r={'module_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'uniform_table':[],'thresholds':[]}
# Directly regenerate every printed policy cell, preserving the actual index construction.
for size in (1,2,3,4,6,10):
 for lag in (5,21,35):
  idx=np.arange(max(0,m.N-lag),min(max(0,m.N-lag)+size,m.N))
  if len(idx)<size:continue
  vals=[m.uniform_policy(e,idx) for e in (.2,.3,.5,.7,.9,1.)]
  line=f'{size:>8} {lag:>6} '+''.join(f'{v:>10.4f}' for v in vals)
  r['uniform_table'].append({'size':size,'lag':lag,'values':vals,'matches_saved_line':line in (p.with_suffix('.txt')).read_text()})
for size in (1,2,3,4,6,10):
 idx=np.arange(m.N-21,m.N-21+size);grid=np.arange(.01,1.001,.005)
 eu=next(e for e in grid if m.uniform_policy(e,idx)>=.5)
 ef=next(e for e in grid if exact(e,idx)>=.5)
 lo,hi=0.,1.
 for _ in range(60):
  mid=(lo+hi)/2
  if exact(mid,idx)>=.5:hi=mid
  else:lo=mid
 checks=[{'eta':float(e),'uniform':float(m.uniform_policy(e,idx)),'sampled_best_4000':float(m.free_bound(e,idx)),'exact':exact(e,idx)} for e in (ef-.005,ef)]
 r['thresholds'].append({'size':size,'uniform_grid_eta':float(eu),'exact_grid_eta':float(ef),'continuous_exact_eta':hi,'sampled_checks':checks})
# Independent formula versus the finite search: finite search can only lower-bound the optimum.
idx=np.array([0,40]);eta=.2;pol=np.zeros(m.N);pol[idx]=[.9174,.0826]
r['counterexample']={'uniform':float(m.uniform_policy(eta,idx)),'specified_policy':float(m.m_eff(eta,pol,idx)),'search_4000':float(m.free_bound(eta,idx)),'exact':exact(eta,idx)}
# Demonstrate a finite Dirichlet search misses the exact max on an admissible support.
rng=np.random.default_rng(17);examples=[]
for _ in range(40):
 idx=np.sort(rng.choice(m.N,size=8,replace=False));eta=float(rng.uniform(.1,.6));found=m.free_bound(eta,idx,tries=100,seed=0);u=exact(eta,idx)
 if u-found>1e-6:examples.append({'answers':idx.tolist(),'eta':eta,'search_100':float(found),'exact':u,'gap':float(u-found)})
 if len(examples)==1:break
if examples:
 e=examples[0];e['search_4000']=float(m.free_bound(e['eta'],np.array(e['answers'])));e['gap_4000']=e['exact']-e['search_4000']
r['finite_search_example']=examples
r['all_uniform_cells_match_saved']=all(x['matches_saved_line'] for x in r['uniform_table'])
out.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='uniform_table'},indent=2))
