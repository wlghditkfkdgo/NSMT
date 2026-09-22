import importlib.util,json,sys
from pathlib import Path
import numpy as np
snap,art=map(Path,sys.argv[1:3]);spec=importlib.util.spec_from_file_location('m',snap/'f_lif_pop_v3/analysis/meff_reachable.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
r=json.loads((art/'reachability_probes.json').read_text())['finite_search_example'][0];idx=np.array(r['answers']);eta=r['eta'];remain=eta*m.MASS;allocation=np.zeros(m.N)
for i in idx:
 take=min(remain,m.B0-(1-eta)*m.PAST[i]);allocation[i]+=take;remain-=take
allocation[idx[0]]+=remain
p=allocation/m.PAST;p/=p.sum();actual=m.m_eff(eta,p,idx)
r.update({'policy':p.tolist(),'policy_sum':float(p.sum()),'constructed_mass':float(actual),'difference_from_exact':float(actual-r['exact'])})
(art/'constructive_bound.json').write_text(json.dumps(r,indent=2)+'\n');print('constructed',actual,'exact',r['exact'],'search4000',r['search_4000'],'gap',actual-r['search_4000'])
