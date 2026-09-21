"""Read-only v3 audit probes. Usage: python audit_f_lif_pop_v3.py <NSMT root>.

No training, source edits, downloads, or checkpoint writes. Findings are measurements,
not assertions that a known faulty result should remain unchanged after a repair.
"""
import sys, json, hashlib
from pathlib import Path
import numpy as np
import torch

ROOT = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(ROOT / 'f_lif_pop_v3/forecasting'))
import layers
from data_provider.synthetic import make_sequence, rng_codes
torch.set_num_threads(2)
torch.manual_seed(7)
out = {'source_root': str(ROOT), 'torch': torch.__version__, 'seed': 7}
out['sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in (ROOT/'f_lif_pop_v3/forecasting').rglob('*.py')}

# A fixed history with one selected slot; compare post-cap mass on the same bank.
sel = layers.Selector().double()
with torch.no_grad():
    sel.eta_hat.fill_(0.)
xi = torch.zeros(1,1,5,dtype=torch.float64)
hist = torch.full((1,1,41,5), 100., dtype=torch.float64)
hist[...,0,:] = 0.
b = layers.fractional_coefficients(.7,42,torch.float64)
cs, auxs = sel(xi,hist,b[1:].flip(0),b[0].item(),'sparse')
cm, auxm = sel(xi,hist,b[1:].flip(0),b[0].item(),'mass_matched')
cf, _ = sel(xi,hist,b[1:].flip(0),b[0].item(),'full')
truth = torch.zeros_like(cs); truth[...,0] = 1.
actual = (cs*truth).sum()/cs.sum()
old_formula = .5*(b[-1]/b[1:].sum())+.5
out['mass_matched'] = {'sparse_kappa': auxs['kappa'].item(), 'control_kappa':auxm['kappa'].item(),
    'control_vs_full_max_error':(cm-cf).abs().max().item(),
    'post_cap_mass_mismatch':(cm.sum()-cs.sum()).item(),
    'oracle_effective_mass_actual':actual.item(), 'uncapped_formula':old_formula.item()}

# Exact per-query chance mass; sequence-mean then dataset-mean.
rows=[]
for nk in (3,5,8):
    rng=np.random.default_rng(20260921)
    vals=[]; active=[]; last=[]; frac=[]; approx=[]; base=[]; m=[]
    for i in range(300):
        x,y,t,r = make_sequence(rng,n_keys=nk,cue_mode='code')
        q=np.flatnonzero(r)
        if not len(q): continue
        counts=t[q].sum(-1)
        vals.append(np.mean(counts/q))
        approx.append(counts.mean()/21.)
        frac.append(r.mean())
        p=x.reshape(42,8)
        codes=rng_codes(nk,4)
        key=np.linalg.norm(p[:,:4,None]-codes.T[None,:,:],axis=1).argmin(-1)
        active.append(len(np.unique(key[r])))
        edge=np.r_[0,np.flatnonzero(np.diff(key))+1]
        runs=key[edge]
        last.append(float(len(set(runs[:max(0,nk-4)]).intersection(set(key[r])))))
        for n in q:
            bh=b[1:n+1].flip(0).numpy()
            base.append(bh[t[n,:n]].sum()/bh.sum())
        # A perfect privileged p, with eta=0.5; compare clipped coefficient allocation.
        for n in q:
            bh=b[1:n+1].flip(0).numpy(); pp=t[n,:n]/t[n,:n].sum()
            rho=.5+.5*bh.sum()*pp/(bh*pp).sum()
            c=np.minimum(bh*rho,b[0].item())
            m.append(float(c[t[n,:n]].sum()/c.sum()))
    rows.append({'n_keys':nk,'sequences_generated':300,'sequences_with_recall':len(vals),
        'exact_uniform_chance_seq_mean':float(np.mean(vals)),
        'reported_approximation':float(np.mean(approx)), 'recall_fraction':float(np.mean(frac)),
        'mean_unique_queried_keys':float(np.mean(active)),
        'max_old_intro_keys_recalled':float(np.max(last)),
        'full_fractional_mass_query_mean':float(np.mean(base)),
        'perfect_oracle_eta_half_actual_mass_query_mean':float(np.mean(m))})
out['synthetic']=rows

# Actual projected/normalized current, not raw input times scale.
rng=np.random.default_rng(20260921)
data=torch.from_numpy(np.stack([make_sequence(rng)[0] for _ in range(256)]))
patch=layers.to_patches(data,8)
emb=layers.Embedding(8,32,8.,input_norm='frozen',max_length=42)
emb.fit_norm(patch)
with torch.no_grad():
    current=emb.current(patch)
    spikes,aux=emb(patch,mode='full',return_aux=True)
out['drive']={'sample_size':256,'raw_patch_times_scale':float(patch.abs().max()*8),
    'actual_max_I':float(current.abs().max()),'max_state':float(aux['state'].abs().max())}
out['diagnostic_keys']=sorted(aux)

# Check whether detached aux voltage/state could train the promised analog readout.
test=layers.PopulationNeuron(2,max_length=4)
x=torch.randn(4,1,2,requires_grad=True)
_,a=test(x,return_aux=True)
out['analog_aux_requires_grad']={k:a[k].requires_grad for k in ('state','voltage')}

# All CLI flags must reach the constructor (static payload diagnostic, no run dirs).
from types import SimpleNamespace
from config import neuron_kwargs
cfg=SimpleNamespace(num_population=4,alpha=.7,tau=[4.,8.,16.,32.],heterogeneous=True,
    num_patches=42,theta=1.,eta_init=-4.,tau_s=2.,threshold=1.,surrogate_scale=5.,
    cap=False,eta_fixed=1.)
out['constructor_payload_for_nocap_eta1']=neuron_kwargs(cfg)

print(json.dumps(out,ensure_ascii=False,indent=2))
