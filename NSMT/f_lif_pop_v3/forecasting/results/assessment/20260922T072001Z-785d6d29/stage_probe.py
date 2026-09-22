"""Read-only snapshot CPU forward audit; no training/backward/optimizer."""
import sys,json,hashlib,importlib.util
from pathlib import Path
from types import SimpleNamespace
import torch,numpy as np
root,out=map(Path,sys.argv[1:3]); src=root/'f_lif_pop_v3/forecasting'; sys.path.insert(0,str(src)); torch.set_num_threads(2); torch.manual_seed(7)
from config import Config
from model import LOAD_MODEL
from data_provider.synthetic import Dataset_Recall
from torch.utils.data import DataLoader
from test import evaluate,selection_diagnostics
from utils import parameter_hash
p=root/'f_lif_pop_v3/analysis/stage_decomposition.py'; spec=importlib.util.spec_from_file_location('stages',p); stages=importlib.util.module_from_spec(spec);spec.loader.exec_module(stages)
# Add two missing diagnostic stages in an audit-only copy, preserving original aggregation.
s=p.read_text().replace("'kernel_mass', 'uniform_slot')","'kernel_mass', 'uniform_slot', 'weighted_p_mass', 'random_best_rank', 'mixture_error', 'coeff_error')")
s=s.replace("'uniform_slot': (answer.sum(-1).double() / n),", """'uniform_slot': (answer.sum(-1).double() / n),
                'weighted_p_mass': (((p*b_hist)*mask).sum(-1)/(p*b_hist).sum(-1).clamp_min(1e-12)).mean(-1).double(),
                'random_best_rank': (n-answer.sum(-1).double()) / ((answer.sum(-1).double()+1)*max(n-1,1)),
                'mixture_error': (((raw*mask).sum(-1)/raw.sum(-1).clamp_min(1e-12)) - ((1-sel.eta)*(b_hist*mask).sum(-1)/b_hist.sum()+sel.eta*((p*b_hist)*mask).sum(-1)/(p*b_hist).sum(-1).clamp_min(1e-12))).abs().amax(-1).double(),
                'coeff_error': (c-aux['coeff'][n]).abs().amax((1,2)).double(),""")
env={'__name__':'audit_stages','__file__':str(p)};exec(compile(s,'audit_extended_stages','exec'),env)
r={'stage':[],'intervention':[]}
def load(suite,variant):
 state=next((src/'log'/suite).glob('**/'+variant+'/model_state'));a=Config();a.load_args(str(state.parent),SimpleNamespace(cpu=True,num_device=0));a.save_model_state_path=str(state);m=LOAD_MODEL[a.model](a,False);m.eval();ds=Dataset_Recall(a,'test');dl=DataLoader(ds,batch_size=a.batch_size);return a,m,ds,dl,state
cases=[('etagrid-113240','seed7_flatten_spike_heterogeneous_sparse'+suffix+'_k3') for suffix in ['', '_eta0.2','_eta0.5','_eta1']]+[('qk2-140541','seed7_flatten_spike_heterogeneous_sparse_qknorm'+suffix+'_k3') for suffix in ['', '_eta0.5']]
for suite,v in cases:
 a,m,ds,dl,state=load(suite,v);z=env['decompose'](m,dl,a);rec={'suite':suite,'variant':v,'checkpoint_sha256':hashlib.sha256((state/'best+model.pt').read_bytes()).hexdigest(),'eta':m.embedding.neuron.selector.eta.item(),'theta':a.theta,'qk_eps':a.qk_eps,'stages':z};r['stage'].append(rec);print('stage',v,z,flush=True);out.write_text(json.dumps(r,indent=2))
for suite,v in [cases[0],cases[4]]:
 a,m,ds,dl,state=load(suite,v);h=hashlib.sha256(b''.join(x.detach().cpu().numpy().tobytes() for x in m.parameters())).hexdigest()
 for eta in [None,.2,.5,1.]:
  m.embedding.neuron.selector.eta_value.fill_(float('nan') if eta is None else eta)
  ev=evaluate(m,dl,a)[0];dg=selection_diagnostics(m,dl,a);z=env['decompose'](m,dl,a)
  unchanged=hashlib.sha256(b''.join(x.detach().cpu().numpy().tobytes() for x in m.parameters())).hexdigest()==h
  rec={'suite':suite,'eta_override':eta,'eta':m.embedding.neuron.selector.eta.item(),'recall_mse':ev['recall']['mse'],'diagnostics':dg,'stages':z,'trainable_parameters_unchanged':unchanged};r['intervention'].append(rec);print('intervention',suite,eta,rec['recall_mse'],dg['m_eff'],dg['hit'],'score',z['score_top1'],'weighted',z['weighted_p_mass'],flush=True);out.write_text(json.dumps(r,indent=2))
print('COMPLETE',flush=True)
