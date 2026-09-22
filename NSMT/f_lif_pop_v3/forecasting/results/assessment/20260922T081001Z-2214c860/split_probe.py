"""CPU forward on snapshot; no training, optimizer or backward."""
import sys,json,hashlib,importlib.util,csv
from pathlib import Path
from types import SimpleNamespace
import torch
root,out=map(Path,sys.argv[1:3]);src=root/'f_lif_pop_v3/forecasting';sys.path.insert(0,str(src));torch.set_num_threads(2);torch.manual_seed(7)
from config import Config
from model import LOAD_MODEL
from data_provider.synthetic import Dataset_Recall
from torch.utils.data import DataLoader
from test import evaluate
from utils import parameter_hash

def mod(name):
 p=root/'f_lif_pop_v3/analysis'/(name+'.py');sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
split=mod('eta_intervention_split');stage=mod('stage_decomposition')
# Add the omitted kernel-weighted policy mass to audit-only function copy.
s=(root/'f_lif_pop_v3/analysis/eta_intervention_split.py').read_text().replace("FIELDS = ('score_top1', 'p_mass', 'precap_mass', 'postcap_mass')","FIELDS = ('score_top1', 'p_mass', 'weighted_p_mass', 'precap_mass', 'postcap_mass')").replace("'p_mass': ((p * mask)","'weighted_p_mass': (((p*b_hist)*mask).sum(-1)/(p*b_hist).sum(-1).clamp_min(1e-12)).mean(-1).double(),\n                'p_mass': ((p * mask)")
env={'__name__':'audit_split','__file__':str(root/'f_lif_pop_v3/analysis/eta_intervention_split.py')};exec(compile(s,'audit_split','exec'),env)
state=next((src/'log/qk2-140541').glob('**/seed7_flatten_spike_heterogeneous_sparse_qknorm_k3/model_state'));a=Config();a.load_args(str(state.parent),SimpleNamespace(cpu=True,num_device=0));a.save_model_state_path=str(state);a.max_eval_batches=0;m=LOAD_MODEL[a.model](a,False);m.eval();sel=m.embedding.neuron.selector;val=DataLoader(Dataset_Recall(a,'val'),batch_size=64);test=DataLoader(Dataset_Recall(a,'test'),batch_size=64);trained=sel.eta.item();initial=parameter_hash(m)
r={'checkpoint_sha256':hashlib.sha256((state/'best+model.pt').read_bytes()).hexdigest(),'config':{k:getattr(a,k,None) for k in ['seed','data_seed','n_train','n_val','n_test','batch_size','theta','qk_eps','g11_bound']},'trained_eta':trained,'rows':[],'revised_test_rank':stage.decompose(m,test,a)}
for fixed in [True,False]:
 for eta in [0.,.2,.5,1.]:
  z=env['stages'](m,val,a,eta,fixed);restored=bool(torch.isnan(sel.eta_value));mse=None
  if not fixed:
   sel.eta_value.fill_(eta);mse=evaluate(m,val,a)[0]['recall']['mse'];sel.eta_value.fill_(float('nan'))
  row={'trajectory_fixed':fixed,'eta':eta,'recall_mse':mse,'stages':z,'buffer_restored':restored};r['rows'].append(row);print(row,flush=True);out.write_text(json.dumps(r,indent=2))
r['trained_baseline']={'recall_mse':evaluate(m,val,a)[0]['recall']['mse'],'stages':env['stages'](m,val,a,None,False)};r['weights_unchanged']=parameter_hash(m)==initial;r['logging_runs']=[]
for suite in ['absmax2-170000','absmax4-170100']:
 state=next((src/'log'/suite).glob('**/model_state'));c=Config();c.load_args(str(state.parent),SimpleNamespace(cpu=True,num_device=0));c.save_model_state_path=str(state);j=json.loads((src/'results'/suite/(c.run_id+'.json')).read_text());rows=list(csv.DictReader((state.parent/'log/best_log_0.csv').open()));hashok=hashlib.sha256((state/'best+model.pt').read_bytes()).hexdigest()==j['provenance']['checkpoint_sha256'];r['logging_runs'].append({'suite':suite,'config':{k:getattr(c,k,None) for k in ['epoch','n_train','n_val','n_test','batch_size','g11_every','seed','data_seed']},'train_json':j['train'],'csv':rows,'checkpoint_hash_matches':hashok,'train_source_sha256':j['provenance']['source_sha256']['train.py']})
out.write_text(json.dumps(r,indent=2));print('baseline',r['trained_baseline']);print('COMPLETE')
