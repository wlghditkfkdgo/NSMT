"""Read-only CPU evaluation of frozen theta pilots and bounded schema probes."""
import ast, contextlib, io, json, sys, tempfile, hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import DataLoader
root, output = map(lambda x:Path(x).resolve(), sys.argv[1:3])
source=root/'f_lif_pop_v3/forecasting';sys.path.insert(0,str(source))
from model import LOAD_MODEL
from test import evaluate
from data_provider.synthetic import Dataset_Recall
from utils import parameter_hash, EpochLog
from train import load_calibration
from layers import PopulationNeuron

torch.set_num_threads(2);torch.manual_seed(7)
result={'torch':torch.__version__,'seed':7,'runs':[]}
for suite in ['pilot-theta-033202','pilot-oracle-033723']:
 for path in sorted((source/'results'/suite).glob('*.json')):
  observed=json.loads(path.read_text()); state=next((source/'log'/suite).glob('**/'+ ('seed7_flatten_heterogeneous_'+observed['mode']+('_eta'+('1' if 'eta1_' in path.name else '0.5') if '_eta' in path.name else '')+'_k3')+'/model_state'))
  args=SimpleNamespace(**torch.load(state/'config.pt',map_location='cpu'));args.device=torch.device('cpu');args.save_model_state_path=str(state)
  model=LOAD_MODEL[args.model](args,train=False)
  dataset=Dataset_Recall(args,'test');loader=DataLoader(dataset,batch_size=128,shuffle=False)
  metrics=evaluate(model,loader,args)[0]
  means=[]; baseline=[];counts=0
  with torch.no_grad():
   for x,y,truth,kind in loader:
    _,aux=model(x,mode=args.mode,truth=truth if args.mode=='oracle' else None,return_aux=True)
    count=torch.zeros(len(x),dtype=torch.float64);mass=count.clone();base=count.clone();b=model.embedding.neuron.b.double()
    for n in range(1,len(aux['coeff'])):
     c=aux['coeff'][n].double();ans=truth[:,n,:n];valid=(kind[:,n]>0)&ans.any(-1)
     share=((c*ans[:,None]).sum(-1)/c.sum(-1).clamp_min(1e-12)).mean(-1)
     kernel=(b[1:n+1].flip(0)*ans).sum(-1)/b[1:n+1].sum()
     mass[valid]+=share[valid];base[valid]+=kernel[valid];count[valid]+=1
    means.extend((mass/count).tolist());baseline.extend((base/count).tolist());counts+=int(count.sum())
  digest=parameter_hash(model)
  row={'suite':suite,'mode':args.mode,'eta_fixed':args.eta_fixed,'theta':args.theta,'key_norm':args.key_norm,'epoch_budget':args.epoch,'epochs_run':observed['train']['epochs_run'],'data':{k:getattr(args,k) for k in ['n_train','n_val','n_test','data_seed','seed','batch_size']},'recall_mse':metrics['recall']['mse'],'recall_first_mse':metrics['recall_first']['mse'],'all_mse':metrics['all']['mse'],'max_mse_abs_difference':max(abs(metrics[k]['mse']-observed['test'][k]['mse']) for k in ['all','copy','recall','recall_first']),'corrected_m_eff':float(np.mean(means)),'kernel_mass':float(np.mean(baseline)),'recall_events':counts,'reported_m_eff':observed['diagnostics']['m_eff'],'parameter_hash':digest,'parameter_hash_matches':digest==observed['provenance']['parameter_hash'],'source_mismatches':[name for name,h in observed['provenance']['source_sha256'].items() if hashlib.sha256((source/name).read_bytes()).hexdigest()!=h]}
  result['runs'].append(row)
calpath=source/'results/calibration/recall_k3_r2_a0.7_norm-frozen_seed7_260922-033146.json'
result['theta_loading']={'stored_theta':json.loads(calpath.read_text())['picked']['theta'],'load_calibration_return':load_calibration(args),'run_theta':args.theta}
# Execute only the pure result-dictionary block from train_one_epoch, no training function.
tree=ast.parse((source/'train.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train_one_epoch');start=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='result' for t in n.targets))
block=ast.Module(body=fn.body[start:-1],type_ignores=[])
ns={'torch':torch,'np':np,'total':torch.tensor([3.,4.,10.]),'rate':[],'peak':0.,'watched':{k:[] for k in ['eta','kappa','would_cap_rate','support_p']}}
exec(compile(block,'<pure-result-schema>','exec'),ns)
with tempfile.TemporaryDirectory(prefix='nsmt_gru_schema_') as temp:
 logger=EpochLog(temp)
 try:
  with contextlib.redirect_stdout(io.StringIO()):
   logger.logging(epoch=0,train_result=ns['result'],val_result={'loss':.3})
   logger.verbose(epoch=0,lr=.001,train_result=ns['result'],val_result={'loss':.3})
  result['gru_result_schema']={'fields':list(ns['result']),'none_values':any(v is None for v in ns['result'].values()),'logging_and_verbose_pass':True}
 finally:logger.close()
# Independently distinguish query-weighted versus pair-weighted score calibration.
neuron=PopulationNeuron(embed_dim=3,max_length=12).double();x=torch.randn(12,2,3,dtype=torch.float64)
with torch.no_grad():
 _,aux=neuron(x,mode='full',return_aux=True);state=aux['state'];sums=[];numbers=[];sel=neuron.selector
 for n in range(1,len(x)):
  q=torch.cat([state[n-1],x[n,...,None]],-1);hist=torch.stack([torch.cat([torch.zeros_like(state[0]) if j==0 else state[j-1],x[j,...,None]],-1) for j in range(n)],-2)
  square=(sel.query(q)[...,None,:]-sel.key(hist)).square().sum(-1)/sel.query_dim;sums.append(float(square.sum()));numbers.append(square.numel())
result['score_scale_aggregation']={'implementation':neuron.score_scale(x),'query_weighted':float(np.mean(np.array(sums)/numbers)),'pair_weighted':sum(sums)/sum(numbers)}
# Historical args did not contain newly required key_norm.
legacy=next((source/'log/pilot-eta-001742').glob('**/model_state'));la=SimpleNamespace(**torch.load(legacy/'config.pt',map_location='cpu'));la.device=torch.device('cpu');la.save_model_state_path=str(legacy)
try:LOAD_MODEL[la.model](la,train=False);result['legacy_load']={'success':True}
except Exception as e:result['legacy_load']={'success':False,'type':type(e).__name__,'detail':str(e)}
output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(output.read_text())
