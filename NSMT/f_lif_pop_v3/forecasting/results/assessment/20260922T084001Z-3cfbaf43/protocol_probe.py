"""Mocked protocol audit. Does not load models or generate/evaluate real confirm data."""
import ast,json,sys,types,io,contextlib,hashlib
from pathlib import Path
import numpy as np
import torch
root,out=map(Path,sys.argv[1:3]);p=root/'f_lif_pop_v3/analysis/eta_selection.py';tree=ast.parse(p.read_text());code=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) or isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ['CANDIDATES','SELECTION_SEED'] for t in n.targets)],type_ignores=[]);ast.fix_missing_locations(code)
env={'__file__':str(p),'np':np,'torch':torch,'Path':Path,'json':json};exec(compile(code,str(p),'exec'),env)
result={'recall':{'mse':1.},'copy':{'mse':2.},'recall_first':{'mse':3.}}
diag={'m_eff':.2,'hit':.3,'kernel_mass':.1,'support_p':.4,'max_abs_state':5.,'finite':False};r={'finite_false_row':env['row']('candidate',.2,result,diag,10.),'missing_bound_row':env['row']('candidate',.2,result,diag,None)}
# Exact at_eta function: evaluation sees full loader, diagnosis receives batches=4.
seen={};fake=types.SimpleNamespace(embedding=types.SimpleNamespace(neuron=types.SimpleNamespace(selector=types.SimpleNamespace(eta_value=torch.tensor(float('nan'))))))
def ev(m,dl,c):seen['evaluate_loader_batches']=len(dl);return result,None
def dg(m,dl,c,**kw):seen['diagnostic_batch_cap']=kw['batches'];return diag
env.update(evaluate=ev,selection_diagnostics=dg);env['at_eta'](fake,list(range(16)),None,.2);r['at_eta_scope']={**seen,'buffer_restored':bool(torch.isnan(fake.embedding.neuron.selector.eta_value))}
# Entire CLI control flow with synthetic metrics and fake paths only.
class Parser:
 def __init__(self,*a,**kw):pass
 def add_argument(self,*a,**kw):pass
 def parse_args(self):return types.SimpleNamespace(suite='AUDIT_MOCK_ONLY',out=None)
for name,seeds,peak in [('one_seed_accepted',[7],5.),('all_confirm_fail_still_significant',[7,13,21,42,123,256,512,1024],20.)]:
 phases=[];cfg=types.SimpleNamespace(num_device=0,eta_fixed=None,g11_bound=10.)
 def load(run):
  seed=int(run.split('seed')[-1].split('_')[0]);cfg2=types.SimpleNamespace(**vars(cfg),seed=seed)
  m=types.SimpleNamespace(embedding=types.SimpleNamespace(neuron=types.SimpleNamespace(selector=types.SimpleNamespace(eta_hat=torch.tensor(-4.)))))
  return m,cfg2
 def provider(c,flag):phases.append([c.seed,flag]);return None,flag
 def at(m,flag,c,eta):
  error=1. if eta==0 else .9 if eta is None else .2+float(eta)+c.seed*1e-5
  z={k:{'mse':error} for k in ['recall','copy','recall_first']};d={**diag,'finite':True,'max_abs_state':peak if flag=='confirm' else 5.};return z,d
 env.update(argparse=types.SimpleNamespace(ArgumentParser=Parser),glob=types.SimpleNamespace(glob=lambda pat:['/AUDIT/seed'+str(x)+'_mock' for x in seeds]),load=load,data_provider=provider,at_eta=at)
 buf=io.StringIO()
 with contextlib.redirect_stdout(buf),np.errstate(all='ignore'):env['main']()
 r[name]={'mock_phases':phases,'stdout':buf.getvalue()}
# Dataset constructor seed/count prefix only: stop before allocation or generation.
t=ast.parse((root/'f_lif_pop_v3/forecasting/data_provider/synthetic.py').read_text());cls=next(x for x in t.body if isinstance(x,ast.ClassDef) and x.name=='Dataset_Recall');fn=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='__init__');prefix=[]
for n in fn.body:
 prefix.append(n)
 if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='rng' for x in n.targets):break
mod=ast.Module(body=prefix,type_ignores=[]);ast.fix_missing_locations(mod);r['split_seed_and_count']={}
for flag in ['train','val','test','confirm']:
 e={'args':types.SimpleNamespace(n_train=2048,n_val=256,n_test=256,n_confirm=1000,data_seed=20260921),'flag':flag,'np':types.SimpleNamespace(random=types.SimpleNamespace(default_rng=lambda x:x))};exec(compile(mod,'dataset-prefix','exec'),e);r['split_seed_and_count'][flag]={'rng_seed':e['rng'],'count':e['count']}
src=root/'f_lif_pop_v3/forecasting';r['new_run_metadata']=[]
for f in (src/'log/seeds-173737').glob('**/config.pt'):
 a=torch.load(f,map_location='cpu');r['new_run_metadata'].append({k:a.get(k) for k in ['seed','data_seed','n_train','n_val','n_test','n_confirm','epoch','batch_size','test','g11_every']})
j=json.loads(next((src/'results/seeds-173737').glob('*.json')).read_text());r['seed7_json']={'test':j['test'],'test_skipped':j.get('test_skipped'),'train':j['train'],'provenance':j['provenance']}
out.write_text(json.dumps(r,indent=2,allow_nan=False));print(json.dumps({k:v for k,v in r.items() if k!='seed7_json'},indent=2))
