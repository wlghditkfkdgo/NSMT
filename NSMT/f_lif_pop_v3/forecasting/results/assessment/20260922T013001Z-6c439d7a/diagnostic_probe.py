"""Frozen checkpoint/forward and isolated trainer-tail audit. No training or optimizer."""
import ast,copy,contextlib,hashlib,io,json,sys,tempfile,time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from torch.utils.data import DataLoader
root,out=map(lambda x:Path(x).resolve(),sys.argv[1:3]);src=root/'f_lif_pop_v3/forecasting';sys.path.insert(0,str(src))
import config,train
from model import LOAD_MODEL
from test import evaluate,selection_diagnostics
from data_provider.synthetic import Dataset_Recall
from utils import parameter_hash

torch.set_num_threads(2);torch.manual_seed(7);r={'snapshot':str(root),'torch':torch.__version__,'runs':[]}
for suite in ['pilot-eta-001742','diag3-102747']:
 state=next(p for p in (src/'log'/suite).glob('**/model_state') if 'analog' not in str(p))
 args=config.Config();args.load_args(str(state.parent),SimpleNamespace(cpu=True,num_device=0));args.save_model_state_path=str(state)
 model=LOAD_MODEL[args.model](args,False);dataset=Dataset_Recall(args,'test')
 loader=DataLoader(dataset,batch_size=128);ev=evaluate(model,loader,args)[0]
 d=selection_diagnostics(model,loader,args,batches=1000)
 dsmall=selection_diagnostics(model,DataLoader(dataset,batch_size=16),args,batches=1000)
 obs=next((src/'results'/suite).glob('*.json'));reported=json.loads(obs.read_text())
 row={'suite':suite,'metrics':ev,'diagnostics':d,'batch_size_16_primary':{k:dsmall[k] for k in ['m_eff','hit','kernel_mass','queries','sequences']},'reported_m_eff':reported['diagnostics']['m_eff'],'mse_difference':ev['mse']-reported['test']['mse'],'primary_batch_difference':max(abs(d[k]-dsmall[k]) for k in ['m_eff','hit','kernel_mass']),'config':{k:getattr(args,k,None) for k in ['epoch','n_train','n_val','n_test','seed','data_seed','theta','input_scale','input_norm','key_norm']}}
 r['runs'].append(row)
# Small ETT-like empty-truth forward; no ETT dataset/training is invoked.
et=copy.copy(args);et.task='forecasting';em=LOAD_MODEL['myModel'](et,True)
x=torch.randn(2,et.seq_len,1);empty=torch.empty(2,0,dtype=torch.bool)
r['empty_truth_diagnostics']=selection_diagnostics(em,[(x,torch.zeros(2,et.pred_len,1),empty,empty)],et)
# Fault-inject only temporary checkpoint copies. Real frozen stats must not default silently.
state_dict=torch.load(state/'best+model.pt',map_location='cpu');x=next(iter(loader))[0][:2].float()
with torch.no_grad(): original=model(x)
with tempfile.TemporaryDirectory(prefix='nsmt_audit_restore_') as tmp:
 a=copy.copy(args);a.save_model_state_path=tmp
 bad=dict(state_dict);removed={k:bad.pop(k) for k in ['embedding.norm_mean','embedding.norm_std']};torch.save(bad,Path(tmp)/'best+model.pt')
 cap=io.StringIO()
 with contextlib.redirect_stdout(cap): restored=LOAD_MODEL[a.model](a,False)
 with torch.no_grad(): delta=(restored(x)-original).abs().max().item()
 r['missing_frozen_stats']={'accepted':True,'input_norm':a.input_norm,'max_prediction_difference':delta,'warning':cap.getvalue(),'original_stats_nonidentity':bool(removed['embedding.norm_mean'].abs().max()>0 or (removed['embedding.norm_std']-1).abs().max()>0)}
 bad=dict(state_dict);key=next(n for n,_ in model.named_parameters());bad.pop(key);torch.save(bad,Path(tmp)/'best+model.pt')
 try: LOAD_MODEL[a.model](a,False);r['missing_parameter_rejected']=False
 except RuntimeError as ex:r['missing_parameter_rejected']=True;r['missing_parameter_error']=str(ex)
# Real calibration loader, no calibration/training execution.
r['calibration']={}
for name,updates in [('matching',{}),('tau_mismatch',{'tau':[40,80,160,320]}),('cue_mismatch',{'cue_mode':'index'}),('key_norm_changed',{'key_norm':'frozen'})]:
 a=copy.copy(args)
 for k,v in updates.items():setattr(a,k,v)
 try:r['calibration'][name]={'accepted':True,'result':train.load_calibration(a)}
 except ValueError as ex:r['calibration'][name]={'accepted':False,'error':str(ex)}
# Execute ONLY the existing post-training AST tail, with a perturbed last-epoch model.
# Neither train() nor train_one_epoch() is called. Test/data loaders are raising sentinels.
fun=next(n for n in ast.parse((src/'train.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='train')
start=next(i for i,n in enumerate(fun.body) if isinstance(n,ast.If) and ast.unparse(n.test)=='not args.test')
code=compile(ast.fix_missing_locations(ast.Module(body=[n for n in fun.body[start:] if not isinstance(n,ast.Return)],type_ignores=[])),'<frozen trainer post-training tail>','exec')
with tempfile.TemporaryDirectory(prefix='nsmt_audit_tail_') as tmp:
 a=copy.copy(args);a.test=False;a.result_path=str(Path(tmp)/'result.json')
 with torch.no_grad(): next(model.parameters()).add_(0.1)
 last=parameter_hash(model);best=parameter_hash(LOAD_MODEL[a.model](a,False))
 ns=dict(vars(train));ns.update(args=a,model=model,epoch=0,started=time.time(),stopper=SimpleNamespace(val_loss_min=.2),train_result={},active=[],train_set=range(2),val_set=range(2))
 def forbidden(*a,**k):raise AssertionError('test split touched under --no-test')
 ns.update(test=forbidden,data_provider=forbidden)
 exec(code,ns);p=ns['payload'];r['post_training_tail']={'test_skipped':p['test_skipped'],'test':p['test'],'last_differs_best':last!=best,'last_hash_correct':p['provenance']['parameter_hash_last_epoch']==last,'evaluated_hash_correct':p['provenance']['parameter_hash_evaluated']==best,'checkpoint_hash_correct':p['provenance']['checkpoint_sha256']==hashlib.sha256((state/'best+model.pt').read_bytes()).hexdigest(),'test_split_untouched':True,'training_invoked':False}
# Execute only the guard, also before any optimizer exists.
guard=next(n for n in fun.body if isinstance(n,ast.If) and ast.unparse(n.test)=='args.g11_bound')
a=copy.copy(args);a.g11_bound=None;a.require_calibration=True
try:exec(compile(ast.fix_missing_locations(ast.Module(body=[guard],type_ignores=[])),'<guard>','exec'),{'args':a});r['missing_calibration_guard_rejected']=False
except ValueError:r['missing_calibration_guard_rejected']=True
r['require_calibration_default']=config.parse_defaults().require_calibration
out.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');print(out.read_text())
