"""Snapshot AST diagnostics only: no train loop, backward or optimizer."""
import ast,json,sys,time,traceback
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
root,out=map(Path,sys.argv[1:3]);src=root/'f_lif_pop_v3/forecasting';torch.set_num_threads(2)
tree=ast.parse((src/'train.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train_one_epoch');reducer=next(n for n in fn.body if isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='rate')
def run(nodes,env):
 mod=ast.Module(body=nodes,type_ignores=[]);ast.fix_missing_locations(mod);exec(compile(mod,'snapshot-extracted','exec'),env)
env={'np':np,'result':{'loss':.5},'rate':[.1,.2],'peak':2.,'watched':{'eta':[.1,.3]},'selector_grad':{'grad_WQ_absmax':[1.,9.],'grad_WK_absmax':[2.,10.],'grad_absmax_all':[1.,9.],'grad_WQ_norm':[1.,9.],'singleton_frac':[0.,1.]},'every_batch':{'grad_total_norm_pre':[.5,5.,.5,5.],'grad_total_norm_post':[.5,1.,.5,1.],'clip_rate':[0.,1.,0.,1.]}}
run([reducer],env);r={'finite_reducer':env['result']};assert r['finite_reducer']['grad_absmax_all']==9.;assert r['finite_reducer']['grad_absmax_all_observations']==2
nr=dict(env);nr.update(result={},selector_grad={'grad_absmax_all':[1.,float('nan'),9.],'grad_WQ_absmax':[float('nan')]});run([reducer],nr);r['nonfinite_reducer']=nr['result']
loop=next(n for n in fn.body if isinstance(n,ast.For));start=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='pre_clip' for t in n.targets));end=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='step');clipnodes=loop.body[start:end];r['clip_probes']=[]
for values in [[3.,4.],[.3,.4]]:
 p=torch.nn.Parameter(torch.zeros(2));p.grad=torch.tensor(values);m=SimpleNamespace(parameters=lambda:[p]);e={'torch':torch,'model':m,'every_batch':{'grad_total_norm_pre':[],'grad_total_norm_post':[],'clip_rate':[]}};run(clipnodes,e);r['clip_probes'].append(e['every_batch']);assert abs(e['post']-min(np.linalg.norm(values),1))<1e-5
train=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train');assign=next(n for n in train.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Subscript) and isinstance(t.value,ast.Name) and t.value.id=='payload' and isinstance(t.slice,ast.Constant) and t.slice.value=='train' for t in n.targets));e={'payload':{},'train_result':env['result'],'epoch':0,'time':time,'started':time.time(),'stopper':SimpleNamespace(val_loss_min=.5)};run([assign],e);r['train_json_projection']=e['payload']['train'];assert r['train_json_projection']['grad_absmax_all_observations']==2
ut=ast.parse((src/'utils.py').read_text());cls=next(n for n in ut.body if isinstance(n,ast.ClassDef) and n.name=='EpochLog');e={'pd':pd};run([cls],e);Log=e['EpochLog'];logger=Log.__new__(Log);logger._logging=lambda *a,**k:None;logger._columns=None;logger._header_written=False;logger.save_log_path=str(out.with_name('synthetic_logger.csv'))
try:
 logger.write(epoch=0,lr=.001,train_result=env['result'],val_result={'loss':.5});r['logger_error']=None
except Exception as exc:
 r['logger_error']={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}
r['logger_csv_created']=Path(logger.save_log_path).exists();print('actual logger error',r['logger_error']);assert r['logger_error']['type']=='AttributeError';assert not r['logger_csv_created']
# Diagnostic type-only comparison, without modifying the source.
converted={k:float(v) for k,v in env['result'].items()};logger.write(epoch=0,lr=.001,train_result=converted,val_result={'loss':.5});r['float_conversion_logger_passed']=Path(logger.save_log_path).exists()
out.write_text(json.dumps(r,indent=2,allow_nan=False));print(json.dumps(r,indent=2))
