"""CPU audit of captured calibration and diagnostic blocks; no model training."""
import sys,json,ast,copy,hashlib,os
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime,timezone
import numpy as np
root,out=map(Path,sys.argv[1:3]);src=root/'f_lif_pop_v3/forecasting';sys.path.insert(0,str(src))
import train
from utils import write_json
art=out.parent
cal=src/'results/calibration/recall_k3_r2_a0.7_norm-frozen_qknorm-eps0.01_seed7_260922-151001.json';payload=json.loads(cal.read_text());a=SimpleNamespace(**payload['args']);train.TASK=src
r={'calibration':[]}
for e in [.01,1.0]:
 aa=copy.copy(a);aa.qk_eps=e
 try:res=train.load_calibration(aa);r['calibration'].append({'requested_eps':e,'natural_lookup':res})
 except Exception as err:r['calibration'].append({'requested_eps':e,'error':str(err)})
original_glob=train.glob.glob;train.glob.glob=lambda pattern:[str(cal)]
aa=copy.copy(a);aa.qk_eps=1.0
try:r['forced_mismatched_eps']=train.load_calibration(aa)
except ValueError as err:r['forced_mismatched_eps']={'rejected':True,'message':str(err)}
train.glob.glob=original_glob
r['calibration_source_hash_matches']={name:hashlib.sha256((src/name).read_bytes()).hexdigest()[:16]==h for name,h in payload['source_sha256_16'].items()}
# Extract actual standalone append calls and reduction from the function AST.
tree=ast.parse((src/'train.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train_one_epoch');loop=next(n for n in fn.body if isinstance(n,ast.For));appends=[n for n in loop.body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and 'every_batch[' in ast.unparse(n)]
reducer=next(n for n in fn.body if isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='rate')
def run_nodes(nodes,env):
 m=ast.Module(body=nodes,type_ignores=[]);ast.fix_missing_locations(m);exec(compile(m,'snapshot-diagnostic-block','exec'),env)
class Number:
 def __init__(self,x):self.x=x
 def item(self):return self.x
env={'np':np,'every_batch':{'grad_total_norm_pre':[],'clip_rate':[]},'watched':{},'selector_grad':{'grad_absmax_all':[1.,9.]},'result':{},'rate':[.1,.2],'peak':1.}
for x in [.5,2.,.9,1.2,1.,.2,3.,.6]:env['pre_clip']=Number(x);run_nodes(appends,env)
run_nodes([reducer],env);r['aggregation']={'input_pre_norms':env['every_batch']['grad_total_norm_pre'],'result':env['result'],'unconditional_append_nodes':len(appends)}
# Exercise only the exception-record block on synthetic numbers, never the training loop.
guard=next(n for n in ast.walk(fn) if isinstance(n,ast.If) and 'args.g11_bound' in ast.unparse(n.test) and 'peak' in ast.unparse(n.test))
auditdir=art/'synthetic_g11';auditdir.mkdir(exist_ok=True)
args=SimpleNamespace(g11_bound=10.,run_id='AUDIT_SYNTHETIC_G11_NOT_A_TRAINING_RUN',run_uuid='audit-only',current_epoch=3,calibration_file='AUDIT_SYNTHETIC',mode='sparse',eta_fixed=1.,qk_norm=True,qk_eps=.01,save_result_path=str(auditdir))
env={'args':args,'peak':11.,'i':7,'os':os,'datetime':datetime,'timezone':timezone,'write_json':write_json}
try:run_nodes([guard],env)
except FloatingPointError as err:r['synthetic_g11']={'exception':str(err),'record':json.loads((auditdir/'G11_violation.json').read_text())}
# Record config-to-epoch assignment and new JSON metadata fields statically.
r['current_epoch_assignment']=any(isinstance(n,ast.Assign) and any(ast.unparse(t)=='args.current_epoch' for t in n.targets) for n in ast.walk(tree))
r['json_fields_present']={k:k in (src/'train.py').read_text() for k in ['grad_total_norm_pre_mean','grad_total_norm_pre_max','clip_rate_all_batches','batches_seen']}
out.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');print(json.dumps(r,indent=2))
