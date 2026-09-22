"""CPU re-evaluation of an existing GRU checkpoint and temporary path checks."""
import copy,json,sys,tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from torch.utils.data import DataLoader
root,output=map(lambda p:Path(p).resolve(),sys.argv[1:3]);source=root/'f_lif_pop_v3/forecasting';sys.path.insert(0,str(source))
from model import LOAD_MODEL
from test import evaluate
from data_provider.synthetic import Dataset_Recall
from utils import parameter_hash
import config

torch.set_num_threads(2);torch.manual_seed(7)
state=next((source/'log/pilot-gru-034153').glob('**/model_state'));saved=torch.load(state/'config.pt',map_location='cpu');args=SimpleNamespace(**saved);args.device=torch.device('cpu');args.save_model_state_path=str(state)
model=LOAD_MODEL[args.model](args,False);dataset=Dataset_Recall(args,'test');loader=DataLoader(dataset,batch_size=128,shuffle=False);metrics=evaluate(model,loader,args)[0];observed=json.loads(next((source/'results/pilot-gru-034153').glob('*.json')).read_text())
x,y,truth,kind=next(iter(DataLoader(dataset,batch_size=2)))
with torch.no_grad():
 z=model(x);moved=x.clone();moved[:,21*args.patch_size:]+=10
 causal=float((z[:,:21]-model(moved)[:,:21]).abs().max());truthdiff=float((model(x,truth=truth)-model(x,truth=~truth)).abs().max())
r={'torch':torch.__version__,'settings':{k:saved[k] for k in ['epoch','batch_size','n_train','n_val','n_test','data_seed','seed','lr','weight_decay','embed_dim','patch_size','readout']},'reevaluated':metrics,'max_mse_abs_difference':max(abs(metrics[k]['mse']-observed['test'][k]['mse']) for k in ['all','copy','recall','recall_first']),'parameter_hash':parameter_hash(model),'parameter_hash_matches':parameter_hash(model)==observed['provenance']['parameter_hash'],'causal_earlier_output_max_error':causal,'truth_change_max_error':truthdiff}
# Counts refer to modules used by the recall architecture, not a matched-capacity experiment.
other_args=copy.copy(args);other_args.model='myModel';other=LOAD_MODEL['myModel'](other_args,True)
r['parameter_counts']={'gru_total':sum(p.numel() for p in model.parameters()),'gru_recall_modules':sum(p.numel() for module in [model.gru,model.recall_head] for p in module.parameters()),'myModel_total':sum(p.numel() for p in other.parameters()),'myModel_recall_modules_including_selector':sum(p.numel() for module in [other.embedding,other.recall_head] for p in module.parameters())}
with tempfile.TemporaryDirectory(prefix='nsmt_run_id_audit_') as tmp:
 with patch.object(config,'TASK',Path(tmp)):
  cases={}
  for name,updates in [('spike',{'model':'myModel','readout':'spike'}),('analog',{'model':'myModel','readout':'analog'}),('GRU',{'model':'GRU','readout':'spike'})]:
   a=copy.copy(args)
   for k,v in updates.items():setattr(a,k,v)
   a.cpu=True;a.suite='audit_only'
   c=config.Config()
   try:c.set_args(a);error=None
   except FileExistsError as e:error=str(e)
   cases[name]={'run_id':c.run_id,'result_path':str(Path(c.result_path).relative_to(tmp)),'save_result_path':str(Path(c.save_result_path).relative_to(tmp)),'collision':error}
  r['path_cases']=cases
output.write_text(json.dumps(r,indent=2)+'\n');print(output.read_text())
