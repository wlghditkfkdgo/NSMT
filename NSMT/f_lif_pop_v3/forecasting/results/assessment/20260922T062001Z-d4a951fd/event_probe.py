"""Forward-only inspection of stopped run's epoch-0 checkpoint; no training."""
import sys,json,csv,hashlib,math
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.utils.data import DataLoader
root,out=map(Path,sys.argv[1:3]);src=root/'f_lif_pop_v3/forecasting';sys.path.insert(0,str(src))
from config import Config
from model import LOAD_MODEL
from data_provider.synthetic import Dataset_Recall

torch.set_num_threads(2);torch.manual_seed(7)
event_path=next((src/'log/g11chk-151001').glob('**/G11_violation.json'));folder=event_path.parent;event=json.loads(event_path.read_text());a=Config();a.load_args(str(folder),SimpleNamespace(cpu=True,num_device=0));a.save_model_state_path=str(folder/'model_state');m=LOAD_MODEL[a.model](a,False);m.eval();ds=Dataset_Recall(a,'train');rows=list(csv.DictReader((folder/'log/best_log_0.csv').open()));cal=src/'results/calibration'/event['calibration_file'];payload=json.loads(cal.read_text());peaks=[]
with torch.no_grad():
 for x,y,truth,kind in DataLoader(ds,batch_size=64,shuffle=False):
  _,aux=m(x,mode=a.mode,return_aux=True)
  peaks.extend(aux['state'].abs().amax(dim=(0,2,3)).tolist())
diffs=[abs(v-event['max_abs_state']) for v in peaks];i=min(range(len(diffs)),key=diffs.__getitem__)
r={'event':event,'config':{k:getattr(a,k,None) for k in ['run_id','run_uuid','seed','data_seed','n_train','n_val','n_test','epoch','batch_size','g11_every','max_train_batches','qk_norm','qk_eps','eta_fixed','theta','calibration_file','g11_bound']},'csv':rows,'event_matches_config':{k:event[k]==getattr(a,k,None) for k in ['run_id','run_uuid','mode','eta_fixed','qk_norm','qk_eps','calibration_file']},'bound_matches_calibration':event['bound']==payload['picked']['declared_bound'],'strict_exceedance':event['max_abs_state']>event['bound'],'checkpoint_sha256':hashlib.sha256((folder/'model_state/best+model.pt').read_bytes()).hexdigest(),'csv_completed_epochs':len(rows),'expected_batches_per_epoch':math.ceil(len(ds)/a.batch_size),'forward_only_train_sequence_peaks':peaks,'forward_max':max(peaks),'count_above_bound':sum(v>event['bound'] for v in peaks),'closest_event_value':{'sequence_index':i,'peak':peaks[i],'absolute_difference':diffs[i]},'completed_result_json_exists':(src/'results/g11chk-151001'/(a.run_id+'.json')).exists(),'scope':'fixed epoch-0 best checkpoint, original train sequences without shuffling; NOT optimizer replay or task performance evaluation'}
out.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='forward_only_train_sequence_peaks'},indent=2))
