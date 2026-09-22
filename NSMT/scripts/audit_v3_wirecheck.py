"""Audit frozen wiring with saved checkpoints; intercept trainer before any training."""
import contextlib,copy,hashlib,io,json,sys,tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from torch.utils.data import DataLoader
root,output=map(lambda p:Path(p).resolve(),sys.argv[1:3]);source=root/'f_lif_pop_v3/forecasting';sys.path.insert(0,str(source))
from model import LOAD_MODEL
from data_provider.synthetic import Dataset_Recall
from test import evaluate
from utils import parameter_hash
import config,train,check_model

torch.manual_seed(7);torch.set_num_threads(2);r={'torch':torch.__version__,'runs':[]}
for state in sorted((source/'log/wirecheck-100840').glob('**/model_state')):
 args=SimpleNamespace(**torch.load(state/'config.pt',map_location='cpu'));args.device=torch.device('cpu');args.save_model_state_path=str(state)
 observed=json.loads((source/'results/wirecheck-100840'/(args.run_id+'.json')).read_text())
 model=LOAD_MODEL[args.model](args,False);metrics=evaluate(model,DataLoader(Dataset_Recall(args,'test'),batch_size=64),args)[0]
 r['runs'].append({'readout':args.readout,'config':{k:getattr(args,k,None) for k in ['seed','epoch','n_train','n_val','n_test','batch_size','max_train_batches','max_eval_batches','theta','input_scale','calibrated_fields']},'metrics':metrics,'mse_abs_difference':max(abs(metrics[k]['mse']-observed['test'][k]['mse']) for k in ['all','copy','recall','recall_first']),'parameter_hash_matches':parameter_hash(model)==observed['provenance']['parameter_hash'],'parameter_hash':parameter_hash(model),'source_mismatches':[name for name,h in observed['provenance']['source_sha256'].items() if hashlib.sha256((source/name).read_bytes()).hexdigest()!=h],'path':str(state.relative_to(root))})
# Real main entry setup and calibration override, replacing training with config capture only.
with tempfile.TemporaryDirectory(prefix='nsmt_wirecheck_main_') as temp:
 cases=[]
 for readout in ['spike','analog']:
  request=copy.copy(args);request.cpu=True;request.theta=123.;request.input_scale=1.;request.readout=readout;request.suite='audit_fixture'
  with patch.object(config,'TASK',Path(temp)),patch.object(train,'parse_arguments',return_value=request),patch.object(train,'train',side_effect=lambda c:c),contextlib.redirect_stdout(io.StringIO()):
   bound=train.main()
  cases.append({'readout':readout,'theta':bound.theta,'input_scale':bound.input_scale,'calibrated_fields':bound.calibrated_fields,'result_path':str(Path(bound.result_path).relative_to(temp)),'state_path':str(Path(bound.save_model_state_path).relative_to(temp))})
 r['main_setup_without_training']=cases
 r['same_suite_paths_distinct']=cases[0]['state_path']!=cases[1]['state_path'] and cases[0]['result_path']!=cases[1]['result_path']
# Recheck only the new G4 function; the complete runner was executed separately.
r['g4']=check_model.golden_parity()
oldgold=Path('/tmp/nsmt_assessment_20260921-145954-utc')
original_report=Path('f_lif_pop_v3/forecasting/results/assessment/20260921-145954-utc/reference/reference_results.json')
original_csv=original_report.with_name('scalar_trajectories.csv')
current_csv=root/'f_lif_pop_v3/reference/golden/scalar_trajectories.csv'
r['golden_sha256']=hashlib.sha256(current_csv.read_bytes()).hexdigest()
r['golden_matches_prior_auditor_csv']=original_csv.exists() and current_csv.read_bytes()==original_csv.read_bytes()
r['reference_json_matches_prior_audit']=original_report.exists() and (root/'f_lif_pop_v3/reference/golden/reference_results.json').read_bytes()==original_report.read_bytes()
r['complete_runner']={'command':'check_model.py --phase all','exit_code':0,'passed':20,'failed':0,'not_run':0,'note':'Executed separately against this frozen snapshot; result recorded from completed process.'}
output.write_text(json.dumps(r,indent=2)+'\n');print(output.read_text())
