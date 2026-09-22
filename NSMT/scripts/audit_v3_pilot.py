"""Read-only CPU audit of a frozen pipeline and existing smoke checkpoint.

Args: frozen NSMT root, output JSON. No optimizer steps or training entrypoints.
"""
import contextlib
import copy
import io
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

root, output = map(lambda p: Path(p).resolve(), sys.argv[1:3])
sys.path.insert(0, str(root / 'f_lif_pop_v3/forecasting'))
from model import LOAD_MODEL
from ours import myModel, truth_to_oracle_p
from test import evaluate, selection_diagnostics
from data_provider.synthetic import Dataset_Recall
from utils import EpochLog, parameter_hash
from train import load_calibration

torch.set_num_threads(2)
torch.manual_seed(7)
state_dir = next(root.glob('f_lif_pop_v3/forecasting/log/pilot-eta-001742/**/model_state'))
saved_args = torch.load(state_dir / 'config.pt', map_location='cpu')
args = SimpleNamespace(**saved_args)
args.device = torch.device('cpu')
args.save_model_state_path = str(state_dir)
model = LOAD_MODEL[args.model](args, train=False)
dataset = Dataset_Recall(args, 'test')
loader = DataLoader(dataset, batch_size=128, shuffle=False)
result = {'snapshot': str(root), 'torch': torch.__version__, 'seed': 7,
          'config': {k: str(saved_args[k]) for k in ['epoch', 'n_train', 'n_val', 'n_test',
                     'batch_size', 'max_train_batches', 'max_eval_batches', 'g11_every']},
          'checkpoint_parameter_hash': parameter_hash(model)}
result['reevaluated_errors'] = evaluate(model, loader, args)[0]
result['reported_diagnostics'] = selection_diagnostics(model, loader, args, batches=len(loader))
small = DataLoader(dataset, batch_size=16, shuffle=False)
result['same_examples_batch16_diagnostics'] = selection_diagnostics(model, small, args, batches=len(small))

sequence_m, sequence_b, sequence_hit, zero_error = [], [], [], []
copy_with_past, recall_count, copy_nonuniform = 0, 0, 0
with torch.no_grad():
    for x, y, truth, kind in loader:
        _, aux = model(x, return_aux=True)
        n_batch = len(x)
        total_m = torch.zeros(n_batch, dtype=torch.float64)
        total_b = total_m.clone()
        total_hit = total_m.clone()
        count = total_m.clone()
        b = model.embedding.neuron.b.double()
        for n in range(1, len(aux['coeff'])):
            c = aux['coeff'][n].double()
            answer = truth[:, n, :n]
            valid = (kind[:, n] > 0) & answer.any(-1)
            has_copy = (kind[:, n] == 0) & answer.any(-1)
            copy_with_past += int(has_copy.sum())
            recall_count += int(valid.sum())
            share = (c * answer[:, None]).sum(-1) / c.sum(-1).clamp_min(1e-12)
            base = (b[1:n+1].flip(0) * answer).sum(-1) / b[1:n+1].sum()
            hit = answer.gather(1, c.argmax(-1)).double().mean(-1)
            total_m[valid] += share.mean(-1)[valid]
            total_b[valid] += base[valid]
            total_hit[valid] += hit[valid]
            count[valid] += 1
            p = truth_to_oracle_p(truth, n, c.shape[1])
            copy_nonuniform += int((has_copy & ((p[:, 0] - 1./n).abs().max(-1).values > 1e-7)).sum())
        good = count > 0
        sequence_m.extend((total_m[good] / count[good]).tolist())
        sequence_b.extend((total_b[good] / count[good]).tolist())
        sequence_hit.extend((total_hit[good] / count[good]).tolist())
        for idx in range(n_batch):
            zero_error.append(float(y[idx][kind[idx] > 0].square().double().mean()))
result['independent_recall_sequence_mean'] = {'m_eff': float(np.mean(sequence_m)),
    'kernel_mass': float(np.mean(sequence_b)), 'coefficient_top1_hit_all_units': float(np.mean(sequence_hit)),
    'sequences': len(sequence_m), 'recall_events': recall_count,
    'copy_events_in_existing_diagnostic_mask': copy_with_past,
    'copy_events_with_nonuniform_oracle': copy_nonuniform,
    'zero_predictor_recall_mse_sequence_mean': float(np.mean(zero_error))}


original = json.loads(next(root.glob('f_lif_pop_v3/forecasting/results/pilot-eta-001742/*.json')).read_text())
result['parameter_hash_matches_report'] = result['checkpoint_parameter_hash'] == original['provenance']['parameter_hash']
result['mse_abs_differences'] = {key: abs(result['reevaluated_errors'][key]['mse']-original['test'][key]['mse']) for key in ['all','copy','recall','recall_first']}
result['interventions_reevaluated'] = {mode: evaluate(model, loader, args, mode=mode)[0] for mode in ['full','recent','mass_matched','oracle']}
result['interventions_recall_abs_difference'] = {mode: abs(item['recall']['mse']-original['interventions'][mode]['recall']['mse']) for mode,item in result['interventions_reevaluated'].items()}
result['additional_config'] = {k:saved_args.get(k) for k in ['data_seed','eta_init','eta_fixed','lr','weight_decay','n_keys','cue_mode','min_gap','readout','patience']}
zero_sq = zero_n = 0
for x,y,truth,kind in loader:
    zero_sq += float(y[kind > 0].double().square().sum())
    zero_n += int((kind > 0).sum())
result['zero_predictor_recall_mse_pooled'] = zero_sq/zero_n
result['note'] = 'Existing pilot checkpoint CPU evaluation only; no optimizer steps. Model and diagnostic API hashes unchanged since audit03.'
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
print(output.read_text())
