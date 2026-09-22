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
state_dir = next(root.glob('f_lif_pop_v3/forecasting/log/smoke-001655/**/model_state'))
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

# Model-level causality and truth separation, including the analog graph.
x, y, truth, kind = next(iter(DataLoader(dataset, batch_size=2)))
changed = x.clone()
changed[:, 21 * args.patch_size:] += 10.
checks = {}
for readout in ('spike', 'analog'):
    model.readout = readout
    with torch.no_grad():
        original = model(x, truth=truth)
        moved = model(changed, truth=truth)
        swapped = model(x, truth=~truth)
    checks[readout] = {'earlier_output_max_error': float((original[:, :21]-moved[:, :21]).abs().max()),
                       'nonoracle_truth_change_max_error': float((original-swapped).abs().max())}
model.zero_grad(set_to_none=True)
pred = model(x)
pred.square().mean().backward()
checks['analog_gradient_norms'] = {n: (None if p.grad is None else float(p.grad.norm()))
    for n, p in model.named_parameters() if n in ['embedding.emb_linear.weight',
    'embedding.neuron.selector.query.weight', 'embedding.neuron.soma.weight', 'recall_head.weight']}
result['model_checks'] = checks

# The ETT loader provides empty truth; exercise exactly that interface without training.
ett_model = myModel(task='ett', embed_dim=4, head_dim=4, input_norm='none', max_length=42)
ett_args = SimpleNamespace(device=torch.device('cpu'), task='ett', mode='sparse', g11_bound=None)
ett_batch = (torch.randn(2, 336, 7), torch.randn(2, 96, 7), torch.empty(2, 0), torch.empty(2, 0))
try:
    selection_diagnostics(ett_model, [ett_batch], ett_args)
    result['ett_empty_truth_diagnostic'] = {'raises': False}
except Exception as exc:
    result['ett_empty_truth_diagnostic'] = {'raises': True, 'type': type(exc).__name__, 'detail': str(exc)}

# Logger receives None metrics in the GRU branch of train_one_epoch.
with tempfile.TemporaryDirectory(prefix='nsmt_gru_logger_audit_') as tmp:
    logger = EpochLog(tmp)
    try:
        logger.logging(epoch=0, train_result={'loss': .3, 'firing_rate': None},
                       val_result={'loss': .3})
        result['gru_logger_none_metric'] = {'raises': False}
    except Exception as exc:
        result['gru_logger_none_metric'] = {'raises': True, 'type': type(exc).__name__, 'detail': str(exc)}
    finally:
        logger.close()

# Calibration loading currently matches only filename stem/alpha/norm, not the full contract.
altered = copy.copy(args)
altered.tau = [40., 80., 160., 320.]
altered.cue_mode = 'code'
altered.eta_fixed = 1.
altered.seed = 123
result['calibration_for_incompatible_request'] = load_calibration(altered)
result['incompatible_requested_fields'] = {'tau': altered.tau, 'cue_mode': altered.cue_mode,
                                         'eta_fixed': altered.eta_fixed, 'seed': altered.seed}
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(result, indent=2, allow_nan=False))
print(output.read_text())
