import os
import numpy as np
import pandas as pd
import torch

from config import Config, parse_arguments, set_random_seed
from model import LOAD_MODEL
from data_provider.data_factory import data_provider
from utils import write_json


@torch.no_grad()
def evaluate(model, data_loader, args, memory_mode=None, baselines=False):
    model.eval()
    totals = torch.zeros(6, dtype=torch.float64, device=args.device)
    horizon = torch.zeros(args.pred_len, 2, dtype=torch.float64, device=args.device)
    count = 0
    horizon_count = 0
    example = None
    for i, batch in enumerate(data_loader):
        if args.max_eval_batches and i >= args.max_eval_batches:
            break
        x, y, _, _ = batch
        x, y = x.float().to(args.device), y.float().to(args.device)
        output = model(x, memory_mode=memory_mode)
        error = (output - y).double()
        totals[0] += error.square().sum()
        totals[1] += error.abs().sum()
        horizon[:, 0] += error.square().sum((0, 2))
        horizon[:, 1] += error.abs().sum((0, 2))
        horizon_count += y.shape[0] * y.shape[2]
        count += y.numel()
        if baselines:
            for offset, prediction in [(2, x[:, -1:]), (4, x.mean(1, keepdim=True))]:
                difference = (prediction - y).double()
                totals[offset] += difference.square().sum()
                totals[offset + 1] += difference.abs().sum()
        if example is None:
            example = {'prediction': output[0].cpu().tolist(), 'target': y[0].cpu().tolist()}
    if count == 0:
        raise ValueError('Empty evaluation')
    result = {'loss': (totals[0] / count).item(), 'mse': (totals[0] / count).item(),
              'mae': (totals[1] / count).item(), 'elements': count}
    if not all(np.isfinite(result[k]) for k in ['loss', 'mse', 'mae']):
        raise FloatingPointError('Nonfinite evaluation')
    if baselines:
        result['baselines'] = {name: {'mse': (totals[k] / count).item(), 'mae': (totals[k+1] / count).item()}
                               for name, k in [('persistence', 2), ('window_mean', 4)]}
    return result, (horizon / horizon_count).cpu().tolist(), example


@torch.no_grad()
def memory_diagnostics(model, data_loader, args):
    model.eval()
    x = next(iter(data_loader))[0][:8].float().to(args.device)
    _, aux = model(x, return_aux=True)
    w = aux['attention']
    steps = w.shape[0]
    entropy = -(w * w.clamp_min(1e-12).log()).sum(-1)
    time = torch.arange(steps, device=w.device)
    lag = (time[:, None] - time[None, :]).clamp_min(0)
    lag_mass = [(w * (lag == d)[:, None, None, :]).sum(-1)[1:].mean().item()
                for d in range(1, steps)]
    return {'scope': f'first {len(x)} test windows, all variables/neurons/patches',
            'spike_rate_per_constituent': aux['spikes'].mean((0, 1, 2)).cpu().tolist(),
            'population_membrane_std': aux['membrane'].std(-1, unbiased=False).mean().item(),
            'max_abs_membrane': aux['membrane'].abs().max().item(),
            'mean_gate_after_first_patch': aux['gate'][1:].mean().item(),
            'mean_abs_evidence': aux['evidence'][1:].abs().mean().item(),
            'evidence_to_charge_abs_ratio': (aux['evidence'][1:].abs().mean() / aux['charge'][1:].abs().mean().clamp_min(1e-8)).item(),
            'mean_normalized_entropy': (entropy[2:] / time[2:, None, None].float().log()).mean().item(),
            'lag_mass': lag_mass,
            'mean_lag_in_patches': sum((i + 1) * v for i, v in enumerate(lag_mass))}


def test(args: Config, model=None, save=True):
    if model is None:
        model = LOAD_MODEL[args.model](args, False)
    _, loader = data_provider(args, flag='test')
    result, horizon, example = evaluate(model, loader, args, baselines=True)
    result['diagnostics'] = memory_diagnostics(model, loader, args)
    result['interventions'] = {}
    if args.retrieval:
        for mode in ['off', 'uniform', 'recent']:
            result['interventions'][mode] = evaluate(model, loader, args, memory_mode=mode)[0]
    if save:
        path = os.path.join(args.save_log_path, 'final+result.csv')
        if os.path.exists(path):
            raise FileExistsError('Final results already exist')
        row = {key: result[key] for key in ['loss', 'mse', 'mae']}
        row.update(parameters=sum(p.numel() for p in model.parameters()),
                   best_epoch=args.best_epoch, seed=args.seed)
        pd.DataFrame([row]).to_csv(path, index=False, float_format='%.6f')
        pd.DataFrame(horizon, columns=['mse', 'mae']).rename_axis('horizon_zero_based').to_csv(
            os.path.join(args.save_log_path, 'horizon_metrics.csv'), float_format='%.12g')
        write_json(os.path.join(args.save_log_path, 'forecast_example.json'), example)
    print('Test was successfully done', result['mse'], result['mae'], flush=True)
    return result


if __name__ == '__main__':
    config = parse_arguments()
    if not config.config:
        raise ValueError('Use --config <existing run directory>')
    args = Config()
    args.load_args(config.config, config)
    set_random_seed(args.seed)
    print(test(args, save=False))
