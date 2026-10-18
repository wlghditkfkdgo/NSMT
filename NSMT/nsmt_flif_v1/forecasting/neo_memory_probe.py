"""Long-term-memory evidence probe for the neocortex path (nsmt_flif_v1, decision 16).

For a trained model and >= 500 test windows, the NEOCORTEX input only (the [N, B*C, 1, P] patch
tensor entering `encoding_neo`; the hippocampus input is untouched) is perturbed and the effect is
measured. No training.

  lag curve (M1):   step n (default N-1); for d = 1..n replace patch j = n-d and record
                    spike_delta_n   = mean |x_neo[n] - x_neo0[n]|   (neocortex spikes at step n)
                    spike_delta_all = mean |x_neo - x_neo0| over steps >= j
                    pred_delta      = mean (yhat - yhat0)^2,  dmse = MSE(yhat) - MSE(yhat0)
  keep-last curve (M2): keep the last k patches (k = 1..N), replace the earlier ones; record MSE(k).
Replacement: 'mean' (window-mean patch per window and channel; default), 'zero', 'shuffle' (the same
patch of another window in the batch). --control hippo repeats the lag curve on the hippocampus
input instead (same-scale reference). Means over windows with a bootstrap 95% interval.

usage: python neo_memory_probe.py --run <run_dir> [-nd 0] [--out DIR] [--windows 512] [--replace mean]
                                  [--step N-1] [--control none|hippo]
"""
import os
import csv
import argparse
from argparse import Namespace

import numpy as np
import torch
from spikingjelly.clock_driven import functional

from config import Config, set_random_seed
from model import LOAD_MODEL
from data_provider.data_factory import data_provider
from analysis import extract_pred_true_for_metrics
from ett_future import cond_name, WEIGHTS

HERE = os.path.dirname(os.path.abspath(__file__))


class Probe:
    """Holds the current perturbation; the pre-hook applies it to the patch tensor."""
    def __init__(self, replace):
        self.replace, self.mask, self.last_neo = replace, None, None

    def perturb(self, inp, patch_dim):
        if self.mask is None:
            return inp
        out = inp.clone()
        idx = [slice(None)] * inp.ndim
        if self.replace == 'mean':
            rep = inp.mean(dim=patch_dim, keepdim=True)              # window-mean patch per (window, channel)
        elif self.replace == 'zero':
            rep = torch.zeros_like(inp.narrow(patch_dim, 0, 1))
        else:                                                        # shuffle: another window's same patch
            rep = inp.roll(1, dims=1)
        for j in self.mask:
            idx[patch_dim] = j
            out[tuple(idx)] = rep[tuple(idx)] if self.replace == 'shuffle' else rep.squeeze(patch_dim)
        return out


def bootstrap(v, n=1000, seed=0):
    v = np.asarray(v, dtype=np.float64)
    if v.size == 0:
        return float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    m = rng.choice(v, size=(n, v.size), replace=True).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True); ap.add_argument('-nd', type=int, default=0)
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'memory_probe'))
    ap.add_argument('--windows', type=int, default=512)
    ap.add_argument('--replace', choices=['mean', 'zero', 'shuffle'], default='mean')
    ap.add_argument('--step', type=int, default=None, help='probed step n (default N-1)')
    ap.add_argument('--control', choices=['none', 'hippo'], default='none')
    ap.add_argument('--weights', choices=['last', 'best'], default='last', help='last-epoch or best-validation weights')
    a = ap.parse_args()
    cfg = Config(); cfg.load_args(os.path.abspath(a.run), Namespace(saved_epoch=[1], num_device=a.nd))
    set_random_seed(cfg.seed)
    model = LOAD_MODEL[getattr(cfg, 'model', 'myModel')](cfg, train=False, ckpt=WEIGHTS[a.weights]); model.eval()
    if getattr(model, 'hippo_only', False):
        raise SystemExit('hippo_only model has no neocortex path; nothing to probe')
    N = model.T; n = N - 1 if a.step is None else a.step
    probe = Probe(a.replace)
    if a.control == 'hippo':
        target, patch_dim = model.encoding, 2                        # hippocampus input [T, B*C, N, P]
    else:
        target, patch_dim = model.encoding_neo, 0                    # neocortex input [N, B*C, 1, P]
    target.register_forward_pre_hook(lambda m, inp: (probe.perturb(inp[0], patch_dim),))
    model.time_block.register_forward_hook(lambda m, i, o: setattr(probe, 'last_neo', o.detach()))
    _, loader = data_provider(cfg, flag='test')

    def run(x, y):
        functional.reset_net(model)
        out = model(x)
        p, t, _, _ = extract_pred_true_for_metrics(out, y, cfg.pred_len, long_pred_len=cfg.pred_len, reduce_T='mean')
        return p, t, probe.last_neo

    lag = {d: {'spike_delta_n': [], 'spike_delta_all': [], 'pred_delta': [], 'dmse': []} for d in range(1, n + 1)}
    keep = {k: {'mse': [], 'pred_delta': []} for k in range(1, N + 1)}
    clean_mse, seen = [], 0
    for x, y, _, _ in loader:
        x, y = x.float().to(cfg.device), y.float().to(cfg.device)
        B = x.shape[0]
        probe.mask = None
        p0, t, s0 = run(x, y)
        mse0 = ((p0 - t) ** 2).mean(dim=(1, 2))                      # per window
        clean_mse += mse0.tolist()
        for d in range(1, n + 1):
            probe.mask = [n - d]
            p, _, s = run(x, y)
            dn = (s[n] - s0[n]).abs().reshape(B, -1).mean(1)          # [B*C...] -> per window (channels folded)
            da = (s[n - d:] - s0[n - d:]).abs().reshape(s.shape[0] - (n - d), B, -1).mean(dim=(0, 2))
            lag[d]['spike_delta_n'] += dn.tolist(); lag[d]['spike_delta_all'] += da.tolist()
            lag[d]['pred_delta'] += ((p - p0) ** 2).mean(dim=(1, 2)).tolist()
            lag[d]['dmse'] += (((p - t) ** 2).mean(dim=(1, 2)) - mse0).tolist()
        for k in range(1, N + 1):
            probe.mask = list(range(0, N - k))
            p, _, _ = run(x, y)
            keep[k]['mse'] += ((p - t) ** 2).mean(dim=(1, 2)).tolist()
            keep[k]['pred_delta'] += ((p - p0) ** 2).mean(dim=(1, 2)).tolist()
        seen += B
        if seen >= a.windows:
            break
    os.makedirs(a.out, exist_ok=True)
    tag = f"{cfg.data}_H{cfg.pred_len}_{cond_name(cfg)}_seed{cfg.seed}_{a.replace}" + ('_ctrlhippo' if a.control == 'hippo' else '') + ('_bestval' if a.weights == 'best' else '')
    path = os.path.join(a.out, tag + '_probe.csv')
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['data', 'cond', 'seed', 'replace', 'control', 'curve', 'x', 'spike_delta_n', 'spike_delta_all',
                    'pred_delta', 'dmse', 'mse', 'ci_lo', 'ci_hi', 'windows', 'step_n', 'clean_mse'])
        cm = float(np.mean(clean_mse))
        for d, v in lag.items():
            lo, hi = bootstrap(v['spike_delta_n'])
            w.writerow([cfg.data, cond_name(cfg), cfg.seed, a.replace, a.control, 'lag', d, np.mean(v['spike_delta_n']),
                        np.mean(v['spike_delta_all']), np.mean(v['pred_delta']), np.mean(v['dmse']), cm + np.mean(v['dmse']),
                        lo, hi, seen, n, cm])
        for k, v in keep.items():
            lo, hi = bootstrap(np.asarray(v['mse']) - np.asarray(clean_mse))
            w.writerow([cfg.data, cond_name(cfg), cfg.seed, a.replace, a.control, 'keep_last', k, '', '',
                        np.mean(v['pred_delta']), np.mean(v['mse']) - cm, np.mean(v['mse']), lo, hi, seen, n, cm])
    print('PROBE', path, 'windows', seen, 'clean_mse', round(cm, 6),
          'S(1)', round(np.mean(lag[1]['spike_delta_n']), 5), f'S({min(12, n)})', round(np.mean(lag[min(12, n)]['spike_delta_n']), 5),
          'dMSE(keep12)', round(np.mean(keep[min(12, N)]['mse']) - cm, 6))


if __name__ == '__main__':
    main()
