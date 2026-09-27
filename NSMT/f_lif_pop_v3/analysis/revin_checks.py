#!/usr/bin/env python3
"""Prereg 2O D-BM pre-checks for the window normalisation R, before any 2O training.

  1. constant and near-constant input windows give finite outputs and states (R on)
  2. window_norm then window_denorm returns the input
  3. splitting the batch or permuting the channels does not change a sequence's output
  4. R off reproduces the existing behaviour: the 48 reused 2N checkpoints give back their
     recorded best validation MSE under the new code (GPU, the device they were trained on)
  5. shifting the input by a constant shifts the R-on forecast by the same constant
     (and does not for R off -- the contrast shows the check can fail)
  6. the training loss is taken on the restored scale: train.py's loss is criterion(output, y)
     on the forward output, and (5) shows that output lives on the input's scale
Checks 1-3 and 5 use freshly initialised models on CPU; no data split beyond validation is read.
"""
import sys
import json
import inspect
import argparse
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'forecasting'))
from config import parse_defaults                               # noqa: E402
from model import LOAD_MODEL                                    # noqa: E402
from ours import window_norm, window_denorm                     # noqa: E402
from data_provider.data_factory import data_provider            # noqa: E402
from test import evaluate                                       # noqa: E402
import train as T                                               # noqa: E402
import ett_test as E                                            # noqa: E402

CONDS = {'q1': dict(model='myModel', mode='hard', hard_stat='pearson', hard_q=1.),
         'pearson': dict(model='myModel', mode='hard', hard_stat='pearson', hard_q=.5),
         'gru': dict(model='GRU', mode='full'),
         'linear': dict(model='Linear', mode='full')}


def build(cond, revin, seed=0):
    args = parse_defaults()
    for k, v in {**CONDS[cond], 'task': 'ett', 'data': 'ETTh1', 'revin': revin, 'input_scale': 6.,
                 'num_patches': 42, 'device': torch.device('cpu')}.items():
        setattr(args, k, v)
    torch.manual_seed(seed)
    model = LOAD_MODEL[args.model](args, train=True).eval()

    return model, args


def forward(model, args, x):
    out, aux = model(x, mode=args.mode, return_aux=True)
    return out, aux.get('state')


@torch.no_grad()
def main(device):
    torch.manual_seed(1)
    x = torch.randn(6, 336, 7)
    report = {}

    # 1 -------------------------------------------------------------------------------------
    flat = x.clone()
    flat[:, :, 0] = 3.
    flat[:, :, 1] = 3. + 1e-9 * torch.randn(6, 336)
    finite = {}
    for cond in CONDS:
        model, args = build(cond, True)
        out, state = forward(model, args, flat)
        finite[cond] = bool(torch.isfinite(out).all() and (state is None or torch.isfinite(state).all()))
    report['1_constant_inputs_finite'] = finite
    print(f"[revin] 1 constant / near-constant channels -> finite output and states: {finite}")

    # 2 -------------------------------------------------------------------------------------
    err = {}
    for dtype in (torch.float32, torch.float64):
        z, mu, s = window_norm(x.to(dtype))
        err[str(dtype)] = (window_denorm(z, mu, s) - x.to(dtype)).abs().max().item()
    report['2_norm_denorm_max_abs'] = err
    print(f"[revin] 2 denorm(norm(x)) - x, max |.|: {err}")

    # 3 and 5 ----------------------------------------------------------------------------------
    perm = torch.tensor([3, 0, 6, 1, 5, 2, 4])
    c = 2.5
    split, shuffle, shift = {}, {}, {}
    for cond in CONDS:
        for revin in (False, True):
            model, args = build(cond, revin)
            full, _ = forward(model, args, x)
            part, _ = forward(model, args, x[:2])
            permuted, _ = forward(model, args, x[:, :, perm])
            moved, _ = forward(model, args, x + c)
            key = f"{cond}{'_R' if revin else ''}"
            split[key] = (full[:2] - part).abs().max().item()
            shuffle[key] = (full[:, :, perm] - permuted).abs().max().item()
            shift[key] = (moved - full - c).abs().max().item()
    report['3_batch_split_max_abs'], report['3_channel_perm_max_abs'] = split, shuffle
    report['5_shift_equivariance_max_abs'] = shift
    print(f"[revin] 3 batch split, max |.|: {split}")
    print(f"[revin] 3 channel permutation, max |.|: {shuffle}")
    print(f"[revin] 5 f(x + {c}) - f(x) - {c}, max |.| (R on must be ~0, R off need not): {shift}")

    # 6 -------------------------------------------------------------------------------------
    src = inspect.getsource(T.train_one_epoch)
    uses_forward = 'loss = criterion(output, y)' in src
    report['6_loss_on_forward_output'] = uses_forward
    print(f"[revin] 6 train_one_epoch computes criterion(output, y) on the forward output: {uses_forward}; "
          f"with (5) the output is on the restored scale")

    # 4 -------------------------------------------------------------------------------------
    worst, n = 0., 0
    for data in E.DATASETS:
        for cond in E.CONDS:
            for seed in E.SEEDS:
                run, result = E.find_run('etthard-20260925', data, 96, cond, seed)
                model, args = E.load_model(run, device)
                assert getattr(args, 'revin', False) is False
                _, val = data_provider(args, 'val')
                now = evaluate(model, val, args)[0]['mse']
                worst = max(worst, abs(now - json.load(open(result))['train']['best_val_loss']))
                n += 1
    report['4_reuse_val_reproduction'] = {'runs': n, 'max_abs_diff': worst}
    print(f"[revin] 4 R off: {n} reused 2N checkpoints reproduce best validation MSE, max |diff| {worst:.2e}")

    Path(__file__).with_suffix('.json').write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description='prereg 2O D-BM pre-checks for the window normalisation R')
    ap.add_argument('-nd', '--num_device', dest='num_device', type=int, default=0)
    cli = ap.parse_args()
    main(torch.device(f'cuda:{cli.num_device}'))
