#!/usr/bin/env python3
"""Prereg 2Q D-CB pre-checks for the model_v1-style single Linear readout, before any 2Q training.

  1. the existing 'flatten' path is unchanged: the 32 R-on spiking 2O checkpoints (q1_R and
     pearson_R, ETTh1/ETTh2, 8 seeds) reproduce their stored best validation MSE within 1e-6
     (GPU, the device they were trained on)
  2. at the same seed, 'linear' and 'flatten' start from the same embedding, neuron, recall head
     and final Linear (every shared parameter and buffer bitwise equal)
  3. 'linear' has exactly 1,056 parameters fewer (the 32x32 compression and its bias)
  4. (what the two heads can express) a trained 'flatten' head folded into one Linear,
     W = W_head (I_42 x W_compress), b = b_head + W_head (1_42 x b_compress), gives the same
     forecast as the two-stage head: the difference is parametrisation, not capacity
check_model.py (all gates) is run separately.
"""
import sys
import json
import argparse
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'forecasting'))
from config import parse_defaults, set_random_seed               # noqa: E402
from layers import to_patches                                   # noqa: E402
from ours import window_norm, window_denorm                     # noqa: E402
from model import LOAD_MODEL                                    # noqa: E402
from data_provider.data_factory import data_provider            # noqa: E402
from test import evaluate                                       # noqa: E402
import ett_alpha as A                                           # noqa: E402


def build(head_mode, seed, q=1.):
    args = parse_defaults()
    for k, v in {'task': 'ett', 'data': 'ETTh1', 'model': 'myModel', 'mode': 'hard', 'hard_axis': 'shared',
                 'hard_stat': 'pearson', 'hard_q': q, 'revin': True, 'input_scale': 10., 'head_mode': head_mode,
                 'num_patches': 42, 'device': torch.device('cpu')}.items():
        setattr(args, k, v)
    set_random_seed(seed)
    return LOAD_MODEL['myModel'](args, train=True), args


def fold(model):
    """The single Linear equal to a 'flatten' head: [pred_len, T*D] weight and [pred_len] bias."""
    Wc, bc = model.head_compress.weight.double(), model.head_compress.bias.double()     # [H, D], [H]
    Wh, bh = model.head.weight.double(), model.head.bias.double()                       # [P, T*H], [P]
    T = model.num_patches
    Wh3 = Wh.reshape(Wh.shape[0], T, Wc.shape[0])                                       # [P, T, H]
    W = torch.einsum('pth,hd->ptd', Wh3, Wc).reshape(Wh.shape[0], -1)                    # [P, T*D]
    b = bh + torch.einsum('pth,h->p', Wh3, bc)

    return W, b


@torch.no_grad()
def main(device):
    report = {}

    # 2 and 3 -------------------------------------------------------------------------------
    same, diffs = True, []
    for seed in (7, 13):
        for q in (1., .5):
            flat, _ = build('flatten', seed, q)
            lin, _ = build('linear', seed, q)
            pf = dict(flat.named_parameters()); pf.update(dict(flat.named_buffers()))
            pl = dict(lin.named_parameters()); pl.update(dict(lin.named_buffers()))
            shared = [k for k in pl if not k.startswith('head_compress')]
            # equal_nan: eta_value is a NaN sentinel buffer when eta is learned (NaN != NaN under torch.equal)
            bad = [k for k in shared if k not in pf or pf[k].shape != pl[k].shape
                   or not torch.allclose(pf[k], pl[k], rtol=0., atol=0., equal_nan=True)]
            same = same and not bad and set(pl) == {k for k in pf if not k.startswith('head_compress')}
            n_f = sum(p.numel() for p in flat.parameters())
            n_l = sum(p.numel() for p in lin.parameters())
            diffs.append(n_f - n_l)
            print(f"[readout] 2 seed {seed} q {q:g}: {len(shared)} shared tensors, unequal {bad}; "
                  f"3 parameters flatten {n_f} linear {n_l} (difference {n_f - n_l})")
    report['2_same_initial_values'] = same
    report['3_parameter_difference'] = sorted(set(diffs))
    print(f"[readout] 2 same initial values at the same seed: {same}; 3 difference always 1,056: {set(diffs) == {1056}}")

    # 4 -------------------------------------------------------------------------------------
    _, run, _ = A.find_cell('ETTh1', 96, 'pearson_R', 7)
    model, args = A.load_model(run, torch.device('cpu'))
    model = model.double().eval()
    W, b = fold(model)
    torch.manual_seed(0)
    x = torch.randn(4, 336, 7, dtype=torch.float64)
    y_two = model(x, mode=args.mode)                                                    # two-stage head
    B, C = 4, 7
    xn, mu, sd = window_norm(x)
    spikes = model.embedding(to_patches(xn, 8), args.mode)                              # [T, B*C, D]
    y_one = (spikes.transpose(0, 1).reshape(B * C, -1) @ W.T + b).reshape(B, C, -1).transpose(1, 2)
    y_one = window_denorm(y_one, mu, sd)
    err = (y_two - y_one).abs().max().item()
    report['4_folded_head_max_abs'] = err
    print(f"[readout] 4 trained flatten head folded into one Linear (ETTh1 pearson_R seed 7): max |diff| {err:.2e}")

    # 1 -------------------------------------------------------------------------------------
    worst, n = 0., 0
    for data in A.DATASETS:
        for cond in ('q1_R', 'pearson_R'):
            for seed in A.SEEDS:
                status, run, result = A.find_cell(data, 96, cond, seed)
                model, args = A.load_model(run, device)
                assert args.head_mode == 'flatten'
                _, val = data_provider(args, 'val')
                now = evaluate(model, val, args)[0]['mse']
                worst = max(worst, abs(now - json.load(open(result))['train']['best_val_loss']))
                n += 1
    report['1_flatten_reproduction'] = {'runs': n, 'max_abs_diff': worst}
    print(f"[readout] 1 flatten path: {n} 2O R-on spiking checkpoints reproduce best validation MSE, max |diff| {worst:.2e}")

    ok = same and set(diffs) == {1056} and err < 1e-9 and worst <= 1e-6
    report['all_pass'] = ok
    print(f"[readout] all pre-checks pass: {ok}")
    Path(__file__).with_suffix('.json').write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='prereg 2Q D-CB pre-checks for the single Linear readout')
    ap.add_argument('-nd', '--num_device', dest='num_device', type=int, default=0)
    main(torch.device(f'cuda:{ap.parse_args().num_device}'))
