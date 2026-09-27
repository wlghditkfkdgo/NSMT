#!/usr/bin/env python3
"""Audit 53 A26-INTERPRETATION point 2: the R-on calibration changed theta as well as the input
scale (ETTh1 1.829 -> 6.164, ETTh2 2.383 -> 7.154). Does theta act on the hard models at all?

theta divides the W_Q/W_K attention score (layers.Selector.forward, after the mode=='hard'
branch has already returned). If that reading is right, a hard model gives the SAME output and
state for any theta. The check has to be able to fail, so the attention mode 'sparse' is run
the same way: there theta must change the output.
CPU, float64, freshly initialised models, random inputs. No data split is read.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'forecasting'))
from config import parse_defaults                               # noqa: E402
from model import LOAD_MODEL                                    # noqa: E402

THETAS = {'ETTh1': (1.8290166825782963, 6.163561140618673), 'ETTh2': (2.382834403980069, 7.153850037877152)}


def build(mode, hard_q, theta, revin, scale):
    args = parse_defaults()
    for k, v in {'task': 'ett', 'data': 'ETTh1', 'model': 'myModel', 'mode': mode, 'hard_axis': 'shared',
                 'hard_stat': 'pearson', 'hard_q': hard_q, 'theta': theta, 'revin': revin,
                 'input_scale': scale, 'num_patches': 42, 'device': torch.device('cpu')}.items():
        setattr(args, k, v)
    torch.manual_seed(0)                                        # 같은 초기 가중치
    return LOAD_MODEL['myModel'](args, train=True).double().eval(), args


@torch.no_grad()
def main():
    torch.manual_seed(1)
    x = torch.randn(4, 336, 7, dtype=torch.float64)
    worst = {}
    for data, (lo, hi) in THETAS.items():
        for name, mode, q in (('q1', 'hard', 1.), ('pearson', 'hard', .5), ('sparse (must differ)', 'sparse', 1.)):
            outs = []
            for theta in (lo, hi):
                model, args = build(mode, q, theta, True, 10.)
                out, aux = model(x, mode=mode, return_aux=True)
                outs.append((out, aux['state']))
            d_out = (outs[0][0] - outs[1][0]).abs().max().item()
            d_state = (outs[0][1] - outs[1][1]).abs().max().item()
            worst[(data, name)] = (d_out, d_state)
            print(f"[theta] {data} {name:<22} theta {lo:.3f} vs {hi:.3f}: max|d output| {d_out:.3e}  "
                  f"max|d state| {d_state:.3e}")
    hard_same = all(v == (0., 0.) for (d, n), v in worst.items() if not n.startswith('sparse'))
    sparse_diff = all(v[0] > 0 for (d, n), v in worst.items() if n.startswith('sparse'))
    print(f"[theta] hard models identical for both thetas: {hard_same}; sparse control differs: {sparse_diff}")


if __name__ == '__main__':
    main()
