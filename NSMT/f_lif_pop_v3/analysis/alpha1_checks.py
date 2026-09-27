#!/usr/bin/env python3
"""Prereg 2P D-BU pre-checks for the alpha=1 (ordinary LIF) control, before any 2P training.

  1. the alpha=1 coefficient table is exactly 1
  2. alpha=1, hard q=1 (R on, shared pearson selector): every branch follows the explicit Euler
     recursion u <- u + (I - u)/tau on the neuron's own current (float64)
  3. alpha=1: the whole myModel (R on) gives the same output and state in mode hard q=1 and mode
     full (G18 at alpha=1)
  4. constant and near-constant (3 + 1e-4 noise) channels: finite output and state for the two
     alpha=1 conditions
  5. an alpha=1 run and an alpha=0.7 run with otherwise identical arguments get different run ids,
     run directories and result files (Config.set_args with directory creation mocked out)
CPU, freshly initialised models and random inputs; no data split is read.
"""
import sys
import json
from pathlib import Path
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'forecasting'))
import layers                                                   # noqa: E402
from config import parse_defaults, Config                       # noqa: E402
from model import LOAD_MODEL                                    # noqa: E402

CONDS = {'q1_R_a1': 1., 'pearson_R_a1': .5}                       # hard_q


def build(hard_q, alpha=1., scale=10.):
    args = parse_defaults()
    for k, v in {'task': 'ett', 'data': 'ETTh1', 'model': 'myModel', 'mode': 'hard', 'hard_axis': 'shared',
                 'hard_stat': 'pearson', 'hard_q': hard_q, 'alpha': alpha, 'revin': True,
                 'input_scale': scale, 'num_patches': 42, 'device': torch.device('cpu')}.items():
        setattr(args, k, v)
    torch.manual_seed(0)
    return LOAD_MODEL['myModel'](args, train=True).double().eval(), args


@torch.no_grad()
def main():
    report = {}

    # 1 -------------------------------------------------------------------------------------
    b = layers.fractional_coefficients(1., 42, torch.float64)
    report['1_alpha1_coefficients_all_one'] = bool(torch.equal(b, torch.ones_like(b)))
    b7 = layers.fractional_coefficients(.7, 42, torch.float64)
    print(f"[alpha1] 1 alpha=1 table == 1 exactly: {report['1_alpha1_coefficients_all_one']} "
          f"(alpha=0.7 for contrast: b_1 {b7[1]:.4f}, b_41 {b7[41]:.4f})")

    # 2 -------------------------------------------------------------------------------------
    torch.manual_seed(0)
    neuron = layers.PopulationNeuron(embed_dim=6, alpha=1., max_length=42, hard_axis='shared',
                                     hard_stat='pearson', hard_q=1., dtype=torch.float64).double()
    torch.manual_seed(1)
    x = 3. * torch.randn(42, 5, 6, dtype=torch.float64)
    state = neuron(x, mode='hard', return_aux=True)[1]['state']
    u, ref = torch.zeros(5, 6, 4, dtype=torch.float64), []
    for t in range(42):
        u = u + (x[t].unsqueeze(-1) - u) / neuron.tau
        ref.append(u.clone())
    err = (state - torch.stack(ref)).abs().max().item()
    report['2_hard_q1_vs_euler_max_abs'] = err
    print(f"[alpha1] 2 alpha=1 hard q=1 branch state vs explicit Euler, max |err| {err:.2e}")

    # 3 -------------------------------------------------------------------------------------
    torch.manual_seed(2)
    xs = torch.randn(4, 336, 7, dtype=torch.float64)
    model, _ = build(1.)
    out_h, aux_h = model(xs, mode='hard', return_aux=True)
    out_f, aux_f = model(xs, mode='full', return_aux=True)
    same = bool(torch.equal(out_h, out_f) and torch.equal(aux_h['state'], aux_f['state']))
    report['3_alpha1_hard_q1_equals_full'] = same
    print(f"[alpha1] 3 alpha=1 myModel (R on) hard q=1 == full, output and state bitwise: {same}")

    # 4 -------------------------------------------------------------------------------------
    torch.manual_seed(3)
    flat = torch.randn(6, 336, 7)
    flat[:, :, 0] = 3.
    flat[:, :, 1] = 3. + 1e-4 * torch.randn(6, 336)
    assert flat[:, :, 1].std(dim=1).min() > 0, 'near-constant channel collapsed to a constant'
    finite = {}
    for cond, q in CONDS.items():
        model, _ = build(q)
        model = model.float()
        out, aux = model(flat, mode='hard', return_aux=True)
        finite[cond] = bool(torch.isfinite(out).all() and torch.isfinite(aux['state']).all())
    report['4_constant_inputs_finite'] = finite
    print(f"[alpha1] 4 constant / near-constant channels -> finite output and state: {finite}")

    # 5 -------------------------------------------------------------------------------------
    names = {}
    for alpha in (.7, 1.):
        args = parse_defaults()
        for k, v in {'task': 'ett', 'data': 'ETTh1', 'model': 'myModel', 'mode': 'hard', 'hard_q': 1.,
                     'revin': True, 'alpha': alpha, 'suite': 'naming-check', 'seed': 7, 'cpu': True}.items():
            setattr(args, k, v)
        config = Config()
        with mock.patch('os.makedirs'), mock.patch('os.path.exists', return_value=False):
            config.set_args(args)
        names[alpha] = (config.run_id, config.save_result_path, config.result_path)
    distinct = all(a != b for a, b in zip(names[.7], names[1.]))
    report['5_names_distinct'] = distinct
    report['5_run_ids'] = {str(k): v[0] for k, v in names.items()}
    print(f"[alpha1] 5 run id / run dir / result file differ between alpha 0.7 and 1: {distinct}")
    for alpha, v in names.items():
        print(f"[alpha1]   alpha {alpha}: {v[0]}")
        print(f"[alpha1]            .../{Path(v[1]).parent.name}/{Path(v[1]).name}")

    ok = (report['1_alpha1_coefficients_all_one'] and report['2_hard_q1_vs_euler_max_abs'] < 1e-12
          and report['3_alpha1_hard_q1_equals_full'] and all(finite.values()) and distinct)
    report['all_pass'] = ok
    print(f"[alpha1] all five checks pass: {ok}")
    Path(__file__).with_suffix('.json').write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
