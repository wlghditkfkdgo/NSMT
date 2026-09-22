"""Separate the two things a change of eta does, on validation.

Audit 23 directs this: "궤적 고정 계수 재계산과 모델 전체 forward 개입을 구분하면 실제
희석과 상태 피드백을 분리할 수 있다."

Changing eta does two things at once. It changes the coefficients directly, and because the
coefficients feed the branch states, it also changes u, hence the query, hence the score
itself -- so weights fixed is NOT score fixed, and the earlier reading that the eta=1 hit
rate "matches" the original trajectory's score mixed statistics from two different
trajectories.

    trajectory-fixed: keep the trained-eta forward pass, take its p and b, recompute only
                      c = min(b * ((1-eta) + eta * B p / sum(b p)), b0). Isolates dilution.
    full forward:     re-run the model at the new eta. Adds the state feedback.

Everything is on the VALIDATION split, because the test split has been observed repeatedly.
"""
import sys
import glob
import argparse
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'forecasting'))

from config import Config, parse_defaults                      # noqa: E402
from model import LOAD_MODEL                                   # noqa: E402
from data_provider.data_factory import data_provider           # noqa: E402
from test import evaluate                                      # noqa: E402
import layers                                                  # noqa: E402

FIELDS = ('score_top1', 'p_mass', 'precap_mass', 'postcap_mass')


@torch.no_grad()
def stages(model, loader, args, eta_override, trajectory_fixed, batches=4):
    """Stage masses at `eta_override`; if trajectory_fixed, reuse the trained-eta forward."""
    neuron = model.embedding.neuron
    sel, b = neuron.selector, neuron.b
    trained = sel.eta_value.clone()
    if not trajectory_fixed and eta_override is not None:
        sel.eta_value.fill_(eta_override)

    acc = {k: [] for k in FIELDS}
    peak = 0.
    for i, batch in enumerate(loader):
        if i >= batches:
            break
        x, y, truth, kind = batch
        x, truth, kind = x.float().to(args.device), truth.to(args.device), kind.to(args.device)
        current = model.embedding.current(layers.to_patches(x, args.patch_size))
        aux = neuron(current, mode=args.mode, return_aux=True)[1]
        state = aux['state']
        peak = max(peak, aux['state'].abs().max().item())
        B = x.shape[0]
        tot = {k: torch.zeros(B, dtype=torch.float64, device=args.device) for k in FIELDS}
        cnt = torch.zeros(B, dtype=torch.float64, device=args.device)

        for n in range(1, current.shape[0]):
            answer = truth[:, n, :n]
            valid = (kind[:, n] > 0) & answer.any(-1)
            if not valid.any():
                continue
            zero = torch.zeros_like(state[0])
            xi = torch.cat([state[n - 1], current[n].unsqueeze(-1)], dim=-1)
            hist = torch.stack([torch.cat([state[j - 1] if j else zero,
                                           current[j].unsqueeze(-1)], dim=-1)
                                for j in range(n)], dim=-2)
            b_hist = b[1:n + 1].flip(0)
            c, sa = sel(xi, hist, b_hist, b[0].item(), args.mode)
            p, score = sa['p'], sa['score']
            eta = eta_override if eta_override is not None else float(sa['eta'].mean())
            if trajectory_fixed:                     # 같은 p, b에서 계수만 다시 만든다
                den = (b_hist * p).sum(-1, keepdim=True).clamp_min(1e-12)
                rho = (1 - eta) + eta * b_hist.sum() * p / den
                raw = b_hist * rho
                c = torch.minimum(raw, b.new_tensor(b[0].item()))
            else:
                raw = b_hist * sa['rho']
            wide = answer.unsqueeze(1).expand(-1, c.shape[1], -1)
            mask = wide.to(c.dtype)
            step = {
                'score_top1': wide.gather(-1, score.argmax(-1, keepdim=True)
                                          ).to(torch.float64).mean((1, 2)),
                'p_mass': ((p * mask).sum(-1) / p.sum(-1).clamp_min(1e-12)).mean(-1).double(),
                'precap_mass': ((raw * mask).sum(-1)
                                / raw.sum(-1).clamp_min(1e-12)).mean(-1).double(),
                'postcap_mass': ((c * mask).sum(-1) / c.sum(-1).clamp_min(1e-12)).mean(-1).double(),
            }
            keep = valid.to(torch.float64)
            for k in FIELDS:
                tot[k] += step[k] * keep
            cnt += keep
        ok = cnt > 0
        for k in FIELDS:
            acc[k].extend((tot[k][ok] / cnt[ok]).cpu().tolist())

    sel.eta_value.copy_(trained)

    return {**{k: float(np.mean(v)) for k, v in acc.items()}, 'max_abs_state': peak}


def main():
    parser = argparse.ArgumentParser(description='separate dilution from state feedback')
    parser.add_argument('--run', required=True)
    parser.add_argument('--etas', nargs='+', type=float, default=[0., .2, .5, 1.])
    cli = parser.parse_args()

    base = parse_defaults()
    base.cpu = True
    config = Config()
    config.load_args(sorted(glob.glob(cli.run))[0], base)
    config.device = torch.device('cpu')
    config.max_eval_batches = 0
    model = LOAD_MODEL[config.model](config, train=False)
    _, val = data_provider(config, 'val')
    sel = model.embedding.neuron.selector
    trained_eta = torch.sigmoid(sel.eta_hat).item()
    bound = getattr(config, 'g11_bound', None)

    print(f"checkpoint: {config.run_id}")
    print(f"validation split, trained eta = {trained_eta:.4f}, G11 bound = {bound}")
    print(f"\n{'개입':>14} {'eta':>6} {'val recall':>11} {'score top1':>11} {'p mass':>8}"
          f" {'pre-cap':>8} {'post-cap':>9} {'max|u|':>9} {'G11':>6}")
    for kind, fixed in (('궤적 고정', True), ('전체 forward', False)):
        for eta in cli.etas:
            s = stages(model, val, config, eta, fixed)
            if fixed:
                mse = float('nan')                   # 궤적 고정은 계수만 바꾸므로 출력이 없다
            else:
                sel.eta_value.fill_(eta)
                mse = evaluate(model, val, config)[0]['recall']['mse']
                sel.eta_value.fill_(float('nan') if config.eta_fixed is None else config.eta_fixed)
            flag = '-' if bound is None else ('OK' if s['max_abs_state'] < bound else 'FAIL')
            shown = f"{mse:>11.4f}" if mse == mse else f"{'-':>11}"
            print(f"{kind:>14} {eta:>6.2f} {shown} {s['score_top1']:>11.4f} {s['p_mass']:>8.4f}"
                  f" {s['precap_mass']:>8.4f} {s['postcap_mass']:>9.4f}"
                  f" {s['max_abs_state']:>9.2f} {flag:>6}")


if __name__ == '__main__':
    main()
