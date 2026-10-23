"""Diagnostic (no training, report only): is the information that the spikes do not carry concentrated in the slow
members of the population, and would a larger reset let their spikes carry it?

    python member_probe.py [-nd <gpu>] [--out <results folder>]

Models: the 2W pick (direct-0.0005-all: soma none, the 4 members tau 4/8/16/32 fire themselves, per-step BN, one output
Linear on all 42 steps), q1_R and q1, ETTh1 and ETTh2, seeds 7/13/21 (12 checkpoints, best validation weights). Splits:
probes are fitted on train windows (every 2nd window; ridge strength picked on the last 10 % of them, chronologically)
and scored on validation. Nothing reads test.

Part A, trained models. For member k, H_t = u_t + (I_t - u_t) / tau_k is the charge the threshold sees (spike = H_t >= 1;
checked exactly), u_t the member state before step t.
  A1 forecast probe: ridge from member k's spikes [42 x 32] vs its charges H [42 x 32] to the 96-step target; also all
     4 members together ([42 x 128], the output Linear's input -- the same function class as the model's head, so its
     MSE sits next to the model's own). Fitted and scored exactly as the model: the forecast of the window-normalised
     target is restored (x std + mean, R on) and the MSE is taken on the dataset scale. Gap = spikes - charges.
     (Revision b, 10-23: the first run standardised a feature that was constant on train by 1e-8 and scored R on in
     the window-normalised space; both blew up on some models. That record is kept, not used.)
  A2 memory probe: from the LAST step only, member k's spikes s_42 [32] vs charges H_42 [32] -> the mean of the input
     patch l steps back (l = 0, 1, 2, 4, 8, 16, 32); validation R^2.
  A3 firing statistics per member: rate, consecutive-spike share, share of steps charged in (0.5, 1) but silent,
     reset per spike.
Part B, the reset (same trained currents, ETTh1 / ETTh2 q1_R seed 7). A standalone member (spikeDE pred / ABM kernel,
alpha 0.7, h 1) with reset term c * s in dv/dt: c = theta / tau (spikeDE, the model; reproduced spike for spike)
vs c = theta (the whole threshold, as an Euler soft reset). The theta-reset member fires less, so it is also run with
its input gain re-matched to the spikeDE member's firing rate. For each: rate, A1 spike probe, A2 spike memory probe.
"""
import sys
import json
import math
import argparse
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / 'forecasting'))

from config import TASK, set_random_seed                       # noqa: E402
from data_provider.data_factory import data_provider           # noqa: E402
import rerun_common as rc                                      # noqa: E402
import layers                                                  # noqa: E402
from ours import window_norm                                   # noqa: E402

SEEDS = (7, 13, 21)
BASES = ('q1_R', 'q1')
DATASETS = ('ETTh1', 'ETTh2')
TAUS = (4., 8., 16., 32.)
LAGS = (0, 1, 2, 4, 8, 16, 32)
LAMBDAS = [10. ** e for e in range(-3, 5)]
SUITE = 'etthard-2w-20261022'


@torch.no_grad()
def collect(model, args, split, device, every=1):
    """Per window x channel row: target, the 42 x 32 x 4 member spikes and charges, patch means, current."""
    set_random_seed(args.seed)
    data_set, loader = data_provider(args, split)
    if split == 'train':                                         # in time order (the loader shuffles train)
        loader = torch.utils.data.DataLoader(data_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    rows = {k: [] for k in ('y', 's', 'h', 'pm', 'cur', 'mu', 'sd')}
    tau = torch.tensor(TAUS, device=device)
    exact = True
    for i, (x, y, _, _) in enumerate(loader):
        x, y = x.float().to(device), y.float().to(device)
        B, L, C = x.shape
        if args.revin:
            xn, mu, sd = window_norm(x)                          # [B, 1, C]: the model forecasts (y - mu) / sd
        else:
            xn, mu, sd = x, torch.zeros_like(x[:, :1]), torch.ones_like(x[:, :1])
        patch = layers.to_patches(xn, args.patch_size, getattr(args, 'patch_stride', 0))   # [T, B*C, 8]
        cur = model.embedding.current(patch)                     # [T, R, 32]
        spikes, aux = model.embedding.neuron(cur, return_aux=True)
        T, R, D = cur.shape
        u = aux['state']                                         # [T, R, 32, 4], u[t] = state after step t
        prev = torch.cat([torch.zeros_like(u[:1]), u[:-1]])
        h = prev + (cur.unsqueeze(-1) - prev) / tau
        s = spikes.reshape(T, R, D, 4)
        exact &= bool(torch.equal(s, (h >= 1.).to(s.dtype)))
        keep = torch.arange(0, B, every, device=device)
        rsel = (keep.unsqueeze(1) * C + torch.arange(C, device=device)).reshape(-1)       # rows of kept windows
        rows['y'].append(y[keep].transpose(1, 2).reshape(-1, y.shape[1]).cpu())    # raw target (dataset scale)
        rows['mu'].append(mu[keep].transpose(1, 2).reshape(-1, 1).cpu())
        rows['sd'].append(sd[keep].transpose(1, 2).reshape(-1, 1).cpu())
        rows['s'].append(s[:, rsel].permute(1, 0, 2, 3).cpu().to(torch.uint8))
        rows['h'].append(h[:, rsel].permute(1, 0, 2, 3).float().cpu())
        rows['pm'].append(patch[:, rsel].mean(-1).transpose(0, 1).cpu())                # [R, T]
        rows['cur'].append(cur[:, rsel].transpose(0, 1).cpu())                           # [R, T, 32]
    out = {k: torch.cat(v) for k, v in rows.items()}
    out['exact'] = exact
    return out


def ridge_fit(X, Y, w, Xh, Yh, wh, device):
    """Weighted ridge with an unpenalised intercept: minimises sum_i w_i^2 |x_i W + c - y_i|^2 + lam N |W|^2. With
    w = the window's std and y = the window-normalised target this is exactly the model's loss on the restored
    scale (R on); w = 1 otherwise. Features are standardised by X's statistics; a feature constant on X (e.g. a
    unit that never fires there) is dropped, so it cannot blow up on other data. lambda is picked on the hold-out
    (Xh, Yh, wh) and the probe refitted on both."""
    X, Y, w, Xh, Yh, wh = (t.to(device, torch.float64) for t in (X, Y, w, Xh, Yh, wh))
    mx, sx = X.mean(0), X.std(0)
    live = sx > 1e-6
    mx, sx = mx[live], sx[live]

    def solve(A, B, wa):
        Z = (A[:, live] - mx) / sx
        q = wa.square() / wa.square().sum()
        mz, mb = (q * Z).sum(0), (q * B).sum(0)
        Zc, Bc = (Z - mz) * wa, (B - mb) * wa
        evals, evecs = torch.linalg.eigh(Zc.T @ Zc)
        Rm = evecs.T @ (Zc.T @ Bc)
        def at(lam):
            W = evecs @ (Rm / (evals + lam * A.shape[0]).unsqueeze(1))
            return lambda C: ((C.to(device, torch.float64)[:, live] - mx) / sx - mz) @ W + mb
        return at

    def err(f, A, B, wa):
        return float(((f(A) - B) * wa).square().mean())

    fit = solve(X, Y, w)
    errs = [err(fit(lam), Xh, Yh, wh) for lam in LAMBDAS]
    lam = LAMBDAS[int(np.argmin(errs))]
    return solve(torch.cat([X, Xh]), torch.cat([Y, Yh]), torch.cat([w, wh]))(lam), lam, int(live.sum())


def probe(train, val, feats, target, device, weighted=True):
    """train / val = (y, s, h, pm, mu, sd). Forecast probes (weighted) score the MSE on the restored scale, the model's
    validation metric; the memory probes (unweighted) score R^2."""
    n = train[0].shape[0]
    cut = int(n * .9)
    X, Y = feats(train).float(), target(train).float()
    w = train[5].float() if weighted else torch.ones(n, 1)
    f, lam, used = ridge_fit(X[:cut], Y[:cut], w[:cut], X[cut:], Y[cut:], w[cut:], device)
    wv = (val[5] if weighted else torch.ones(val[0].shape[0], 1)).to(device, torch.float64)
    P, Yv = f(feats(val).float()), target(val).to(device, torch.float64)
    mse = float(((P - Yv) * wv).square().mean())
    r2 = float(1. - (P - Yv).square().sum() / (Yv - Yv.mean(0)).square().sum())
    return {'mse': mse, 'r2': r2, 'lambda': lam, 'dim': X.shape[1], 'features_used': used}


def as_tuple(d, s=None):
    return (d['y'], d['s'] if s is None else s, d['h'], d['pm'], d['mu'], d['sd'])


def ynorm(d):
    return (d[0] - d[4]) / d[5]                                  # the target the model forecasts (R on); y for R off


def part_a(tr, va, device):
    train, val = as_tuple(tr), as_tuple(va)
    T = tr['s'].shape[1]
    out = {'A1': {}, 'A2': {}, 'A3': {}}
    y = ynorm
    for k, tau in enumerate(TAUS):
        sp = probe(train, val, lambda d, k=k: d[1][..., k].reshape(d[1].shape[0], -1), y, device)
        mb = probe(train, val, lambda d, k=k: d[2][..., k].reshape(d[2].shape[0], -1), y, device)
        out['A1'][f'tau{tau:g}'] = {'spikes': sp, 'charges': mb, 'gap_mse': sp['mse'] - mb['mse']}
        lag = {}
        for l in LAGS:
            tgt = lambda d, l=l: d[3][:, T - 1 - l:T - l]
            lag[l] = {'spikes': probe(train, val, lambda d, k=k: d[1][:, -1, :, k], tgt, device, weighted=False)['r2'],
                      'charges': probe(train, val, lambda d, k=k: d[2][:, -1, :, k], tgt, device, weighted=False)['r2']}
        out['A2'][f'tau{tau:g}'] = lag
        s, h = va['s'][..., k].float(), va['h'][..., k].float()
        out['A3'][f'tau{tau:g}'] = {'rate': float(s.mean()), 'pair_rate': float((s[:, 1:] * s[:, :-1]).mean()),
                                    'silent_charged': float(((h > .5) & (h < 1.)).float().mean()),
                                    'charge_mean': float(h.mean()), 'charge_p99': float(torch.quantile(h.flatten()[::97], .99)),
                                    'reset_per_spike': 1. / tau}
    sp = probe(train, val, lambda d: d[1].reshape(d[1].shape[0], -1), y, device)
    mb = probe(train, val, lambda d: d[2].reshape(d[2].shape[0], -1), y, device)
    out['A1']['all'] = {'spikes': sp, 'charges': mb, 'gap_mse': sp['mse'] - mb['mse']}
    return out


# ---- part B: a standalone member with a chosen reset ----------------------------------------------------------
def kernel(T, alpha, device):
    d = torch.arange(T, dtype=torch.float64, device=device)
    return ((d + 1.) ** alpha - d ** alpha) / math.gamma(alpha + 1.)


def member(cur, tau, reset, alpha=.7, theta=1.):
    """cur [R, T, D] -> spikes [R, T, D]. spikeDE pred (ABM): v_{t+1} = sum_j b_{t-j} f_j, f = (I - v)/tau - reset s,
    s_t = H(v_t + (I_t - v_t)/tau - theta)."""
    R, T, D = cur.shape
    b = kernel(T, alpha, cur.device)
    cur = cur.double()
    v = torch.zeros(R, D, dtype=torch.float64, device=cur.device)
    fs, out = [], []
    for t in range(T):
        drift = (cur[:, t] - v) / tau
        s = ((v + drift - theta) >= 0.).double()
        fs.append(drift - reset * s)
        out.append(s)
        F = torch.stack(fs, 0)                                   # [t+1, R, D]
        v = torch.einsum('j,jrd->rd', b[:t + 1].flip(0), F)
    return torch.stack(out, 1)


def matched_gain(cur, tau, reset, want, lo=.25, hi=16., steps=18):
    sub = cur[::7]
    for _ in range(steps):
        mid = math.sqrt(lo * hi)
        if float(member(sub * mid, tau, reset).mean()) < want:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def part_b(tr, va, device):
    out = {}
    T = tr['s'].shape[1]
    for k, tau in enumerate(TAUS):
        ct, cv = tr['cur'].to(device), va['cur'].to(device)
        same = float((member(cv, tau, 1. / tau) == va['s'][..., k].to(device).double()).double().mean())
        rate0 = float(member(cv, tau, 1. / tau).mean())
        g = matched_gain(ct, tau, 1., float(tr['s'][..., k].float().mean()))
        res = {'spikeDE_reproduced': same}
        for name, gain, reset in (('theta_over_tau', 1., 1. / tau), ('theta', 1., 1.), ('theta_rate_matched', g, 1.)):
            st, sv = member(ct * gain, tau, reset).float().cpu(), member(cv * gain, tau, reset).float().cpu()
            train, val = as_tuple(tr, st), as_tuple(va, sv)
            a1 = probe(train, val, lambda d: d[1].reshape(d[1].shape[0], -1), ynorm, device)
            a2 = {l: probe(train, val, lambda d: d[1][:, -1], lambda d, l=l: d[3][:, T - 1 - l:T - l], device,
                           weighted=False)['r2'] for l in LAGS}
            res[name] = {'gain': gain, 'rate': float(sv.mean()), 'pair_rate': float((sv[:, 1:] * sv[:, :-1]).mean()),
                         'A1_mse': a1['mse'], 'A2_r2': a2}
        res['rate_model'] = rate0
        out[f'tau{tau:g}'] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-nd', '--num_device', type=int, default=0)
    ap.add_argument('--out', default='member-probe-20261023b')
    a = ap.parse_args()
    device = torch.device(f'cuda:{a.num_device}')
    out_dir = TASK / 'results' / a.out
    out_dir.mkdir(parents=True, exist_ok=False)
    record = {'what': __doc__.split('\n')[0], 'suite': SUITE, 'seeds': list(SEEDS), 'per': {}, 'part_b': {}}
    for base in BASES:
        for data in DATASETS:
            for seed in SEEDS:
                spec = rc.cell_2w(f'{base}-direct-0.0005-all')
                status, run, result = rc.find_cell(SUITE, 'ett', data, spec, seed)
                if status != 'done':
                    raise SystemExit(f'{base} {data} {seed}: {status}')
                model, args = rc.load_model(run, device)
                model.eval()
                tr = collect(model, args, 'train', device, every=2)
                va = collect(model, args, 'val', device)
                if not (tr['exact'] and va['exact']):
                    raise SystemExit(f'{base} {data} {seed}: spikes != (charge >= 1); the charge is not the thresholded value')
                key = f'{base}/{data}/seed{seed}'
                record['per'][key] = part_a(tr, va, device)
                record['per'][key]['model_val_mse'] = json.loads(Path(result).read_text())['train']['best_val_loss']
                r = record['per'][key]
                print(f"[probe] {key}: model val {r['model_val_mse']:.4f}; A1 spikes / charges by tau " +
                      ' '.join(f"{t}: {r['A1'][t]['spikes']['mse']:.4f}/{r['A1'][t]['charges']['mse']:.4f}" for t in r['A1']),
                      flush=True)
                if base == 'q1_R' and seed == 7:
                    record['part_b'][f'{base}/{data}/seed{seed}'] = part_b(tr, va, device)
                    for t, v in record['part_b'][f'{base}/{data}/seed{seed}'].items():
                        print(f"[reset] {data} {t}: reproduced {v['spikeDE_reproduced']:.6f}; " + '; '.join(
                            f"{n} rate {v[n]['rate']:.3f} A1 {v[n]['A1_mse']:.4f} A2(l=4) {v[n]['A2_r2'][4]:.3f}"
                            for n in ('theta_over_tau', 'theta', 'theta_rate_matched')), flush=True)
    record['analysis_sha256'] = rc.sha256(Path(__file__))
    record['source_sha256'] = {f: rc.sha256(TASK / f) for f in rc.SOURCES}
    json.dump(record, open(out_dir / 'member_probe.json', 'x'), indent=1, allow_nan=False)
    print(f'[probe] written to {out_dir / "member_probe.json"}')


if __name__ == '__main__':
    main()
