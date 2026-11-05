"""MODI 2026-10-28: post-hoc correction of the forecast-head energy of the Spikformer-style runs of the ETT generalization batch.

forecast_wrap.Forecaster averages the Spikformer-style spikes over the T_h repeats *before* the flat head Linear(N*D -> H), so the
head input is non-binary (0, 1/T_h, ...) and syops counts the head as MAC (4.6 pJ) on every input. The Neo-mixer head reads binary
spikes and is counted as AC (0.9 pJ) at the measured firing rate. The head is linear (no bias here: --no-bias), so
head(mean_t s_t) = (1/T_h) sum_t head(s_t): the same model can apply the head to each binary step (AC per spike, 1/T_h folded into
the weights) and add the T_h outputs. This script recounts the Spikformer-style head that way:

    AC_head = T_h * mean(s) * (N*D) * H * C   (spike-driven additions)  +  (T_h - 1) * H * C   (summing the T_h outputs)
    E_corr  = total_fft_uJ - MAC_head * 4.6 pJ + AC_head * 0.9 pJ

mean(s) is measured on the test set with the best-validation weights (model_state/checkpoint.pth, the weights of final+result.csv).
Checks (fail-closed): the recomputed test MSE equals the logged one (relative 1e-5), the per-step head input is exactly binary,
MAC_head = N*D*H*C matches the syops per-layer line, and every Neo-mixer head line has 0 MAC (its energy is kept as logged).
MODI 2026-10-29: the dual-path forecasters (B form spkneo_*, [NSMT hippocampus T=4] nsmt4_*) have the same averaged head and are
recounted the same way (per-step head input from Forecaster.step_features, which also runs the neocortex).
usage: python energy_recount_ett.py --suite flif-v1-gen-ett   ->  forecasting/results/<suite>/energy_recount.csv
       --missing_only: keep the rows already in energy_recount.csv and process only RESULT rows that are not there (e.g. a rerun)
"""
import argparse
import csv
import math
import os
import re
import sys

import torch
from spikingjelly.clock_driven import functional

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, '..', 'forecasting'))

from load_model import LOAD_MODEL
from forecast_wrap import Forecaster
from summarize_rhythm import parse
from train_forecast import data_args
from data_provider.data_factory import data_provider

E_AC, E_MAC = 0.9e-6, 4.6e-6                                          # µJ per operation (utils.get_energy_consumption)
UNIT = {'': 1., 'K': 1e3, 'M': 1e6, 'G': 1e9}


def recount(cond):
    """conditions whose forecast head reads the T_h-averaged hippocampus spikes (Spikformer-style and dual-path forecasters)."""
    return cond == 'spk_ssa' or cond.startswith(('spkall_', 'spkneo_', 'nsmt4_', 'nsmt4q_'))   # MODI 2026-10-30: + nsmt4q_


def head_line(run):
    """(AC ops, MAC ops) of the top-level forecast head from the syops per-layer file (rounded as printed)."""
    for line in open(os.path.join(run, 'log', 'model+info+per+layer.txt'), encoding='utf-8'):
        if line.startswith('  (head): Linear('):
            m = re.search(r'%\s*Params,\s*([\d.]+)\s*([KMG]?)\s*Ops,\s*[\d.]+%\s*ACs,\s*([\d.]+)\s*([KMG]?)\s*Ops', line)
            if m:
                return float(m.group(1)) * UNIT[m.group(2)], float(m.group(3)) * UNIT[m.group(4)]
    raise RuntimeError(f'no head line in {run}')


@torch.no_grad()
def measure(run, device):
    cfg = torch.load(os.path.join(run, 'model_state', 'config.pt'), map_location='cpu')
    args = argparse.Namespace(**cfg)
    args.device = device
    model = Forecaster(LOAD_MODEL[args.model](args, True), args.embed_dim, args.pred_len, bias=args.bias)
    model.load_state_dict(torch.load(os.path.join(run, 'model_state', 'checkpoint.pth'), map_location='cpu'))
    model = model.to(device).eval(); model.train_mode = 'testing'
    if model.kind not in ('ssa', 'dual') or model.head.bias is not None:
        raise RuntimeError(f'{run}: expected a bias-free Spikformer-style or dual-path forecaster')
    _, dl = data_provider(data_args(args, args), 'test')
    m = model.inner
    se = 0.; cnt = 0; spikes = 0.; n_el = 0; binary = True; T_h = None; C = None
    functional.reset_net(model)
    for x, y, _, _ in dl:
        x = x.float().to(device); y = y.float().to(device)[:, -args.pred_len:, :]
        B, L, C = x.shape
        means = x.mean(1, keepdim=True); xc = x - means
        stdev = torch.sqrt(torch.var(xc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        z = (xc / stdev).permute(0, 2, 1).reshape(B * C, 1, L)
        xh = model.step_features(z)                                    # [T_h, B*C, N, D], per-step head input
        functional.reset_net(model)
        T_h = xh.shape[0]
        binary &= bool(((xh == 0) | (xh == 1)).all())
        spikes += float(xh.double().sum()); n_el += xh.numel()
        out = model.head(xh.mean(0).reshape(B * C, -1)).reshape(B, C, args.pred_len).permute(0, 2, 1) * stdev + means
        d = (out - y).double(); se += float((d ** 2).sum()); cnt += d.numel()
    return dict(mse=se / cnt, rate=spikes / n_el, binary=binary, T_h=T_h, C=C, ND=m.num_patches * args.embed_dim, H=args.pred_len)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--suite', default='flif-v1-gen-ett')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--missing_only', action='store_true')
    a = ap.parse_args()
    res = os.path.join(HERE, '..', 'forecasting', 'results', a.suite)
    rows = parse(os.path.join(res, 'raw.txt'))
    out, bad = [], []
    if a.missing_only:                                                 # MODI 2026-10-30: e.g. one rerun after the full recount
        out = list(csv.DictReader(open(os.path.join(res, 'energy_recount.csv'))))
        have = {(o['ds'], int(o['pl']), o['cond'], o.get('tag', ''), int(o['seed'])) for o in out}
        rows = [r for r in rows if (r['ds'], int(r['pl']), r['cond'], r.get('tag', ''), int(r['seed'])) not in have]
        print(f'{len(out)} rows kept, {len(rows)} to process')
    for r in rows:
        run = r['dir']; ac_p, mac_p = head_line(run)
        rec = dict(ds=r['ds'], pl=int(r['pl']), cond=r['cond'], seed=int(r['seed']), mse_logged=r['mse'], total_fft_uJ=r['total_fft_uJ'])
        if 'tag' in r:                                                 # MODI 2026-10-30: tuning suite (one cond, several settings)
            rec = dict(ds=rec.pop('ds'), pl=rec.pop('pl'), cond=rec.pop('cond'), tag=r['tag'], **rec)
        if not recount(r['cond']):                                     # MODI 2026-10-29: + spkall_*, dual-path spkneo_* / nsmt4_*
            if mac_p != 0:
                bad.append(f'{run}: Neo-mixer head has MAC {mac_p}')
            rec.update(mse_recomputed='', rate='', binary='', head_MAC=0, head_AC_corr='', total_fft_uJ_corr=r['total_fft_uJ'])
            out.append(rec); continue
        q = measure(run, a.device)
        mac = q['ND'] * q['H'] * q['C']
        ac = q['T_h'] * q['rate'] * mac + (q['T_h'] - 1) * q['H'] * q['C']
        corr = r['total_fft_uJ'] - mac * E_MAC + ac * E_AC
        rel = abs(q['mse'] - r['mse']) / r['mse']
        if not (math.isfinite(q['mse']) and rel < 1e-5):
            bad.append(f"{run}: test MSE {q['mse']:.6f} != logged {r['mse']:.6f}")
        if not q['binary']:
            bad.append(f'{run}: per-step head input is not binary')
        if abs(mac - mac_p) > 0.006 * mac:                             # syops prints 3 significant digits
            bad.append(f'{run}: MAC_head {mac} != syops {mac_p}')
        if not (math.isfinite(corr) and corr > 0):
            bad.append(f'{run}: corrected energy {corr}')
        rec.update(mse_recomputed=f"{q['mse']:.6f}", rate=f"{q['rate']:.6f}", binary=q['binary'], head_MAC=mac,
                   head_AC_corr=f'{ac:.1f}', total_fft_uJ_corr=f'{corr:.6f}')
        out.append(rec)
        print(f"{r['ds']} H{int(r['pl'])} seed{int(r['seed'])}: mse {q['mse']:.6f} (logged {r['mse']:.6f}) rate {q['rate']:.4f} "
              f"head MAC {mac / 1e6:.2f}M -> AC {ac / 1e6:.2f}M  energy {r['total_fft_uJ']:.2f} -> {corr:.2f} µJ", flush=True)
    path = os.path.join(res, 'energy_recount.csv')
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    print(f'{len(out)} runs ({sum(recount(r["cond"]) for r in out)} T_h-averaged heads recounted) -> {path}')
    if bad:
        print('FAILED CHECKS:\n  ' + '\n  '.join(bad)); sys.exit(1)
    print('all checks passed')


if __name__ == '__main__':
    main()
