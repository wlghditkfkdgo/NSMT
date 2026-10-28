"""MODI 2026-10-28: smoothing controls for PPG-DaLiA (pre-registration docs/MODI_PROJECT_LOG.md 2026-10-28). No training.

For every run of a suite (best-validation weights, preds_val.npz / preds_test.npz written by train_ppg.py): causal smoothing of the
390 predictions of each window along the labels -- median of the last k (k = 1, 3, 5, 9, 15, 25) or exponential smoothing
s_t = a p_t + (1 - a) s_{t-1} (a = 1, 0.5, 0.3, 0.2, 0.1, 0.05). The (method, parameter) with the lowest validation MSE is applied
to the test predictions. k = 1 / a = 1 is no smoothing, so the chosen control can only match or beat the raw model on validation.
usage: python ppg_smooth.py --suite flif-v1-ppg   ->  results/<suite>/smooth.csv
"""
import argparse
import csv
import os

import numpy as np

from summarize_rhythm import parse

HERE = os.path.dirname(os.path.abspath(__file__))
KS, AS = (1, 3, 5, 9, 15, 25), (1., .5, .3, .2, .1, .05)


def causal_median(p, k):
    if k == 1:
        return p
    out = np.empty_like(p)
    for t in range(p.shape[1]):
        out[:, t] = np.median(p[:, max(0, t - k + 1):t + 1], axis=1)
    return out


def ema(p, a):
    out = np.empty_like(p); s = p[:, 0].copy()
    for t in range(p.shape[1]):
        s = a * p[:, t] + (1 - a) * s if t else s
        out[:, t] = s
    return out


def smooth(p, method, v):
    return causal_median(p, v) if method == 'median' else ema(p, v)


def metrics(pred, d):
    e = pred.astype(np.float64) - d['y'].astype(np.float64)
    half = (d['bpm_range'][d['subject'], 1] - d['bpm_range'][d['subject'], 0]) / 2.
    return float((e ** 2).mean()), float((np.abs(e) * half[:, None]).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--suite', default='flif-v1-ppg')
    a = ap.parse_args()
    res = os.path.join(HERE, 'results', a.suite)
    rows = []
    for r in parse(os.path.join(res, 'raw.txt')):
        va, te = (np.load(os.path.join(r['dir'], 'log', f'preds_{s}.npz')) for s in ('val', 'test'))
        best = None
        for method, grid in (('median', KS), ('ema', AS)):
            for v in grid:
                m = metrics(smooth(va['pred'], method, v), va)[0]
                if best is None or m < best[0]:
                    best = (m, method, v)
        raw_mse, raw_bpm = metrics(te['pred'], te)
        mse, bpm = metrics(smooth(te['pred'], best[1], best[2]), te)
        if abs(raw_mse - r['mse']) > 1e-6 * max(1., abs(r['mse'])):
            raise RuntimeError(f"{r['dir']}: saved test predictions give mse {raw_mse} != logged {r['mse']}")
        rows.append(dict(cond=r['cond'], seed=int(r['seed']), method=best[1], param=best[2], val_mse=f'{best[0]:.8f}',
                         test_mse_raw=f'{raw_mse:.8f}', test_mse=f'{mse:.8f}', test_mae_bpm=f'{bpm:.6f}', dir=r['dir']))
        print(rows[-1])
    with open(os.path.join(res, 'smooth.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == '__main__':
    main()
