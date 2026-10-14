"""Does the auxiliary low-pass target need memory? (read-only measurement behind decision 6, 2026-10-15)

Mirrors ours.py exactly: per-window standardisation, ReplicationPad1d(stride) -> length 100, DCT over the
padded window, keep ceil(100 * keep_ratio) lowest coefficients with centre=True, patches of 8 with stride 4
(N = 24). Fits a linear least-squares map from the last `ctx` input patches to the low-pass target of patch
n (the current L_aux target) and of patch n + lead (a forward-looking alternative), pooled over n >= 6, and
reports R^2. Train split only (StandardScaler on the first 12 months, windows sampled from it).

usage: python analysis_aux_target.py [--data ETTh1] [--keep_ratio 0.25] [--windows 3000] [--seed 0]
"""
import os
import math
import argparse

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def r2(Xf, Y):
    Xf = np.concatenate([Xf, np.ones((len(Xf), 1))], 1)
    beta, *_ = np.linalg.lstsq(Xf, Y, rcond=None)
    res = Y - Xf @ beta
    return 1 - res.var(0).sum() / Y.var(0).sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='ETTh1'); ap.add_argument('--keep_ratio', type=float, default=0.25)
    ap.add_argument('--windows', type=int, default=3000); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--seq_len', type=int, default=96); ap.add_argument('--patch', type=int, default=8)
    a = ap.parse_args()
    df = pd.read_csv(os.path.join(HERE, 'dataset', 'ETT-small', f'{a.data}.csv'))
    x = df.iloc[:, 1:].values.astype(np.float64)
    f = 4 if a.data.startswith('ETTm') else 1
    n_train = 12 * 30 * 24 * f
    x = x[:n_train]
    x = (x - x.mean(0)) / x.std(0)
    L, P, S, k = a.seq_len, a.patch, a.patch // 2, a.keep_ratio
    rng = np.random.default_rng(a.seed)
    starts = rng.choice(n_train - L, a.windows, replace=False)
    W = np.stack([x[s:s + L].T for s in starts]).reshape(-1, L)
    W = (W - W.mean(1, keepdims=True)) / np.sqrt(W.var(1, keepdims=True) + 1e-5)
    Wp = np.concatenate([W, np.repeat(W[:, -1:], S, 1)], 1)
    Tn = Wp.shape[1]
    n_ = np.arange(Tn); kk = np.arange(Tn)[:, None]
    C = np.cos(math.pi * (n_ + 0.5) * kk / Tn) * math.sqrt(2 / Tn); C[0] /= math.sqrt(2)
    m = Wp.mean(1, keepdims=True); Xc = (Wp - m) @ C.T; K = max(1, math.ceil(Tn * k)); Xc[:, K:] = 0
    low = Xc @ C + m

    def patches(arr):
        return np.stack([arr[:, i * S:i * S + P] for i in range((Tn - P) // S + 1)], 1)
    Pin, Plow = patches(Wp), patches(low); N = Pin.shape[1]
    print(f'{a.data}: N = {N} patches, {len(W)} window-channels, keep_ratio {k} -> {K} of {Tn} DCT coefficients')
    for lead in (0, 1, 2, 3, 4):
        out = []
        for ctx in (1, 2, 4, 7):
            Xs, Ys = [], []
            for n in range(6, N - lead):
                Xs.append(Pin[:, n - ctx + 1:n + 1].reshape(len(Pin), -1)); Ys.append(Plow[:, n + lead])
            out.append((ctx, r2(np.concatenate(Xs), np.concatenate(Ys))))
        label = 'low-pass(patch n)   = current L_aux target' if lead == 0 else f'low-pass(patch n+{lead})'
        print(f'  {label:45s}: ' + '  '.join(f'ctx={c}: R2={v:.4f}' for c, v in out))
    print(f'  share of the current target variance explained by the patch mean alone: '
          f'{r2(Pin[:, 6:].mean(2).reshape(-1, 1), Plow[:, 6:].reshape(-1, P)):.4f}')


if __name__ == '__main__':
    main()
