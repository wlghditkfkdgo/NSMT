"""MODI 2026-11-13: share of the signal energy in each of the 4 DCT bands of model.band_split, on the Neocortex input as the models
see it (descriptive; used for the predictions of the frequency-band test, docs/MODI_PROJECT_LOG.md 2026-11-13).
  ETT: training part of the csv, 4,000 random windows of 96 (seed 0), every channel, window normalisation as forecast_wrap.Forecaster
       (mean / std of the window), replication padding by the stride (4) -> 100 samples
  MIT-BIH: every beat of the training file (187 samples), replication padding by the stride (16) -> 203 samples
share_i = sum(band_i^2) / sum_i sum(band_i^2); 'no mean' removes each row's mean from band 0 first.
usage (classification folder): python band_energy.py
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.getcwd())
from model import band_split                                              # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def shares(x, pad):
    xp = torch.nn.functional.pad(x, (0, pad), mode='replicate')
    b = band_split(xp, 4, 2).double()
    e = (b ** 2).sum(dim=(1, 2, 3))
    b0 = b.clone(); b0[0] = b0[0] - xp.double().mean(-1, keepdim=True)
    e0 = (b0 ** 2).sum(dim=(1, 2, 3))
    return (e / e.sum()).tolist(), (e0 / e0.sum()).tolist(), xp.shape[-1]


def main():
    rng = np.random.default_rng(0)
    out = ['| data | samples | rows | band 0 (lowest) | band 1 | band 2 | band 3 (highest) | without the mean: band 0 / 1 / 2 / 3 |', '|---|---|---|---|---|---|---|---|']
    for ds, n_train in (('ETTh2', 12 * 30 * 24), ('ETTm2', 12 * 30 * 24 * 4)):
        df = pd.read_csv(os.path.join(HERE, '..', 'forecasting', 'dataset', 'ETT-small', ds + '.csv'))
        v = df.iloc[:n_train, 1:].to_numpy(dtype=np.float64)
        st = rng.integers(0, n_train - 96, size=4000)
        w = np.stack([v[s:s + 96] for s in st])                           # [4000, 96, 7]
        w = (w - w.mean(1, keepdims=True)) / np.sqrt(w.var(1, keepdims=True) + 1e-5)
        x = torch.tensor(w, dtype=torch.float32).permute(0, 2, 1).reshape(-1, 1, 96)
        s, s0, n = shares(x, 4)
        out.append(f'| {ds} (train, window-normalised) | {n} | {x.shape[0]} | ' + ' | '.join(f'{100 * a:.2f} %' for a in s) + ' | ' + ' / '.join(f'{100 * a:.2f}' for a in s0) + ' |')
    p = os.path.join(HERE, '..', '..', 'nsmt_base', 'classification', 'ECG', 'data', '5class_mitbih_train_7_187.csv')
    a = pd.read_csv(p, header=None).to_numpy(dtype=np.float64)
    if a.shape[1] != 188:
        raise RuntimeError(f'unexpected MIT-BIH shape {a.shape}')
    x = torch.tensor(a[:, :187], dtype=torch.float32).unsqueeze(1)
    s, s0, n = shares(x, 16)
    out.append(f'| MIT-BIH (train beats) | {n} | {x.shape[0]} | ' + ' | '.join(f'{100 * v_:.2f} %' for v_ in s) + ' | ' + ' / '.join(f'{100 * v_:.2f}' for v_ in s0) + ' |')
    txt = '\n'.join(out)
    print(txt)
    open(os.path.join(HERE, 'results', 'band-energy-20261113.md'), 'w').write(
        '# Energy share of the 4 DCT bands (model.band_split) on the Neocortex input\n\nSee band_energy.py for the definition.\n\n' + txt + '\n')


if __name__ == '__main__':
    main()
