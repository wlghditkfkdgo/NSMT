"""Post-hoc, report-only (2026-11-03, exp/f-lif-pop-v3; PROJECT_LOG 'head information loss' entry): does the AD head pass the
information needed to reconstruct a window? Compares the holdout validation MSE of the tuning runs (trained on the first 80 % of
the training file, validated on the last 20 %) with two training-free reconstructions of the same validation windows:
    win-mean   each window's channel means (what a model whose decoded output is 0 after the window normalisation returns)
    zero       0 (the training-file mean; x is standardised with the training-file statistics)
Same loader (flag 'val' = the last 20 % of the training file, step-1 windows) and the same element-mean MSE as train_ad.evaluate.
usage: python ad_winmean_val.py [--data_root <NSMT/anomaly_detection/dataset>]
"""
import argparse
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ad_trivial_baselines import ROOT, WIN                               # noqa: E402
from data_provider.data_factory import data_dict                         # noqa: E402

OUT = os.path.join(HERE, 'results', 'flif-v1-ad-spkmux-tune', 'posthoc_winmean_val.md')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--data_root', default=ROOT); a = ap.parse_args()
    L = ['# Validation MSE of training-free reconstructions (post-hoc, report only)', '',
         'Holdout validation windows (last 20 % of the training file, step 1); compare with val_mse_best of the holdout tuning runs '
         '(summary_lr.md, summary_layers.md, summary_res.md).', '',
         '| dataset | validation windows | win-mean MSE | zero MSE |', '|---|---:|---:|---:|']
    for ds in ('MSL', 'SMAP', 'PSM'):
        d = data_dict[ds](types.SimpleNamespace(data=ds), os.path.join(a.data_root, ds), WIN, flag='val')
        v, n = np.asarray(d.val, np.float32).astype(np.float64), len(d)
        assert np.array_equal(np.float32(v[n - 1:n - 1 + WIN]), d[n - 1][0])
        se = sz = 0.; cnt = 0
        for s in range(0, n, 5000):
            w = v[np.arange(s, min(s + 5000, n))[:, None] + np.arange(WIN)[None, :]]
            se += float(((w - w.mean(1, keepdims=True)) ** 2).sum()); sz += float((w ** 2).sum()); cnt += w.size
        L.append(f'| {ds} | {n} | {se / cnt:.5f} | {sz / cnt:.5f} |')
        print(L[-1], flush=True)
    open(OUT, 'w').write('\n'.join(L) + '\n')
    print('written', OUT)


if __name__ == '__main__':
    main()
