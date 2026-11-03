"""Post-hoc, report-only reference scores for the anomaly-detection results (2026-11-03, exp/f-lif-pop-v3; PROJECT_LOG AD tuning
results). Not pre-registered and not part of any verdict: written after the final AUC-ROC on SMAP came out below 50 for every
trained model, to show where chance and a training-free score sit on the same test points.

No training, no learned parameters. Same loaders (TSLib, train-file standardisation), same concatenation of the overlapping step-1
test windows, same threshold rule and the same ad_metrics as train_ad.final_test:
    random      uniform random score per concatenated step (numpy seed 0)
    train-dev   channel mean of x^2 (x is standardised with the training-file statistics: distance from the training mean)
    win-mean    channel mean of (x - window channel mean)^2 (the reconstruction error of a model that outputs each window's
                channel means; the trained models normalise each window by these means before encoding)
usage: python ad_trivial_baselines.py [--data_root <NSMT/anomaly_detection/dataset>]
"""
import argparse
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ad_metrics import ad_metrics                                        # noqa: E402
from data_provider.data_factory import data_dict                         # noqa: E402

ROOT = ('/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/'
        'NSMT/anomaly_detection/dataset')
OUT = os.path.join(HERE, 'results', 'flif-v1-ad-spkmux-tune', 'posthoc_trivial_baselines.md')
WIN, RATIO = 100, 1.0


def win_idx(n):
    """time indices w + j over windows w < n, j < WIN (the loader's step-1 windows, concatenated)."""
    return (np.arange(n)[:, None] + np.arange(WIN)[None, :]).reshape(-1)


def win_mean_scores(x, n, chunk=20000):
    out = []
    for s in range(0, n, chunk):
        idx = np.arange(s, min(s + chunk, n))[:, None] + np.arange(WIN)[None, :]
        w = x[idx]                                                       # [b, WIN, C]
        out.append(((w - w.mean(1, keepdims=True)) ** 2).mean(-1).reshape(-1))
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--data_root', default=ROOT); a = ap.parse_args()
    rng = np.random.default_rng(0)
    L = ['# Post-hoc reference scores (report only, not pre-registered)', '',
         'Same test points, threshold rule and metrics as the final runs; no training. Values in %.', '',
         '| dataset | score | AUC-PR | AUC-ROC | raw F1 | point-adjusted F1 | anomaly share |', '|---|---|---:|---:|---:|---:|---:|']
    for ds_name in ('MSL', 'SMAP', 'PSM'):
        args = types.SimpleNamespace(data=ds_name)
        ds_tr = data_dict[ds_name](args, os.path.join(a.data_root, ds_name), WIN, flag='train')
        ds_te = data_dict[ds_name](args, os.path.join(a.data_root, ds_name), WIN, flag='test')
        xtr, xte = np.asarray(ds_tr.train, np.float32).astype(np.float64), np.asarray(ds_te.test, np.float32).astype(np.float64)
        ytes = np.asarray(ds_te.test_labels).reshape(len(xte), -1)[:, 0]
        ntr, nte = len(ds_tr), len(ds_te)
        for i in (0, nte - 1):                                           # the indexing equals the loader's windows
            xw, yw = ds_te[i]
            assert np.array_equal(np.float32(xte[i:i + WIN]), xw) and np.array_equal(ytes[i:i + WIN], np.asarray(yw).reshape(-1))
        assert np.array_equal(np.float32(xtr[ntr - 1:ntr - 1 + WIN]), ds_tr[ntr - 1][0])
        itr, ite = win_idx(ntr), win_idx(nte)
        y = ytes[ite]
        dev_tr, dev_te = (xtr ** 2).mean(-1), (xte ** 2).mean(-1)
        cand = {'random': (rng.random(itr.size), rng.random(ite.size)),
                'train-dev': (dev_tr[itr], dev_te[ite]),
                'win-mean': (win_mean_scores(xtr, ntr), win_mean_scores(xte, nte))}
        for name, (s_tr, s_te) in cand.items():
            m = ad_metrics(s_tr, s_te, y, RATIO)
            L.append(f"| {ds_name} | {name} | {100 * m['aucpr']:.2f} | {100 * m['aucroc']:.2f} | {100 * m['raw_f1']:.2f} | "
                     f"{100 * m['pa_f1']:.2f} | {100 * m['anomaly_frac']:.2f} |")
            print(L[-1], flush=True)
    open(OUT, 'w').write('\n'.join(L) + '\n')
    print('written', OUT)


if __name__ == '__main__':
    main()
