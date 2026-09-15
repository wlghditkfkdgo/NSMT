"""Read-only reference: closed-form channel-independent linear forecaster under the exact v2
splits/scaler/windows (all test windows, no drop_last). Ridge strength chosen on validation only."""
import sys

import numpy as np

TASK = '/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/f_lif_pop_v2/forecasting'
sys.path.insert(0, TASK)
from data_provider.data_loader import Dataset_ETT_hour  # noqa: E402


class Args:
    seq_len = 336
    root_path = '/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/forecasting/dataset/ETT-small'


def windows(ds, H):
    values = ds.data_x.numpy().astype(np.float64)
    view = np.lib.stride_tricks.sliding_window_view(values, 336 + H, axis=0)[:len(ds)]
    return view[..., :336].reshape(-1, 336), view[..., 336:].reshape(-1, H)


def fit(X, Y, lam):
    X1 = np.hstack([X, np.ones((len(X), 1))])
    gram = X1.T @ X1
    gram[np.diag_indices(336)] += lam
    return np.linalg.solve(gram, X1.T @ Y)


def predict(W, X):
    return np.hstack([X, np.ones((len(X), 1))]) @ W


for data in ['ETTh1', 'ETTh2']:
    for H in [96, 720]:
        args = Args()
        args.pred_len, args.data_path = H, data + '.csv'
        (Xtr, Ytr), (Xva, Yva), (Xte, Yte) = [windows(Dataset_ETT_hour(args, f), H) for f in ['train', 'val', 'test']]
        for name, offset in [('linear', lambda X: 0.), ('linear_last_value_norm', lambda X: X[:, -1:])]:
            best = None
            for lam in [1e-2, 1., 1e1, 1e2, 1e3, 1e4, 1e5]:
                W = fit(Xtr - offset(Xtr), Ytr - offset(Xtr), lam)
                val = np.mean((predict(W, Xva - offset(Xva)) + offset(Xva) - Yva) ** 2)
                if best is None or val < best[0]:
                    best = (val, lam, W)
            val, lam, W = best
            error = predict(W, Xte - offset(Xte)) + offset(Xte) - Yte
            print(f'{data} H{H} {name:24s} lambda={lam:g} val_mse={val:.4f} test_mse={np.mean(error ** 2):.4f} '
                  f'test_mae={np.mean(np.abs(error)):.4f} test_windows={len(Xte) // 7}')
