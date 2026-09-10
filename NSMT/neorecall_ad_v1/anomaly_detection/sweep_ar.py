"""Anomaly-ratio sweep tool.

`anomaly_ratio` only changes the threshold percentile -- training is unaffected.
So we re-implement just steps (1)-(2) of ``test()`` (energy collection on
train+test) ONCE per dataset, then iterate over multiple AR values to compute
metrics. This avoids touching ``test.py`` and avoids retraining for each AR.

CLI mirrors the existing scripts/<DATA>.sh interface (same args, same imports,
same device handling, same logging style).
"""

import os
import argparse

import numpy as np
import torch
import torch.nn as nn

from spikingjelly.clock_driven import functional
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from config import set_random_seed, parse_arguments, Config
from utils import adjustment
from load_model import LOAD_MODEL
from data_provider.data_factory import data_provider


@torch.no_grad()
def collect_energy(model, loader, device, criterion, with_labels=False):
    """Run the model over ``loader`` and collect the per-sample energy.

    Returns a flat np.ndarray of shape ``[N_samples * win_size]`` matching the
    convention used in test.py. If ``with_labels`` is True, also returns a flat
    ndarray of ground-truth labels.
    """
    model.eval()
    model.train_mode = 'testing'

    energies = []
    labels = []
    for batch in loader:
        if with_labels:
            x, y = batch
            x = x.float().to(device)
            y = y.float().to(device)
        else:
            x, _ = batch
            x = x.float().to(device)
        out = model(x)
        if isinstance(out, tuple):
            out = out[0]
        # score: [B, win_size]
        score = criterion(x, out).mean(-1)
        energies.append(score.detach().cpu().numpy())
        if with_labels:
            labels.append(y.detach().cpu().numpy())
        functional.reset_net(model)
    energy = np.concatenate(energies, axis=0).reshape(-1)
    if with_labels:
        labels = np.concatenate(labels, axis=0).reshape(-1).astype(int)
        return energy, labels
    return energy


def evaluate_threshold(test_energy, gt, threshold):
    """Compute (raw_*, adj_*) metrics at a given threshold.

    test_energy : [N], gt : [N], threshold : float
    """
    assert test_energy.shape == gt.shape, (
        f"shape mismatch: energy {test_energy.shape} vs gt {gt.shape}"
    )
    pred = (test_energy > threshold).astype(int)

    raw_acc = accuracy_score(gt, pred)
    raw_p, raw_r, raw_f, _ = precision_recall_fscore_support(
        gt, pred, average='binary', zero_division=0)

    gt_adj, pred_adj = adjustment(gt.copy(), pred.copy())
    adj_acc = accuracy_score(gt_adj, pred_adj)
    adj_p, adj_r, adj_f, _ = precision_recall_fscore_support(
        gt_adj, pred_adj, average='binary', zero_division=0)

    return {
        'raw_acc': raw_acc, 'raw_pre': raw_p, 'raw_rec': raw_r, 'raw_f1': raw_f,
        'adj_acc': adj_acc, 'adj_pre': adj_p, 'adj_rec': adj_r, 'adj_f1': adj_f,
    }


def sweep_ar(args, ar_grid, model=None):
    """Train-once, evaluate-many for anomaly_ratio.

    args     : Config (already populated; expects args.save_log_path etc.)
    ar_grid  : iterable of float anomaly_ratio values (in % of samples flagged)
    model    : optional pre-loaded model (else loaded via LOAD_MODEL)

    Returns the best (anomaly_ratio, metrics) by adjusted F1, and writes a
    full sweep CSV to ``args.save_log_path/anomaly_ratio_sweep.csv``.
    """
    set_random_seed(args.seed)
    _, train_loader = data_provider(args, flag='train')
    _, test_loader = data_provider(args, flag='test')

    if model is None:
        model = LOAD_MODEL[args.model](args, train=False)
    model = model.to(args.device)
    functional.reset_net(model)

    criterion = nn.MSELoss(reduction='none')

    train_energy = collect_energy(model, train_loader, args.device,
                                  criterion, with_labels=False)
    test_energy, gt_full = collect_energy(model, test_loader, args.device,
                                          criterion, with_labels=True)
    combined_energy = np.concatenate([train_energy, test_energy], axis=0)

    rows = []  # list of (ar, threshold, raw_acc, raw_pre, raw_rec, raw_f1, adj_acc, ...)
    for ar in ar_grid:
        thr = float(np.percentile(combined_energy, 100.0 - float(ar)))
        m = evaluate_threshold(test_energy, gt_full, thr)
        rows.append((float(ar), thr,
                     m['raw_acc'], m['raw_pre'], m['raw_rec'], m['raw_f1'],
                     m['adj_acc'], m['adj_pre'], m['adj_rec'], m['adj_f1']))
        print(f"  ar={float(ar):>5} thr={thr:.4f} | "
              f"raw f1={m['raw_f1']:.4f} adj f1={m['adj_f1']:.4f}")

    best = max(rows, key=lambda r: r[9])  # by adj_f1
    print(f"[sweep] best anomaly_ratio = {best[0]} (adj F1 = {best[9]:.4f})")

    out_csv = os.path.join(args.save_log_path, "anomaly_ratio_sweep.csv")
    os.makedirs(args.save_log_path, exist_ok=True)
    header = ("anomaly_ratio,threshold,raw_acc,raw_pre,raw_rec,raw_f1,"
              "adj_acc,adj_pre,adj_rec,adj_f1\n")
    with open(out_csv, "w") as f:
        f.write(header)
        for r in rows:
            f.write(",".join(f"{v:.6f}" if isinstance(v, float) else str(v)
                             for v in r) + "\n")
    print(f"[sweep] full table saved to `{out_csv}`")

    return best[0], best, rows


def main():
    # Pull out --ar_grid first so the underlying parse_arguments() (which is
    # strict on unknown args) doesn't reject it.
    import sys
    extra_parser = argparse.ArgumentParser(add_help=False)
    extra_parser.add_argument(
        '--ar_grid', nargs='+', type=float,
        default=[0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0],
        help='anomaly_ratio values to sweep at evaluation',
    )
    extra, remaining = extra_parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    parent = parse_arguments()
    setattr(parent, 'ar_grid', extra.ar_grid)

    args = Config()
    if parent.config:
        args.load_args(parent.config, parent)
    else:
        args.set_args(parent)
    set_random_seed(args.seed)
    args.print_info()

    sweep_ar(args, ar_grid=parent.ar_grid)


if __name__ == '__main__':
    main()
