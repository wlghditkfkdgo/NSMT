#!/usr/bin/env python3
"""Cause analysis (a): does ETT need "retrieval of a similar past"? (exploratory, after 2N)

The plan and every definition were written to PROJECT_LOG before this ran. Nothing here is a
verdict; 2N stands as recorded.

  (1) data only -- analog forecasting inside the 336-step window. The last patch is compared
      with every earlier patch by Pearson correlation (the model's rule, applied to raw patch
      values). The continuation that followed a past patch j predicts the future, level-aligned:
          pred = last value + mean_j (continuation_j - last value of patch j)
      Sets of j: all / top-50% similar / most recent 50% / random 50% (Monte Carlo) /
      daily-aligned (offset a multiple of 24 h) / "hindsight" 50%: the k candidates whose OWN
      forecast errors against the target are smallest. That is not the set whose averaged
      forecast is best, so it is not a ceiling (audit 47 A23-HINDSIGHT-BOUND).
      Horizons h=8 (next patch, j<=40) and h=96 (12 patches, continuation inside the window, j<=29).
  (2) what the trained pearson mask picks on validation: overlap of its last-step mask with the
      "hindsight" 50% for h=8, agreement with the data-space top-50%, the share of kept slots in
      the recent half, and the analog error of its own selection. A random k-subset of J slots
      overlaps a fixed k-subset by k/J on average: 20/41 = 0.488 at h=8, 0.5 at h=96; the recent
      half j>=21 is also 20 of 41 slots (audit 47 A23-RANDOM-REFERENCE).
  (3) period shift: channel means/stds of the train-standardised series in the train, val and
      test periods, and the (1) gap similar-minus-all per period. The test period was opened for
      2N. Here no model touches it, but (1) does compute target-using analog errors and the
      "hindsight" selection on it -- more than descriptive statistics, exploratory only (audit 47
      A23-EXPLORATORY-PROTOCOL).

Docstring corrected after the run (audit 48); the code and the saved results are unchanged.
"""
import sys
import json
import glob
import argparse
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'forecasting'))
from data_provider.data_loader import Dataset_ETT_hour          # noqa: E402
from layers import to_patches                                  # noqa: E402
import ett_test as E                                           # noqa: E402

PATCH, SEQ, PRED = 8, 336, 96
T = SEQ // PATCH                                                # 42 patches; the last one is index 41
SUITE = 'etthard-20260925'


def load_windows(data, flag):
    """All windows of a split -> patches [M, 42, 8] and targets [M, 96], M = windows x channels."""
    ds = Dataset_ETT_hour(SimpleNamespace(seq_len=SEQ, pred_len=PRED, root_path=str(E.DATA_ROOT),
                                          data_path=f'{data}.csv'), flag=flag)
    x = ds.data_x.unfold(0, SEQ + PRED, 1).permute(0, 2, 1)     # [N, 432, 7], same windows as __getitem__
    assert x.shape[0] == len(ds)
    past, future = x[:, :SEQ], x[:, SEQ:]
    patches = to_patches(past, PATCH).permute(1, 0, 2)         # [M, 42, 8], M = N*C in the model's order
    target = future.transpose(1, 2).reshape(-1, PRED)          # [M, 96], same (window, channel) order

    return patches.double(), target.double()


def pearson(a, b):
    """a: [M, 8], b: [M, J, 8] -> [M, J]; a constant patch gets 0, not NaN."""
    a = a - a.mean(-1, keepdim=True)
    b = b - b.mean(-1, keepdim=True)
    return (a.unsqueeze(1) * b).sum(-1) / (a.norm(dim=-1, keepdim=True) * b.norm(dim=-1)).clamp_min(1e-12)


def continuations(patches, h):
    """Level-aligned continuation of every usable past patch: [M, J, h], J = 41 (h=8) or 30 (h=96)."""
    if h == PATCH:
        follow = patches[:, 1:T, :]                            # patch j+1 follows patch j, j = 0..40
    else:
        J = T - 1 - h // PATCH + 1                             # j + 12 <= 41  ->  j <= 29
        follow = torch.stack([patches[:, j + 1:j + 1 + h // PATCH, :].reshape(len(patches), h)
                              for j in range(J)], dim=1)
    J = follow.shape[1]

    return follow - patches[:, :J, -1:]                        # 뒤따른 구간 - patch j의 마지막 값


def topk_mask(score, k):
    idx = score.topk(k, dim=-1).indices
    return torch.zeros_like(score).scatter_(-1, idx, 1.)


def analog(patches, target, h, draws=64, seed=0, chunk=8192):
    """(1): MSE of each selection rule, and overlap of the similar set with the "hindsight" set."""
    gen = torch.Generator().manual_seed(seed)
    sums = {k: 0. for k in ('persistence', 'all', 'similar', 'recent', 'random', 'daily', 'hindsight')}
    overlap, n_el = 0., 0
    for s in range(0, len(patches), chunk):
        P, Y = patches[s:s + chunk], target[s:s + chunk, :h]
        delta = continuations(P, h)                            # [m, J, h]
        J = delta.shape[1]
        k = max(1, int(round(.5 * J)))                         # the model's budget rule (round half even)
        last = P[:, -1, -1:]                                   # [m, 1]
        sim = pearson(P[:, -1, :], P[:, :J, :])
        err_j = ((last.unsqueeze(1) + delta - Y.unsqueeze(1)) ** 2).mean(-1)   # 목표를 본 사후 오차
        j = torch.arange(J, dtype=P.dtype)
        masks = {'all': torch.ones_like(sim),
                 'similar': topk_mask(sim, k),
                 'recent': (j >= J - k).to(P.dtype).expand_as(sim),
                 'daily': (((T - 1 - j) % 3) == 0).to(P.dtype).expand_as(sim),
                 'hindsight': topk_mask(-err_j, k)}
        for name, w in masks.items():
            pred = last + (w.unsqueeze(-1) * delta).sum(1) / w.sum(1, keepdim=True)
            sums[name] += ((pred - Y) ** 2).sum().item()
        rnd = 0.
        for _ in range(draws):
            w = topk_mask(torch.rand(sim.shape, generator=gen, dtype=P.dtype), k)
            pred = last + (w.unsqueeze(-1) * delta).sum(1) / w.sum(1, keepdim=True)
            rnd += ((pred - Y) ** 2).sum().item()
        sums['random'] += rnd / draws
        sums['persistence'] += ((last - Y) ** 2).sum().item()
        overlap += ((masks['similar'] * masks['hindsight']).sum(1) / k).sum().item()
        n_el += Y.numel()

    result = {k: v / n_el for k, v in sums.items()}
    result['overlap_similar_hindsight'] = overlap / len(patches)
    result['J'], result['k'] = J, k

    return result


@torch.no_grad()
def model_masks(data, device, seeds=E.SEEDS):
    """(2): the trained pearson model's last-step mask on validation, per seed."""
    patches, target = load_windows(data, 'val')
    delta = continuations(patches, PATCH)                      # [M, 41, 8]
    last = patches[:, -1, -1:]
    Y = target[:, :PATCH]
    err_j = ((last.unsqueeze(1) + delta - Y.unsqueeze(1)) ** 2).mean(-1)
    k = max(1, int(round(.5 * 41)))
    hindsight = topk_mask(-err_j, k)
    similar = topk_mask(pearson(patches[:, -1, :], patches[:, :41, :]), k)
    rows = []
    for seed in seeds:
        run, _ = E.find_run(SUITE, data, PRED, 'pearson', seed)
        model, args = E.load_model(run, device)
        model.eval()
        _, loader = E.data_provider(args, flag='val')
        masks = []
        for x, _, _, _ in loader:
            _, aux = model(x.float().to(device), mode=args.mode, return_aux=True)
            masks.append((aux['coeff'][T - 1][:, 0, :] > 0).double().cpu())   # step 41: slots j = 0..40
        m = torch.cat(masks)
        assert m.shape == hindsight.shape and bool((m.sum(1) == k).all())
        pred = last + (m.unsqueeze(-1) * delta).sum(1) / k
        rows.append({'seed': seed,
                     'overlap_hindsight': ((m * hindsight).sum(1) / k).mean().item(),
                     'agree_data_similar': ((m * similar).sum(1) / k).mean().item(),
                     'recent_half_share': (m[:, 21:].sum(1) / k).mean().item(),
                     'analog_mse_of_mask': ((pred - Y) ** 2).mean().item()})

    return rows


def period_stats(data):
    """(3): train-standardised channel means/stds per period (no model)."""
    ds = Dataset_ETT_hour(SimpleNamespace(seq_len=SEQ, pred_len=PRED, root_path=str(E.DATA_ROOT),
                                          data_path=f'{data}.csv'), flag='train')
    import pandas as pd
    frame = pd.read_csv(E.DATA_ROOT / f'{data}.csv')
    values = ds.scaler.transform(frame[ds.columns].to_numpy(dtype=np.float64))
    out = {}
    for name, (a, b) in {'train': (0, 8640), 'val': (8640, 11520), 'test': (11520, 14400)}.items():
        out[name] = {'mean': values[a:b].mean(0).round(4).tolist(), 'std': values[a:b].std(0).round(4).tolist()}

    return ds.columns, out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description='cause analysis (a): does ETT need retrieval of a similar past?')
    ap.add_argument('--data', dest='data', nargs='+', default=list(E.DATASETS))
    ap.add_argument('-nd', '--num_device', dest='num_device', type=int, default=0)
    ap.add_argument('--out', dest='out', default=str(Path(__file__).with_suffix('.json')))
    cli = ap.parse_args()
    device = torch.device(f'cuda:{cli.num_device}')
    record = {}
    for data in cli.data:
        print(f"{f' {data} ':=^100s}")
        record[data] = {'analog': {}, 'model': None, 'period': None}
        # (1) train과 val (그리고 (3)을 위해 test 기간도, 모델 없이)
        for flag in ('train', 'val', 'test'):
            patches, target = load_windows(data, flag)
            for h in (PATCH, PRED):
                r = analog(patches, target, h)
                record[data]['analog'][f'{flag}/h{h}'] = r
                print(f"[analog] {data} {flag:>5} h={h:<3} J={r['J']} k={r['k']}  "
                      + '  '.join(f"{k} {r[k]:.4f}" for k in ('persistence', 'all', 'similar', 'recent', 'random',
                                                             'daily', 'hindsight'))
                      + f"  overlap(similar,hindsight) {r['overlap_similar_hindsight']:.3f}")
        # (2) 학습된 pearson 마스크 (val)
        rows = model_masks(data, device)
        record[data]['model'] = rows
        for key in ('overlap_hindsight', 'agree_data_similar', 'recent_half_share', 'analog_mse_of_mask'):
            v = [r[key] for r in rows]
            print(f"[model] {data} val pearson mask, 8 seeds: {key} mean {np.mean(v):.4f} (min {min(v):.4f}, max {max(v):.4f})")
        # (3) 기간별 분포
        columns, stats = period_stats(data)
        record[data]['period'] = {'columns': columns, **stats}
        for name, s in stats.items():
            print(f"[period] {data} {name:>5} mean {s['mean']}  std {s['std']}")
    Path(cli.out).write_text(json.dumps(record, indent=1))
    print(f"Record saved to `{cli.out}`")
