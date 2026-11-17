"""Checks of the ETT input-length test on the SpkMixer (2026-11-17, exp/f-lif-pop-v3; PROJECT_LOG pre-registration). No training.
  d1  data: with input length L the validation and test sets have as many windows as with 96 and every target window is the same array
      (first, last and 50 spread indices, all 7 channels); the input window of length 96 is the tail of the input window of length L;
      the training set has L - 96 fewer windows. ETTh2 / ETTm2 x H96 / H720.
  d2  model: the SpkMixer (LIF and population LIF, tuned ETT settings) builds with L, has (L - 8) / 4 + 2 tokens, runs forward and backward
      with finite values; the body parameter count does not depend on L.
  d3  gains: the population LIF SpkMixer with the gain table made at input 96 (calib/stft/..._bdnone_txrep.json), run on real training
      windows of length L: firing of every population layer against the same layer of the LIF SpkMixer at L. Pre-fixed rule: the 96
      table is reused iff the worst relative error is < 0.10 (the acceptance threshold of calibrate_stft.py); otherwise a table is
      calibrated at L with the same procedure. The check prints the errors and the decision; it fails only on a silent / non-finite layer.
usage: python -B check_len.py --len 336        (GPU; exit 1 on any failure)
"""
import argparse
import math
import os
import sys
import types

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import calibrate_spk as cs                                                 # noqa: E402
import stft_common as sc                                                   # noqa: E402
from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute  # noqa: E402
from spk_mixer import apply_gain_table                                     # noqa: E402

FAIL = []


def ok(cond, msg):
    print(('pass  ' if cond else 'FAIL  ') + msg, flush=True)
    if not cond:
        FAIL.append(msg)


def dataset(ds, flag, L, H):
    a = types.SimpleNamespace(augmentation_ratio=0)
    cls = Dataset_ETT_hour if ds == 'ETTh2' else Dataset_ETT_minute
    return cls(a, sc.ETT_DIR, flag=flag, size=[L, 48, H], features='M', data_path=f'{ds}.csv', target='OT', timeenc=1, freq='h' if ds == 'ETTh2' else 't')


def batch(ds, L, n=32):
    """n real training windows of length L as the model receives them: [n * 7, 1, L], window-normalised (as stft_common.batch at 96)."""
    import pandas as pd
    v = pd.read_csv(os.path.join(sc.ETT_DIR, f'{ds}.csv')).iloc[:, 1:].to_numpy(dtype=np.float32)
    n_train = 12 * 30 * 24 * (4 if ds.startswith('ETTm') else 1)
    starts = np.linspace(0, n_train - L, n).astype(int)
    x = torch.from_numpy(np.stack([v[s:s + L] for s in starts]))
    x = x - x.mean(1, keepdim=True)
    x = x / torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
    return x.permute(0, 2, 1).reshape(-1, 1, L).contiguous()


def d1(L):
    for ds in ('ETTh2', 'ETTm2'):
        for H in (96, 720):
            for flag in ('val', 'test'):
                a, b = dataset(ds, flag, 96, H), dataset(ds, flag, L, H)
                same = len(a) == len(b) and len(a) > 0
                if same:
                    for i in sorted(set([0, len(a) - 1] + list(np.linspace(0, len(a) - 1, 50).astype(int)))):
                        xa, ya = a[i][0], a[i][1]; xb, yb = b[i][0], b[i][1]
                        same = same and np.array_equal(ya[-H:], yb[-H:]) and np.array_equal(xa, xb[-96:]) and xb.shape[0] == L
                ok(same, f'd1 {ds} H{H} {flag}: {len(a)} windows with 96, {len(b)} with {L}; targets equal, input 96 = tail of input {L}')
            a, b = dataset(ds, 'train', 96, H), dataset(ds, 'train', L, H)
            ok(len(a) - len(b) == L - 96, f'd1 {ds} H{H} train: {len(a)} windows with 96, {len(b)} with {L} (difference {len(a) - len(b)})')


def body_params(m):
    return sum(p.numel() for n, p in m.named_parameters() if not n.startswith('head'))


def d2(L):
    for ds in ('ETTh2', 'ETTm2'):
        x = batch(ds, L, 8).cuda()
        for mixer in ('lif', 'pop_lif'):
            gt = sc.table_path(ds, mixer, 'none', 'rep') if mixer == 'pop_lif' else None
            m0 = sc.build(ds, mixer, seed=0, gain_table=gt)
            m = sc.build(ds, mixer, seed=0, gain_table=gt, seq_len=L).cuda().train()
            out = m(x); out.float().pow(2).mean().backward()
            g = [p.grad for p in m.parameters() if p.grad is not None]
            fin = bool(torch.isfinite(out).all()) and all(bool(torch.isfinite(z).all()) for z in g) and any(float(z.abs().sum()) > 0 for z in g)
            ok(m.num_patches == (L - 8) // 4 + 2 and fin and body_params(m) == body_params(m0),
               f'd2 {ds} {mixer}: {m.num_patches} tokens at input {L}, forward / backward finite, body parameters {body_params(m)} (= input 96: {body_params(m0)})')
            del m, m0; torch.cuda.empty_cache()


def rates(m, x):
    r, hs = {}, []
    for name, mod in m.named_modules():
        if isinstance(mod, cs.NeoNeuron):
            hs.append(mod.register_forward_hook(lambda md, i, o, k=cs.layer_kind(name): r.__setitem__(k, float((o[0] if isinstance(o, tuple) else o).float().mean()))))
    cs.forward(m, x)
    for h in hs: h.remove()
    return r


def d3(L):
    worst_all = 0.
    for ds in ('ETTh2', 'ETTm2'):
        for LL in (96, L):
            x = batch(ds, LL).cuda()
            ref = rates(sc.build(ds, 'lif', seed=0, seq_len=LL).cuda(), x)
            path = sc.table_path(ds, 'pop_lif', 'none', 'rep')
            m = sc.build(ds, 'pop_lif', seed=0, gain_table=path, seq_len=LL).cuda()
            apply_gain_table(m, path)
            got = rates(m, x)
            err = {k: abs(got[k] - ref[k]) / ref[k] if ref[k] > 0 else float('nan') for k in ref}
            worst = max(err.values())
            fin = all(math.isfinite(v) and v > 0 for v in list(got.values()) + list(ref.values())) and math.isfinite(worst)
            ok(fin, f'd3 {ds} input {LL}: table of input 96, worst relative rate error {worst:.3f}  ' +
               ' '.join(f'{k} {got[k]:.4f}/{ref[k]:.4f}' for k in ref))
            if LL == L:
                worst_all = max(worst_all, worst) if math.isfinite(worst) else float('nan')
            del m; torch.cuda.empty_cache()
    dec = 'reuse the input-96 tables' if worst_all < 0.10 else f'calibrate new tables at input {L}'
    print(f'd3 decision (pre-fixed rule, threshold 0.10): worst at input {L} = {worst_all:.3f} -> {dec}', flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--len', type=int, required=True)
    a = ap.parse_args()
    d1(a.len); d2(a.len); d3(a.len)
    print(f'{"ALL PASSED" if not FAIL else "FAILED"}: {len(FAIL)} failed', flush=True)
    sys.exit(1 if FAIL else 0)


if __name__ == '__main__':
    main()
