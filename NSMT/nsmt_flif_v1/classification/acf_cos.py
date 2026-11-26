"""MODI 2026-11-26: ACF-CoS of our datasets (Kim et al., "PPT: Patch Order Do Matters In Time Series Pretext Task", ICLR 2025,
Sec. 4 Eq. 4 and App. L; papers/ICLR-2025-ppt-...pdf). Measurement only, no training (docs/MODI_PROJECT_LOG.md 2026-11-26).

ACF-CoS(x) = 1 - cos(a, a'), a = autocorrelation of x at lags 0..T-1 (Eq. 4: a_j = sum_{t<=T-j} (x_t - m)(x_{t+j} - m) / sum (x_t - m)^2),
a' = the same for x' = x cut into non-overlapping patches, patches shuffled at random, put back together. Each channel is its own series;
the score is the mean over samples (and channels), averaged over three patch sizes (the paper: patch sizes giving 10-40 patches; here
T // 10, T // 20, T // 40, the tail that does not fill a patch is dropped). Paper anchors: white noise 0.001, step function 0.902.
Anchors checked against the authors' public notebook (github.com/eai-lab/PPT, minimal_implementation_for_acf_cos.ipynb, read 11-26):
  - its autocorrelation is np.correlate(x, x, 'full')[n-1:] without removing the mean, on a z-normalized series (the staircase cell);
    removing the mean here gives the same cosine for a z-normalized series (cosine ignores scale);
  - staircase (50 steps of length 10, z-normalized, patch 10, 1000 random swaps with random.seed(42)): the notebook prints 0.928 for
    that one permutation, and acf() here gives 0.9279 with the same permutation (checked 11-26); over 2000 seeds the mean is 0.820
    (range 0.154-0.964), and 'anchor staircase' below is the mean over random shuffles (about 0.82);
  - white noise: the notebook cell takes the "permuted" ACF from the unpermuted series (permute_acf = autocorrelation(sample_gaussian_noise)),
    so it prints 0.000; with a real patch shuffle the same functions give about 0.27 (T 100, patch 5, 10 swaps, mean of 2000 draws).
    'anchor white noise' below uses a real shuffle, so the noise level of this score is about 0.3, not 0.
A high score means shuffling patches changes the autocorrelation a lot (the paper's "orderness"); it is a property of the signal, not of
the labels.
usage (classification folder): python acf_cos.py [--n 4000] [--sets mitbih,cinc,ett,anchors]
"""
import argparse
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.abspath(os.path.join(HERE, '..', '..'))


def acf(x):
    """x [n, T] -> a [n, T] (lags 0..T-1), Eq. 4 of the paper; rows with zero variance -> nan."""
    n, T = x.shape
    x = x - x.mean(1, keepdims=True)
    f = np.fft.rfft(x, 2 * T, axis=1)
    c = np.fft.irfft(f * np.conj(f), axis=1)[:, :T]
    with np.errstate(invalid='ignore', divide='ignore'):
        return c / c[:, :1]


def shuffle_patches(x, p, rng):
    n, T = x.shape
    k = T // p
    xp = x[:, :k * p].reshape(n, k, p)
    perm = np.argsort(rng.random((n, k)), axis=1)
    return np.take_along_axis(xp, perm[:, :, None], axis=1).reshape(n, k * p)


def score(x, p, rng):
    """mean ACF-CoS over rows of x [n, T] with patch size p (both series cut to k*p points)."""
    k = x.shape[1] // p
    xo = x[:, :k * p]
    a, b = acf(xo), acf(shuffle_patches(xo, p, rng))
    ok = np.isfinite(a).all(1) & np.isfinite(b).all(1)
    cos = (a[ok] * b[ok]).sum(1) / (np.linalg.norm(a[ok], axis=1) * np.linalg.norm(b[ok], axis=1))
    return float(np.mean(1 - cos)), int(ok.sum())


def report(name, x, rng, sizes=None):
    T = x.shape[1]
    sizes = sizes or sorted({max(1, T // 10), max(1, T // 20), max(1, T // 40)})
    vals = [score(x, p, rng) for p in sizes]
    m = float(np.mean([v for v, _ in vals]))
    print(f'{name:<38s} T={T:<5d} n={vals[0][1]:<6d} ' + ' '.join(f'p{p}={v:.3f}' for p, (v, _) in zip(sizes, vals)) + f'  ACF-CoS={m:.3f}')
    return m


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--n', type=int, default=4000); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--sets', default='anchors,mitbih,cinc,ett')
    a = ap.parse_args(); rng = np.random.default_rng(a.seed); sets = a.sets.split(',')
    if 'anchors' in sets:                                               # paper anchors: white noise 0.001, step 0.902
        report('anchor white noise (T 512)', rng.standard_normal((a.n, 512)), rng)
        st = np.zeros((a.n, 512)); st[:, 256:] = 1
        report('anchor single step (T 512)', st, rng)
        sc = np.tile(np.repeat(np.arange(1, 51), 10).astype(float), (a.n, 1))                 # the notebook's staircase, patch 10
        report('anchor staircase (T 500, notebook)', sc, rng, [10])
    if 'mitbih' in sets:                                                # Kachuee et al. 2018 beats, 125 Hz, 187 points, zero-padded
        d = np.loadtxt(os.path.join(R, 'nsmt_base/classification/ECG/data/5class_mitbih_test_3_187.csv'), delimiter=',')
        x = d[rng.permutation(len(d))[:a.n], :187]
        report('MIT-BIH beat (187, as the model sees it)', x, rng)
        report('MIT-BIH beat, patch 16 (11 patches)', x, rng, [16])
        nz = np.array([187 - np.argmax(r[::-1] != 0) for r in x])
        L = int(np.percentile(nz, 5))                                   # common unpadded length: cut every beat to the 5th percentile
        report(f'MIT-BIH beat, first {L} points (no padding)', x[nz >= L, :L], rng)
    if 'cinc' in sets:                                                  # CinC 2017, 125 Hz cache of rhythm_data.py, centre 30 s
        d = np.load(os.path.join(HERE, 'dataset', 'cinc2017', 'cinc2017_v3_125hz.npz'))
        sig, off = d['signal'], d['offsets']; W = 3750
        rows = [sig[off[i] + (off[i + 1] - off[i] - W) // 2: off[i] + (off[i + 1] - off[i] - W) // 2 + W]
                for i in range(len(off) - 1) if off[i + 1] - off[i] >= W]
        x = np.stack(rows); x = x[rng.permutation(len(x))[:a.n]]
        report('CinC 2017 record, centre 30 s (3750)', x, rng)
        report('CinC 2017, model token 32 (117 patches)', x, rng, [32])
    if 'ett' in sets:                                                   # ETTh2 / ETTm2 input windows of 96, each channel a series
        import pandas as pd
        for ds in ('ETTh2', 'ETTm2'):
            v = pd.read_csv(os.path.join(R, 'forecasting/dataset/ETT-small', f'{ds}.csv')).values[:, 1:].astype(float)
            w = np.stack([v[i:i + 96] for i in range(0, len(v) - 96, 96)])           # [n, 96, 7]
            x = w.transpose(0, 2, 1).reshape(-1, 96)
            report(f'{ds} input window (96, per channel)', x[rng.permutation(len(x))[:a.n]], rng)


if __name__ == '__main__':
    main()
