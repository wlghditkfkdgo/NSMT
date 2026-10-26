"""Stage 9 (MODI_PROJECT_LOG 2026-10-19): SHD HDF5 -> binned npz, run ONCE with the system python (h5py 2.10):
    /usr/bin/python3 shd_convert.py [--bin_ms 10] [--dur_s 1.0]
The snn_recall training env has no h5py; training reads only the npz.

Spiking Heidelberg Digits (Cramer et al., TNNLS 2022): 700 cochlear channels, spike times in seconds, 20 classes.
Output dataset/shd/shd_bin<bin>ms_<dur>s.npz with
    x_train [n, steps, 700] uint8 spike counts per bin (spikes at t >= dur are dropped; shorter samples are zero-padded)
    y_train [n] int64, spk_train [n] speaker id, n_spikes_train [n] spikes kept (sum of x, for the loader check)
    (same for test)
"""
import argparse
import os

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def convert(path, bin_ms, dur_s, n_ch=700):
    f = h5py.File(path, 'r')
    times, units = f['spikes']['times'], f['spikes']['units']
    labels = np.asarray(f['labels'], dtype=np.int64)
    speakers = np.asarray(f['extra']['speaker'], dtype=np.int64)
    steps = int(round(dur_s * 1000 / bin_ms))
    n = len(labels)
    x = np.zeros((n, steps, n_ch), dtype=np.uint8)
    kept = np.zeros(n, dtype=np.int64)
    for i in range(n):
        t, u = np.asarray(times[i]), np.asarray(units[i]).astype(np.int64)
        m = t < dur_s
        b = np.floor(t[m] * 1000 / bin_ms).astype(np.int64)
        np.add.at(x[i], (b, u[m]), 1)
        kept[i] = int(m.sum())
    if int(x.max()) == 255:
        raise RuntimeError('uint8 overflow in a bin')
    return x, labels, speakers, kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bin_ms', type=float, default=10.)
    ap.add_argument('--dur_s', type=float, default=1.0)
    a = ap.parse_args()
    d = os.path.join(HERE, 'dataset', 'shd')
    out = {}
    for split in ('train', 'test'):
        x, y, s, k = convert(os.path.join(d, f'shd_{split}.h5'), a.bin_ms, a.dur_s)
        out.update({f'x_{split}': x, f'y_{split}': y, f'spk_{split}': s, f'n_spikes_{split}': k})
        print(split, x.shape, 'classes', len(np.unique(y)), 'max count', int(x.max()), 'spikes kept', int(k.sum()))
    path = os.path.join(d, f'shd_bin{int(a.bin_ms)}ms_{a.dur_s:g}s.npz')
    np.savez_compressed(path, **out)
    print('->', path)


if __name__ == '__main__':
    main()
