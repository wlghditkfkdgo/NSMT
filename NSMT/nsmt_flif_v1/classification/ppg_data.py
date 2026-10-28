"""MODI 2026-10-28: PPG-DaLiA heart-rate regression with the Log-NCDE protocol (Walker et al., ICML 2024; data_dir/process_ppg.py
of github.com/Benjamin-Walker/log-neural-cdes, identical in tk-rusch/linoss), re-implemented here.

Per subject S1..S15 (PPG_FieldStudy/S{i}/S{i}.pkl, UCI dataset 495): wrist ACC (32 Hz, 3 axes, repeated x2), BVP (64 Hz), EDA and
TEMP (4 Hz, repeated x16) -> 6 channels at 64 Hz, each min-max scaled to [-1, 1] over the subject's whole recording (ACC: the three
axes together); label = mean heart rate of the 8-s ECG window every 2 s, with three copies of the first value prepended (label k
then ends with input sample 128 (k + 1)), min-max scaled to [-1, 1] per subject. One of six contiguous 70 / 15 / 15 block orders
per subject. Windows: 49,920 input samples with stride 4,992, 390 labels with stride 39 (one label per 128 samples = 2 s).
Difference from the original code: the block order is drawn with random.Random(seed).randint(0, 5) subject by subject; the
original calls random.randint without a seed, so its split is not reproducible (the published numbers use unknown splits).
The pickles are read with an unpickler that only rebuilds numpy arrays and builtin containers.
usage: python ppg_data.py --prepare [--seed 0]   -> dataset/ppg_dalia/processed/logncde_seed<seed>.npz
"""
import argparse
import io
import os
import pickle
import random
import zipfile

import numpy as np
import torch
from torch.utils.data import Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, 'dataset', 'ppg_dalia')
WIN, STRIDE, LWIN, LSTRIDE, SPL = 49920, 4992, 390, 39, 128
VARIANTS = {0: ((0, .7), (.7, .85), (.85, 1)), 1: ((0, .7), (.85, 1), (.7, .85)), 2: ((.15, .85), (0, .15), (.85, 1)),
            3: ((.15, .85), (.85, 1), (0, .15)), 4: ((.30, 1), (0, .15), (.15, .30)), 5: ((.30, 1), (.15, .30), (0, .15))}


class SafeUnpickler(pickle.Unpickler):
    ALLOWED = {('numpy.core.multiarray', '_reconstruct'), ('numpy', 'ndarray'), ('numpy', 'dtype'),
               ('numpy.core.multiarray', 'scalar'), ('numpy._core.multiarray', '_reconstruct'), ('numpy._core.multiarray', 'scalar'),
               ('builtins', 'dict'), ('builtins', 'list'), ('builtins', 'tuple'), ('__builtin__', 'dict'), ('__builtin__', 'list'),
               ('copy_reg', '_reconstructor'), ('copyreg', '_reconstructor'), ('__builtin__', 'object'), ('builtins', 'object')}

    def find_class(self, module, name):
        if (module, name) in self.ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f'blocked {module}.{name}')


def _mm(a):
    return 2 * (a - np.min(a)) / (np.max(a) - np.min(a)) - 1


def _cut(a, lo, hi):
    return a[int(lo * len(a)):] if hi == 1 else a[int(lo * len(a)):int(hi * len(a))]


def load_subject(raw):
    w = raw['signal']['wrist']
    acc = np.repeat(w['ACC'], 2, axis=0); bvp = w['BVP']; eda = np.repeat(w['EDA'], 16, axis=0); temp = np.repeat(w['TEMP'], 16, axis=0)
    x = np.concatenate([_mm(acc), _mm(bvp), _mm(eda), _mm(temp)], axis=1)
    lab = raw['label']
    y = _mm(np.concatenate([[lab[0]], [lab[0]], [lab[0]], lab], axis=0))
    return x, y, lab


def prepare(seed=0):
    src = zipfile.ZipFile(os.path.join(ROOT, 'data.zip'))                # inner archive of the UCI download (unzip ppg_dalia.zip data.zip)
    rng = random.Random(seed)
    out, info = {}, []
    for i in range(1, 16):
        member = [n for n in src.namelist() if n.endswith(f'S{i}/S{i}.pkl')]
        if len(member) != 1:
            raise RuntimeError(f'S{i}.pkl: {member}')
        raw = SafeUnpickler(io.BytesIO(src.read(member[0])), encoding='latin1').load()
        x, y, lab = load_subject(raw)
        out[f'S{i}_bpm_range'] = np.array([np.min(lab), np.max(lab)], dtype=np.float64)   # undoes the label scaling (BPM)
        v = rng.randint(0, 5)
        row = dict(subject=i, variant=v, n_in=len(x), n_lab=len(y), n_in_over_128=len(x) / SPL)
        for split, (lo, hi) in zip(('train', 'val', 'test'), VARIANTS[v]):
            xs, ys = _cut(x, lo, hi), _cut(y, lo, hi)
            n_x = (len(xs) - WIN) // STRIDE + 1 if len(xs) >= WIN else 0
            n_y = (len(ys) - LWIN) // LSTRIDE + 1 if len(ys) >= LWIN else 0
            off = int(lo * len(x)) - SPL * int(lo * len(y))                # input sample offset vs label grid (as in the original)
            row[f'{split}_windows_x'], row[f'{split}_windows_y'], row[f'{split}_offset'] = n_x, n_y, off
            out[f'S{i}_{split}_x'] = xs.astype(np.float32); out[f'S{i}_{split}_y'] = ys.astype(np.float32)
        info.append(row)
    os.makedirs(os.path.join(ROOT, 'processed'), exist_ok=True)
    path = os.path.join(ROOT, 'processed', f'logncde_seed{seed}.npz')
    np.savez(path, info=np.array([str(r) for r in info]), **out)
    return path, info


class PPGWindows(Dataset):
    """windows of one split: x [WIN, 6] float32, y [LWIN] float32, subject index 0..14 (the original pairs input window j with
    label window j). bpm_range[s] = (min, max) heart rate of subject s, to undo the label scaling."""
    def __init__(self, npz, split):
        d = np.load(npz)
        self.x, self.y, self.idx = [], [], []
        self.bpm_range = np.stack([d[f'S{i}_bpm_range'] for i in range(1, 16)])
        for i in range(1, 16):
            xs, ys = d[f'S{i}_{split}_x'], d[f'S{i}_{split}_y']
            n = min((len(xs) - WIN) // STRIDE + 1 if len(xs) >= WIN else 0, (len(ys) - LWIN) // LSTRIDE + 1 if len(ys) >= LWIN else 0)
            self.x.append(xs); self.y.append(ys)
            self.idx += [(i - 1, j) for j in range(n)]

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        s, j = self.idx[k]
        x = self.x[s][j * STRIDE:j * STRIDE + WIN]; y = self.y[s][j * LSTRIDE:j * LSTRIDE + LWIN]
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y)), s


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--prepare', action='store_true'); ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    if a.prepare:
        path, info = prepare(a.seed)
        for r in info:
            print(r)
        tot = {s: sum(r[f'{s}_windows_x'] for r in info) for s in ('train', 'val', 'test')}
        mism = [r['subject'] for r in info for s in ('train', 'val', 'test') if r[f'{s}_windows_x'] != r[f'{s}_windows_y']]
        print('windows', tot, 'x/y window-count mismatch in subjects', sorted(set(mism)) or 'none', '->', path)
