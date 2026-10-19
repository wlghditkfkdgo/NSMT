"""Stage 9 data (MODI_PROJECT_LOG 2026-10-19): PhysioNet/CinC Challenge 2017 and SHD, protocols adopted in stage 8
(docs/surveys/{ecg_rhythm,shd_ssc}_adopted.md).

CinC 2017  8,528 single-lead AliveCor recordings (300 Hz, 9-61 s), REFERENCE-v3 labels N/A/O/~ -> 0/1/2/3.
           Preprocessing: resample to 125 Hz (polyphase 5/12), 0.5-40 Hz 4th-order Butterworth (zero-phase), per-record z-score.
           Record-wise stratified 5-fold (StratifiedKFold, random_state 0); test = fold `fold`; 10% of the rest held out for
           validation (stratified, random_state 0). Training: a random 30 s crop per record per epoch (records shorter than
           30 s are tiled). Evaluation: 30 s windows every 15 s, softmax averaged per record (rhythm_eval.py).
SHD        Spiking Heidelberg Digits binned by shd_convert.py (10 ms, first 1 s -> 100 x 700 spike counts). Official train /
           test; 10% of train held out for validation (stratified, random_state = seed).
Pre-check  crop_tokens = k > 0 limits every input to the length that yields k tokens; CinC: random crop for training and
           one centred crop per record for evaluation; SHD: the k bins centred on the sample's spike centre of mass.
A sample is x [C, L] float32 (C = 1 for ECG, 700 for SHD); evaluation samples also carry the record / sample index.
"""
import hashlib
import os

import numpy as np
import torch
from scipy.io import loadmat
from scipy.signal import butter, resample_poly, sosfiltfilt
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
CINC_DIR = os.path.join(HERE, 'dataset', 'cinc2017')
SHD_NPZ = os.path.join(HERE, 'dataset', 'shd', 'shd_bin10ms_1s.npz')
CINC_LABELS = {'N': 0, 'A': 1, 'O': 2, '~': 3}
CINC_NAMES = ('N', 'AF', 'Other', 'Noisy')


def tokens_to_len(k, patch, stride):
    """input length that gives k tokens with the ECG model's patching (end padding by one stride)."""
    return (k - 2) * stride + patch


# ---------------------------------------------------------------------------------------------------------------- CinC
def cinc_cache(fs_target=125.0):
    path = os.path.join(CINC_DIR, f'cinc2017_v3_{int(fs_target)}hz.npz')
    if os.path.exists(path):
        d = np.load(path, allow_pickle=False)
        return d['signal'], d['offsets'], d['labels'], d['names']
    ref = [l.strip().split(',') for l in open(os.path.join(CINC_DIR, 'REFERENCE-v3.csv')) if l.strip()]
    sos = butter(4, [0.5, 40.0], btype='band', fs=fs_target, output='sos')
    sigs, labels, names = [], [], []
    up, down = (5, 12) if int(fs_target) == 125 else (int(fs_target), 300)
    for name, lab in ref:
        x = loadmat(os.path.join(CINC_DIR, 'training2017', name + '.mat'))['val'][0].astype(np.float64)
        x = resample_poly(x, up, down)
        x = sosfiltfilt(sos, x)
        x = (x - x.mean()) / (x.std() + 1e-8)
        sigs.append(x.astype(np.float32)); labels.append(CINC_LABELS[lab]); names.append(name)
    offsets = np.cumsum([0] + [len(s) for s in sigs]).astype(np.int64)
    np.savez(path, signal=np.concatenate(sigs), offsets=offsets, labels=np.array(labels, dtype=np.int64), names=np.array(names))
    return np.concatenate(sigs), offsets, np.array(labels, dtype=np.int64), np.array(names)


def cinc_split(labels, fold, val_frac=0.1):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    folds = list(skf.split(np.zeros(len(labels)), labels))
    trval, test = folds[fold]
    tr, va = train_test_split(trval, test_size=val_frac, stratify=labels[trval], random_state=0)
    return np.sort(tr), np.sort(va), np.sort(test)


def _tile(x, n):
    return x[:n] if len(x) >= n else np.resize(x, n)       # np.resize repeats the record


class CincTrain(Dataset):
    def __init__(self, sig, off, labels, idx, win, seed):
        self.sig, self.off, self.labels, self.idx, self.win, self.seed, self.epoch = sig, off, labels, idx, int(win), int(seed), 0

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        r = self.idx[i]
        x = self.sig[self.off[r]:self.off[r + 1]]
        if len(x) > self.win:                                   # crop offset from (seed, epoch, record): reproducible, varies by epoch
            rng = np.random.default_rng([self.seed, self.epoch, int(r)])
            s = int(rng.integers(0, len(x) - self.win + 1))
            x = x[s:s + self.win]
        else:
            x = _tile(x, self.win)
        return torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0), int(self.labels[r])


class CincEval(Dataset):
    """Windows of `win` samples every `hop`; the last window is aligned to the record end. single=True: one centred window."""
    def __init__(self, sig, off, labels, idx, win, hop, single=False):
        self.sig, self.off, self.labels, self.win = sig, off, labels, int(win)
        self.items = []
        for r in idx:
            n = int(off[r + 1] - off[r])
            if single or n <= self.win:
                starts = [max(0, (n - self.win) // 2)] if n > self.win else [0]
            else:
                starts = list(range(0, n - self.win + 1, int(hop)))
                if starts[-1] != n - self.win:
                    starts.append(n - self.win)
            self.items += [(int(r), s) for s in starts]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        r, s = self.items[i]
        x = self.sig[self.off[r]:self.off[r + 1]]
        x = _tile(x[s:s + self.win], self.win) if len(x) >= self.win else _tile(x, self.win)
        return torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0), int(self.labels[r]), r


# ----------------------------------------------------------------------------------------------------------------- SHD
class ShdSet(Dataset):
    def __init__(self, x, y, idx, k=0, with_index=False):
        self.x, self.y, self.idx, self.k, self.with_index = x, y, idx, int(k), with_index
        if self.k:
            T = x.shape[1]
            w = x[idx].astype(np.float32).sum(2)                   # [n, T] spikes per bin
            com = (w * np.arange(T)).sum(1) / np.maximum(w.sum(1), 1)
            self.start = np.clip(np.round(com - self.k / 2).astype(int), 0, T - self.k)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        x = self.x[j].astype(np.float32)                           # [T, 700]
        if self.k:
            x = x[self.start[i]:self.start[i] + self.k]
        x = torch.from_numpy(np.ascontiguousarray(x.T))            # [700, T]
        return (x, int(self.y[j]), int(j)) if self.with_index else (x, int(self.y[j]))


# --------------------------------------------------------------------------------------------------------------- build
def build(args, seed):
    """Returns dict(train=Dataset, val=Dataset(eval), test=Dataset(eval), num_classes, num_channels, seq_len, counts, info)."""
    data = str(args.data).upper()
    k = int(getattr(args, 'crop_tokens', 0) or 0)
    P = int(args.patch_size); S = P // 2
    if data == 'CINC2017':
        sig, off, labels, names = cinc_cache(args.fs_target)
        tr, va, te = cinc_split(labels, int(args.fold), float(args.val_frac))
        win = int(round(args.win_s * args.fs_target)) if not k else tokens_to_len(k, P, S)
        hop = int(round(args.hop_s * args.fs_target))
        single = bool(k)
        out = dict(train=CincTrain(sig, off, labels, tr, win, seed), val=CincEval(sig, off, labels, va, win, hop, single),
                   test=CincEval(sig, off, labels, te, win, hop, single), num_classes=4, num_channels=1, seq_len=win,
                   counts=np.bincount(labels[tr], minlength=4),
                   info=dict(n_records=len(labels), split_sizes=(len(tr), len(va), len(te)), win=win, hop=hop,
                             label_counts=np.bincount(labels, minlength=4).tolist(), names_md5=hashlib.md5(''.join(names[te]).encode()).hexdigest()))
        return out
    if data == 'SHD':
        d = np.load(SHD_NPZ)
        xtr, ytr, xte, yte = d['x_train'], d['y_train'], d['x_test'], d['y_test']
        tr, va = train_test_split(np.arange(len(ytr)), test_size=float(args.val_frac), stratify=ytr, random_state=int(seed))
        T = xtr.shape[1] if not k else k
        out = dict(train=ShdSet(xtr, ytr, np.sort(tr), k), val=ShdSet(xtr, ytr, np.sort(va), k, with_index=True),
                   test=ShdSet(xte, yte, np.arange(len(yte)), k, with_index=True), num_classes=20, num_channels=xtr.shape[2],
                   seq_len=T, counts=np.bincount(ytr[tr], minlength=20),
                   info=dict(n_train=len(ytr), n_test=len(yte), split_sizes=(len(tr), len(va), len(yte)), steps=T))
        return out
    raise ValueError(f'unknown data {args.data}')


def loader(ds, bs, shuffle, seed=0, epoch=0, workers=4):
    g = torch.Generator(); g.manual_seed(int(seed) * 1000 + int(epoch))
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, generator=g if shuffle else None, num_workers=workers,
                      drop_last=False, persistent_workers=False)
