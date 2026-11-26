"""MODI 2026-11-27 (user request 11-27: PPT의 dataset 검토 -> 확인 시험): PTB (PhysioNet PTB Diagnostic ECG, myocardial infarction vs
healthy control) as used by COMET (Wang et al., NeurIPS 2023), Medformer (NeurIPS 2024) and PPT (Kim et al., ICLR 2025).

Signals: the COMET / Medformer processed PTB (Medformer README, Google Drive file 14fBIXc2gSHm00wLaejNIsPgitc-wZdXu; COMET's figshare
copy did not download from this server): 198 patients, 64,356 heartbeats of 300 points (250 Hz, from the R-peak, zero padded) x 15 leads,
standard-scaled by COMET; label per patient (Label/label.npy: [label, patient id]). Read with allow_pickle=False only.
Split: PPT's patient split (eai-lab/PPT Google Drive, PTB/{train,val,test}_label.csv: 142 / 28 / 28 patients, disjoint). PPT used 62,370
samples grouped by a trial_id that does not index these heartbeats; its feature files are not public, so every heartbeat of a patient is
used here (64,356). Patient ids and labels agree with PPT for 198 / 198 patients (checked 11-27).
usage (classification folder): python ppt_data.py      (builds dataset/ppt_ptb/ptb_x.npy + ptb_meta.npz)
"""
import csv
import glob
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PTB_DIR = os.path.join(HERE, 'dataset', 'ppt_ptb')
PTB_X = os.path.join(PTB_DIR, 'ptb_x.npy')
PTB_META = os.path.join(PTB_DIR, 'ptb_meta.npz')


def build_ptb():
    lab = np.load(os.path.join(PTB_DIR, 'medformer', 'PTB', 'Label', 'label.npy'), allow_pickle=False)
    lab_by = {int(p): int(c) for c, p in lab}
    split_of = {}
    for k, s in enumerate(('train', 'val', 'test')):
        for r in csv.DictReader(open(os.path.join(PTB_DIR, 'split', f'{s}_label.csv'))):
            p = int(r['patient_id'])
            if split_of.setdefault(p, k) != k:
                raise SystemExit(f'patient {p} in two PPT splits')
            if lab_by[p] != int(r['y_true']):
                raise SystemExit(f'patient {p}: label {lab_by[p]} vs PPT {r["y_true"]}')
    files = sorted(glob.glob(os.path.join(PTB_DIR, 'medformer', 'PTB', 'Feature', 'feature_*.npy')))
    X, P = [], []
    for f in files:
        a = np.load(f, allow_pickle=False)
        if a.shape[1:] != (300, 15):
            raise SystemExit(f'{f}: shape {a.shape}')
        p = int(os.path.basename(f)[8:11]); X.append(a.astype(np.float32)); P.append(np.full(len(a), p))
    X, P = np.concatenate(X), np.concatenate(P)
    y = np.array([lab_by[p] for p in P]); sp = np.array([split_of[p] for p in P])
    np.save(PTB_X, X)
    np.savez(PTB_META, y=y, pid=P, split=sp)
    for k, s in enumerate(('train', 'val', 'test')):
        m = sp == k
        print(f'{s}: heartbeats {m.sum()}, patients {len(np.unique(P[m]))}, MI / healthy {int((y[m] == 1).sum())} / {int((y[m] == 0).sum())}')
    print('X', X.shape, 'finite', bool(np.isfinite(X).all()))


def load_ptb():
    m = np.load(PTB_META)
    return np.load(PTB_X, mmap_mode='r'), m['y'], m['pid'], m['split']


if __name__ == '__main__':
    build_ptb()
