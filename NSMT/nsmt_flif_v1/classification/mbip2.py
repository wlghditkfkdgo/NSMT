"""MODI 2026-11-27 ((가) v2, user decision 11-27 "(가) 우선 시작해보자"): MIT-BIH inter-patient beat classification with a waveform path
and a separate RR-interval path (docs/MODI_PROJECT_LOG.md 2026-11-27 02:34, stage 0 = validation only).

Why two paths (11-26 23:00 analysis): with the beat waveform in the input, the MBIP models memorised the S morphology of the few training
patients and ignored timing (validation S F1 4-17); a model on the RR sequence alone reached 49-54, but V needs the waveform.

data: oc_data.load_mitdb() (MLII, causal 0.5 Hz high-pass, window R - 0.25 .. R + 0.40 s at 180 Hz, pre-RR from the annotations). Targets
= N / S / V / F beats with at least KMAX = 80 beats before them in the record (so every past window used later fits; same target set for
all conditions), window inside the signal. Splits: training = DS1 minus 201 / 118 / 223 / 114, validation = those four, test = DS2
(stage 0 never evaluates the test split).
model: waveform path = current beat window (117 points, standardised per beat) -> Linear-GELU-Linear-LayerNorm (128) [+ dropout];
RR path = pre-RR of the K past beats and of the current beat (K + 1 values, seconds) -> Linear(1 -> 32) -> 2-layer GRU (64), last step;
head = concat -> Linear(192 -> 128)-GELU-Linear(-> 4). --paths both | rr | wave.
training: AdamW 1e-3, wd 1e-2, 1 warm-up epoch + cosine, 30 epochs, batch 256, clip 1.0, CE with or without balanced class weights;
best validation 4-class macro-F1 epoch.
usage (classification folder): python mbip2.py --loss w|u --wdrop 0|0.5 --paths both|rr|wave [--K 16] [--seed 42] --split val
"""
import argparse
import json
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import oc_data as D

KMAX = 80
MB_VAL = [201, 118, 223, 114]


class Data2:
    def __init__(self, K, dev):
        m = D.load_mitdb()
        rec, cls, rr, ok = m['rec'], m['cls'], m['rr'], m['ok']
        idx = np.zeros(len(rec), dtype=np.int64)
        for r in np.unique(rec):
            w = np.where(rec == r)[0]; idx[w] = np.arange(len(w))
        tgt = (idx >= KMAX + 1) & (cls <= 3) & ok & np.isin(rec, D.DS1 + D.DS2)
        t = np.where(tgt)[0]
        if (rec[t - KMAX] != rec[t]).any() or np.isnan(rr[t[:, None] - np.arange(KMAX + 1)[None]]).any():
            raise SystemExit('context crosses a record or lacks a pre-RR')
        w = m['w']
        self.wave = torch.from_numpy(((w - w.mean(1, keepdims=True)) / (w.std(1, keepdims=True) + 1e-6)).astype(np.float32)).to(dev)
        self.rr = torch.from_numpy(np.nan_to_num(rr).astype(np.float32)).to(dev)
        self.t, self.rec_t, self.dev, self.K = t, rec[t], dev, K
        self.y = torch.from_numpy(cls[t]).to(dev)
        trm = np.isin(rec[t], D.DS1) & ~np.isin(rec[t], MB_VAL)
        self.split = {'train': np.where(trm)[0], 'val': np.where(np.isin(rec[t], MB_VAL))[0], 'test': np.where(np.isin(rec[t], D.DS2))[0]}
        self.rrctx = torch.from_numpy(t[:, None] - np.arange(K, -1, -1)[None]).to(dev)     # [n, K + 1], current beat last
        cnt = np.bincount(cls[t[trm]], minlength=4).astype(np.float64)
        self.cw = torch.tensor(cnt.sum() / (4 * cnt), dtype=torch.float32, device=dev)

    def batch(self, i):
        i = torch.as_tensor(i, device=self.dev)
        tt = torch.as_tensor(self.t, device=self.dev)[i]
        return self.wave[tt], self.rr[self.rrctx[i]].unsqueeze(-1), self.y[i]


class TwoPath(nn.Module):
    def __init__(self, paths, wdrop, d=128, h=64):
        super().__init__()
        self.paths = paths
        self.wave = nn.Sequential(nn.Linear(117, d), nn.GELU(), nn.Linear(d, d), nn.LayerNorm(d), nn.Dropout(wdrop)) if paths != 'rr' else None
        if paths != 'wave':
            self.rr_in = nn.Linear(1, 32)
            self.rr = nn.GRU(32, h, num_layers=2, batch_first=True, dropout=0.1)
        fin = (d if paths != 'rr' else 0) + (h if paths != 'wave' else 0)
        self.head = nn.Sequential(nn.Linear(fin, 128), nn.GELU(), nn.Linear(128, 4))

    def forward(self, wave, rr):
        z = []
        if self.wave is not None:
            z.append(self.wave(wave))
        if self.paths != 'wave':
            o, _ = self.rr(self.rr_in(rr)); z.append(o[:, -1])
        return self.head(torch.cat(z, -1))


def scores(p, y):
    out = {}
    for k, n in enumerate(('N', 'S', 'V', 'F')):
        tp, fp, fn = int(((p == k) & (y == k)).sum()), int(((p == k) & (y != k)).sum()), int(((p != k) & (y == k)).sum())
        out[f'{n}_f1'] = 0.0 if tp == 0 else 200.0 * tp / (2 * tp + fp + fn)
        if n in ('S', 'V'):
            out[f'{n}_se'] = 100.0 * tp / max(1, tp + fn); out[f'{n}_pp'] = 100.0 * tp / max(1, tp + fp)
    out['macro_f1'] = float(np.mean([out[f'{n}_f1'] for n in ('N', 'S', 'V', 'F')]))
    out['S_pred'], out['S_true'] = int((p == 1).sum()), int((y == 1).sum())
    return out


@torch.no_grad()
def predict(model, data, idx):
    model.eval(); ps = []
    for s in range(0, len(idx), 2048):
        w, r, _ = data.batch(idx[s:s + 2048]); ps.append(model(w, r).argmax(1))
    return torch.cat(ps).cpu().numpy(), data.y[torch.as_tensor(idx, device=data.dev)].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--loss', choices=['w', 'u'], required=True); ap.add_argument('--wdrop', type=float, default=0.0)
    ap.add_argument('--paths', choices=['both', 'rr', 'wave'], default='both'); ap.add_argument('--K', type=int, default=16)
    ap.add_argument('--seed', type=int, default=42); ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--split', choices=['val'], default='val')            # stage 0: validation only (no test evaluation in this script)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device('cuda')
    data = Data2(a.K, dev)
    model = TwoPath(a.paths, a.wdrop).to(dev)
    tr, va = data.split['train'], data.split['val']
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    spe = math.ceil(len(tr) / 256); total, warm = a.epochs * spe, spe
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    wt = data.cw if a.loss == 'w' else None
    best = None
    for ep in range(a.epochs):
        model.train(); order = np.random.permutation(tr)
        for b in range(spe):
            w, r, y = data.batch(order[b * 256:(b + 1) * 256])
            loss = F.cross_entropy(model(w, r), y, weight=wt)
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        s = scores(*predict(model, data, va))
        if best is None or s['macro_f1'] > best['macro_f1']:
            best = dict(epoch=ep, **s)
    print('V2VAL ' + json.dumps(dict(loss=a.loss, wdrop=a.wdrop, paths=a.paths, K=a.K, seed=a.seed, n_train=len(tr), n_val=len(va),
                                     **{k: (round(v, 2) if isinstance(v, float) else v) for k, v in best.items()})), flush=True)


if __name__ == '__main__':
    main()
