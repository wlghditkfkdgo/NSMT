"""MODI 2026-11-26 (user question 11-26: "S를 거의 찾지 못하는 이유가 뭐야? 손실함수?"): validation-only probes of the MBIP failure
(docs/MODI_PROJECT_LOG.md 2026-11-26). DS1 only: training records = DS1 minus 201 / 118 / 223 / 114, validation = those four; DS2 is
not loaded for evaluation. Same model family and training as order_check.py (MLP token encoder + 2-layer GRU, AdamW 1e-3, wd 1e-2,
1 warm-up epoch + cosine, 30 epochs, batch 256, clip 1.0, best validation 4-class macro-F1 epoch), seed 42. What changes:
  --loss  w = class-balanced CE (as MBIP) | u = plain CE
  --inp   full = beat window (117, per-beat standardised) + pre-RR      (as MBIP)
          rr   = pre-RR only (no waveform)
          full_rel / rr_rel = the same + pre-RR / mean pre-RR of the 16 beats before (relative timing; past only)
  --L     1 (current beat) or 17 (16 past beats + current); for L1 the "_rel" value is computed from the 16 beats before.
usage (classification folder): python mbip_probe.py --loss w --inp full --L 17
"""
import argparse
import json
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import order_check as O


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--loss', choices=['w', 'u'], required=True)
    ap.add_argument('--inp', choices=['full', 'rr', 'full_rel', 'rr_rel'], required=True); ap.add_argument('--L', type=int, choices=[1, 17], required=True)
    ap.add_argument('--seed', type=int, default=42); ap.add_argument('--epochs', type=int, default=30)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device('cuda')
    d = O.Data('MBIP', O.parse_cond('MBIP', f'gru_orig_L{a.L}'), 0, dev)
    tr, va = d.split['train'], d.split['val']
    rel_t = d.B[torch.from_numpy(d.t).to(dev), -1] / d.mrr                       # pre-RR / mean of the 16 before, per target

    def tokens(i):
        i = torch.as_tensor(i, device=dev)
        x = d.B[d.ctx[i]]                                                         # [B, L, 118]: window + pre-RR
        if a.inp.startswith('rr'):
            x = x[..., -1:]                                                       # pre-RR only
        if a.inp.endswith('_rel'):
            r = torch.zeros(x.shape[0], x.shape[1], 1, device=dev); r[:, -1, 0] = rel_t[i]   # relative timing on the current token
            x = torch.cat([x, r], -1)
        return x, d.y[i]
    x0, _ = tokens(tr[:2])
    model = O.RefModel('MBIP', dict(model='gru', order='orig', k=None, L=a.L), a.L, x0.shape[-1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    spe = math.ceil(len(tr) / 256); total, warm = a.epochs * spe, spe
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    w = d.cw if a.loss == 'w' else None
    best = None
    for ep in range(a.epochs):
        model.train(); order = np.random.permutation(tr)
        for b in range(spe):
            x, y = tokens(order[b * 256:(b + 1) * 256])
            loss = F.cross_entropy(model(x), y, weight=w)
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        model.eval(); ps = []
        with torch.no_grad():
            for s in range(0, len(va), 2048):
                ps.append(model(tokens(va[s:s + 2048])[0]).argmax(1))
        p = torch.cat(ps).cpu().numpy(); y = d.y[torch.as_tensor(va, device=dev)].cpu().numpy()
        f1 = []
        for k in range(4):
            tp, fp, fn = ((p == k) & (y == k)).sum(), ((p == k) & (y != k)).sum(), ((p != k) & (y == k)).sum()
            f1.append(0.0 if tp == 0 else 200.0 * tp / (2 * tp + fp + fn))
        mf = float(np.mean(f1))
        if best is None or mf > best['val_mf1']:
            tp = int(((p == 1) & (y == 1)).sum())
            best = dict(epoch=ep, val_mf1=mf, S_f1=f1[1], V_f1=f1[2], N_f1=f1[0], S_se=100.0 * tp / max(1, (y == 1).sum()),
                        S_pp=100.0 * tp / max(1, (p == 1).sum()), S_pred=int((p == 1).sum()), S_true=int((y == 1).sum()))
    print('PROBE ' + json.dumps(dict(loss=a.loss, inp=a.inp, L=a.L, **{k: (round(v, 2) if isinstance(v, float) else v) for k, v in best.items()})), flush=True)


if __name__ == '__main__':
    main()
