"""MODI 2026-11-27: checks for the PTB levels of order_check.py (ppt_data.py). Fail closed.
usage (classification folder): python check_oc3.py
"""
import csv
import os
import sys

import numpy as np
import torch

import order_check as O
import ppt_data as PD

FAILS = []


def ok(name, cond, info=''):
    print(f'{"PASS" if cond else "FAIL"} {name} {info}', flush=True)
    if not cond:
        FAILS.append(name)


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X, y, pid, sp = PD.load_ptb()
    ok('a1 64,356 heartbeats of 300 x 15, finite', X.shape == (64356, 300, 15) and bool(np.isfinite(X[::97]).all()))
    pats = [set(np.unique(pid[sp == k])) for k in range(3)]
    ok('a2 patients 142 / 28 / 28, disjoint', [len(p) for p in pats] == [142, 28, 28] and not (pats[0] & pats[1] or pats[0] & pats[2] or pats[1] & pats[2]))
    ppt = {}
    for k, s in enumerate(('train', 'val', 'test')):
        for r in csv.DictReader(open(os.path.join(PD.PTB_DIR, 'split', f'{s}_label.csv'))):
            ppt[int(r['patient_id'])] = (k, int(r['y_true']))
    same = all(ppt[p] == (int(sp[pid == p][0]), int(y[pid == p][0])) for p in np.unique(pid))
    ok('a3 every patient in the PPT split and label', same and len(ppt) == 198)
    ok('a4 one label per patient', all(len(np.unique(y[pid == p])) == 1 for p in np.unique(pid)))
    for ds, P in (('PTB', 10), ('PTB30', 30)):
        d = O.Data(ds, O.parse_cond(ds, 'gru_orig'), 0, dev)
        i = int(d.split['test'][11])
        xi = d.x[i].cpu().numpy()
        ok(f'b1 {ds}: {300 // P} tokens, token j = points jP..jP+P-1 of all 15 leads', xi.shape == (300 // P, P * 15)
           and np.array_equal(xi[3], np.asarray(X[i, 3 * P:4 * P, :]).reshape(-1)))
        ok(f'b2 {ds}: splits = the PPT patients', all(np.array_equal(d.split[k], np.where(sp == v)[0]) for k, v in (('train', 0), ('val', 1), ('test', 2))))
        g = torch.Generator(device=dev); g.manual_seed(0)
        xb, _ = d.tokens(d.split['train'][:8])
        s = O.shuffle_tokens(xb, g, False)
        ok(f'b3 {ds}: shuffle keeps each sample\'s tokens', all(sorted(map(tuple, xb[k].cpu().numpy().round(5))) == sorted(map(tuple, s[k].cpu().numpy().round(5))) for k in range(8)))
        del d
    print('ALL PASS' if not FAILS else f'FAILED: {FAILS}')
    sys.exit(1 if FAILS else 0)


if __name__ == '__main__':
    main()
