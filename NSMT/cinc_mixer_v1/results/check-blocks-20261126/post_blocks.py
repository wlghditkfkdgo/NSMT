"""Block count (number of SpkMixer blocks) in the existing ETT tuning runs (2026-11-26, exp/f-lif-pop-v3; user question
"2-layer로 연산하면 성능이 개선될 수 있을까?"; no training, post hoc). For every tuning suite, finds the comparisons where only `blocks`
differs (same ds, pl, lr, layers, patch, emb, mlpr, wd; blocks 1 / 2 / 3, seeds 42 / 7 / 2026 all present) and reports, per dataset
(mean over its horizons and seeds): best validation L1, test MSE / MAE and syops energy (syn_energy_uJ), and blocks 2 / 3 minus blocks 1.
The validation change is given in the selection form (mean over horizons of seed-mean validation ratio to blocks 1, minus 1), so its
sign matches the selection tables. The end lists blocks 2 - blocks 1 over all comparisons.
Test values were never used for selection; they are reported here after the fact. Fails closed on a non-finite value or a missing seed.
usage (from cinc_mixer_v1/): python -B results/check-blocks-20261126/post_blocks.py
"""
import math
import os
import sys
from collections import defaultdict

C = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SUITES = ('flif-v1-ett-spkmux-ettune', 'flif-v1-ett-spkmux-ettune2', 'flif-v1-ett-spkmux-retune')
KEY = ('ds', 'lr', 'layers', 'patch', 'emb', 'mlpr', 'wd')
SEEDS = {'42', '7', '2026'}
mean = lambda xs: sum(xs) / len(xs)
ALL = {'2': [], '3': []}
for su in SUITES:
    p = os.path.join(C, 'results', su, 'raw.txt')
    if not os.path.exists(p):
        sys.exit(f'STOP: {p} missing')
    rows = [dict(t.split('=', 1) for t in l.split()[1:] if '=' in t) for l in open(p) if l.startswith('RESULT')]
    g = defaultdict(lambda: defaultdict(dict))          # (KEY) -> blocks -> (pl, seed) -> row
    for r in rows:
        for k in ('mse', 'mae', 'val_loss_best', 'syn_energy_uJ'):
            if not math.isfinite(float(r[k])):
                sys.exit(f'STOP: {su} {k} not finite')
        g[tuple(r[k] for k in KEY)][r['blocks']][(r['pl'], r['seed'])] = r
    print(f'== {su} ({len(rows)} rows)')
    found = 0
    for k, byb in sorted(g.items()):
        if not {'1', '2', '3'} <= set(byb):
            continue
        cells = set(byb['1']) & set(byb['2']) & set(byb['3'])
        pls = sorted({pl for pl, s in cells}, key=int)
        if not pls or any({s for q, s in cells if q == pl} != SEEDS for pl in pls):
            continue
        found += 1
        ks = dict(zip(KEY, k))
        print(f'  {ks["ds"]} (H {"/".join(pls)}; lr {ks["lr"]}, layers {ks["layers"]}, patch {ks["patch"]}, d_model {ks["emb"]}, '
              f'MLP {ks["mlpr"]}, wd {ks["wd"]})')
        base = {}
        vh = {b: {pl: mean([float(byb[b][(pl, sd)]['val_loss_best']) for sd in SEEDS]) for pl in pls} for b in ('1', '2', '3')}
        for b in ('1', '2', '3'):
            v = {m: mean([float(byb[b][c][m]) for c in sorted(cells)]) for m in ('val_loss_best', 'mse', 'mae', 'syn_energy_uJ')}
            score = mean([vh[b][pl] / vh['1'][pl] for pl in pls])
            if b == '1':
                base = v
            else:
                ALL[b].append((su, ks['ds'], score, v['mse'] - base['mse'], v['mae'] - base['mae'], v['syn_energy_uJ'] / base['syn_energy_uJ']))
            d = '' if b == '1' else (f'  | vs 1: val score {100 * (score - 1):+.2f} %, '
                                     f'MSE {v["mse"] - base["mse"]:+.4f}, MAE {v["mae"] - base["mae"]:+.4f}, '
                                     f'energy x{v["syn_energy_uJ"] / base["syn_energy_uJ"]:.2f}')
            print(f'    blocks {b}: val {v["val_loss_best"]:.6f}  MSE {v["mse"]:.4f}  MAE {v["mae"]:.4f}  energy {v["syn_energy_uJ"]:.2f} uJ{d}')
    print(f'  comparisons found: {found}')
for b in ('2', '3'):
    L = ALL[b]
    print(f'== blocks {b} - blocks 1 over {len(L)} comparisons: val score lower in {sum(x[2] < 1 for x in L)}, '
          f'test MSE lower in {sum(x[3] < 0 for x in L)} (mean {mean([x[3] for x in L]):+.4f}, range {min(x[3] for x in L):+.4f} .. '
          f'{max(x[3] for x in L):+.4f}), MAE lower in {sum(x[4] < 0 for x in L)} (mean {mean([x[4] for x in L]):+.4f}), '
          f'energy x{min(x[5] for x in L):.2f} .. x{max(x[5] for x in L):.2f}')
