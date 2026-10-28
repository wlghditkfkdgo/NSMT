"""MODI 2026-10-28: summary of the PPG-DaLiA suite with the pre-registered rules (docs/MODI_PROJECT_LOG.md 2026-10-28).

Unit: test MSE on the scaled labels (tables x 1e-2), best-validation weights, seeds 42 / 7 / 2026, paired by seed.
Relative difference r = (MSE_A - MSE_B) / MSE_B per seed. 'better' = mean r <= -10 % and r < 0 in 3 of 3 seeds; 'worse' = mean
r >= +10 % and r > 0 in 3 of 3; otherwise 'no clear difference'. Missing / non-finite runs -> 'incomplete'.
  V1 (main)  P5 n (window 3 + T = N neocortex) vs P6 (window 3 + memoryless LIF neocortex), n = LIF, f-LIF, population LIF,
             population f-LIF: 'better' = the neuron state carries information from beyond the 8-s attention window
  V2         beyond smoothing: V1 'better' AND P5 n MSE < MSE of P1 + its validation-chosen smoothing (smooth.csv) in 3 of 3 seeds
  V3         P4 n (Neo-mixer forward) vs P1 and vs P3 (report)
  V4         context for attention: P2 vs P1, P3 vs P1 (report)
  V5         non-spiking recurrent reference: GRU vs P1 (report)
usage: python summarize_ppg.py [--suite flif-v1-ppg]
"""
import argparse
import csv
import math
import os

import numpy as np

from summarize_rhythm import parse

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = (42, 7, 2026)
NEURONS = ('lif', 'flif', 'pop_lif', 'pop_flif')
NL = {'lif': 'LIF', 'flif': 'f-LIF', 'pop_lif': 'population LIF', 'pop_flif': 'population f-LIF'}
LAB = {'ppg_tok': 'P0 token only (window 1)', 'ppg_ssa_w8': 'P1 Spikformer-style, window 3 (8 s)', 'ppg_ssa_w32': 'P2 window 15 (32 s)',
       'ppg_ssa_full': 'P3 all past', 'ppg_spkneo_lif_nomem': 'P6 window 3 + memoryless LIF neocortex', 'ppg_gru': 'GRU (non-spiking)'}
LAB.update({f'ppg_mix_{n}': f'P4 Neo-mixer forward, {NL[n]}' for n in NEURONS})
LAB.update({f'ppg_spkneo_{n}': f'P5 window 3 + T = N neocortex, {NL[n]}' for n in NEURONS})
ORDER = ['ppg_tok', 'ppg_ssa_w8', 'ppg_ssa_w32', 'ppg_ssa_full'] + [f'ppg_mix_{n}' for n in NEURONS] + \
        [f'ppg_spkneo_{n}' for n in NEURONS] + ['ppg_spkneo_lif_nomem', 'ppg_gru']


def per_seed(rows, cond, key):
    out = []
    for s in SEEDS:
        v = [r.get(key) for r in rows if r.get('cond') == cond and int(r.get('seed', -1)) == s]
        v = v[-1] if v else None
        out.append(float(v) if isinstance(v, (int, float)) and math.isfinite(float(v)) else None)
    return out


def rel(a, b):
    if any(x is None for x in a + b) or any(y == 0 for y in b):
        return None
    return [(x - y) / y for x, y in zip(a, b)]


def verdict(r):
    if r is None:
        return 'incomplete'
    m = np.mean(r)
    if m <= -0.10 and all(x < 0 for x in r):
        return 'better'
    if m >= 0.10 and all(x > 0 for x in r):
        return 'worse'
    return 'no clear difference'


def ms(v, sc=1., nd=2):
    return 'incomplete' if v is None or any(x is None for x in v) else f'{np.mean(v) * sc:.{nd}f} (' + ' / '.join(f'{x * sc:.{nd}f}' for x in v) + ')'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--suite', default='flif-v1-ppg')
    a = ap.parse_args()
    res = os.path.join(HERE, 'results', a.suite)
    rows = parse(os.path.join(res, 'raw.txt'))
    sm = []
    if os.path.exists(os.path.join(res, 'smooth.csv')):
        for r in csv.DictReader(open(os.path.join(res, 'smooth.csv'))):
            sm.append(dict(cond=r['cond'], seed=float(r['seed']), test_mse=float(r['test_mse']), test_mae_bpm=float(r['test_mae_bpm']),
                           method=f"{r['method']} {r['param']}"))
    L = [f'# PPG-DaLiA heart-rate regression `{a.suite}` (Log-NCDE protocol, seed-0 split, causal models) — summary', '',
         'Test MSE x 1e-2 on the scaled labels and MAE in BPM; best-validation weights; mean (seed 42 / 7 / 2026). Literature numbers '
         'use other (unseeded) splits and mostly non-causal settings: reference only.', '',
         '| model | MSE x 1e-2 | MAE BPM | + smoothing MSE x 1e-2 | total µJ (FFT) | params M |', '|---|---|---|---|---|---|']
    for c in ORDER:
        L.append(f"| {LAB[c]} | {ms(per_seed(rows, c, 'mse'), 100)} | {ms(per_seed(rows, c, 'mae_bpm'), 1)} | "
                 f"{ms(per_seed(sm, c, 'test_mse'), 100)} | {ms(per_seed(rows, c, 'total_fft_uJ'), 1, 1) if c != 'ppg_gru' else '—'} | "
                 f"{ms(per_seed(rows, c, 'params_M'), 1, 3)} |")
    L += ['', '## Pre-registered comparisons (relative MSE difference per seed; better = mean <= -10 % and 3 / 3 seeds)', '',
          '| question | A | B | (A − B) / B per seed | verdict |', '|---|---|---|---|---|']
    mse = lambda c: per_seed(rows, c, 'mse')
    def line(q, ca, cb, rb=None):
        r = rel(mse(ca), mse(cb) if rb is None else rb)
        L.append(f"| {q} | {LAB.get(ca, ca)} | {LAB.get(cb, cb)} | {' / '.join(f'{x * 100:+.1f} %' for x in r) if r else '—'} | {verdict(r)} |")
        return verdict(r)
    v1 = {n: line('V1 state beyond the window (main)', f'ppg_spkneo_{n}', 'ppg_spkneo_lif_nomem') for n in NEURONS}
    p1s = per_seed(sm, 'ppg_ssa_w8', 'test_mse')
    for n in NEURONS:
        a_ = mse(f'ppg_spkneo_{n}')
        if v1[n] == 'incomplete' or any(x is None for x in a_ + p1s):
            v2 = 'incomplete'
        else:
            v2 = 'yes' if v1[n] == 'better' and all(x < y for x, y in zip(a_, p1s)) else 'no'
        L.append(f"| V2 beyond smoothing | {LAB[f'ppg_spkneo_{n}']} | P1 + smoothing | "
                 f"{' / '.join(f'{(x - y) / y * 100:+.1f} %' for x, y in zip(a_, p1s)) if v2 != 'incomplete' else '—'} | {v2} |")
    for n in NEURONS:
        line('V3 state only vs attention 8 s', f'ppg_mix_{n}', 'ppg_ssa_w8'); line('V3 state only vs attention all past', f'ppg_mix_{n}', 'ppg_ssa_full')
    line('V4 context for attention', 'ppg_ssa_w32', 'ppg_ssa_w8'); line('V4 context for attention', 'ppg_ssa_full', 'ppg_ssa_w8')
    line('V4 context for attention', 'ppg_ssa_w8', 'ppg_tok')
    line('V5 recurrent reference', 'ppg_gru', 'ppg_ssa_w8')
    for n in ('flif', 'pop_lif', 'pop_flif'):
        line('neuron (report)', f'ppg_spkneo_{n}', 'ppg_spkneo_lif'); line('neuron (report)', f'ppg_mix_{n}', 'ppg_mix_lif')
    L.append('')
    open(os.path.join(res, 'summary.md'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
