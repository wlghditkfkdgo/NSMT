"""Membrane shortcut test, LIF only (2026-11-08, exp/f-lif-pop-v3; PROJECT_LOG membrane shortcut pre-registration). Tuned settings, not re-tuned; pairs by cell and seed.
    Controls: SpkMixer muxlin with the tuned residual (ETT IAND, MIT-BIH ADD), tuned Spikformer-style (IAND); both existing runs.
    Questions: M1 SpkMixer + MS vs SpkMixer, M2 Spikformer-style + MS vs Spikformer-style, M3 SpkMixer + MS vs Spikformer-style + MS. Report only: M4 SpkMixer + MS vs Spikformer-style.
Rules (each task's existing one):
    ETT      on test MSE: r = 100 (A - B) / B per seed, R = mean over the 4 cells of the seed-mean r, better = R <= -1 % and >= 8 of 12 pairs lower, worse = mirror.
    MIT-BIH  on test macro-F1: difference in %p per seed, better = mean >= +0.5 and >= 2 of 3 seeds higher, worse = mirror.
Energy: total_fft_uJ_corr of energy_recount, and for the shortcut runs plus the shortcut additions, which no hook counts:
    2 shortcuts x T_h x N x D x C real-valued additions at E_AC = 0.9 pJ per window (N * D is taken from the recount's head_MAC on ETT and checked against 24 * 64; MIT-BIH 11 * 384, C = 1).
Report only: firing rates, best validation epoch, last-epoch ETT MSE, per-class F1, default-lr Spikformer-style on MIT-BIH.
Fail closed (exit 2, nothing written): a missing run, a SKIP line of a used condition, a duplicate, a non-finite value, a recount that does not reproduce the logged metric,
a non-binary head input, a recount rule that is not binary on a shortcut run, or a best epoch that is not the argmin of validation loss.
usage: python -B summarize_ms.py --task ett|mitbih [--suite <suite>] [--out <file>] [--tags <A1>,<A2>]   ->   results/<suite>/summary.md
"""
import argparse
import csv
import math
import os
import re
import sys

from summarize_l2 import CELLS, SEEDS, RES, TUNE, compare, mean, num, read_result, spikformer, stop

HERE = os.path.dirname(os.path.abspath(__file__))
LR_MX = {'ETTh2': '0.0005', 'ETTm2': '0.0001'}
LR_SF = {'ETTh2': '0.0005', 'ETTm2': '0.00025'}
LY_SF = {'ETTh2': '1', 'ETTm2': '2'}
E_AC_UJ = 0.9e-6
T_H = 4
MITB = os.path.join(RES, 'flif-v1-mitbih-spkmux-tune')
MIT_BASE = os.path.join(RES, 'flif-v1-mitbih-spkmux')


def ms_add_ett(e):
    """Shortcut additions per window, µJ; N * D from the recount's head_MAC = N * D * H * C."""
    H, C = int(e['pl']), 7
    nd = int(e['head_MAC']) / (H * C)
    if nd != 24 * 64:
        stop(f'N * D = {nd} on {e["ds"]} H{H}, expected {24 * 64}')
    return 2 * T_H * nd * C * E_AC_UJ


def load_ett(raw, en, wanted, cond, add=False):
    E = {}
    for e in csv.DictReader(open(en)):
        k = (e.get('tag', ''), e['ds'], int(e['pl']), int(e['seed']))
        if k in E:
            stop(f'duplicate energy row {k} in {en}')
        E[k] = e
    T = {}
    for r in read_result(raw, lambda c: c == cond):
        if r['tag'] != wanted(r['ds']):
            continue
        k = (r['ds'], int(r['pl']), int(r['seed']))
        if k in T:
            stop(f'duplicate run {k} in {raw}')
        e = E.get((r['tag'], r['ds'], int(r['pl']), int(r['seed'])))
        if e is None:
            stop(f'no energy row for {k} in {en}')
        if abs(float(e['mse_recomputed']) - float(e['mse_logged'])) > 1e-5 or e['binary'] != 'True' or e['rule'] != 'binary':
            stop(f'recount fails for {k} in {en}: {e["mse_recomputed"]} vs {e["mse_logged"]}, binary {e["binary"]}, rule {e["rule"]}')
        r['E'] = float(e['total_fft_uJ_corr']) + (ms_add_ett(e) if add else 0.)
        T[k] = r
    for ds, pl in CELLS:
        for s in SEEDS:
            if (ds, pl, s) not in T:
                stop(f'missing {ds} H{pl} seed {s} of {cond} in {raw}')
    return T


def best_ok(k, r):
    """The logged best epoch is the argmin of val_loss over the logged epochs."""
    p = os.path.join(r['dir'], 'log', 'best_log_0.csv')
    if not os.path.exists(p):
        stop(f'no best_log_0.csv in {r["dir"]}')
    be, val, ep = r['best_epoch_val_epochs'].split(':')
    v = [float(x['val_loss']) for x in csv.DictReader(open(p))]
    if len(v) != int(ep) or abs(v[int(be)] - float(val)) > 1e-6 or abs(min(v) - float(val)) > 1e-6:
        stop(f'{k}: best {be} {val} is not the argmin of val_loss {v}')


def ett(a):
    S = os.path.join(RES, a.suite)
    raw, en = os.path.join(S, 'raw.txt'), os.path.join(S, 'energy_recount.csv')
    t1 = lambda ds: f'spkmx_lif_muxlin_ms_lr{LR_MX[ds]}_L1'
    t2 = lambda ds: f'spk_ssa_lr{LR_SF[ds]}_L{LY_SF[ds]}_ms'
    A1 = load_ett(raw, en, t1, 'spkmx_lif_muxlin_ms', add=True)
    A2 = load_ett(raw, en, t2, 'spk_ssa', add=True)
    if a.check_cells is None:
        for T in (A1, A2):
            for k, r in T.items():
                best_ok(k, r)
    B1 = load_ett(os.path.join(TUNE, 'raw.txt'), os.path.join(TUNE, 'energy_recount.csv'), lambda ds: f'spkmx_lif_muxlin_iand_lr{LR_MX[ds]}_L1', 'spkmx_lif_muxlin_iand')
    B2 = spikformer()
    L = ['# Membrane shortcut test, ETT (LIF only)', '', 'Rules and sources: module docstring of summarize_ms.py. Seeds 42 / 7 / 2026. Tuned settings, not re-tuned for the shortcut.', '']
    L += ['## Test values (seed mean)', '', '| cell | SpkMixer | SpkMixer + MS | Spikformer-style | Spikformer-style + MS |', '|---|--:|--:|--:|--:|']
    for ds, pl in CELLS:
        f = lambda T, key: mean([num(T[(ds, pl, s)], key) for s in SEEDS])
        L.append(f"| {ds} H{pl} | {f(B1, 'mse'):.4f} | {f(A1, 'mse'):.4f} | {f(B2, 'mse'):.4f} | {f(A2, 'mse'):.4f} |")
    Lq = [('M1 SpkMixer + MS vs SpkMixer', A1, B1), ('M2 Spikformer-style + MS vs Spikformer-style', A2, B2), ('M3 SpkMixer + MS vs Spikformer-style + MS', A1, A2),
          ('M4 (report only) SpkMixer + MS vs Spikformer-style', A1, B2), ('(reference) SpkMixer vs Spikformer-style', B1, B2)]
    L += ['', '## Pre-registered comparisons', '', '| comparison | R % | lower pairs | verdict | energy A/B | per cell r % | MAE R % |', '|---|--:|--:|---|--:|---|--:|']
    for name, x, y in Lq:
        c, m = compare(x, y, 'mse'), compare(x, y, 'mae')
        L.append(f"| {name} | {c['R']:+.2f} | {c['lo']}/12 | {c['verdict']} | {c['energy']:.2f} | " + ', '.join(f'{z:+.2f}' for z in c['cells']) + f" | {m['R']:+.2f} |")
    L += ['', 'Pairs r % (test MSE), seed 42, 7, 2026 per cell:', '']
    for name, x, y in Lq[:3]:
        c = compare(x, y, 'mse')
        L.append(f'- {name}: ' + '; '.join(f"{ds} H{pl} " + ', '.join(f'{z:+.2f}' for z in c['pairs'][3 * i:3 * i + 3]) for i, (ds, pl) in enumerate(CELLS)))
    L += ['', '## Report only: stopping, last-epoch weights, firing rates (seed mean)', '', '| cell | model | best epoch (seeds) | epochs run (seeds) | last-epoch MSE | rho_hippo | rho_neo | energy µJ |', '|---|---|---|---|--:|--:|--:|--:|']
    for ds, pl in CELLS:
        for name, T in (('SpkMixer', B1), ('SpkMixer + MS', A1), ('Spikformer-style', B2), ('Spikformer-style + MS', A2)):
            rr = [T[(ds, pl, s)] for s in SEEDS]
            be = [x['best_epoch_val_epochs'].split(':') for x in rr]
            neo = [x.get('rho_neo', 'None') for x in rr]
            rn = f'{mean([float(v) for v in neo]):.4f}' if 'None' not in neo else '-'
            L.append(f"| {ds} H{pl} | {name} | {', '.join(b[0] for b in be)} | {', '.join(b[2] for b in be)} | {mean([num(x, 'last_mse') for x in rr]):.4f} | {mean([num(x, 'rho_hippo') for x in rr]):.4f} | {rn} | {mean([x['E'] for x in rr]):.2f} |")
    return S, L


def load_mit(raw, en, tag, add=False):
    E = {}
    for e in csv.DictReader(open(en)):
        k = (e.get('tag', ''), int(e['seed']))
        if k in E:
            stop(f'duplicate energy row {k} in {en}')
        E[k] = e
    T = {}
    for r in read_result(raw, lambda c: True):
        if r['tag'] != tag or int(r.get('val_fold', -1)) != -1:
            continue
        s = int(r['seed'])
        if s in T:
            stop(f'duplicate run {tag} seed {s} in {raw}')
        e = E.get((tag, s))
        if e is None:
            stop(f'no energy row for {tag} seed {s} in {en}')
        if abs(float(e['f1_recomputed']) - float(e['f1_logged'])) > 1e-5:
            stop(f'recount fails for {tag} seed {s}: {e["f1_recomputed"]} vs {e["f1_logged"]}')
        if add and e['rule'] != 'binary':
            stop(f'{tag} seed {s}: rule {e["rule"]}, expected binary')
        r['E'] = float(e['total_fft_uJ_corr']) + (2 * T_H * 11 * 384 * E_AC_UJ if add else 0.)
        T[s] = r
    for s in SEEDS:
        if s not in T:
            stop(f'missing {tag} seed {s} in {raw}')
    return T


def cmp_mit(A, B):
    d = [100 * (num(A[s], 'f1') - num(B[s], 'f1')) for s in SEEDS]
    m, hi, lo = mean(d), sum(x > 0 for x in d), sum(x < 0 for x in d)
    v = 'better' if m >= .5 and hi >= 2 else 'worse' if m <= -.5 and lo >= 2 else 'no clear difference'
    return dict(d=d, mean=m, hi=hi, verdict=v, energy=mean([A[s]['E'] for s in SEEDS]) / mean([B[s]['E'] for s in SEEDS]))


def mitbih(a):
    S = os.path.join(RES, a.suite)
    raw, en = os.path.join(S, 'raw.txt'), os.path.join(S, 'energy_recount.csv')
    A1 = load_mit(raw, en, 'spkmx_lif_muxlin_ms_lr0.02_L2', add=True)
    A2 = load_mit(raw, en, 'spk_ssa_lr0.005_L2_ms', add=True)
    tr, te = os.path.join(MITB, 'raw.txt'), os.path.join(MITB, 'energy_recount.csv')
    B1, B2 = load_mit(tr, te, 'spkmx_lif_muxlin_add_lr0.02_L2'), load_mit(tr, te, 'spk_ssa_lr0.005_L2')
    L = ['# Membrane shortcut test, MIT-BIH (LIF only)', '', 'Rules and sources: module docstring of summarize_ms.py. Seeds 42 / 7 / 2026. Tuned settings, not re-tuned for the shortcut; last-epoch weights on the 30 % test file.', '']
    L += ['## Test values (seed mean)', '', '| model | macro-F1 % | accuracy % | F1 N / S / V / F / Q | rho_hippo | energy µJ |', '|---|--:|--:|---|--:|--:|']
    for name, T in (('SpkMixer (ADD)', B1), ('SpkMixer + MS', A1), ('Spikformer-style', B2), ('Spikformer-style + MS', A2)):
        rr = [T[s] for s in SEEDS]
        cls = ' / '.join(f"{100 * mean([num(x, f'f1_{i}') for x in rr]):.2f}" for i in range(5))
        L.append(f"| {name} | {100 * mean([num(x, 'f1') for x in rr]):.2f} | {100 * mean([num(x, 'overall_acc') for x in rr]):.2f} | {cls} | {mean([num(x, 'rho_hippo') for x in rr]):.4f} | {mean([x['E'] for x in rr]):.2f} |")
    L += ['', '## Pre-registered comparisons', '', '| comparison | mean %p | seeds higher | per seed 42, 7, 2026 | verdict | energy A/B |', '|---|--:|--:|---|---|--:|']
    for name, x, y in (('M1 SpkMixer + MS vs SpkMixer', A1, B1), ('M2 Spikformer-style + MS vs Spikformer-style', A2, B2), ('M3 SpkMixer + MS vs Spikformer-style + MS', A1, A2),
                       ('M4 (report only) SpkMixer + MS vs Spikformer-style', A1, B2), ('(reference) SpkMixer vs Spikformer-style', B1, B2)):
        c = cmp_mit(x, y)
        L.append(f"| {name} | {c['mean']:+.2f} | {c['hi']}/3 | " + ', '.join(f'{z:+.2f}' for z in c['d']) + f" | {c['verdict']} | {c['energy']:.2f} |")
    return S, L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True, choices=['ett', 'mitbih'])
    ap.add_argument('--suite')
    ap.add_argument('--out')
    ap.add_argument('--check_cells', help='check_ms.py dry run on synthetic rows: skips the best-epoch check, which those rows cannot pass')
    a = ap.parse_args()
    a.suite = a.suite or {'ett': 'flif-v1-ett-spkmux-ms', 'mitbih': 'flif-v1-mitbih-spkmux-ms'}[a.task]
    S, L = (ett if a.task == 'ett' else mitbih)(a)
    p = a.out or os.path.join(S, 'summary.md')
    with open(p, 'w') as f:
        f.write('\n'.join(L) + '\n')
    print('wrote', p)
    print('\n'.join(L))


if __name__ == '__main__':
    main()
