"""MODI 2026-11-17: report-only extras of the input-length test, computed from the raw RESULT lines only (no code shared with
summarize_len.py, so the L1-L4 block also serves as an independent re-derivation of its verdicts).

  L1-L4      same rules as summarize_len.py (r = A / B - 1 of test MSE, 12 cell x seed pairs; +-1 % and 8/12; per cell +-1 % and 2/3)
  extras     per-seed L1 ratios and test MSE; L1 without ETTh2 H96; seed sd / mean; controls - Spikformer-style (not pre-registered);
             the L1 ratio on validation L1 and on the last-epoch test MSE; L4 on MAE; best epoch / epochs run; rho_neo, rho_hippo
usage (classification folder): python len_extra.py [--len 336] [--ref 96] > results/flif-v1-len336-extra.txt
"""
import argparse, collections, math, os, re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, '..', 'forecasting', 'results')
CELLS = [('ETTh2', '96'), ('ETTh2', '720'), ('ETTm2', '96'), ('ETTm2', '720')]; SEEDS = ['42', '7', '2026']
N = {'pop LIF': 'nsmt4q_pop_lif_msn', 'pop f-LIF': 'nsmt4q_pop_flif_msn'}
ALL = ['spk_ssa'] + [c + v for c in N.values() for v in ('', '_nomem', '_nshuf')]
T = {}


def suite(cond, L):
    if L != 96:
        return f'flif-v1-ett-len{L}'
    return 'flif-v1-gen-ett' if cond == 'spk_ssa' else 'flif-v1-ett-ms' if cond.endswith('_msn') else 'flif-v1-ett-len96ctl'


def get(cond, L, ds, pl, sd):
    su = suite(cond, L)
    if su not in T:
        T[su] = collections.defaultdict(list)
        for ln in open(os.path.join(R, su, 'raw.txt')):
            if ln.startswith('RESULT'):
                d = dict(re.findall(r'(\w+)=(\S*)', ln)); T[su][d['cond'], d['ds'], d['pl'], d['seed']].append(d)
    x = T[su][cond, ds, pl, sd]
    if len(x) != 1:
        raise RuntimeError(f'{su} {cond} {ds} {pl} seed {sd}: {len(x)} rows')
    return x[0]


def f(d, k):
    v = float(d[k])
    if not math.isfinite(v):
        raise RuntimeError(f'{k}={d[k]}')
    return v


def ratios(a, La, b, Lb, k='mse'):
    return {c: [100 * (f(get(a, La, *c, s), k) / f(get(b, Lb, *c, s), k) - 1) for s in SEEDS] for c in CELLS}


def show(tag, per):
    allr = [x for c in CELLS for x in per[c]]; Rm = np.mean(allr); lo = sum(x < 0 for x in allr); hi = sum(x > 0 for x in allr)
    v = 'better' if Rm <= -1 and lo >= 8 else 'worse' if Rm >= 1 and hi >= 8 else 'no clear difference'
    cv = ['B' if np.mean(per[c]) <= -1 and sum(x < 0 for x in per[c]) >= 2 else 'W' if np.mean(per[c]) >= 1 and sum(x > 0 for x in per[c]) >= 2
          else '=' for c in CELLS]
    print(f'{tag:60s} ' + ' '.join(f'{np.mean(per[c]):+6.2f}{x}' for c, x in zip(CELLS, cv)) + f' | R {Rm:+.2f} lower {lo}/12 {v}')
    return per


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--len', type=int, default=336); ap.add_argument('--ref', type=int, default=96)
    a = ap.parse_args(); L, L0 = a.len, a.ref
    print(f'input {L} against input {L0}; cells: ETTh2 H96, ETTh2 H720, ETTm2 H96, ETTm2 H720 (B better / W worse / = no clear difference)')
    print('\n== L1 base - Spikformer-style')
    L1 = {(n, x): show(f'L1 {n} base - Spikformer-style, input {x}', ratios(c, x, 'spk_ssa', x)) for n, c in N.items() for x in (L0, L)}
    print('\n== L2 gap change g = r_L - r_ref (%p)')
    for n in N:
        g = {c: [L1[n, L][c][i] - L1[n, L0][c][i] for i in range(3)] for c in CELLS}; allg = [x for c in CELLS for x in g[c]]
        G = np.mean(allg); neg = sum(x < 0 for x in allg); pos = sum(x > 0 for x in allg)
        print(f'L2 {n}: ' + ' '.join(f'{np.mean(g[c]):+6.2f}' for c in CELLS) + f' | G {G:+.2f} negative {neg}/12 positive {pos}/12 -> '
              + ('grew' if G <= -1 and neg >= 8 else 'shrank' if G >= 1 and pos >= 8 else 'no clear change'))
        print('   per pair (seeds 42 / 7 / 2026): ' + '; '.join(f'{c[0]} H{c[1]} ' + ' '.join(f'{x:+.2f}' for x in g[c]) for c in CELLS))
    print('\n== L3 controls')
    for x, y in (('', '_nshuf'), ('', '_nomem'), ('_nshuf', '_nomem')):
        for n, c in N.items():
            for l in (L0, L):
                show(f'L3 {n} msn{x} - msn{y}, input {l}', ratios(c + x, l, c + y, l))
    print(f'\n== L4 every model, input {L} - input {L0} (test MSE, then MAE)')
    for c in ALL:
        show(f'L4 {c}', ratios(c, L, c, L0)); show(f'   MAE {c}', ratios(c, L, c, L0, 'mae'))
    print(f'\n== per-seed L1 ratios at input {L} (seeds 42 / 7 / 2026, %)')
    for n in N:
        print(f'{n}: ' + '; '.join(f'{c[0]} H{c[1]} ' + ' '.join(f'{x:+.2f}' for x in L1[n, L][c]) for c in CELLS))
    print('\n== L1 without ETTh2 H96 (9 pairs)')
    for n in N:
        for l in (L0, L):
            x = [v for c in CELLS[1:] for v in L1[n, l][c]]
            print(f'{n} input {l}: mean {np.mean(x):+.2f} %, lower {sum(v < 0 for v in x)}/9')
    print('\n== per-seed test MSE (seeds 42 / 7 / 2026 per cell) and seed sd / mean (%)')
    for c in ['spk_ssa'] + list(N.values()):
        for l in (L0, L):
            ms = [[f(get(c, l, *cl, s), 'mse') for s in SEEDS] for cl in CELLS]
            print(f'{c:22s} L{l:<3d} ' + ' | '.join(' '.join(f'{v:.4f}' for v in m) for m in ms)
                  + '  sd/mean ' + ' '.join(f'{100 * np.std(m, ddof=1) / np.mean(m):.2f}' for m in ms))
    print('\n== report only, not pre-registered: controls - Spikformer-style')
    for n, c in N.items():
        for v in ('_nomem', '_nshuf'):
            for l in (L0, L):
                show(f'{n} msn{v} - Spikformer-style, input {l}', ratios(c + v, l, 'spk_ssa', l))
    print('\n== the L1 ratio on validation L1 and on the last-epoch test MSE')
    for k, nm in (('val_loss_best', 'validation L1'), ('last_mse', 'last-epoch MSE')):
        for n, c in N.items():
            for l in (L0, L):
                show(f'{nm}: {n} base - Spikformer-style, input {l}', ratios(c, l, 'spk_ssa', l, k))
    print('\n== best epoch / epochs run (0-based best epoch; seeds 42, 7, 2026 per cell)')
    for c in ALL:
        for l in (L0, L):
            be = [[get(c, l, *cl, s)['best_epoch_val_epochs'].split(':') for s in SEEDS] for cl in CELLS]
            print(f'{c:30s} L{l:<3d} ' + ' | '.join(','.join(f'{b[0]}/{b[2]}' for b in x) for x in be))
    print('\n== rho_neo / rho_hippo / mean validation L1 (mean of seeds per cell)')
    for c in ALL:
        for l in (L0, L):
            rn = [np.mean([f(get(c, l, *cl, s), 'rho_neo') for s in SEEDS]) if c != 'spk_ssa' else float('nan') for cl in CELLS]
            rh = [np.mean([f(get(c, l, *cl, s), 'rho_hippo') for s in SEEDS]) for cl in CELLS]
            va = [np.mean([f(get(c, l, *cl, s), 'val_loss_best') for s in SEEDS]) for cl in CELLS]
            print(f'{c:30s} L{l:<3d} rho_neo ' + ' '.join(f'{x:.4f}' for x in rn) + f' mean {np.mean(rn):.4f} | rho_hippo '
                  + ' '.join(f'{x:.4f}' for x in rh) + ' | val ' + ' '.join(f'{x:.4f}' for x in va))


if __name__ == '__main__':
    main()
