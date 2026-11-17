"""Mixer variants of the LIF SpkMixer (2026-11-17, exp/f-lif-pop-v3; PROJECT_LOG pre-registration; user request 11-17). Tuned LIF SpkMixer settings,
not re-tuned; each arm changes one thing; pairs by cell and seed.
Arms:  base     = the tuned LIF SpkMixer (ETT: flif-v1-ett-spkmux-tune, 1 mixer layer per direction; MIT-BIH: flif-v1-mitbih-spkmux-tune, 2 layers)
       max/mean = the T_h repeats pooled (OR / average) before the mixer, mixer + fusion once, output copied to the repeats   (tags _mpmax, _mpmean)
       L2       = ETT only: 2 mixer layers per direction, no shortcut (tag _L2)
       MS       = membrane shortcut between the layers of each direction (ETT: 2 layers, tag _L2_mms; MIT-BIH: tag _mms)
       subtract / zero = reset of the directional-stack LIF neurons: minus theta / to 0 (base: minus theta / tau)   (tags _mrsubtract, _mrzero)
Comparisons:  V1 max - base, V2 mean - base;  V3 MS - base;  V4 (ETT) MS - L2;  V5 subtract - base, V6 zero - base.
       Report only: max - mean, (ETT) L2 - base, subtract - zero, every arm - tuned Spikformer-style.
Rules (each task's existing one):
    ETT      on test MSE: r = 100 (A - B) / B per seed, R = mean over the 4 cells of the seed-mean r, better = R <= -1 % and >= 8 of 12 pairs lower, worse = mirror.
    MIT-BIH  on test macro-F1: difference in %p per seed, better = mean >= +0.5 and >= 2 of 3 seeds higher, worse = mirror.
Energy: total_fft_uJ_corr of energy_recount (per window / beat). The mean-pooled mixer input is not 0/1, so its first Linear is counted as MAC (syops rule).
Fail closed (exit 2, nothing written): a missing run, a SKIP line of a used condition, a duplicate, a non-finite value, a recount that does not reproduce the logged
metric, a non-binary ETT head input, or (ETT) a new run whose best epoch is not the argmin of its validation loss.
usage: python -B summarize_mx3.py --task ett|mitbih [--suite <suite>] [--out <file>]   ->   results/<suite>/summary.md
"""
import argparse
import os

from summarize_l2 import CELLS, SEEDS, RES, TUNE, compare, mean, num, spikformer
from summarize_ms import LR_MX, MITB, best_ok, cmp_mit, load_ett, load_mit

ETT_ARMS = (('base', '_L1'), ('max', '_L1_mpmax'), ('mean', '_L1_mpmean'), ('L2', '_L2'), ('MS', '_L2_mms'), ('subtract', '_L1_mrsubtract'), ('zero', '_L1_mrzero'))
MIT_ARMS = (('base', ''), ('max', '_mpmax'), ('mean', '_mpmean'), ('MS', '_mms'), ('subtract', '_mrsubtract'), ('zero', '_mrzero'))


def questions(T, ett):
    Q = [('V1 max vs base', T['max'], T['base']), ('V2 mean vs base', T['mean'], T['base']), ('V3 MS vs base', T['MS'], T['base'])]
    if ett:
        Q.append(('V4 MS vs L2', T['MS'], T['L2']))
    return Q + [('V5 subtract vs base', T['subtract'], T['base']), ('V6 zero vs base', T['zero'], T['base'])]


def reports(T, SF, ett):
    R = [('(report) max vs mean', T['max'], T['mean'])] + ([('(report) L2 vs base', T['L2'], T['base'])] if ett else [])
    R.append(('(report) subtract vs zero', T['subtract'], T['zero']))
    return R + [(f'(report) {n} vs Spikformer-style', X, SF) for n, X in T.items()]


def ett(a):
    S = os.path.join(RES, a.suite)
    raw, en = os.path.join(S, 'raw.txt'), os.path.join(S, 'energy_recount.csv')
    T = {}
    for n, suf in ETT_ARMS:
        tag = lambda ds, suf=suf: f'spkmx_lif_muxlin_iand_lr{LR_MX[ds]}{suf}'
        if n == 'base':
            T[n] = load_ett(os.path.join(TUNE, 'raw.txt'), os.path.join(TUNE, 'energy_recount.csv'), tag, 'spkmx_lif_muxlin_iand')
        else:
            T[n] = load_ett(raw, en, tag, 'spkmx_lif_muxlin_iand')
            for k, r in T[n].items():
                best_ok(k, r)
    SF = spikformer()
    L = ['# Mixer variants of the LIF SpkMixer, ETT', '', 'Rules and sources: module docstring of summarize_mx3.py. Seeds 42 / 7 / 2026. Tuned LIF SpkMixer settings, not re-tuned.', '',
         '## Test MSE (seed mean) and energy', '', '| arm | ' + ' | '.join(f'{ds} H{pl}' for ds, pl in CELLS) + ' | mean | mean best validation L1 | energy µJ (4-cell mean) | params (ETTh2 H96) |',
         '|---|--:|--:|--:|--:|--:|--:|--:|--:|']
    for n, X in list(T.items()) + [('Spikformer-style', SF)]:
        v = [mean([num(X[(ds, pl, s)], 'mse') for s in SEEDS]) for ds, pl in CELLS]
        val = mean([num(X[(ds, pl, s)], 'val_loss_best') for ds, pl in CELLS for s in SEEDS])
        e = mean([mean([X[(ds, pl, s)]['E'] for s in SEEDS]) for ds, pl in CELLS])
        L.append(f'| {n} | ' + ' | '.join(f'{z:.4f}' for z in v) + f" | {mean(v):.4f} | {val:.5f} | {e:.2f} | {X[('ETTh2', 96, 42)].get('params', '-')} |")
    L += ['', '## Pre-registered comparisons', '', '| comparison | R % | lower pairs | verdict | energy A/B | per cell r % | MAE R % | validation L1 R % |', '|---|--:|--:|---|--:|---|--:|--:|']
    for name, x, y in questions(T, True) + reports(T, SF, True):
        c, m, v = compare(x, y, 'mse'), compare(x, y, 'mae'), compare(x, y, 'val_loss_best')
        L.append(f"| {name} | {c['R']:+.2f} | {c['lo']}/12 | {c['verdict']} | {c['energy']:.2f} | " + ', '.join(f'{z:+.2f}' for z in c['cells']) + f" | {m['R']:+.2f} | {v['R']:+.2f} |")
    L += ['', 'Pairs r % (test MSE), seed 42, 7, 2026 per cell:', '']
    for name, x, y in questions(T, True):
        c = compare(x, y, 'mse')
        L.append(f'- {name}: ' + '; '.join(f"{ds} H{pl} " + ', '.join(f'{z:+.2f}' for z in c['pairs'][3 * i:3 * i + 3]) for i, (ds, pl) in enumerate(CELLS)))
    L += ['', '## Report only: stopping and firing rates (seed mean)', '', '| cell | arm | best epoch (seeds) | epochs run (seeds) | last-epoch MSE | rho_hippo | energy µJ |', '|---|---|---|---|--:|--:|--:|']
    for ds, pl in CELLS:
        for n, X in T.items():
            rr = [X[(ds, pl, s)] for s in SEEDS]
            be = [x['best_epoch_val_epochs'].split(':') for x in rr]
            L.append(f"| {ds} H{pl} | {n} | {', '.join(b[0] for b in be)} | {', '.join(b[2] for b in be)} | {mean([num(x, 'last_mse') for x in rr]):.4f} | "
                     f"{mean([num(x, 'rho_hippo') for x in rr]):.4f} | {mean([x['E'] for x in rr]):.2f} |")
    return S, L


def mitbih(a):
    S = os.path.join(RES, a.suite)
    raw, en = os.path.join(S, 'raw.txt'), os.path.join(S, 'energy_recount.csv')
    tr, te = os.path.join(MITB, 'raw.txt'), os.path.join(MITB, 'energy_recount.csv')
    T = {}
    for n, suf in MIT_ARMS:
        tag = f'spkmx_lif_muxlin_add_lr0.02_L2{suf}'
        T[n] = load_mit(tr, te, tag) if n == 'base' else load_mit(raw, en, tag)
    SF = load_mit(tr, te, 'spk_ssa_lr0.005_L2')
    L = ['# Mixer variants of the LIF SpkMixer, MIT-BIH', '', 'Rules and sources: module docstring of summarize_mx3.py. Seeds 42 / 7 / 2026. Tuned LIF SpkMixer settings, not re-tuned; '
         'last-epoch weights on the 30 % test file.', '', '## Test values (seed mean)', '',
         '| arm | macro-F1 % | per seed 42, 7, 2026 | accuracy % | F1 N / S / V / F / Q | rho_hippo | energy µJ | params |', '|---|--:|---|--:|---|--:|--:|--:|']
    for n, X in list(T.items()) + [('Spikformer-style', SF)]:
        rr = [X[s] for s in SEEDS]
        cls = ' / '.join(f"{100 * mean([num(x, f'f1_{i}') for x in rr]):.2f}" for i in range(5))
        L.append(f"| {n} | {100 * mean([num(x, 'f1') for x in rr]):.2f} | " + ', '.join(f"{100 * num(x, 'f1'):.2f}" for x in rr) + f" | {100 * mean([num(x, 'overall_acc') for x in rr]):.2f} | {cls} | "
                 f"{mean([num(x, 'rho_hippo') for x in rr]):.4f} | {mean([x['E'] for x in rr]):.2f} | {rr[0].get('params', '-')} |")
    L += ['', '## Pre-registered comparisons', '', '| comparison | mean %p | seeds higher | per seed 42, 7, 2026 | verdict | energy A/B |', '|---|--:|--:|---|---|--:|']
    for name, x, y in questions(T, False) + reports(T, SF, False):
        c = cmp_mit(x, y)
        L.append(f"| {name} | {c['mean']:+.2f} | {c['hi']}/3 | " + ', '.join(f'{z:+.2f}' for z in c['d']) + f" | {c['verdict']} | {c['energy']:.2f} |")
    return S, L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True, choices=['ett', 'mitbih']); ap.add_argument('--suite'); ap.add_argument('--out')
    a = ap.parse_args()
    a.suite = a.suite or {'ett': 'flif-v1-ett-spkmux-mx3', 'mitbih': 'flif-v1-mitbih-spkmux-mx3'}[a.task]
    S, L = (ett if a.task == 'ett' else mitbih)(a)
    p = a.out or os.path.join(S, 'summary.md')
    with open(p, 'w') as f:
        f.write('\n'.join(L) + '\n')
    print('wrote', p)
    print('\n'.join(L))


if __name__ == '__main__':
    main()
