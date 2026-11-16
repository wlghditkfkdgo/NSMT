"""STFT band test on the SpkMixer (2026-11-16, exp/f-lif-pop-v3; PROJECT_LOG pre-registration). Tuned LIF SpkMixer settings, not re-tuned; pairs by cell and seed.
Forms (8): {LIF, A = LIF + STFT band per channel group, POP = population LIF, B = population LIF + STFT band per member} x
           {rep = token axis only in the mixer (T_h 4 repeats elsewhere), token = token axis in every neuron layer (no repeats)}.
           LIF rep = the existing tuned runs (flif-v1-ett-spkmux-tune / flif-v1-mitbih-spkmux-tune); the other 7 are this suite.
Comparisons, for each layout p: S1 A_p vs LIF_p, S2 B_p vs POP_p, S3 A_p vs B_p (the user's comparison);
             for each form X: S4 X_token vs X_rep (the user's item 3).  Report only: POP_p vs LIF_p, every form vs tuned Spikformer-style.
Rules (each task's existing one):
    ETT      on test MSE: r = 100 (A - B) / B per seed, R = mean over the 4 cells of the seed-mean r, better = R <= -1 % and >= 8 of 12 pairs lower, worse = mirror.
    MIT-BIH  on test macro-F1: difference in %p per seed, better = mean >= +0.5 and >= 2 of 3 seeds higher, worse = mirror.
Layout choice (item 3): ETT by the mean best validation L1 of the 12 runs (the lower layout is 'chosen'; test is not used); MIT-BIH has no validation split
    (trained on the whole training file), so no layout is chosen there and both are shown.
Energy: total_fft_uJ_corr of energy_recount (per window). The STFT itself is not counted by any hook; its upper estimate (a P x P real matrix product per frame,
    E_MAC 4.6 pJ) is printed once per task and not added.
Fail closed (exit 2, nothing written): a missing run, a SKIP line of a used condition, a duplicate, a non-finite value, a recount that does not reproduce the logged
metric, a non-binary ETT head input, or (ETT) a best epoch that is not the argmin of validation loss.
usage: python -B summarize_stft.py --task ett|mitbih [--suite <suite>] [--out <file>]   ->   results/<suite>/summary.md
"""
import argparse
import os

from summarize_l2 import CELLS, SEEDS, RES, TUNE, compare, mean, num, spikformer, stop
from summarize_ms import LR_MX, MITB, best_ok, cmp_mit, load_ett, load_mit

FORMS = (('LIF', 'lif', ''), ('A', 'lif', '_bdgroup'), ('POP', 'pop_lif', ''), ('B', 'pop_lif', '_bdmember'))
LAYOUTS = (('rep', ''), ('token', '_txtoken'))
E_MAC_UJ = 4.6e-6


def names():
    return [f'{f} {p}' for p, _ in LAYOUTS for f, _, _ in FORMS]


def questions(T):
    Q = []
    for p, _ in LAYOUTS:
        Q += [(f'S1 A {p} vs LIF {p}', T[f'A {p}'], T[f'LIF {p}']), (f'S2 B {p} vs POP {p}', T[f'B {p}'], T[f'POP {p}']), (f'S3 A {p} vs B {p}', T[f'A {p}'], T[f'B {p}'])]
    Q += [(f'S4 {f} token vs {f} rep', T[f'{f} token'], T[f'{f} rep']) for f, _, _ in FORMS]
    return Q


def reports(T, SF):
    R = [(f'(report) POP {p} vs LIF {p}', T[f'POP {p}'], T[f'LIF {p}']) for p, _ in LAYOUTS]
    return R + [(f'(report) {n} vs Spikformer-style', T[n], SF) for n in names()]


def ett(a):
    S = os.path.join(RES, a.suite)
    raw, en = os.path.join(S, 'raw.txt'), os.path.join(S, 'energy_recount.csv')
    T = {}
    for f, nrn, bt in FORMS:
        for p, pt in LAYOUTS:
            tag = lambda ds, nrn=nrn, bt=bt, pt=pt: f'spkmx_{nrn}_muxlin_iand_lr{LR_MX[ds]}_L1{bt}{pt}'
            if (f, p) == ('LIF', 'rep'):
                T['LIF rep'] = load_ett(os.path.join(TUNE, 'raw.txt'), os.path.join(TUNE, 'energy_recount.csv'), tag, 'spkmx_lif_muxlin_iand')
            else:
                T[f'{f} {p}'] = load_ett(raw, en, tag, f'spkmx_{nrn}_muxlin_iand')
                for k, r in T[f'{f} {p}'].items():
                    best_ok(k, r)
    SF = spikformer()
    L = ['# STFT band test on the SpkMixer, ETT', '', 'Rules and sources: module docstring of summarize_stft.py. Seeds 42 / 7 / 2026. Tuned LIF SpkMixer settings, not re-tuned.', '',
         f'STFT cost not counted by the hooks (upper estimate, 24 frames x 8 x 8 x 7 channels at E_MAC): {24 * 64 * 7 * E_MAC_UJ:.3f} µJ per window.', '']
    L += ['## Test MSE (seed mean) and energy', '', '| form | ' + ' | '.join(f'{ds} H{pl}' for ds, pl in CELLS) + ' | mean | energy µJ (4-cell mean) | params (ETTh2 H96) |', '|---|--:|--:|--:|--:|--:|--:|--:|']
    for n, X in list(T.items()) + [('Spikformer-style', SF)]:
        v = [mean([num(X[(ds, pl, s)], 'mse') for s in SEEDS]) for ds, pl in CELLS]
        e = mean([mean([X[(ds, pl, s)]['E'] for s in SEEDS]) for ds, pl in CELLS])
        L.append(f'| {n} | ' + ' | '.join(f'{z:.4f}' for z in v) + f" | {mean(v):.4f} | {e:.2f} | {X[('ETTh2', 96, 42)].get('params', '-')} |")
    L += ['', '## Pre-registered comparisons', '', '| comparison | R % | lower pairs | verdict | energy A/B | per cell r % | MAE R % |', '|---|--:|--:|---|--:|---|--:|']
    for name, x, y in questions(T) + reports(T, SF):
        c, m = compare(x, y, 'mse'), compare(x, y, 'mae')
        L.append(f"| {name} | {c['R']:+.2f} | {c['lo']}/12 | {c['verdict']} | {c['energy']:.2f} | " + ', '.join(f'{z:+.2f}' for z in c['cells']) + f" | {m['R']:+.2f} |")
    L += ['', 'Pairs r % (test MSE), seed 42, 7, 2026 per cell:', '']
    for name, x, y in questions(T):
        c = compare(x, y, 'mse')
        L.append(f'- {name}: ' + '; '.join(f"{ds} H{pl} " + ', '.join(f'{z:+.2f}' for z in c['pairs'][3 * i:3 * i + 3]) for i, (ds, pl) in enumerate(CELLS)))
    L += ['', '## Layout choice by validation (item 3; test not used)', '', '| form | mean best validation L1, rep | token | token / rep - 1 % | chosen | test R % token vs rep |', '|---|--:|--:|--:|---|--:|']
    for f, _, _ in FORMS:
        v = {p: mean([num(T[f'{f} {p}'][(ds, pl, s)], 'val_loss_best') for ds, pl in CELLS for s in SEEDS]) for p, _ in LAYOUTS}
        c = compare(T[f'{f} token'], T[f'{f} rep'], 'mse')
        L.append(f"| {f} | {v['rep']:.5f} | {v['token']:.5f} | {100 * (v['token'] / v['rep'] - 1):+.2f} | {'token' if v['token'] < v['rep'] else 'rep'} | {c['R']:+.2f} |")
    L += ['', '## Report only: stopping and firing rates (seed mean)', '', '| cell | form | best epoch (seeds) | epochs run (seeds) | last-epoch MSE | rho_hippo | energy µJ |', '|---|---|---|---|--:|--:|--:|']
    for ds, pl in CELLS:
        for n, X in T.items():
            rr = [X[(ds, pl, s)] for s in SEEDS]
            be = [x['best_epoch_val_epochs'].split(':') for x in rr]
            L.append(f"| {ds} H{pl} | {n} | {', '.join(b[0] for b in be)} | {', '.join(b[2] for b in be)} | {mean([num(x, 'last_mse') for x in rr]):.4f} | {mean([num(x, 'rho_hippo') for x in rr]):.4f} | {mean([x['E'] for x in rr]):.2f} |")
    return S, L


def mitbih(a):
    S = os.path.join(RES, a.suite)
    raw, en = os.path.join(S, 'raw.txt'), os.path.join(S, 'energy_recount.csv')
    tr, te = os.path.join(MITB, 'raw.txt'), os.path.join(MITB, 'energy_recount.csv')
    T = {}
    for f, nrn, bt in FORMS:
        for p, pt in LAYOUTS:
            tag = f'spkmx_{nrn}_muxlin_add_lr0.02_L2{bt}{pt}'
            T[f'{f} {p}'] = load_mit(tr, te, tag) if (f, p) == ('LIF', 'rep') else load_mit(raw, en, tag)
    SF = load_mit(tr, te, 'spk_ssa_lr0.005_L2')
    L = ['# STFT band test on the SpkMixer, MIT-BIH', '', 'Rules and sources: module docstring of summarize_stft.py. Seeds 42 / 7 / 2026. Tuned LIF SpkMixer settings, not re-tuned; last-epoch weights on the 30 % test file.', '',
         f'STFT cost not counted by the hooks (upper estimate, 11 frames x 32 x 32 at E_MAC): {11 * 1024 * E_MAC_UJ:.3f} µJ per beat.', '']
    L += ['## Test values (seed mean)', '', '| form | macro-F1 % | per seed 42, 7, 2026 | accuracy % | F1 N / S / V / F / Q | rho_hippo | energy µJ | params |', '|---|--:|---|--:|---|--:|--:|--:|']
    for n, X in list(T.items()) + [('Spikformer-style', SF)]:
        rr = [X[s] for s in SEEDS]
        cls = ' / '.join(f"{100 * mean([num(x, f'f1_{i}') for x in rr]):.2f}" for i in range(5))
        L.append(f"| {n} | {100 * mean([num(x, 'f1') for x in rr]):.2f} | " + ', '.join(f"{100 * num(x, 'f1'):.2f}" for x in rr) + f" | {100 * mean([num(x, 'overall_acc') for x in rr]):.2f} | {cls} | "
                 f"{mean([num(x, 'rho_hippo') for x in rr]):.4f} | {mean([x['E'] for x in rr]):.2f} | {rr[0].get('params', '-')} |")
    L += ['', '## Pre-registered comparisons', '', '| comparison | mean %p | seeds higher | per seed 42, 7, 2026 | verdict | energy A/B |', '|---|--:|--:|---|---|--:|']
    for name, x, y in questions(T) + reports(T, SF):
        c = cmp_mit(x, y)
        L.append(f"| {name} | {c['mean']:+.2f} | {c['hi']}/3 | " + ', '.join(f'{z:+.2f}' for z in c['d']) + f" | {c['verdict']} | {c['energy']:.2f} |")
    L += ['', 'Layout choice (item 3): not made on MIT-BIH (no validation split); both layouts are shown above.']
    return S, L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True, choices=['ett', 'mitbih'])
    ap.add_argument('--suite')
    ap.add_argument('--out')
    a = ap.parse_args()
    a.suite = a.suite or {'ett': 'flif-v1-ett-spkmux-stft', 'mitbih': 'flif-v1-mitbih-spkmux-stft'}[a.task]
    S, L = (ett if a.task == 'ett' else mitbih)(a)
    p = a.out or os.path.join(S, 'summary.md')
    with open(p, 'w') as f:
        f.write('\n'.join(L) + '\n')
    print('wrote', p)
    print('\n'.join(L))


if __name__ == '__main__':
    main()
