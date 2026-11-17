"""ETT input-length test on the SpkMixer (2026-11-17, exp/f-lif-pop-v3; PROJECT_LOG pre-registration). Tuned input-96 settings, not re-tuned; pairs by cell and seed.
Models (3): LIF = LIF SpkMixer (muxlin, IAND, 1 mixer layer), POP = population LIF SpkMixer (same structure, taus 4 / 8 / 16 / 32, gain table of the same
            input length), SF = tuned Spikformer-style.  Input lengths: 96 (existing runs) and L (this suite, flif-v1-ett-spkmux-len<L>, tags ..._s<L>).
            Input-96 rows: LIF flif-v1-ett-spkmux-tune, POP flif-v1-ett-spkmux-stft (the band-less rep rows), SF summarize_l2.spikformer().
Comparisons (test MSE; the test targets are the same windows at every input length, check_len.py d1):
    P1  POP_L vs LIF_L                       does the population help at the longer input (the user's question)
    P2  POP_L vs SF_L, LIF_L vs SF_L         against Spikformer-style at the same input length
    P3  X_L vs X_96 for X = LIF, POP, SF     the effect of the longer input itself
    G   gap change of each P1 / P2 pair: g = r(L) - r(96) per cell and seed (same seed at both lengths), G = mean over the 4 cells of the seed-mean g
Rules: r = 100 (A - B) / B per seed, R = mean over the 4 cells of the seed-mean r; better = R <= -1 % and >= 8 of 12 pairs lower, worse = mirror.
       G: 'toward A' = G <= -1 %p and >= 8 of 12 pairs negative, 'toward B' = mirror, otherwise 'no change'.
Energy: total_fft_uJ_corr of energy_recount (per window), mean over cells of the ratio of the seed means.
Report only: validation L1, MAE, best epoch / epochs run, last-epoch MSE, firing rate, parameters.
Fail closed (exit 2, nothing written): a missing run, a SKIP line of a used condition, a duplicate, a non-finite value, a recount that does not reproduce the logged
MSE, a non-binary head input, or a new run whose best epoch is not the argmin of its validation loss.
usage: python -B summarize_len.py --len 336 [--suite <suite>] [--out <file>]   ->   results/<suite>/summary.md
"""
import argparse
import os

from summarize_l2 import CELLS, SEEDS, RES, TUNE, compare, mean, num, spikformer, stop
from summarize_ms import LR_MX, LR_SF, LY_SF, best_ok, load_ett

STFT = os.path.join(RES, 'flif-v1-ett-spkmux-stft')
MODELS = ('LIF', 'POP', 'SF')
NAME = {'LIF': 'LIF SpkMixer', 'POP': 'population LIF SpkMixer', 'SF': 'Spikformer-style'}


def tables(suite, L):
    raw, en = os.path.join(RES, suite, 'raw.txt'), os.path.join(RES, suite, 'energy_recount.csv')
    T = {('LIF', 96): load_ett(os.path.join(TUNE, 'raw.txt'), os.path.join(TUNE, 'energy_recount.csv'), lambda ds: f'spkmx_lif_muxlin_iand_lr{LR_MX[ds]}_L1', 'spkmx_lif_muxlin_iand'),
         ('POP', 96): load_ett(os.path.join(STFT, 'raw.txt'), os.path.join(STFT, 'energy_recount.csv'), lambda ds: f'spkmx_pop_lif_muxlin_iand_lr{LR_MX[ds]}_L1', 'spkmx_pop_lif_muxlin_iand'),
         ('SF', 96): spikformer(),
         ('LIF', L): load_ett(raw, en, lambda ds: f'spkmx_lif_muxlin_iand_lr{LR_MX[ds]}_L1_s{L}', 'spkmx_lif_muxlin_iand'),
         ('POP', L): load_ett(raw, en, lambda ds: f'spkmx_pop_lif_muxlin_iand_lr{LR_MX[ds]}_L1_s{L}', 'spkmx_pop_lif_muxlin_iand'),
         ('SF', L): load_ett(raw, en, lambda ds: f'spk_ssa_lr{LR_SF[ds]}_L{LY_SF[ds]}_s{L}', 'spk_ssa')}
    for m in MODELS:
        for k, r in T[(m, L)].items():
            best_ok(k, r)
    return T


def gap(T, a, b, L):
    cells, pairs = [], []
    for ds, pl in CELLS:
        g = []
        for s in SEEDS:
            k = (ds, pl, s)
            r = lambda LL: 100 * (num(T[(a, LL)][k], 'mse') - num(T[(b, LL)][k], 'mse')) / num(T[(b, LL)][k], 'mse')
            g.append(r(L) - r(96))
        cells.append(mean(g)); pairs += g
    G, neg, pos = mean(cells), sum(z < 0 for z in pairs), sum(z > 0 for z in pairs)
    v = f'toward {a}' if G <= -1 and neg >= 8 else f'toward {b}' if G >= 1 and pos >= 8 else 'no change'
    return dict(G=G, neg=neg, verdict=v, cells=cells, pairs=pairs)


def row(name, c, m):
    return f"| {name} | {c['R']:+.2f} | {c['lo']}/12 | {c['verdict']} | {c['energy']:.2f} | " + ', '.join(f'{z:+.2f}' for z in c['cells']) + f" | {m['R']:+.2f} |"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--len', type=int, required=True); ap.add_argument('--suite'); ap.add_argument('--out')
    a = ap.parse_args()
    L = a.len
    if L == 96:
        stop('--len must differ from 96')
    suite = a.suite or f'flif-v1-ett-spkmux-len{L}'
    T = tables(suite, L)
    O = [f'# ETT input-length test on the SpkMixer, input {L} against input 96', '',
         'Rules and sources: module docstring of summarize_len.py. Seeds 42 / 7 / 2026. Tuned input-96 settings, not re-tuned.', '']
    O += ['## Test MSE (seed mean) and energy', '', '| model | input | ' + ' | '.join(f'{ds} H{pl}' for ds, pl in CELLS) + ' | mean | mean best validation L1 | energy µJ (4-cell mean) | params (ETTh2 H96) |',
          '|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|']
    for m in MODELS:
        for LL in (96, L):
            X = T[(m, LL)]
            v = [mean([num(X[(ds, pl, s)], 'mse') for s in SEEDS]) for ds, pl in CELLS]
            val = mean([num(X[(ds, pl, s)], 'val_loss_best') for ds, pl in CELLS for s in SEEDS])
            e = mean([mean([X[(ds, pl, s)]['E'] for s in SEEDS]) for ds, pl in CELLS])
            O.append(f'| {NAME[m]} | {LL} | ' + ' | '.join(f'{z:.4f}' for z in v) + f" | {mean(v):.4f} | {val:.5f} | {e:.2f} | {X[('ETTh2', 96, 42)].get('params', '-')} |")
    Q = [(f'P1 POP {L} vs LIF {L}', ('POP', L), ('LIF', L)), (f'P2 POP {L} vs SF {L}', ('POP', L), ('SF', L)), (f'P2 LIF {L} vs SF {L}', ('LIF', L), ('SF', L))]
    Q += [(f'P3 {m} {L} vs {m} 96', (m, L), (m, 96)) for m in MODELS]
    REF = [('(input 96) POP vs LIF', ('POP', 96), ('LIF', 96)), ('(input 96) POP vs SF', ('POP', 96), ('SF', 96)), ('(input 96) LIF vs SF', ('LIF', 96), ('SF', 96))]
    O += ['', '## Pre-registered comparisons', '', '| comparison | R % | lower pairs | verdict | energy A/B | per cell r % | MAE R % |', '|---|--:|--:|---|--:|---|--:|']
    for name, x, y in Q + REF:
        O.append(row(name, compare(T[x], T[y], 'mse'), compare(T[x], T[y], 'mae')))
    O += ['', '## Gap change from input 96 (g = r(L) - r(96), %p; negative = the first model gains more from the longer input)', '',
          '| pair | G %p | negative pairs | verdict | per cell g %p |', '|---|--:|--:|---|---|']
    for x, y in (('POP', 'LIF'), ('POP', 'SF'), ('LIF', 'SF')):
        g = gap(T, x, y, L)
        O.append(f"| {x} vs {y} | {g['G']:+.2f} | {g['neg']}/12 | {g['verdict']} | " + ', '.join(f'{z:+.2f}' for z in g['cells']) + ' |')
    O += ['', 'Pairs r % (test MSE), seed 42, 7, 2026 per cell:', '']
    for name, x, y in Q:
        c = compare(T[x], T[y], 'mse')
        O.append(f'- {name}: ' + '; '.join(f"{ds} H{pl} " + ', '.join(f'{z:+.2f}' for z in c['pairs'][3 * i:3 * i + 3]) for i, (ds, pl) in enumerate(CELLS)))
    O += ['', '## Report only: the same comparisons on best validation L1 and on last-epoch test MSE', '', '| comparison | validation L1 R % | lower pairs | last-epoch MSE R % | lower pairs |', '|---|--:|--:|--:|--:|']
    for name, x, y in Q:
        cv, cl = compare(T[x], T[y], 'val_loss_best'), compare(T[x], T[y], 'last_mse')
        O.append(f"| {name} | {cv['R']:+.2f} | {cv['lo']}/12 | {cl['R']:+.2f} | {cl['lo']}/12 |")
    O += ['', '## Report only: stopping, firing rate, energy per cell (seed mean)', '', '| cell | model | input | best epoch (seeds) | epochs run (seeds) | last-epoch MSE | rho_hippo | energy µJ | params |', '|---|---|--:|---|---|--:|--:|--:|--:|']
    for ds, pl in CELLS:
        for m in MODELS:
            for LL in (96, L):
                rr = [T[(m, LL)][(ds, pl, s)] for s in SEEDS]
                be = [x['best_epoch_val_epochs'].split(':') for x in rr]
                O.append(f"| {ds} H{pl} | {m} | {LL} | {', '.join(b[0] for b in be)} | {', '.join(b[2] for b in be)} | {mean([num(x, 'last_mse') for x in rr]):.4f} | "
                         f"{mean([num(x, 'rho_hippo') for x in rr]):.4f} | {mean([x['E'] for x in rr]):.2f} | {rr[0].get('params', '-')} |")
    p = a.out or os.path.join(RES, suite, 'summary.md')
    with open(p, 'w') as f:
        f.write('\n'.join(O) + '\n')
    print('wrote', p)
    print('\n'.join(O))


if __name__ == '__main__':
    main()
