"""MSE-loss test of the tuned SpkMixer on ETT (2026-11-07, exp/f-lif-pop-v3; PROJECT_LOG L2 pre-registration). Pairs by cell and seed:
    A = muxlin LIF, IAND residual, tuned lr and layers, --train_loss mse (results/<suite>/raw.txt, tag ..._mse)
    B = the same model with the L1 loss (results/flif-v1-ett-spkmux-tune/raw.txt, tags ..._lr0.0005_L1 on ETTh2, ..._lr0.0001_L1 on ETTm2)
Rule (the task's existing one), on test MSE: r = 100 (A - B) / B per seed, R = mean over the 4 cells of the seed-mean r, better = R <= -1 % and >= 8 of 12 pairs lower,
worse = mirror. The same rule is reported on test MAE. Energy: total_fft_uJ_corr of energy_recount.csv, mean over cells of the ratio of the seed means.
Report only: best validation epoch and epochs run, validation MSE (A's checkpoint, B's checkpoint, B's lowest over its epochs), firing rates, last-epoch weights,
the two losses against tuned Spikformer-style (which is trained with L1 only, so the A row mixes losses).
Fail closed (exit 2, nothing written): a missing run, a SKIP line of a used condition, a duplicate, a non-finite value, a recount that does not reproduce the logged MSE,
a non-binary head input, or an A run whose best epoch is not the argmin of its validation MSE column.
usage: python -B summarize_l2.py [--suite flif-v1-ett-spkmux-l2] [--dry]   ->   results/<suite>/summary.md
--dry: dry run of the summarizer on a suite whose A rows are copies of B; skips the checkpoint selection check, which those rows cannot pass.
"""
import argparse
import csv
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
TUNE = os.path.join(RES, 'flif-v1-ett-spkmux-tune')
BASE = os.path.join(RES, 'flif-v1-ett-spkmux')
CELLS = (('ETTh2', 96), ('ETTh2', 720), ('ETTm2', 96), ('ETTm2', 720))
SEEDS = (42, 7, 2026)
COND = 'spkmx_lif_muxlin_iand'
LR = {'ETTh2': '0.0005', 'ETTm2': '0.0001'}


def stop(msg):
    print('STOP:', msg)
    sys.exit(2)


def read_result(path, used=lambda c: True):
    out = []
    for line in open(path):
        if line.startswith('SKIP'):
            m = re.search(r'\bcond=(\S+)', line)
            if m is None or used(m[1]):
                stop(f'SKIP line of a used condition in {path}: {line[:140]}')
        if line.startswith('RESULT'):
            r = dict(t.split('=', 1) for t in line.split()[1:] if '=' in t)
            if used(r['cond']):
                out.append(r)
    return out


def mean(v):
    return sum(v) / len(v)


def num(r, k):
    x = float(r[k])
    if not math.isfinite(x):
        stop(f'non-finite {k}: {r.get("tag")} {r.get("ds")} {r.get("pl")} seed {r.get("seed")}')
    return x


def energy(path):
    M = {}
    for e in csv.DictReader(open(path)):
        k = (e.get('tag', ''), e['ds'], int(e['pl']), int(e['seed']))
        if k in M:
            stop(f'duplicate energy row {k}')
        M[k] = e
    return M


def load(raw, en, wanted):
    """(ds, pl, seed) -> row with E, for the rows whose tag is wanted(ds)."""
    E = energy(en)
    T = {}
    for r in read_result(raw, lambda c: c == COND):
        if r['tag'] != wanted(r['ds']):
            continue
        k = (r['ds'], int(r['pl']), int(r['seed']))
        if k in T:
            stop(f'duplicate run {k} in {raw}')
        e = E.get((r['tag'], r['ds'], int(r['pl']), int(r['seed'])))
        if e is None:
            stop(f'no energy row for {k} in {en}')
        if abs(float(e['mse_recomputed']) - float(e['mse_logged'])) > 1e-5 or e['binary'] != 'True':
            stop(f'recount fails for {k} in {en}: {e["mse_recomputed"]} vs {e["mse_logged"]}, binary {e["binary"]}')
        r['E'] = float(e['total_fft_uJ_corr'])
        T[k] = r
    for ds, pl in CELLS:
        for s in SEEDS:
            if (ds, pl, s) not in T:
                stop(f'missing {ds} H{pl} seed {s} in {raw}')
    return T


def spikformer():
    en = {(e['ds'], int(e['pl']), int(e['seed'])): float(e['total_fft_uJ_corr']) for e in csv.DictReader(open(os.path.join(BASE, 'baseline_spk_ssa_energy.csv')))}
    ent = {(e['tag'], e['ds'], int(e['pl']), int(e['seed'])): float(e['total_fft_uJ_corr']) for e in csv.DictReader(open(os.path.join(TUNE, 'energy_recount.csv')))}
    T = {}
    for r in read_result(os.path.join(BASE, 'baseline_spk_ssa_raw.txt')):
        if r['ds'] == 'ETTh2':
            T[(r['ds'], int(r['pl']), int(r['seed']))] = dict(r, E=en[(r['ds'], int(r['pl']), int(r['seed']))])
    for r in read_result(os.path.join(TUNE, 'raw.txt'), lambda c: c == 'spk_ssa'):
        if r['ds'] == 'ETTm2' and abs(float(r['lr']) - 2.5e-4) < 1e-12:
            T[(r['ds'], int(r['pl']), int(r['seed']))] = dict(r, E=ent[(r['tag'], r['ds'], int(r['pl']), int(r['seed']))])
    for ds, pl in CELLS:
        for s in SEEDS:
            if (ds, pl, s) not in T:
                stop(f'missing Spikformer-style {ds} H{pl} seed {s}')
    return T


def compare(A, B, key='mse'):
    cell, pairs, ers = [], [], []
    for ds, pl in CELLS:
        a, b = [A[(ds, pl, s)] for s in SEEDS], [B[(ds, pl, s)] for s in SEEDS]
        r = [100 * (num(x, key) - num(y, key)) / num(y, key) for x, y in zip(a, b)]
        cell.append(mean(r))
        pairs += r
        ers.append(mean([x['E'] for x in a]) / mean([y['E'] for y in b]))
    R, lo, hi = mean(cell), sum(z < 0 for z in pairs), sum(z > 0 for z in pairs)
    v = 'better' if R <= -1 and lo >= 8 else 'worse' if R >= 1 and hi >= 8 else 'no clear difference'
    return dict(R=R, lo=lo, verdict=v, energy=mean(ers), cells=cell, pairs=pairs)


def blog(r):
    p = os.path.join(r['dir'], 'log', 'best_log_0.csv')
    if not os.path.exists(p):
        stop(f'no best_log_0.csv in {r["dir"]}')
    return list(csv.DictReader(open(p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--suite', default='flif-v1-ett-spkmux-l2')
    ap.add_argument('--dry', action='store_true')
    a = ap.parse_args()
    S = os.path.join(RES, a.suite)
    A = load(os.path.join(S, 'raw.txt'), os.path.join(S, 'energy_recount.csv'), lambda ds: f'{COND}_lr{LR[ds]}_L1_mse')
    B = load(os.path.join(TUNE, 'raw.txt'), os.path.join(TUNE, 'energy_recount.csv'), lambda ds: f'{COND}_lr{LR[ds]}_L1')
    SP = spikformer()
    if not a.dry:
        for k, r in A.items():
            be, val, ep = r['best_epoch_val_epochs'].split(':')
            v = [float(x['val_mse']) for x in blog(r)]
            if len(v) != int(ep) or abs(v[int(be)] - float(val)) > 1e-6 or abs(min(v) - float(val)) > 1e-6:
                stop(f'{k}: best {be} {val} is not the argmin of val_mse {v}')
    for k, r in B.items():
        be, val, ep = r['best_epoch_val_epochs'].split(':')
        v = [float(x['val_loss']) for x in blog(r)]
        if len(v) != int(ep) or abs(v[int(be)] - float(val)) > 1e-6 or abs(min(v) - float(val)) > 1e-6:
            stop(f'{k}: best {be} {val} is not the argmin of val_loss {v}')

    L = ['# MSE-loss test of the tuned SpkMixer on ETT', '', 'Rules and sources: module docstring. A = MSE loss, B = L1 loss; same model, lr, seeds, patience. Seeds 42 / 7 / 2026.', '']
    L += ['## Test values (seed mean)', '', '| cell | MSE, L1 loss | MSE, MSE loss | r % | MAE, L1 loss | MAE, MSE loss | r % | energy µJ, L1 / MSE |', '|---|---:|---:|---:|---:|---:|---:|---|']
    mse, mae = compare(A, B, 'mse'), compare(A, B, 'mae')
    for i, (ds, pl) in enumerate(CELLS):
        ra, rb = [A[(ds, pl, s)] for s in SEEDS], [B[(ds, pl, s)] for s in SEEDS]
        L.append(f"| {ds} H{pl} | {mean([num(x, 'mse') for x in rb]):.4f} | {mean([num(x, 'mse') for x in ra]):.4f} | {mse['cells'][i]:+.2f} | "
                 f"{mean([num(x, 'mae') for x in rb]):.4f} | {mean([num(x, 'mae') for x in ra]):.4f} | {mae['cells'][i]:+.2f} | {mean([x['E'] for x in rb]):.2f} / {mean([x['E'] for x in ra]):.2f} |")
    L += ['', '## Pre-registered comparison: MSE loss vs L1 loss', '', '| metric | R % | lower pairs | verdict | energy A/B | per cell r % |', '|---|---:|---:|---|---:|---|']
    L.append(f"| test MSE (verdict) | {mse['R']:+.2f} | {mse['lo']}/12 | {mse['verdict']} | {mse['energy']:.2f} | " + ', '.join(f'{z:+.2f}' for z in mse['cells']) + ' |')
    L.append(f"| test MAE (same rule, report) | {mae['R']:+.2f} | {mae['lo']}/12 | {mae['verdict']} | {mae['energy']:.2f} | " + ', '.join(f'{z:+.2f}' for z in mae['cells']) + ' |')
    L += ['', 'Pairs r % (test MSE), seed 42, 7, 2026 per cell:', '']
    for i, (ds, pl) in enumerate(CELLS):
        L.append(f'- {ds} H{pl}: ' + ', '.join(f'{z:+.2f}' for z in mse['pairs'][3 * i:3 * i + 3]))

    L += ['', '## Report only: stopping and validation', '', '| cell | best epoch, L1 loss (seeds) | best epoch, MSE loss (seeds) | epochs run, L1 / MSE (seeds) | validation MSE: L1 checkpoint | L1 lowest over its epochs | MSE checkpoint |', '|---|---|---|---|---:|---:|---:|']
    for ds, pl in CELLS:
        ra, rb = [A[(ds, pl, s)] for s in SEEDS], [B[(ds, pl, s)] for s in SEEDS]
        bea = [int(x['best_epoch_val_epochs'].split(':')[0]) for x in ra]; eaa = [x['best_epoch_val_epochs'].split(':')[2] for x in ra]
        beb = [int(x['best_epoch_val_epochs'].split(':')[0]) for x in rb]; ebb = [x['best_epoch_val_epochs'].split(':')[2] for x in rb]
        va = mean([float(blog(x)[b]['val_mse']) for x, b in zip(ra, bea)])
        vb = mean([float(blog(x)[b]['val_mse']) for x, b in zip(rb, beb)])
        vm = mean([min(float(y['val_mse']) for y in blog(x)) for x in rb])
        L.append(f"| {ds} H{pl} | {', '.join(map(str, beb))} | {', '.join(map(str, bea))} | {'/'.join(ebb)} ; {'/'.join(eaa)} | {vb:.4f} | {vm:.4f} | {va:.4f} |")

    L += ['', '## Report only: firing rates and last-epoch weights (seed mean)', '', '| cell | rho_neo, L1 / MSE | rho_hippo, L1 / MSE | last-epoch test MSE, L1 / MSE |', '|---|---|---|---|']
    for ds, pl in CELLS:
        ra, rb = [A[(ds, pl, s)] for s in SEEDS], [B[(ds, pl, s)] for s in SEEDS]
        f = lambda rr, k: mean([num(x, k) for x in rr])
        L.append(f"| {ds} H{pl} | {f(rb, 'rho_neo'):.4f} / {f(ra, 'rho_neo'):.4f} | {f(rb, 'rho_hippo'):.4f} / {f(ra, 'rho_hippo'):.4f} | {f(rb, 'last_mse'):.4f} / {f(ra, 'last_mse'):.4f} |")

    L += ['', '## Report only: against tuned Spikformer-style (trained with L1 only; the MSE-loss row mixes losses)', '', '| comparison | metric | R % | lower pairs | verdict | energy A/B | per cell r % |', '|---|---|---:|---:|---|---:|---|']
    for name, T in (('SpkMixer L1 loss vs Spikformer-style L1 loss', B), ('SpkMixer MSE loss vs Spikformer-style L1 loss', A)):
        for key in ('mse', 'mae'):
            c = compare(T, SP, key)
            L.append(f"| {name} | {key.upper()} | {c['R']:+.2f} | {c['lo']}/12 | {c['verdict']} | {c['energy']:.2f} | " + ', '.join(f'{z:+.2f}' for z in c['cells']) + ' |')

    p = os.path.join(S, 'summary.md')
    with open(p, 'w') as f:
        f.write('\n'.join(L) + '\n')
    print('wrote', p)
    print('\n'.join(L))


if __name__ == '__main__':
    main()
