"""Summary of the T test (2026-11-27, exp/f-lif-pop-v3; PROJECT_LOG T-test pre-registration; user "T를 8로 늘려보자"; rules fixed
before training). Same cells, seeds, verdict rule and energy rules as the fusion stages.
  M8    muxlin IAND, T 8 (repetition steps of the embedding / MLP neurons; first-chain settings, one block): this suite, 12 rows
  P8    msxdpca IAND, T 8 (Pearson mask added to the spike map + MUX; stage 4): this suite, 12 rows
  M4    muxlin IAND, T 4: the existing rows of flif-v1-ett-spkmux-retune (the reference of every fusion stage)
  P4    msxdpca IAND, T 4: stage 4 (flif-v1-ett-sxa-s4)
  comparisons (test MSE verdict; MAE and energy ratios reported): T effect M8 vs M4, P8 vs P4; structure at T 8 P8 vs M8
  verdict  r = 100 (A - B) / B per (cell, seed), R = mean over the 4 cells of the seed-mean r; 'better' if R <= -1 % and >= 8 of 12
           lower, 'worse' if R >= +1 % and >= 8 higher, else 'no clear difference'
  energy   syops (judging rule) = logged syn_energy_uJ (with the attention ledger); integer rule = body_syn_int_uJ + head_AC_corr x 0.9 pJ
Fails closed (exit 2, nothing written): SKIP line, not exactly 24 rows (12 + 12) of the two expected T-8 arms, a missing / duplicate
(cell, seed), a non-finite value, mixed commits, blocks != 1, a missing energy row or the energy identity off (new rows), reference rows
missing.
usage (from cinc_mixer_v1/): python -B summarize_t8.py --suite flif-v1-ett-t8
"""
import argparse
import csv
import json
import math
import os
import sys

C = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--suite', default='flif-v1-ett-t8')
ap.add_argument('--ref_root', default=os.path.join(C, 'results'))
a = ap.parse_args()
OUT = os.path.join(C, 'results', a.suite)
CELLS = [(d, p) for d in ('ETTh1', 'ETTm1') for p in ('96', '720')]
SEEDS = ('42', '7', '2026')
NUM = ('mse', 'mae', 'val_loss_best', 'syn_energy_uJ')
REF_TAG = {'ETTh1': 'spkmx_lif_muxlin_iand_lr0.00025_L1_e32_mlp4_wd0.1', 'ETTm1': 'spkmx_lif_muxlin_iand_lr0.0005_L2_pt16_e32_mlp0.5'}


def stop(msg):
    print(f'STOP: {msg}')
    sys.exit(2)


def fin(r, k):
    try:
        v = float(r[k])
    except (KeyError, ValueError):
        stop(f'{k} missing ({r.get("tag")} {r.get("ds")} H{r.get("pl")} seed {r.get("seed")})')
    if not math.isfinite(v):
        stop(f'{k} not finite ({r.get("tag")} {r.get("ds")} H{r.get("pl")} seed {r.get("seed")})')
    return v


def load(d, pick, strict=False, identity=False):
    """pick(row) -> True for the wanted rows; returns {(ds, pl, seed): values}, the SKIP count and the commits."""
    lines = open(os.path.join(d, 'raw.txt')).read().splitlines()
    nskip = sum(l.startswith('SKIP') for l in lines)
    rows = [dict(t.split('=', 1) for t in l.split()[1:] if '=' in t) for l in lines if l.startswith('RESULT')]
    ep = os.path.join(d, 'energy_recount.csv')
    if not os.path.exists(ep):
        stop(f'{d}: energy_recount.csv missing')
    en = {(e['tag'], e['ds'], e['pl'], e['seed']): e for e in csv.DictReader(open(ep))}
    V, shas = {}, set()
    for r in rows:
        if not pick(r):
            if strict:
                stop(f'unexpected row {r.get("cond")} {r.get("tag")}')
            continue
        k = (r['ds'], r['pl'], r['seed'])
        if k in V:
            stop(f'duplicate run {k} in {d}')
        if r.get('blocks') != '1':
            stop(f'blocks {r.get("blocks")} in {k}')
        v = {m: fin(r, m) for m in NUM}
        e = en.get((r['tag'], r['ds'], r['pl'], r['seed']))
        if e is None:
            stop(f'no energy_recount row for {k} in {d}')
        if identity and abs(fin(e, 'body_syn_uJ') + fin(e, 'head_MAC') * 4.6e-6 - v['syn_energy_uJ']) > 1e-3 * v['syn_energy_uJ']:
            stop(f'energy identity off for {k}')
        v['E_int'] = fin(e, 'body_syn_int_uJ') + fin(e, 'head_AC_corr') * 0.9e-6
        V[k] = v
        shas.add(r['sha'])
    want = {(dd, p, s) for dd, p in CELLS for s in SEEDS}
    if set(V) != want:
        stop(f'{d}: {len(V)} of 12 wanted runs')
    return V, nskip, shas


lines = open(os.path.join(OUT, 'raw.txt')).read().splitlines()
if any(l.startswith('SKIP') for l in lines):
    stop('SKIP line(s)')
rr = [dict(t.split('=', 1) for t in l.split()[1:] if '=' in t) for l in lines if l.startswith('RESULT')]
keys = sorted({(r.get('cond'), r.get('tag', '').endswith('_T8')) for r in rr})
if len(rr) != 24 or keys != [('spkmx_lif_msxdpca_iand', True), ('spkmx_lif_muxlin_iand', True)] or sum(r['cond'] == 'spkmx_lif_muxlin_iand' for r in rr) != 12:
    stop(f'{len(rr)} rows of {keys} (want 12 + 12 of muxlin / msxdpca IAND with _T8)')
M8, _, sh1 = load(OUT, lambda r: r['cond'] == 'spkmx_lif_muxlin_iand' and r['tag'].endswith('_T8'), identity=True)
P8, _, sh2 = load(OUT, lambda r: r['cond'] == 'spkmx_lif_msxdpca_iand' and r['tag'].endswith('_T8'), identity=True)
shas = sh1 | sh2
if len(shas) != 1:
    stop(f'mixed training commits {sorted(shas)}')
R = a.ref_root
M4, _, _ = load(os.path.join(R, 'flif-v1-ett-spkmux-retune'), lambda r: r['ds'] in REF_TAG and r['tag'] == REF_TAG[r['ds']]
                and r['pl'] in ('96', '720') and r.get('blocks') == '1')
P4, _, _ = load(os.path.join(R, 'flif-v1-ett-sxa-s4'), lambda r: r['cond'] == 'spkmx_lif_msxdpca_iand')
ARMS = {'muxlin IAND T8': M8, 'Pearson add + MUX IAND T8': P8, 'muxlin IAND T4 (ref)': M4, 'Pearson add + MUX IAND T4 (stage 4)': P4}
mean = lambda x: sum(x) / len(x)


def compare(A, B, met):
    rr, lo, hi = [], 0, 0
    for d, p in CELLS:
        rr.append(mean([100. * (A[(d, p, s)][met] - B[(d, p, s)][met]) / B[(d, p, s)][met] for s in SEEDS]))
        lo += sum(A[(d, p, s)][met] < B[(d, p, s)][met] for s in SEEDS)
        hi += sum(A[(d, p, s)][met] > B[(d, p, s)][met] for s in SEEDS)
    Rr = mean(rr)
    return dict(R_pct=Rr, pairs_lower=lo, pairs_higher=hi, r_cells=rr,
                verdict='better' if Rr <= -1 and lo >= 8 else ('worse' if Rr >= 1 and hi >= 8 else 'no clear difference'))


def er(A, B, k):
    return mean([mean([A[(d, p, s)][k] for s in SEEDS]) / mean([B[(d, p, s)][k] for s in SEEDS]) for d, p in CELLS])


COMPS = [('T 8 vs 4', 'muxlin IAND T8', 'muxlin IAND T4 (ref)'), ('T 8 vs 4', 'Pearson add + MUX IAND T8', 'Pearson add + MUX IAND T4 (stage 4)'),
         ('structure at T 8', 'Pearson add + MUX IAND T8', 'muxlin IAND T8')]
out = dict(suite=a.suite, sha=sorted(shas)[0], cells={}, mean4={}, comparisons=[])
for nm, V in ARMS.items():
    out['cells'][nm] = {f'{d} {p}': {m: mean([V[(d, p, s)][m] for s in SEEDS]) for m in NUM + ('E_int',)} for d, p in CELLS}
    out['mean4'][nm] = {m: mean([out['cells'][nm][f'{d} {p}'][m] for d, p in CELLS]) for m in NUM + ('E_int',)}
for kind, A, B in COMPS:
    out['comparisons'].append(dict(kind=kind, A=A, B=B, mse=compare(ARMS[A], ARMS[B], 'mse'), mae=compare(ARMS[A], ARMS[B], 'mae'),
                                   energy_syops=er(ARMS[A], ARMS[B], 'syn_energy_uJ'), energy_int=er(ARMS[A], ARMS[B], 'E_int')))
json.dump(out, open(os.path.join(OUT, 'summary.json'), 'w'), indent=1)
L = ['# T test: repetition steps T 8 vs 4 for muxlin IAND and Pearson add + MUX IAND (LIF, one block; ETTh1 / ETTm1 x H96 / H720 x 3 seeds)',
     '', f'Rules: summarize_t8.py docstring. 24 new runs, commit `{out["sha"]}`.', '',
     '## 4-cell means (seed means)', '', '| arm | MSE | MAE | best val L1 | energy µJ (syops) | energy µJ (integer) |', '|---|--:|--:|--:|--:|--:|']
for nm in ARMS:
    m = out['mean4'][nm]
    L.append(f'| {nm} | {m["mse"]:.4f} | {m["mae"]:.4f} | {m["val_loss_best"]:.4f} | {m["syn_energy_uJ"]:.2f} | {m["E_int"]:.2f} |')
L += ['', '## Comparisons (test)', '', '| kind | A | B | MSE R % | lower / higher | MSE verdict | MAE R % | MAE verdict | energy A/B syops | integer |',
      '|---|---|---|--:|---|---|--:|---|--:|--:|']
for c in out['comparisons']:
    L.append(f'| {c["kind"]} | {c["A"]} | {c["B"]} | {c["mse"]["R_pct"]:+.2f} | {c["mse"]["pairs_lower"]} / {c["mse"]["pairs_higher"]} | '
             f'**{c["mse"]["verdict"]}** | {c["mae"]["R_pct"]:+.2f} | {c["mae"]["verdict"]} | {c["energy_syops"]:.3f} | {c["energy_int"]:.3f} |')
L += ['', '## Per cell (seed-mean test MSE / MAE, syops energy µJ)', '', '| arm | ' + ' | '.join(f'{d} H{p}' for d, p in CELLS) + ' |',
      '|---|' + '---|' * len(CELLS)]
for nm in ARMS:
    cc = out['cells'][nm]
    L.append(f'| {nm} | ' + ' | '.join(f'{cc[f"{d} {p}"]["mse"]:.4f} / {cc[f"{d} {p}"]["mae"]:.4f} / {cc[f"{d} {p}"]["syn_energy_uJ"]:.2f}' for d, p in CELLS) + ' |')
L += [''] + [f'Per-cell r (MSE, %) {c["A"]} vs {c["B"]}: ' + ' '.join(f'{x:+.2f}' for x in c['mse']['r_cells']) for c in out['comparisons']]
open(os.path.join(OUT, 'summary.md'), 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
