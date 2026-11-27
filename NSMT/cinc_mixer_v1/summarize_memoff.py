"""Summary of the memoryless control (2026-11-27, exp/f-lif-pop-v3; PROJECT_LOG memoryless-control pre-registration; user "(a)도
진행해줘"; rules fixed before training). Same cells, seeds, verdict rule and energy rules as the fusion stages.
  M0    muxlin IAND T4 with token_memory off (every token-axis neuron of the mixer resets after each token: no information moves along
        the token axis inside the mixer; weights, embedding / MLP neurons and head unchanged): this suite, 12 rows
  M4    muxlin IAND T4 (normal LIF memory along the token axis): the existing rows of flif-v1-ett-spkmux-retune (paired seeds, the same
        initial weights)
  comparison (test MSE verdict; MAE and energy ratios reported): M0 vs M4. 'worse' (R >= +1 % and >= 8 of 12 higher) = the token-axis
        neuron dynamics contribute; 'no clear difference' = no evidence that they do; 'better' = they hurt
  verdict  r = 100 (A - B) / B per (cell, seed), R = mean over the 4 cells of the seed-mean r
  energy   syops (judging rule) = logged syn_energy_uJ; integer rule = body_syn_int_uJ + head_AC_corr x 0.9 pJ
Fails closed (exit 2, nothing written): SKIP line, not exactly 12 rows of muxlin IAND with _tmoff, a missing / duplicate (cell, seed),
a non-finite value, mixed commits, blocks != 1, a missing energy row or the energy identity off, reference rows missing.
usage (from cinc_mixer_v1/): python -B summarize_memoff.py --suite flif-v1-ett-memoff
"""
import argparse
import csv
import json
import math
import os
import sys

C = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--suite', default='flif-v1-ett-memoff')
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
if len(rr) != 12 or any(r.get('cond') != 'spkmx_lif_muxlin_iand' or not r.get('tag', '').endswith('_tmoff') for r in rr):
    stop(f'{len(rr)} rows; want 12 of muxlin IAND with _tmoff')
M0, _, shas = load(OUT, lambda r: r['cond'] == 'spkmx_lif_muxlin_iand' and r['tag'].endswith('_tmoff'), identity=True)
if len(shas) != 1:
    stop(f'mixed training commits {sorted(shas)}')
R = a.ref_root
M4, _, _ = load(os.path.join(R, 'flif-v1-ett-spkmux-retune'), lambda r: r['ds'] in REF_TAG and r['tag'] == REF_TAG[r['ds']]
                and r['pl'] in ('96', '720') and r.get('blocks') == '1')
ARMS = {'muxlin IAND memoryless': M0, 'muxlin IAND (ref, LIF memory)': M4}
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


COMPS = [('memory off vs on', 'muxlin IAND memoryless', 'muxlin IAND (ref, LIF memory)')]
out = dict(suite=a.suite, sha=sorted(shas)[0], cells={}, mean4={}, comparisons=[])
for nm, V in ARMS.items():
    out['cells'][nm] = {f'{d} {p}': {m: mean([V[(d, p, s)][m] for s in SEEDS]) for m in NUM + ('E_int',)} for d, p in CELLS}
    out['mean4'][nm] = {m: mean([out['cells'][nm][f'{d} {p}'][m] for d, p in CELLS]) for m in NUM + ('E_int',)}
for kind, A, B in COMPS:
    out['comparisons'].append(dict(kind=kind, A=A, B=B, mse=compare(ARMS[A], ARMS[B], 'mse'), mae=compare(ARMS[A], ARMS[B], 'mae'),
                                   energy_syops=er(ARMS[A], ARMS[B], 'syn_energy_uJ'), energy_int=er(ARMS[A], ARMS[B], 'E_int')))
json.dump(out, open(os.path.join(OUT, 'summary.json'), 'w'), indent=1)
L = ['# Memoryless control: muxlin IAND T4 with every token-axis mixer neuron reset after each token vs normal LIF (one block; ETTh1 / ETTm1 x H96 / H720 x 3 seeds)',
     '', f'Rules: summarize_memoff.py docstring. 12 new runs, commit `{out["sha"]}`.', '',
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
