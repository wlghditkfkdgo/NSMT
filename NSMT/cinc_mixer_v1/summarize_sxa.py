"""Summary of stage 1 of the fusion redesign (2026-11-27, exp/f-lif-pop-v3; PROJECT_LOG pre-registration; rules fixed before training).

Reads results/<suite>/raw.txt (120 RESULT rows) and results/<suite>/energy_recount.csv (energy_recount_ett.py), and the muxlin IAND
one-block reference rows of results/flif-v1-ett-spkmux-retune/raw.txt (no new runs).
  arms      (comb, shared) for comb in muxlin / sxd / sxdpc / sxp / sxppc and shared in {no, yes}; arm 1 = (muxlin, no)
  cells     ETTh1 / ETTm1 x H 96 / 720, seeds 42 / 7 / 2026
  verdict   per comparison A vs B on test MSE: r = 100 (A - B) / B per (cell, seed); R = mean over the 4 cells of the seed-mean r;
            'better' if R <= -1 % and >= 8 of 12 pairs lower, 'worse' if R >= +1 % and >= 8 higher, else 'no clear difference';
            MAE reported with the same rule
  energy    syops rule (the judging rule, user decision 11-24) = logged syn_energy_uJ, which now includes the attention ledger
            (sxa.attn_syops_hook); checked: energy_recount body_syn_uJ + head_MAC x 4.6 pJ = syn_energy_uJ (relative 1e-3).
            Integer rule (reported only) = body_syn_int_uJ + head_AC_corr x 0.9 pJ. Ratio = mean over cells of seed-mean E_A / E_B.
  comparisons  main: each attention arm vs arm 1 (8); mask: *pc vs the same arm without it (4); projection: sxp* vs sxd* (4);
            sharing: shared vs independent per comb (5); ADD: arm 1 vs the IAND one-block reference (1)
  selection (validation only) among the Pearson arms (sxdpc / sxppc x no / yes): score = mean over the 4 cells of
            seed-mean best validation L1 (arm) / seed-mean best validation L1 (arm 1); lowest wins; |difference| < 1e-12 -> lower
            mean syops energy. Written to selected_s1.json (refuses to overwrite a different selection).
Fails closed (exit 2, nothing written): a SKIP line, not exactly 120 distinct runs, a missing or duplicate (arm, cell, seed), a
non-finite value, mixed training commits, blocks != 1, a missing energy_recount row, the energy identity off, reference rows missing.
usage (from cinc_mixer_v1/): python -B summarize_sxa.py --suite flif-v1-ett-sxa-s1
"""
import argparse
import csv
import json
import math
import os
import sys

C = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--suite', default='flif-v1-ett-sxa-s1')
ap.add_argument('--ref_suite', default='flif-v1-ett-spkmux-retune')
a = ap.parse_args()
OUT = os.path.join(C, 'results', a.suite)
CELLS = [(d, p) for d in ('ETTh1', 'ETTm1') for p in ('96', '720')]
SEEDS = ('42', '7', '2026')
COMBS = ('muxlin', 'sxd', 'sxdpc', 'sxp', 'sxppc')
ARMS = [(c, s) for c in COMBS for s in (False, True)]
REF_TAG = {'ETTh1': 'spkmx_lif_muxlin_iand_lr0.00025_L1_e32_mlp4_wd0.1', 'ETTm1': 'spkmx_lif_muxlin_iand_lr0.0005_L2_pt16_e32_mlp0.5'}
NUM = ('mse', 'mae', 'val_loss_best', 'syn_energy_uJ')


def stop(msg):
    print(f'STOP: {msg}')
    sys.exit(2)


def rows(path):
    out, nskip = [], 0
    for line in open(path):
        if line.startswith('SKIP'):
            nskip += 1
        if line.startswith('RESULT'):
            out.append(dict(t.split('=', 1) for t in line.split()[1:] if '=' in t))
    return out, nskip


def fin(r, k):
    try:
        v = float(r[k])
    except (KeyError, ValueError):
        stop(f'{k} missing in {r.get("tag")} {r.get("ds")} H{r.get("pl")} seed {r.get("seed")}')
    if not math.isfinite(v):
        stop(f'{k} not finite in {r.get("tag")} {r.get("ds")} H{r.get("pl")} seed {r.get("seed")}')
    return v


def arm_of(r):
    cond = r['cond']
    if not (cond.startswith('spkmx_lif_') and cond.endswith('_add')):
        stop(f'unexpected cond {cond}')
    comb = cond[len('spkmx_lif_'):-len('_add')]
    if comb not in COMBS:
        stop(f'unexpected comb {comb}')
    return comb, r['tag'].endswith('_ws')


R_, nskip = rows(os.path.join(OUT, 'raw.txt'))
if nskip:
    stop(f'{nskip} SKIP line(s)')
if len(R_) != 120:
    stop(f'{len(R_)} RESULT rows (want 120)')
V = {}
for r in R_:
    k = (arm_of(r), r['ds'], r['pl'], r['seed'])
    if k in V:
        stop(f'duplicate run {k}')
    if r.get('blocks') != '1':
        stop(f'blocks {r.get("blocks")} in {k}')
    V[k] = {m: fin(r, m) for m in NUM}
    V[k]['tag'] = r['tag']
missing = [(arm, d, p, s) for arm in ARMS for d, p in CELLS for s in SEEDS if (arm, d, p, s) not in V]
if missing:
    stop(f'{len(missing)} missing runs, e.g. {missing[0]}')
shas = sorted({r['sha'] for r in R_})
if len(shas) != 1:
    stop(f'mixed training commits {shas}')
ep = os.path.join(OUT, 'energy_recount.csv')
if not os.path.exists(ep):
    stop('energy_recount.csv missing')
E = {}
for e in csv.DictReader(open(ep)):
    E[(e['tag'], e['ds'], e['pl'], e['seed'])] = e
for k, v in V.items():
    e = E.get((v['tag'], k[1], k[2], k[3]))
    if e is None:
        stop(f'no energy_recount row for {k}')
    body, hmac, bint, hac = (fin(e, x) for x in ('body_syn_uJ', 'head_MAC', 'body_syn_int_uJ', 'head_AC_corr'))
    if abs(body + hmac * 4.6e-6 - v['syn_energy_uJ']) > 1e-3 * v['syn_energy_uJ']:
        stop(f'energy identity off for {k}: body {body} + head {hmac * 4.6e-6} != syn {v["syn_energy_uJ"]}')
    v['E_int'] = bint + hac * 0.9e-6
ref_rows, rs = rows(os.path.join(C, 'results', a.ref_suite, 'raw.txt'))
REF = {}
for r in ref_rows:
    if r['ds'] in REF_TAG and r['tag'] == REF_TAG[r['ds']] and r['pl'] in ('96', '720') and r.get('blocks') == '1':
        k = (r['ds'], r['pl'], r['seed'])
        if k in REF:
            stop(f'duplicate reference row {k}')
        REF[k] = {m: fin(r, m) for m in NUM}
if sorted(REF) != sorted((d, p, s) for d, p in CELLS for s in SEEDS):
    stop(f'reference rows: {len(REF)} of 12')


def val(arm, d, p, s):
    return REF[(d, p, s)] if arm == 'ref' else V[(arm, d, p, s)]


def mean(x):
    return sum(x) / len(x)


def compare(A, B, metric):
    rr, lower, higher = [], 0, 0
    for d, p in CELLS:
        rs_ = []
        for s in SEEDS:
            x, y = val(A, d, p, s)[metric], val(B, d, p, s)[metric]
            rs_.append(100. * (x - y) / y)
            lower += x < y
            higher += x > y
        rr.append(mean(rs_))
    R = mean(rr)
    verdict = 'better' if R <= -1 and lower >= 8 else ('worse' if R >= 1 and higher >= 8 else 'no clear difference')
    return dict(R_pct=R, pairs_lower=lower, pairs_higher=higher, verdict=verdict, r_cells=rr)


def eratio(A, B, key):
    if A == 'ref' or B == 'ref':
        if key != 'syn_energy_uJ':
            return None
    return mean([mean([val(A, d, p, s)[key] for s in SEEDS]) / mean([val(B, d, p, s)[key] for s in SEEDS]) for d, p in CELLS])


def name(arm):
    return 'IAND ref' if arm == 'ref' else f'{arm[0]} {"shared" if arm[1] else "indep."}'


ONE = ('muxlin', False)
COMPS = [('main', x, ONE) for x in ARMS if x[0] != 'muxlin']
COMPS += [('mask', (pc, s), (no, s)) for pc, no in (('sxdpc', 'sxd'), ('sxppc', 'sxp')) for s in (False, True)]
COMPS += [('projection', (p, s), (d, s)) for p, d in (('sxp', 'sxd'), ('sxppc', 'sxdpc')) for s in (False, True)]
COMPS += [('sharing', (c, True), (c, False)) for c in COMBS]
COMPS += [('ADD', ONE, 'ref')]
out = dict(suite=a.suite, sha=shas[0], n_rows=len(R_), comparisons=[], cells={}, mean4={})
for arm in ARMS + ['ref']:
    out['cells'][name(arm)] = {f'{d} {p}': {m: mean([val(arm, d, p, s)[m] for s in SEEDS]) for m in NUM + (('E_int',) if arm != 'ref' else ())}
                               for d, p in CELLS}
    out['mean4'][name(arm)] = {m: mean([out['cells'][name(arm)][f'{d} {p}'][m] for d, p in CELLS])
                               for m in out['cells'][name(arm)]['ETTh1 96']}
for kind, A, B in COMPS:
    out['comparisons'].append(dict(kind=kind, A=name(A), B=name(B), mse=compare(A, B, 'mse'), mae=compare(A, B, 'mae'),
                                   energy_syops=eratio(A, B, 'syn_energy_uJ'), energy_int=eratio(A, B, 'E_int')))
cands = [(c, s) for c in ('sxdpc', 'sxppc') for s in (False, True)]
score = {arm: mean([mean([V[(arm, d, p, s)]['val_loss_best'] for s in SEEDS]) / mean([V[(ONE, d, p, s)]['val_loss_best'] for s in SEEDS])
                    for d, p in CELLS]) for arm in cands}
best = min(cands, key=lambda x: score[x])
ties = [x for x in cands if abs(score[x] - score[best]) < 1e-12]
if len(ties) > 1:
    best = min(ties, key=lambda x: out['mean4'][name(x)]['syn_energy_uJ'])
out['selection'] = dict(scores={name(x): score[x] for x in cands}, selected=name(best), comb=best[0], shared=best[1])
# report only: the uncounted elementwise |H_f - H_r| of muxlin per window (T * C * N * D additions), C = 7, T = 4
out['muxlin_uncounted_uJ'] = {'ETTh1': 4 * 7 * 24 * 32 * 0.9e-6, 'ETTm1': 4 * 7 * 12 * 32 * 0.9e-6}
sel_path = os.path.join(OUT, 'selected_s1.json')
sel = dict(comb=best[0], shared=best[1], score=score[best], rule='lowest validation score among the Pearson arms')
if os.path.exists(sel_path):
    old = json.load(open(sel_path))
    if (old.get('comb'), old.get('shared')) != (best[0], best[1]):
        stop(f'selected_s1.json exists with a different selection {old}')
json.dump(sel, open(sel_path, 'w'), indent=1)
json.dump(out, open(os.path.join(OUT, 'summary.json'), 'w'), indent=1)
L = [f'# Fusion redesign stage 1 (LIF, ADD, one block; ETTh1 / ETTm1 x H96 / H720 x 3 seeds)', '',
     f'Rules: summarize_sxa.py docstring. 120 runs, training commit `{shas[0]}`. Energy = syops (judging rule), integer rule reported.', '',
     '## 4-cell means (seed means)', '', '| arm | MSE | MAE | best val L1 | energy µJ (syops) | energy µJ (integer) |', '|---|--:|--:|--:|--:|--:|']
for arm in ARMS + ['ref']:
    m = out['mean4'][name(arm)]
    L.append(f'| {name(arm)} | {m["mse"]:.4f} | {m["mae"]:.4f} | {m["val_loss_best"]:.4f} | {m["syn_energy_uJ"]:.2f} | '
             f'{m["E_int"]:.2f} |' if 'E_int' in m else f'| {name(arm)} | {m["mse"]:.4f} | {m["mae"]:.4f} | {m["val_loss_best"]:.4f} | '
             f'{m["syn_energy_uJ"]:.2f} | — |')
L += ['', '## Comparisons (test; R = mean over cells of the seed-mean % change, A vs B)', '',
      '| kind | A | B | MSE R % | lower / higher | MSE verdict | MAE R % | MAE verdict | energy A/B syops | integer |',
      '|---|---|---|--:|---|---|--:|---|--:|--:|']
for c in out['comparisons']:
    ei = f'{c["energy_int"]:.3f}' if c['energy_int'] is not None else '—'
    L.append(f'| {c["kind"]} | {c["A"]} | {c["B"]} | {c["mse"]["R_pct"]:+.2f} | {c["mse"]["pairs_lower"]} / {c["mse"]["pairs_higher"]} | '
             f'**{c["mse"]["verdict"]}** | {c["mae"]["R_pct"]:+.2f} | {c["mae"]["verdict"]} | {c["energy_syops"]:.3f} | {ei} |')
L += ['', '## Per cell (seed-mean test MSE / MAE, syops energy µJ)', '', '| arm | ' + ' | '.join(f'{d} H{p}' for d, p in CELLS) + ' |',
      '|---|' + '---|' * len(CELLS)]
for arm in ARMS + ['ref']:
    cc = out['cells'][name(arm)]
    L.append(f'| {name(arm)} | ' + ' | '.join(f'{cc[f"{d} {p}"]["mse"]:.4f} / {cc[f"{d} {p}"]["mae"]:.4f} / {cc[f"{d} {p}"]["syn_energy_uJ"]:.2f}'
                                             for d, p in CELLS) + ' |')
L += ['', '## Selection for stage 2 (validation only)', '']
for k, v in out['selection']['scores'].items():
    L.append(f'- {k}: score {v:.6f}')
L.append(f'- **selected: {out["selection"]["selected"]}** (selected_s1.json)')
L += ['', f'Report only: muxlin\'s uncounted |H_f - H_r| per window: ETTh1 {out["muxlin_uncounted_uJ"]["ETTh1"]:.4f} µJ, '
      f'ETTm1 {out["muxlin_uncounted_uJ"]["ETTm1"]:.4f} µJ.']
open(os.path.join(OUT, 'summary.md'), 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
