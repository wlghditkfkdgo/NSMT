"""SF-de with patch 16 (fair-patch comparison; 2026-11-23, exp/f-lif-pop-v3; PROJECT_LOG pre-registration; user request 11-23 "(가) 실행").
Rows: results/flif-v1-ett-sfde-pt16/raw.txt (12 runs: ETTh2 / ETTm2 x H96 / H720 x seeds 42 / 7 / 2026; SF-de = COND=spk_ssa NALL=lif with
its selected lr (ETTh2 5e-4, ETTm2 1e-4) and layers (ETTh2 1, ETTm2 2), PATCH=16, everything else as the SF-de of the criteria) and its
energy_recount.csv (energy_recount_ett.py: recomputed test MSE within 1e-5, syops body + head = logged synaptic energy within 1e-3).
Other rows through summarize_ettune.py's own loaders and checks: the first-chain final (results/flif-v1-ett-spkmux-ettune/final.json,
ET_CHAIN unset), the extension final (ET_CHAIN=x), SF-de patch 8 (the SF-de of the criteria).
Comparisons (ETT rule of summarize_ettune.cmp: r = 100 (A - B) / B per seed on test MSE, R = 4-cell mean, better = R <= -1 % and >= 8 of
12 lower, worse = mirror; MAE R; energy A/B = mean over cells of the seed-mean ratio):
  main     first-chain final vs SF-de patch 16
  report   extension final vs SF-de patch 16; SF-de patch 16 vs SF-de patch 8
Energy under two rules: syops+neuron = logged total_fft_uJ (the rule used so far: syops as published + the f-LIF-paper neuron term);
syops only = logged syn_energy_uJ (syops as published, no neuron term; the rule of NSMT Table 6 / the SpikF runs).
Pass criteria (report): pass_ett_nsmt.judge (NSMT-paper DNN rows) with criterion (2) against SF-de patch 8 (as defined) and against
SF-de patch 16 (what the criterion would say if its SF-de had the same patch; the definition is the user's to change).
Fail closed (exit 2, nothing written): SKIP line, duplicate, unexpected row, missing run, non-finite value, missing / non-reproducing
energy row. env SF16_DIR=<folder>: read the patch-16 suite from <folder> (dry runs).
usage (from cinc_mixer_v1/): python -B results/sfde-pt16-20261123/summarize_sfpt16.py  ->  <suite>/summary.md, <suite>/summary.json
"""
import csv
import importlib.util
import json
import math
import os
import sys

import numpy as np

C = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, C)
sys.path.insert(0, os.path.join(C, 'results', 'ett-pass-20261123'))
SUITE = 'flif-v1-ett-sfde-pt16'
DIR = os.environ.get('SF16_DIR') or os.path.join(C, 'results', SUITE)
LR, LY = {'ETTh2': '0.0005', 'ETTm2': '0.0001'}, {'ETTh2': '1', 'ETTm2': '2'}


def stop(m):
    print('STOP:', m); sys.exit(2)


def load_se(chain):
    os.environ.pop('ET_DRY', None)
    os.environ['ET_CHAIN'] = chain
    spec = importlib.util.spec_from_file_location(f'se_{chain or "first"}', os.path.join(C, 'summarize_ettune.py'))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


s1, sx = load_se(''), load_se('x')
import pass_ett_nsmt as pn                                                           # noqa: E402
pe = pn.pe
CELLS, SEEDS = s1.CELLS, s1.SEEDS


def final_arm(se):
    D, E = se.load(), se.energy_rows()
    cfg = json.load(open(os.path.join(se.OUT, 'final.json')))['config']
    return se.arm(D, E, cfg), E


T1, E1 = final_arm(s1)
TX, _ = final_arm(sx)
S8 = s1.sfde_rows(E1)


def sf16():
    p = os.path.join(DIR, 'raw.txt')
    if not os.path.exists(p):
        stop(f'{p} missing')
    en = {}
    for e in csv.DictReader(open(os.path.join(DIR, 'energy_recount.csv'))):
        k = (e['tag'], e['ds'], int(e['pl']), int(e['seed']))
        if k in en:
            stop(f'duplicate energy row {k}')
        en[k] = e
    T = {}
    for r in s1.raw(p):                                                             # stops on a SKIP line
        ds = r.get('ds')
        want = f'spk_ssa_lr{LR.get(ds)}_L{LY.get(ds)}_pt16_nalif'
        if r.get('cond') != 'spk_ssa' or r.get('tag') != want or r.get('patch') != '16' or r.get('lr') != LR[ds] or r.get('layers') != LY[ds] \
                or r.get('emb') != '64' or r.get('wd') != '0.06' or r.get('blocks') != '1':
            stop(f'unexpected row {r.get("tag")} {ds} {r.get("pl")} {r.get("seed")}')
        k = (ds, int(r['pl']), int(r['seed']))
        if k[1] not in (96, 720) or k[2] not in SEEDS:
            stop(f'unexpected cell {k}')
        if k in T:
            stop(f'duplicate row {k}')
        r = s1.finite_row(r, f'SF-de pt16 {k}')
        e = en.get((r['tag'], ds, k[1], k[2]))
        if e is None:
            stop(f'no energy row for {k}')
        if abs(float(e['mse_recomputed']) - r['mse']) > 1e-5:
            stop(f'recomputed MSE {e["mse_recomputed"]} != {r["mse"]} {k}')
        body = float(e['body_syn_uJ']) + float(e['head_MAC']) * 4.6e-6
        if abs(body - r['syn_energy_uJ']) > 1e-3 * r['syn_energy_uJ']:
            stop(f'syops body + head {body} != {r["syn_energy_uJ"]} {k}')
        r['Es'], r['Eo'] = r['total_fft_uJ'], float(e['total_fft_uJ_corr'])
        T[k] = r
    miss = [(d, p_, s) for d, p_ in CELLS for s in SEEDS if (d, p_, s) not in T]
    if miss or len(T) != 12:
        stop(f'missing runs {miss} ({len(T)} rows)')
    return T


S16 = sf16()


def cm(T, k):
    return float(np.mean([np.mean([T[(d, p, s)][k] for s in SEEDS]) for d, p in CELLS]))


def cell(T, k):
    return [float(np.mean([T[(d, p, s)][k] for s in SEEDS])) for d, p in CELLS]


def eratio(A, B, k):
    return float(np.mean([a / b for a, b in zip(cell(A, k), cell(B, k))]))


for T in (T1, TX, S8, S16):
    for v in T.values():
        if not all(math.isfinite(float(v[x])) for x in ('mse', 'mae', 'total_fft_uJ', 'syn_energy_uJ')):
            stop('non-finite value')
MODELS = [('first-chain final (LIF SpkMixer, patch 16)', T1), ('extension final (LIF SpkMixer, patch 24)', TX),
          ('SF-de patch 16', S16), ('SF-de patch 8 (criteria SF-de)', S8)]
L = ['# SF-de with patch 16 (fair-patch comparison; test values, seeds 42 / 7 / 2026, input 96)', '',
     'Rules and sources: module docstring of summarize_sfpt16.py.', '', '## Test MSE / MAE (seed mean) and energy (µJ per input window)', '',
     '| model | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | 4-cell MSE / MAE | energy syops+neuron | energy syops only | params ETTh2 H96 / H720 |',
     '|---|---|---|---|---|---|--:|--:|---|']
for nm, T in MODELS:
    ms, ma = cell(T, 'mse'), cell(T, 'mae')
    L.append(f'| {nm} | ' + ' | '.join(f'{a:.4f} / {b:.4f}' for a, b in zip(ms, ma)) +
             f' | {cm(T, "mse"):.4f} / {cm(T, "mae"):.4f} | {cm(T, "total_fft_uJ"):.2f} | {cm(T, "syn_energy_uJ"):.2f} | '
             f'{T[("ETTh2", 96, 42)]["params"]} / {T[("ETTh2", 720, 42)]["params"]} |')
L += ['', 'Energy per cell (seed mean, µJ; syops+neuron / syops only):', '', '| model | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 |',
      '|---|--:|--:|--:|--:|']
for nm, T in MODELS:
    L.append(f'| {nm} | ' + ' | '.join(f'{a:.2f} / {b:.2f}' for a, b in zip(cell(T, 'total_fft_uJ'), cell(T, 'syn_energy_uJ'))) + ' |')
L += ['', '## Comparisons (ETT rule; test MSE)', '',
      '| comparison | R % | lower pairs | verdict | per cell r % | MAE R % | energy A/B syops+neuron | energy A/B syops only |',
      '|---|--:|--:|---|---|--:|--:|--:|']
res = {}
for nm, A, B in (('**main** first-chain final vs SF-de patch 16', T1, S16), ('(report) extension final vs SF-de patch 16', TX, S16),
                 ('(report) SF-de patch 16 vs SF-de patch 8', S16, S8), ('(report) first-chain final vs SF-de patch 8', T1, S8)):
    c = s1.cmp(A, B)
    c['Es_syops_only'] = eratio(A, B, 'syn_energy_uJ')
    res[nm] = {k: c[k] for k in ('R', 'lo', 'verdict', 'cells', 'R_mae', 'Es')} | {'Es_syops_only': c['Es_syops_only']}
    L.append(f'| {nm} | {c["R"]:+.2f} | {c["lo"]}/12 | {c["verdict"]} | {", ".join(f"{x:+.2f}" for x in c["cells"])} | {c["R_mae"]:+.2f} | '
             f'{c["Es"]:.4f} | {c["Es_syops_only"]:.4f} |')
VS8 = {c: {k: [S8[(c[0], c[1], s)][k] for s in SEEDS] for k in ('mse', 'mae')} for c in CELLS}
ES8 = {c: [S8[(c[0], c[1], s)]['Es'] for s in SEEDS] for c in CELLS}
VS16 = {c: {k: [S16[(c[0], c[1], s)][k] for s in SEEDS] for k in ('mse', 'mae')} for c in CELLS}
ES16 = {c: [S16[(c[0], c[1], s)]['Es'] for s in SEEDS] for c in CELLS}
L += ['', '## Pass criteria (report; NSMT-paper DNN rows; pass_ett_nsmt.judge)', '']
passes = {}
for nm, T in MODELS[:2]:
    V = {c: {k: [T[(c[0], c[1], s)][k] for s in SEEDS] for k in ('mse', 'mae')} for c in CELLS}
    EA = {c: [T[(c[0], c[1], s)]['Es'] for s in SEEDS] for c in CELLS}
    for ref, VS, ES in (('SF-de patch 8 (as defined)', VS8, ES8), ('SF-de patch 16 (if the criterion used it)', VS16, ES16)):
        r, lines, om, er = pn.judge(f'{nm}; criterion (2) against {ref}', V, VS, EA, ES, pn.DNN_NSMT)
        passes[f'{nm} | {ref}'] = r
        L += lines
json.dump({'comparisons': res, 'pass': passes,
           'means': {nm: {'mse': cm(T, 'mse'), 'mae': cm(T, 'mae'), 'E_syops_neuron': cm(T, 'total_fft_uJ'), 'E_syops_only': cm(T, 'syn_energy_uJ')}
                     for nm, T in MODELS}}, open(os.path.join(DIR, 'summary.json'), 'w'), indent=1)
open(os.path.join(DIR, 'summary.md'), 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
