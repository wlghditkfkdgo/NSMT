"""Paper bar under the syops library rule as published (2026-11-19, exp/f-lif-pop-v3; PROJECT_LOG; user instruction 11-19
"energy 계산에 대해서: 기존 synops 라이브러리의 규칙을 그대로 따라줘. (혹은 해당 원본 github 저장소를 찾아서 그대로 이행)").
No training: the logged values of the existing final runs are re-read.

Two energies per run:
  current  total_fft_uJ_corr: the rule used so far -- syops, but (i) an ADD-residual body (inputs 0-3) as integer accumulates
           (user decision 10-29) and (ii) the forecast / reconstruction head that reads the T_h-averaged spikes recounted as T_h binary
           steps of AC (ETT, AD; 10-28 MODI recount).
  syops    the syops counter as published (github.com/iCGY96/syops-counter HEAD 4232f77; the installed syops/ops.py is byte-identical):
           a Linear / BN input with <= 2 distinct values in [0, 1] -> AC x firing rate, otherwise MAC; one AC per neuron input
           element; plus the neuron energy of the f-LIF paper (kept: not a syops rule, the same for both models).
           ETT, AD: the logged total_fft_uJ (syops' own count in the run). MIT-BIH: total_fft_uJ - syn_energy_uJ + syn_syops_uJ
           (the recorder's syops rule; it also counts neuron classes syops was not registered for, which only matters for population
           neurons; for the LIF / SF-de rows used here syn_syops_uJ = syn_energy_uJ is checked to the 2 logged decimals).
Rows: the comparisons of the paper-bar table (LIF SpkMixer vs SF-de per task), loaded with the existing summarizers' own loaders and
checks (summarize_pn, summarize_adsf, summarize_ad_tune). Paper bar (user 11-18): mean better than SF-de and energy A/B < 1, sign only.
Fail closed: a missing energy value, a non-finite value, or syn_syops_uJ != syn_energy_uJ on a used MIT-BIH row stops (exit 2).
usage (from cinc_mixer_v1/): python -B results/syops-exact-20261119/paper_bar_syops.py  ->  results/syops-exact-20261119/summary.md
"""
import csv
import json
import math
import os
import sys

import numpy as np

C = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, C)
import summarize_pn as sp                                                          # noqa: E402
import summarize_adsf as sa                                                        # noqa: E402
import summarize_ad_tune as sat                                                    # noqa: E402
from summarize_l2 import compare                                                   # noqa: E402
from summarize_ms import cmp_mit                                                   # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))


def stop(m):
    print('STOP:', m); sys.exit(2)


def fin(v, what):
    v = float(v)
    if not math.isfinite(v) or v <= 0:
        stop(f'{what}: {v}')
    return v


def ett_syops(T, suite):
    """T from summarize_pn loaders (E = current rule); returns a copy with E = logged syops total."""
    en = {}
    for e in csv.DictReader(open(os.path.join(suite, 'energy_recount.csv'))):
        en[(e['tag'], e['ds'], int(e['pl']), int(e['seed']))] = e
    out = {}
    for k, r in T.items():
        e = en.get((r['tag'], r['ds'], int(r['pl']), int(r['seed'])))
        if e is None:
            stop(f'no energy row {k} {r["tag"]}')
        out[k] = dict(r, E=fin(e['total_fft_uJ'], f'{r["tag"]} {k} total_fft_uJ'))
    return out


def mit_syops(T, suite):
    en = {}
    for e in csv.DictReader(open(os.path.join(suite, 'energy_recount.csv'))):
        if str(e.get('val_fold', '-1')) == '-1':
            en[(e['tag'], int(e['seed']))] = e
    out = {}
    for s, r in T.items():
        e = en.get((r['tag'], int(s)))
        if e is None:
            stop(f'no energy row {r["tag"]} seed {s}')
        a, b = fin(e['syn_energy_uJ'], 'syn_energy_uJ'), fin(e['syn_syops_uJ'], 'syn_syops_uJ')
        if abs(a - b) > 0.005 + 1e-6 * a:                                    # syn_energy_uJ is logged with 2 decimals
            stop(f'{r["tag"]} seed {s}: syn_syops_uJ {b} != syn_energy_uJ {a}')
        out[s] = dict(r, E=fin(e['total_fft_uJ'], 'total_fft_uJ') - a + b)
    return out


def emean(T):
    return float(np.mean([r['E'] for r in T.values()]))


rows = []

# ETT input 96: A1 LIF SpkMixer vs SF-de (summarize_pn.stage_sf_ett rows)
A, B = sp.arms_ett_one('LIF'), sp.load_sfde_ett()
A_s, B_s = ett_syops(A, sp.TUNE), ett_syops(B, sp.SFE)
c0, c1 = compare(A, B, 'mse'), compare(A_s, B_s, 'mse')
rows.append(('ETT input 96', 'LIF SpkMixer vs SF-de', f"R {c0['R']:+.3f} % (MSE; < 0 better)", c0['R'] < 0, c0['energy'], c1['energy'],
             emean(A), emean(B), emean(A_s), emean(B_s)))

# ETT input 336: D1 LIF 336 vs SF-de 336 (summarize_pn.stage_sf336 rows)
raw, en = os.path.join(sp.L336, 'raw.txt'), os.path.join(sp.L336, 'energy_recount.csv')
A = sp.load_ett(raw, en, lambda ds: f'spkmx_lif_muxlin_iand_lr{sp.LR_MX[ds]}_L1_s336', 'spkmx_lif_muxlin_iand')
B = sp.load_sfde_ett('_s336')
A_s, B_s = ett_syops(A, sp.L336), ett_syops(B, sp.P336)
c0, c1 = compare(A, B, 'mse'), compare(A_s, B_s, 'mse')
rows.append(('ETT input 336', 'LIF SpkMixer vs SF-de', f"R {c0['R']:+.3f} % (MSE; < 0 better)", c0['R'] < 0, c0['energy'], c1['energy'],
             emean(A), emean(B), emean(A_s), emean(B_s)))

# MIT-BIH: A1 LIF SpkMixer (ADD, lr 0.02, M 2) vs SF-de (summarize_pn.stage_sf_mit rows)
A = sp.load_mit(os.path.join(sp.MITB, 'raw.txt'), os.path.join(sp.MITB, 'energy_recount.csv'), 'spkmx_lif_muxlin_add_lr0.02_L2')
B = sp.load_sfde_mit()
A_s, B_s = mit_syops(A, sp.MITB), mit_syops(B, sp.SFM)
c0, c1 = cmp_mit(A, B), cmp_mit(A_s, B_s)
rows.append(('MIT-BIH', 'LIF SpkMixer (ADD) vs SF-de', f"{c0['mean']:+.3f} %p (macro-F1; > 0 better)", c0['mean'] > 0, c0['energy'], c1['energy'],
             emean(A), emean(B), emean(A_s), emean(B_s)))

# AD: E1 muxlin(tuned) vs SF-de(tuned), report muxlin(default) vs SF-de(tuned) (summarize_adsf.stage_final rows)
D = sa.sfde_rows()
sel = json.load(open(os.path.join(sa.SUITE, 'selected_lr.json')))['selected']
T = sat.load()
tsel = {s: json.load(open(os.path.join(sat.TUNE, f'selected_{s}.json')))['selected'] for s in ('lr', 'layers', 'res')}
mk = sat.CONDKEY[sat.MUX]
mux_t = lambda ds: sat.seeds(T, sat.MUX, ds, (tsel['res'][mk][ds], tsel['lr'][mk][ds], tsel['layers'][mk][ds]), 0)
mux_d = lambda ds: sat.seeds(T, sat.MUX, ds, (sat.R0, sat.LR0, sat.L0), 0)
sfde = lambda ds: sa.three(D, sel[ds], ds, 0)


def syops_rows(f):
    return lambda ds: [dict(r, total_fft_uJ_corr=fin(r['total_fft_uJ'], f"{r['tag']} total_fft_uJ")) for r in f(ds)]


ad_cfg = {ds: f"{tsel['res'][mk][ds]} / lr {tsel['lr'][mk][ds]} / M {tsel['layers'][mk][ds]}" for ds in sat.DSS}
for name, f in (('muxlin(tuned) vs SF-de', mux_t), ('muxlin(default: OR / 5e-4 / M 2) vs SF-de', mux_d)):
    c0, c1 = sat.compare(f, sfde), sat.compare(syops_rows(f), syops_rows(sfde))
    if c0 is None or c1 is None:
        stop(f'AD {name}: missing rows or energy_check != ok')
    ea = float(np.mean([np.mean([r['total_fft_uJ_corr'] for r in f(ds)]) for ds in sat.DSS]))
    eb = float(np.mean([np.mean([r['total_fft_uJ_corr'] for r in sfde(ds)]) for ds in sat.DSS]))
    ea_s = float(np.mean([np.mean([r['total_fft_uJ'] for r in f(ds)]) for ds in sat.DSS]))
    eb_s = float(np.mean([np.mean([r['total_fft_uJ'] for r in sfde(ds)]) for ds in sat.DSS]))
    rows.append(('AD', name, f"R {c0['R']:+.3f} % (AUC-PR; > 0 better)", c0['R'] > 0, c0['energy'], c1['energy'], ea, eb, ea_s, eb_s))
    per = []
    for ds in sat.DSS:
        a0 = np.mean([r['total_fft_uJ_corr'] for r in f(ds)]); b0 = np.mean([r['total_fft_uJ_corr'] for r in sfde(ds)])
        a1 = np.mean([r['total_fft_uJ'] for r in f(ds)]); b1 = np.mean([r['total_fft_uJ'] for r in sfde(ds)])
        per.append(f'{ds} {a0 / b0:.3f} -> {a1 / b1:.3f}')
    rows[-1] = rows[-1] + ('; '.join(per),)

L = ['# Paper bar under the syops library rule as published', '', 'Rules and sources: module docstring of paper_bar_syops.py. Seeds 42 / 7 / 2026. '
     'Performance values are unchanged; only the energy rule differs.', '',
     f'AD tuned settings (11-03, validation): {json.dumps(ad_cfg)}', '',
     '| task | comparison | performance (seed mean) | mean better | energy A/B current rule | paper bar current | energy A/B syops | paper bar syops | '
     'A / B µJ current | A / B µJ syops |', '|---|---|---|---|--:|---|--:|---|---|---|']
for r in rows:
    task, name, perf, better, e0, e1, ea, eb, ea_s, eb_s = r[:10]
    pb0 = 'met' if better and e0 < 1 else 'not met'
    pb1 = 'met' if better and e1 < 1 else 'not met'
    L.append(f"| {task} | {name} | {perf} | {'yes' if better else 'no'} | {e0:.4f} | {pb0} | {e1:.4f} | {pb1} | {ea:.2f} / {eb:.2f} | {ea_s:.2f} / {eb_s:.2f} |")
L += ['', 'Energy A/B is the mean over cells (ETT) / datasets (AD) of the per-cell ratio of seed means, as in the summarizers.', '']
for r in rows:
    if len(r) > 10:
        L.append(f'- AD {r[1]} per dataset (current -> syops): {r[10]}')
open(os.path.join(OUT, 'summary.md'), 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
