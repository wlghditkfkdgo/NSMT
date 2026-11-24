"""ETT re-judgment with the SNN baselines of the SpikF paper only (no Spikformer row) (2026-11-24, exp/f-lif-pop-v3; PROJECT_LOG; user
11-24: "그렇다면 바로 직전 선행 연구인 SpikF에서도 Spikformer와의 예측성능을 비교하지 않았다는 것이니, 현재 SpikMIxer 연구를 논문으로 작성할 때
SNN baseline들에서는 SPikformer를 제외하고 SpikF의 논문에 실린 baseline들만 비교해도 될 것 같아."). No training; report only (the new
criterion set is not fixed yet -- open points in the PROJECT_LOG entry).
SNN rows (test MSE / MAE, input 96; SpikF Table 12 = NSMT Table 14, transcription verified 11-19): SpikF, SpikeRNN, iSpikformer. These are
all SNN baselines the SpikF paper reports on ETT (Table 1 holds ANN baselines only; iSpikformer also appears in Table 5, Solar-energy).
NSMT (NSMT Table 14) is not in the SpikF paper: shown as a report row, not judged as a criterion.
DNN rows: the 5 NSMT Table 13 rows (pass_ett_nsmt.DNN_NSMT; user decision 11-23).
Ours: seed means (42 / 7 / 2026) of the first-chain final and the extension final, loaded with summarize_ettune.py's loaders and checks.
Rule per reference row: 4-cell mean of (ref) - 4-cell mean of (ours) >= 0.001 for MSE and for MAE (same as pass_ett: MARGIN 0.001, EPS 1e-12).
Energy (syops only = syn_energy_uJ; user 11-24): mean over cells of the per-cell ratio ours / ref, against SpikF and NSMT (the only SNN rows
with a syops count; SpikeRNN / iSpikformer: not available). The removed criterion (2) (SF-de patch 8 + energy <= 1.05x) is shown for
reference only.
usage (from cinc_mixer_v1/): python -B results/ett-pass-20261124/spikf_rows_only.py > results/ett-pass-20261124/spikf_rows_only.md
"""
import csv
import importlib.util
import json
import math
import os
import sys

import numpy as np

C = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, C)
sys.path.insert(0, os.path.join(C, 'results', 'ett-pass-20261123'))
import pass_ett_nsmt as pn                                                           # noqa: E402
pe = pn.pe
CELLS = (('ETTh2', 96), ('ETTh2', 720), ('ETTm2', 96), ('ETTm2', 720))
SNN = {'SpikF': ((0.290, 0.416, 0.175, 0.405), (0.336, 0.436, 0.251, 0.397)),
       'SpikeRNN': ((0.295, 0.428, 0.176, 0.409), (0.335, 0.438, 0.251, 0.396)),
       'iSpikformer': ((0.308, 0.438, 0.178, 0.406), (0.349, 0.439, 0.256, 0.398))}
NSMT = ((0.288, 0.409, 0.175, 0.399), (0.334, 0.429, 0.251, 0.393))


def stop(m):
    print('STOP:', m); sys.exit(2)


ref = open(os.path.join(C, 'results', 'ref-snn-lb96-20261119', 'compare_snn_lb96.py')).read()
ns = {}
exec(ref[ref.index('P = {'):ref.index('MODELS =')], ns)
for m, v in list(SNN.items()) + [('NSMT', NSMT)]:
    for j, c in enumerate(CELLS):
        if ns['P'][c][m] != (v[0][j], v[1][j]):
            stop(f'{m} {c}: {ns["P"][c][m]} != {(v[0][j], v[1][j])}')


def load_se(chain):
    os.environ.pop('ET_DRY', None); os.environ['ET_CHAIN'] = chain
    spec = importlib.util.spec_from_file_location(f'se_{chain or "first"}', os.path.join(C, 'summarize_ettune.py'))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


s1, sx = load_se(''), load_se('x')
SEEDS = s1.SEEDS


def final_arm(se):
    D, E = se.load(), se.energy_rows()
    return se.arm(D, E, json.load(open(os.path.join(se.OUT, 'final.json')))['config']), E


T1, E1 = final_arm(s1)
TX, _ = final_arm(sx)
S8 = s1.sfde_rows(E1)


def cell(T, k):
    v = [float(np.mean([float(T[(d, p, s)][k]) for s in SEEDS])) for d, p in CELLS]
    if not all(math.isfinite(x) for x in v):
        stop(f'non-finite {k}')
    return v


ec = {r['ds'] + r['pl']: float(r['energy_uJ']) for r in csv.DictReader(open(os.path.join(C, 'results', 'energy-common-20261123', 'spikf_syops.csv')))}
EN = {'SpikF': [ec['ETTh296'], ec['ETTh2720'], ec['ETTm296'], ec['ETTm2720']], 'NSMT': [33.41, 44.90, 38.46, 46.72]}
L = ['# ETT: SNN baselines of the SpikF paper only (no Spikformer row); report, no training', '',
     'Rules and sources: module docstring of spikf_rows_only.py. Margin = ref 4-cell mean − ours (≥ +0.0010 needed for MSE and MAE).', '']
out = {}
for nm, T in (('first-chain final (patch 16)', T1), ('extension final (patch 24)', TX)):
    om, oa, oe = cell(T, 'mse'), cell(T, 'mae'), cell(T, 'syn_energy_uJ')
    L += [f'## {nm}: 4-cell MSE / MAE {np.mean(om):.4f} / {np.mean(oa):.4f}; energy syops µJ per cell '
          + ', '.join(f'{x:.2f}' for x in oe) + f' (mean {np.mean(oe):.2f})', '',
          '| reference | role | ref MSE / MAE | margin MSE / MAE | passes 0.001 rule | energy ours / ref (syops; mean of cell ratios; cells) |',
          '|---|---|---|---|---|---|']
    res = {}
    rows = [(m, 'SNN (SpikF paper)', v) for m, v in SNN.items()] + [(m, 'DNN (NSMT Table 13)', v[:2]) for m, v in pn.DNN_NSMT.items()] \
        + [('NSMT', 'report only (not in the SpikF paper)', NSMT)]
    for m, role, (ms, ma) in rows:
        dm, da = float(np.mean(ms) - np.mean(om)), float(np.mean(ma) - np.mean(oa))
        ok = dm >= pe.MARGIN - pe.EPS and da >= pe.MARGIN - pe.EPS
        er = [a / b for a, b in zip(oe, EN[m])] if m in EN else None
        es = f'{np.mean(er):.4f} ({", ".join(f"{x:.3f}" for x in er)})' if er else 'not available'
        L.append(f'| {m} | {role} | {np.mean(ms):.4f} / {np.mean(ma):.4f} | {dm:+.4f} / {da:+.4f} | {"yes" if ok else "no"} | {es} |')
        res[m] = {'role': role, 'margin_mse': dm, 'margin_mae': da, 'passes': ok, 'energy_ratio': float(np.mean(er)) if er else None}
    crit = [m for m, r in res.items() if not r['role'].startswith('report')]
    allpass = all(res[m]['passes'] for m in crit)
    sm, sa, se_ = cell(S8, 'mse'), cell(S8, 'mae'), cell(S8, 'syn_energy_uJ')
    e8 = float(np.mean([a / b for a, b in zip(oe, se_)]))
    L += ['', f'Without criterion (2): passes every SpikF-paper SNN row and every NSMT DNN row on accuracy: **{"yes" if allpass else "no"}**. '
          f'Energy vs SpikF (syops): {res["SpikF"]["energy_ratio"]:.4f}.',
          f'Removed criterion (2), for reference: SF-de patch 8 margin {np.mean(sm) - np.mean(om):+.4f} / {np.mean(sa) - np.mean(oa):+.4f}, '
          f'energy ours / SF-de (syops) {e8:.4f} (bound was 1.05).', '']
    out[nm] = {'rows': res, 'passes_all_without_2': allpass, 'mse': float(np.mean(om)), 'mae': float(np.mean(oa)),
               'energy_cells': oe, 'energy_vs_sfde8': e8}
json.dump(out, open(os.path.join(OUT, 'spikf_rows_only.json'), 'w'), indent=1)
print('\n'.join(L))
