"""Summary of SF-de on ETTh1 / ETTm1 (2026-11-27, exp/f-lif-pop-v3; PROJECT_LOG pre-registration; user "SF-de에 대해서 ETTh1, ETTm1의
순서 섞기 실험을 진행해줘"; rules fixed before training).
  Order (main, no new training beyond SF-de): per arm the 23:03 rule ('order used' if R >= +1 % and >= 8 of 12 runs higher) for the
     SF-de SSA / head / full shuffles (order_sfde1.json); paired contrast with the base muxlin IAND mixer shuffle of 23:10
     (order_base.json): D = d_base(mixer) - d_SF-de(SSA) per (cell, seed); 'muxlin depends more' if mean D >= +1 %p and >= 8 pairs
     larger, 'SF-de depends more' if <= -1 %p and >= 8 smaller, else 'no difference'.
  Accuracy (report only): SF-de vs base muxlin IAND on test MSE with the usual rule (R = mean over cells of seed-mean %, +-1 % and 8 / 12),
     plus syops energy ratio.
Fails closed on missing / non-finite values, a SKIP line, not 12 runs, mixed commits, or unmatched pairs.
usage (from cinc_mixer_v1/): python -B results/check-sfde1-20261127/summarize_order3.py
"""
import json
import math
import os
import sys

C = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
O = os.path.join(C, 'results', 'check-order-20261127')
OUT = os.path.dirname(os.path.abspath(__file__))
CELLS = [('ETTh1', '96'), ('ETTh1', '720'), ('ETTm1', '96'), ('ETTm1', '720')]
SEEDS = ('42', '7', '2026')
avg = lambda x: sum(x) / len(x)


def stop(m):
    print(f'STOP: {m}')
    sys.exit(2)


J = {}
for arm in ('base', 'sfde1'):
    runs = json.load(open(os.path.join(O, f'order_{arm}.json')))['runs']
    J[arm] = {(r['ds'], r['pl'], r['seed']): r for r in runs}
    if len(J[arm]) != 12 or set(J[arm]) != {(d, p, s) for d, p in CELLS for s in SEEDS}:
        stop(f'{arm}: {len(J[arm])} runs')
out = {'order': {}, 'accuracy': {}}
for arm in ('base', 'sfde1'):
    for var in ('mixer', 'head', 'full'):
        vals = {k: r[var]['d_mse_pct'] for k, r in J[arm].items()}
        if not all(math.isfinite(v) for v in vals.values()):
            stop('non-finite')
        cell = [avg([vals[(d, p, s)] for s in SEEDS]) for d, p in CELLS]
        R, hi = avg(cell), sum(v > 0 for v in vals.values())
        out['order'][f'{arm}_{var}'] = dict(R_pct=R, runs_higher=hi, cells=cell, verdict='order used' if R >= 1 and hi >= 8 else 'no evidence')
D = [J['base'][k]['mixer']['d_mse_pct'] - J['sfde1'][k]['mixer']['d_mse_pct'] for k in sorted(J['base'])]
mD, lg, sm = avg(D), sum(x > 0 for x in D), sum(x < 0 for x in D)
out['order']['contrast'] = dict(mean_D=mD, larger=lg, smaller=sm,
                                verdict='muxlin depends more' if mD >= 1 and lg >= 8 else ('SF-de depends more' if mD <= -1 and sm >= 8 else 'no difference'))


def rows(suite, pick):
    lines = open(os.path.join(C, 'results', suite, 'raw.txt')).read().splitlines()
    if suite == 'flif-v1-ett-sfde-h1m1' and any(l.startswith('SKIP') for l in lines):
        stop('SKIP line')
    rr = [dict(t.split('=', 1) for t in l.split()[1:] if '=' in t) for l in lines if l.startswith('RESULT')]
    V = {}
    for r in rr:
        if pick(r):
            k = (r['ds'], r['pl'], r['seed'])
            if k in V:
                stop(f'duplicate {k}')
            V[k] = {m: float(r[m]) for m in ('mse', 'mae', 'syn_energy_uJ')}
            V[k]['sha'] = r['sha']
    if set(V) != {(d, p, s) for d, p in CELLS for s in SEEDS}:
        stop(f'{suite}: {len(V)} runs')
    return V


REF = {'ETTh1': 'spkmx_lif_muxlin_iand_lr0.00025_L1_e32_mlp4_wd0.1', 'ETTm1': 'spkmx_lif_muxlin_iand_lr0.0005_L2_pt16_e32_mlp0.5'}
SF = rows('flif-v1-ett-sfde-h1m1', lambda r: r['cond'] == 'spk_ssa')
if len({v['sha'] for v in SF.values()}) != 1:
    stop('mixed training commits')
BA = rows('flif-v1-ett-spkmux-retune', lambda r: r['ds'] in REF and r['tag'] == REF[r['ds']] and r['pl'] in ('96', '720') and r['blocks'] == '1')
for met in ('mse', 'mae'):
    rc = [avg([100 * (SF[(d, p, s)][met] - BA[(d, p, s)][met]) / BA[(d, p, s)][met] for s in SEEDS]) for d, p in CELLS]
    lo = sum(SF[k][met] < BA[k][met] for k in SF)
    hi = sum(SF[k][met] > BA[k][met] for k in SF)
    R = avg(rc)
    out['accuracy'][met] = dict(R_pct=R, lower=lo, higher=hi, cells=rc,
                                verdict='SF-de better' if R <= -1 and lo >= 8 else ('SF-de worse' if R >= 1 and hi >= 8 else 'no clear difference'))
out['accuracy']['energy_ratio'] = avg([avg([SF[(d, p, s)]['syn_energy_uJ'] for s in SEEDS]) / avg([BA[(d, p, s)]['syn_energy_uJ'] for s in SEEDS]) for d, p in CELLS])
out['accuracy']['mean4'] = {nm: {m: avg([V[k][m] for k in V]) for m in ('mse', 'mae', 'syn_energy_uJ')} for nm, V in (('SF-de', SF), ('muxlin IAND', BA))}
out['sha'] = sorted({v['sha'] for v in SF.values()})
json.dump(out, open(os.path.join(OUT, 'summary_order3.json'), 'w'), indent=1)
L = ['# SF-de (Spikformer-style, SSA + relative position bias) on ETTh1 / ETTm1: order shuffle and accuracy vs the base muxlin IAND', '',
     '| model | shuffle | MSE R % (ETTh1 96 / 720, ETTm1 96 / 720) | runs higher | verdict |', '|---|---|---|--:|---|']
for arm, nm in (('base', 'muxlin IAND (base)'), ('sfde1', 'SF-de')):
    for var in ('mixer', 'head', 'full'):
        m = out['order'][f'{arm}_{var}']
        L.append(f'| {nm} | {"SSA" if (arm == "sfde1" and var == "mixer") else var} | {m["R_pct"]:+.2f} ({" / ".join(f"{c:+.2f}" for c in m["cells"])}) | '
                 f'{m["runs_higher"]} / 12 | **{m["verdict"]}** |')
c = out['order']['contrast']
L += ['', f'Paired contrast d_base(mixer) - d_SF-de(SSA): mean {c["mean_D"]:+.2f} %p, larger {c["larger"]} / smaller {c["smaller"]} -> **{c["verdict"]}**', '',
      '## Accuracy (report only): SF-de vs muxlin IAND', '']
for met in ('mse', 'mae'):
    m = out['accuracy'][met]
    L.append(f'- {met.upper()}: R {m["R_pct"]:+.2f} % (cells ' + ' / '.join(f'{x:+.2f}' for x in m['cells']) + f'), lower {m["lower"]} / higher {m["higher"]} -> {m["verdict"]}')
mm = out['accuracy']['mean4']
L.append(f'- 4-cell means: SF-de MSE {mm["SF-de"]["mse"]:.4f} / MAE {mm["SF-de"]["mae"]:.4f} / {mm["SF-de"]["syn_energy_uJ"]:.2f} uJ; '
         f'muxlin IAND {mm["muxlin IAND"]["mse"]:.4f} / {mm["muxlin IAND"]["mae"]:.4f} / {mm["muxlin IAND"]["syn_energy_uJ"]:.2f} uJ; '
         f'energy ratio SF-de / muxlin {out["accuracy"]["energy_ratio"]:.3f}')
open(os.path.join(OUT, 'summary_order3.md'), 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
