"""Summary of the order-shuffle extension (2026-11-27, exp/f-lif-pop-v3; PROJECT_LOG 23:17 pre-registration; no training): SF-de
(Spikformer-style, SSA with relative position bias, patch 16) and the first-chain muxlin IAND (patch 16) on ETTh2 / ETTm2 x H96 / H720 x
3 seeds. Per arm the 23:03 rule ('order used' if R >= +1 % and >= 8 of 12 runs higher). Paired contrast on the 12 (cell, seed) pairs:
D = d_muxlin(mixer) - d_SF-de(SSA); 'muxlin depends more' if mean D >= +1 %p and >= 8 pairs larger, 'SF-de depends more' if <= -1 %p
and >= 8 smaller, else 'no difference'. Fails closed on missing / non-finite values or unmatched pairs.
usage (from this folder): python -B summarize_order2.py
"""
import json
import math
import sys

CELLS = [('ETTh2', '96'), ('ETTh2', '720'), ('ETTm2', '96'), ('ETTm2', '720')]
avg = lambda x: sum(x) / len(x)
J = {arm: {(r['ds'], r['pl'], r['seed']): r for r in json.load(open(f'order_{arm}.json'))['runs']} for arm in ('sfde', 'base2')}
for arm, d in J.items():
    if len(d) != 12 or sorted({(k[0], k[1]) for k in d}) != sorted(CELLS):
        sys.exit(f'STOP: {arm}: {len(d)} runs')
out = {}
for arm in ('base2', 'sfde'):
    out[arm] = {}
    for var in ('mixer', 'head', 'full'):
        for met in ('mse', 'mae'):
            vals = {k: r[var][f'd_{met}_pct'] for k, r in J[arm].items()}
            if not all(math.isfinite(v) for v in vals.values()):
                sys.exit('STOP: non-finite')
            cell = [avg([v for k, v in vals.items() if (k[0], k[1]) == c]) for c in CELLS]
            R, hi = avg(cell), sum(v > 0 for v in vals.values())
            out[arm][f'{var}_{met}'] = dict(R_pct=R, runs_higher=hi, cells=cell, min=min(vals.values()), max=max(vals.values()),
                                            verdict='order used' if R >= 1 and hi >= 8 else 'no evidence')
if set(J['sfde']) != set(J['base2']):
    sys.exit('STOP: unmatched (cell, seed) pairs')
D = [J['base2'][k]['mixer']['d_mse_pct'] - J['sfde'][k]['mixer']['d_mse_pct'] for k in sorted(J['sfde'])]
mD, larger, smaller = avg(D), sum(x > 0 for x in D), sum(x < 0 for x in D)
out['contrast'] = dict(mean_D=mD, larger=larger, smaller=smaller,
                       verdict='muxlin depends more' if mD >= 1 and larger >= 8 else ('SF-de depends more' if mD <= -1 and smaller >= 8 else 'no difference'))
json.dump(out, open('summary_order2.json', 'w'), indent=1)
print('| model | shuffle | MSE R % (cells ETTh2 96 / 720, ETTm2 96 / 720) | runs higher | range | verdict | MAE R % |')
print('|---|---|---|--:|---|---|--:|')
for arm, nm in (('base2', 'muxlin (first chain, pt16)'), ('sfde', 'SF-de (SSA + rel. bias, pt16)')):
    for var in ('mixer', 'head', 'full'):
        m, a = out[arm][f'{var}_mse'], out[arm][f'{var}_mae']
        print(f'| {nm} | {"SSA" if (arm == "sfde" and var == "mixer") else var} | {m["R_pct"]:+.2f} ({" / ".join(f"{c:+.2f}" for c in m["cells"])}) | '
              f'{m["runs_higher"]} / 12 | {m["min"]:+.2f} .. {m["max"]:+.2f} | **{m["verdict"]}** | {a["R_pct"]:+.2f} |')
print(f'\nPaired contrast d_muxlin(mixer) - d_SF-de(SSA): mean {mD:+.2f} %p, larger {larger} / smaller {smaller} of 12 -> **{out["contrast"]["verdict"]}**')
