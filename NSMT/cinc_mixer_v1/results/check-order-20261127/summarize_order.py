"""Summary of the token-order shuffle evaluation (2026-11-27, exp/f-lif-pop-v3; PROJECT_LOG order-eval pre-registration; no training).
Reads order_base.json / order_pmux.json (order_eval.py). Per run d = change of test MSE in % against no shuffle (mean of 3 permutation
seeds); R = mean over the 4 cells of the seed-mean d; 'order used' if R >= +1 % and >= 8 of 12 runs higher, else 'no evidence'.
Primary: base, mixer shuffle. MAE reported with the same rule. Fails closed on a missing / non-finite value or not 12 runs.
usage (from this folder): python -B summarize_order.py
"""
import json
import math
import sys

CELLS = [('ETTh1', '96'), ('ETTh1', '720'), ('ETTm1', '96'), ('ETTm1', '720')]
avg = lambda x: sum(x) / len(x)
out = {}
for arm in ('base', 'pmux'):
    J = json.load(open(f'order_{arm}.json'))
    runs = J['runs']
    if len(runs) != 12 or len({(r['ds'], r['pl'], r['seed']) for r in runs}) != 12:
        sys.exit(f'STOP: {arm}: {len(runs)} runs')
    out[arm] = {}
    for var in ('mixer', 'head', 'full'):
        for met in ('mse', 'mae'):
            vals = [r[var][f'd_{met}_pct'] for r in runs]
            if not all(math.isfinite(v) for v in vals):
                sys.exit(f'STOP: non-finite {arm} {var} {met}')
            cell = [avg([r[var][f'd_{met}_pct'] for r in runs if (r['ds'], r['pl']) == c]) for c in CELLS]
            R = avg(cell)
            hi = sum(v > 0 for v in vals)
            out[arm][f'{var}_{met}'] = dict(R_pct=R, runs_higher=hi, cells=cell, min=min(vals), max=max(vals),
                                            verdict='order used' if R >= 1 and hi >= 8 else 'no evidence')
json.dump(out, open('summary_order.json', 'w'), indent=1)
print('| model | shuffle | MSE R % (cells ETTh1 96 / 720, ETTm1 96 / 720) | runs higher | range | verdict | MAE R % | MAE verdict |')
print('|---|---|---|--:|---|---|--:|---|')
for arm in ('base', 'pmux'):
    for var in ('mixer', 'head', 'full'):
        m, a = out[arm][f'{var}_mse'], out[arm][f'{var}_mae']
        print(f'| {arm} | {var} | {m["R_pct"]:+.2f} ({" / ".join(f"{c:+.2f}" for c in m["cells"])}) | {m["runs_higher"]} / 12 | '
              f'{m["min"]:+.2f} .. {m["max"]:+.2f} | **{m["verdict"]}** | {a["R_pct"]:+.2f} | {a["verdict"]} |')
