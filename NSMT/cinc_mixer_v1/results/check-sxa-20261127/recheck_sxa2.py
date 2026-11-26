"""Independent recheck of summarize_sxa2.py (2026-11-27, exp/f-lif-pop-v3; PROJECT_LOG stage-2 pre-registration). Imports nothing
from this repository. Recomputes the 4-cell means, the 7 comparisons (R, pair counts, verdicts for MSE / MAE, energy ratios) and the
selection from raw.txt / energy_recount.csv of both suites and selected_s1.json, and compares with summary.json / selected_s2.json.
--sha a,b: every stage-2 run trained at a, and commit b changes no .py / .sh under NSMT/ since a.
usage (from cinc_mixer_v1/): python -B results/check-sxa-20261127/recheck_sxa2.py [--suite flif-v1-ett-sxa-s2] [--s1_suite ...] [--sha a,b]
"""
import argparse
import csv
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
C = os.path.dirname(os.path.dirname(HERE))
ap = argparse.ArgumentParser()
ap.add_argument('--suite', default='flif-v1-ett-sxa-s2')
ap.add_argument('--s1_suite', default='flif-v1-ett-sxa-s1')
ap.add_argument('--sha', default='')
a = ap.parse_args()
bad = 0


def ok(c, msg):
    global bad
    bad += not c
    print(('PASS ' if c else 'FAIL ') + msg)


def close(x, y):
    return abs(x - y) <= 1e-9 * max(1., abs(x), abs(y))


def read(suite):
    d = os.path.join(C, 'results', suite)
    lines = open(os.path.join(d, 'raw.txt')).read().splitlines()
    rows = [dict(t.split('=', 1) for t in l.split()[1:] if '=' in t) for l in lines if l.startswith('RESULT')]
    en = {(e['tag'], e['ds'], e['pl'], e['seed']): e for e in csv.DictReader(open(os.path.join(d, 'energy_recount.csv')))}
    out = {}
    for r in rows:
        e = en[(r['tag'], r['ds'], r['pl'], r['seed'])]
        arm = f'{r["cond"][len("spkmx_lif_"):-len("_add")]} {"shared" if r["tag"].endswith("_ws") else "indep."}'
        out[(arm, r['ds'], r['pl'], r['seed'])] = dict(mse=float(r['mse']), mae=float(r['mae']), val=float(r['val_loss_best']),
                                                       es=float(r['syn_energy_uJ']),
                                                       ei=float(e['body_syn_int_uJ']) + float(e['head_AC_corr']) * 0.9e-6, sha=r['sha'])
    return out, sum(l.startswith('SKIP') for l in lines), len(rows)


D2, sk2, n2 = read(a.suite)
D1, sk1, _ = read(a.s1_suite)
ok(sk2 == 0 and n2 == 36 and len(D2) == 36, f'stage 2: {n2} rows, {len(D2)} distinct, {sk2} SKIP')
data = {**D1, **D2}
S = json.load(open(os.path.join(C, 'results', a.suite, 'summary.json')))
cells = [('ETTh1', '96'), ('ETTh1', '720'), ('ETTm1', '96'), ('ETTm1', '720')]
seeds = ['42', '7', '2026']
avg = lambda xs: sum(xs) / len(xs)
sm = lambda arm, d, p, k: avg([data[(arm, d, p, s)][k] for s in seeds])
for arm, m in S['mean4'].items():
    ok(close(avg([sm(arm, d, p, 'mse') for d, p in cells]), m['mse']) and close(avg([sm(arm, d, p, 'es') for d, p in cells]), m['syn_energy_uJ']),
       f'{arm}: 4-cell MSE {m["mse"]:.6f}, energy {m["syn_energy_uJ"]:.4f} = summary')
for c in S['comparisons']:
    A, B = c['A'], c['B']
    for met in ('mse', 'mae'):
        R = avg([avg([100 * (data[(A, d, p, s)][met] - data[(B, d, p, s)][met]) / data[(B, d, p, s)][met] for s in seeds]) for d, p in cells])
        lo = sum(data[(A, d, p, s)][met] < data[(B, d, p, s)][met] for d, p in cells for s in seeds)
        hi = sum(data[(A, d, p, s)][met] > data[(B, d, p, s)][met] for d, p in cells for s in seeds)
        v = 'better' if (R <= -1 and lo >= 8) else ('worse' if (R >= 1 and hi >= 8) else 'no clear difference')
        ok(close(R, c[met]['R_pct']) and lo == c[met]['pairs_lower'] and hi == c[met]['pairs_higher'] and v == c[met]['verdict'],
           f'{A} vs {B} {met}: R {R:+.4f} %, {lo} / {hi}, {v} = summary')
    ok(close(avg([sm(A, d, p, 'es') / sm(B, d, p, 'es') for d, p in cells]), c['energy_syops'])
       and close(avg([sm(A, d, p, 'ei') / sm(B, d, p, 'ei') for d, p in cells]), c['energy_int']), f'{A} vs {B} energy ratios = summary')
names = S['names']
cand = [names[x] for x in ('S', 'OT', 'PM', 'OM')]
sc = {x: avg([sm(x, d, p, 'val') / sm(names['A1'], d, p, 'val') for d, p in cells]) for x in cand}
best = sorted(cand, key=lambda x: (sc[x], avg([sm(x, d, p, 'es') for d, p in cells])))[0]
sel = json.load(open(os.path.join(C, 'results', a.suite, 'selected_s2.json')))
ok(best == S['selection']['selected'] and best == f'{sel["comb"]} {"shared" if sel["shared"] else "indep."}', f'selection {best} = summary / selected_s2.json')
if a.sha:
    sa, sb = a.sha.split(',')
    shas = sorted({v['sha'] for v in D2.values()})
    ok(shas == [sa], f'all stage-2 runs trained at {sa} ({shas})')
    ch = subprocess.run(['git', 'diff', '--name-only', sa, sb, '--', '.'], cwd=os.path.dirname(C), capture_output=True, text=True).stdout.split()
    code = [x for x in ch if x.endswith(('.py', '.sh'))]
    ok(not code, f'commit {sb} changes no .py / .sh under NSMT/ since {sa} ({code})')
print('ALL CHECKS PASSED' if bad == 0 else f'{bad} FAIL')
sys.exit(0 if bad == 0 else 1)
