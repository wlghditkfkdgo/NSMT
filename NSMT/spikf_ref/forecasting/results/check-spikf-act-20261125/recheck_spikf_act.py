"""Independent recheck of the SpikF-LIF summary (2026-11-26, exp/f-lif-pop-v3; PROJECT_LOG SpikF-LIF pre-registration). Imports
nothing from this repo; re-reads raw.txt, refs16.json and judge.json and recomputes every pre-registered number of summary.json:
16-cell means per activation, the paper means, R / pair counts / verdicts for MSE and MAE, the energy ratios, and for each of our two
models against SpikF-LIF and SpikF-GELU the margins, accuracy pass, energy cells below and verdict. --sha a,b: every run trained at a,
and b changes no .py / .sh under NSMT/ since a. Ends with ALL CHECKS PASSED or exits 1.
usage (from spikf_ref/forecasting/): python -B results/check-spikf-act-20261125/recheck_spikf_act.py [--suite spikf-act-ett] [--sha a,b]
"""
import argparse
import json
import math
import os
import subprocess
import sys

F = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
N = os.path.dirname(os.path.dirname(F))
ap = argparse.ArgumentParser()
ap.add_argument('--suite', default='spikf-act-ett')
ap.add_argument('--sha', default='')
a = ap.parse_args()
OUT = os.path.join(F, 'results', a.suite)
bad = 0


def ok(c, msg):
    global bad
    bad += not c
    print(('PASS ' if c else 'FAIL ') + msg)


R = []
nskip = 0
for line in open(os.path.join(OUT, 'raw.txt')):
    if line.startswith('SKIP'):
        nskip += 1
    if line.startswith('RESULT'):
        R.append({k: v for k, v in (t.split('=', 1) for t in line.split()[1:] if '=' in t)})
ok(nskip == 0, f'no SKIP line ({nskip})')
cells = [(d, p) for d in ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2') for p in ('96', '192', '336', '720')]
key = {(r['ds'], r['pl'], r['act'], r['seed']): r for r in R}
ok(len(R) == 96 and len(key) == 96 and all((d, p, x, s) in key for d, p in cells for x in ('gelu', 'lif') for s in ('42', '7', '2026')),
   f'96 distinct runs ({len(R)} rows)')
S = json.load(open(os.path.join(OUT, 'summary.json')))
refs = json.load(open(os.path.join(N, 'cinc_mixer_v1', 'results', 'ett-full16-20261124', 'refs16.json')))['models']['SpikF']['rows']
ours = json.load(open(os.path.join(N, 'cinc_mixer_v1', 'results', 'flif-v1-ett-spkmux-retune', 'judge.json')))['models']


def cm(d, p, x, m):
    return sum(float(key[(d, p, x, s)][m]) for s in ('42', '7', '2026')) / 3


def close(u, v):
    return abs(u - v) <= 1e-9 * max(1.0, abs(v))


for x in ('gelu', 'lif'):
    for m in ('mse', 'mae', 'energy_uJ'):
        v = sum(cm(d, p, x, m) for d, p in cells) / 16
        ok(close(v, S['mean16'][x][m]), f'{x} 16-cell {m} {v:.6f} = summary')
for i, m in enumerate(('mse', 'mae')):
    v = sum(refs[f'{d} {p}'][i] for d, p in cells) / 16
    ok(close(v, S['published16'][m]), f'paper 16-cell {m} {v:.6f} = summary')
for m in ('mse', 'mae'):
    hi = lo = 0
    rr = []
    for d, p in cells:
        t = 0.0
        for s in ('42', '7', '2026'):
            g, l = float(key[(d, p, 'gelu', s)][m]), float(key[(d, p, 'lif', s)][m])
            t += 100 * (l - g) / g / 3
            hi += l > g
            lo += l < g
        rr.append(t)
    Rp = sum(rr) / 16
    vd = 'LIF worse' if Rp >= 1 and hi >= 32 else 'LIF better' if Rp <= -1 and lo >= 32 else 'no clear difference'
    b = S['B'][m]
    ok(close(Rp, b['R_pct']) and hi == b['pairs_higher'] and lo == b['pairs_lower'] and vd == b['verdict'],
       f'B {m}: R {Rp:+.4f} %, {hi} higher / {lo} lower, {vd} = summary')
er = sum(cm(d, p, 'lif', 'energy_uJ') / cm(d, p, 'gelu', 'energy_uJ') for d, p in cells) / 16
ok(close(er, S['energy_ratio_lif_gelu_mean']), f'energy LIF / GELU mean ratio {er:.6f} = summary')
for mk, mv in ours.items():
    for x in ('lif', 'gelu'):
        mm = sum(cm(d, p, x, 'mse') for d, p in cells) / 16 - mv['mse16']
        ma = sum(cm(d, p, x, 'mae') for d, p in cells) / 16 - mv['mae16']
        below = sum(mv['cells'][f'{d} {p}']['syn_energy_uJ'] < cm(d, p, x, 'energy_uJ') for d, p in cells)
        acc = mm >= 0.001 - 1e-12 and ma >= 0.001 - 1e-12
        vd = 'PASS' if acc and below == 16 else 'NOT PASSED'
        j = S['C'][mk][x]
        ok(close(mm, j['margin_mse']) and close(ma, j['margin_mae']) and acc == j['accuracy_pass'] and below == j['energy_cells_below'] and vd == j['verdict'],
           f'C {mk} vs SpikF-{x}: margins {mm:+.6f} / {ma:+.6f}, energy below {below}/16, {vd} = summary')
if a.sha:
    sh = a.sha.split(',')
    ok(all(r['sha'] == sh[0] for r in R), f'all {len(R)} runs trained at {sh[0]} ({sorted({r["sha"] for r in R})})')
    for s in sh[1:]:
        ch = subprocess.run(['git', 'diff', '--name-only', sh[0], s, '--', '.'], cwd=N, capture_output=True, text=True).stdout.split()
        code = [c for c in ch if c.endswith(('.py', '.sh'))]
        ok(not code, f'commit {s} changes no .py / .sh under NSMT/ since {sh[0]} ({code})')
print('ALL CHECKS PASSED' if bad == 0 else f'{bad} FAIL')
sys.exit(0 if bad == 0 else 1)
