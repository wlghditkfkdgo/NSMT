"""Independent recheck of summarize_sxa.py (2026-11-27, exp/f-lif-pop-v3; PROJECT_LOG stage-1 pre-registration). Imports nothing
from this repository. Re-reads raw.txt, energy_recount.csv and the IAND reference rows, recomputes the 4-cell means, every comparison
(R, pair counts, verdict for MSE and MAE, syops / integer energy ratios) and the selection, and compares them with summary.json /
selected_s1.json (relative 1e-9). --sha a,b: every run trained at a, and commit b changes no .py / .sh under NSMT/ since a.
usage (from cinc_mixer_v1/): python -B results/check-sxa-20261127/recheck_sxa.py [--suite flif-v1-ett-sxa-s1] [--sha a,b]
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
ap.add_argument('--suite', default='flif-v1-ett-sxa-s1')
ap.add_argument('--sha', default='')
a = ap.parse_args()
D = os.path.join(C, 'results', a.suite)
bad = 0


def ok(c, msg):
    global bad
    bad += not c
    print(('PASS ' if c else 'FAIL ') + msg)


def close(x, y):
    return abs(x - y) <= 1e-9 * max(1., abs(x), abs(y))


raw = [l for l in open(os.path.join(D, 'raw.txt'))]
ok(not any(l.startswith('SKIP') for l in raw), 'no SKIP line')
rows = [dict(t.split('=', 1) for t in l.split()[1:] if '=' in t) for l in raw if l.startswith('RESULT')]
ok(len(rows) == 120, f'{len(rows)} RESULT rows')
en = {(e['tag'], e['ds'], e['pl'], e['seed']): e for e in csv.DictReader(open(os.path.join(D, 'energy_recount.csv')))}
data = {}
for r in rows:
    comb = r['cond'].replace('spkmx_lif_', '').replace('_add', '')
    arm = f'{comb} {"shared" if r["tag"].endswith("_ws") else "indep."}'
    e = en[(r['tag'], r['ds'], r['pl'], r['seed'])]
    data[(arm, r['ds'], r['pl'], r['seed'])] = dict(mse=float(r['mse']), mae=float(r['mae']), val=float(r['val_loss_best']),
                                                    es=float(r['syn_energy_uJ']),
                                                    ei=float(e['body_syn_int_uJ']) + float(e['head_AC_corr']) * 0.9e-6)
ok(len(data) == 120, f'{len(data)} distinct (arm, cell, seed)')
REF = {'ETTh1': 'spkmx_lif_muxlin_iand_lr0.00025_L1_e32_mlp4_wd0.1', 'ETTm1': 'spkmx_lif_muxlin_iand_lr0.0005_L2_pt16_e32_mlp0.5'}
for l in open(os.path.join(C, 'results', 'flif-v1-ett-spkmux-retune', 'raw.txt')):
    if not l.startswith('RESULT'):
        continue
    r = dict(t.split('=', 1) for t in l.split()[1:] if '=' in t)
    if r['ds'] in REF and r['tag'] == REF[r['ds']] and r['pl'] in ('96', '720') and r['blocks'] == '1':
        data[('IAND ref', r['ds'], r['pl'], r['seed'])] = dict(mse=float(r['mse']), mae=float(r['mae']), val=float(r['val_loss_best']),
                                                               es=float(r['syn_energy_uJ']))
ok(len(data) == 132, f'{len(data) - 120} reference rows (want 12)')
S = json.load(open(os.path.join(D, 'summary.json')))
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
        rc = [avg([100 * (data[(A, d, p, s)][met] - data[(B, d, p, s)][met]) / data[(B, d, p, s)][met] for s in seeds]) for d, p in cells]
        lo = sum(data[(A, d, p, s)][met] < data[(B, d, p, s)][met] for d, p in cells for s in seeds)
        hi = sum(data[(A, d, p, s)][met] > data[(B, d, p, s)][met] for d, p in cells for s in seeds)
        R = avg(rc)
        v = 'better' if (R <= -1 and lo >= 8) else ('worse' if (R >= 1 and hi >= 8) else 'no clear difference')
        ok(close(R, c[met]['R_pct']) and lo == c[met]['pairs_lower'] and hi == c[met]['pairs_higher'] and v == c[met]['verdict'],
           f'{c["kind"]} {A} vs {B} {met}: R {R:+.4f} %, {lo} lower / {hi} higher, {v} = summary')
    er = avg([sm(A, d, p, 'es') / sm(B, d, p, 'es') for d, p in cells])
    ok(close(er, c['energy_syops']), f'{c["kind"]} {A} vs {B} energy ratio {er:.4f} = summary')
cand = ['sxdpc indep.', 'sxdpc shared', 'sxppc indep.', 'sxppc shared']
sc = {x: avg([sm(x, d, p, 'val') / sm('muxlin indep.', d, p, 'val') for d, p in cells]) for x in cand}
best = sorted(cand, key=lambda x: (sc[x], avg([sm(x, d, p, 'es') for d, p in cells])))[0]
sel = json.load(open(os.path.join(D, 'selected_s1.json')))
ok(best == S['selection']['selected'] and best == f'{sel["comb"]} {"shared" if sel["shared"] else "indep."}'
   and all(close(sc[x], S['selection']['scores'][x]) for x in cand), f'selection {best} = summary / selected_s1.json')
if a.sha:
    sa, sb = a.sha.split(',')
    shas = sorted({r['sha'] for r in rows})
    ok(shas == [sa], f'all runs trained at {sa} ({shas})')
    ch = subprocess.run(['git', 'diff', '--name-only', sa, sb, '--', '.'], cwd=os.path.dirname(C), capture_output=True, text=True).stdout.split()
    code = [f for f in ch if f.endswith(('.py', '.sh'))]
    ok(not code, f'commit {sb} changes no .py / .sh under NSMT/ since {sa} ({code})')
print('ALL CHECKS PASSED' if bad == 0 else f'{bad} FAIL')
sys.exit(0 if bad == 0 else 1)
