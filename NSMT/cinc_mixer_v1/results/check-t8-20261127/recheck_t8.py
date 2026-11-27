"""Independent recheck of summarize_t8.py (2026-11-27, exp/f-lif-pop-v3; PROJECT_LOG T-test pre-registration). Imports nothing from
this repository. Re-reads the new suite and the three reference suites, recomputes the 4-cell means and the 3 comparisons (R, pair
counts, verdicts for MSE / MAE, syops / integer energy ratios), and compares with summary.json. --sha a,b: every new run trained at a,
and commit b changes no .py / .sh under NSMT/ since a.
usage (from cinc_mixer_v1/): python -B results/check-t8-20261127/recheck_t8.py [--suite flif-v1-ett-t8] [--sha a,b]
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
ap.add_argument('--suite', default='flif-v1-ett-t8')
ap.add_argument('--sha', default='')
a = ap.parse_args()
bad = 0


def ok(c, msg):
    global bad
    bad += not c
    print(('PASS ' if c else 'FAIL ') + msg)


def close(x, y):
    return abs(x - y) <= 1e-9 * max(1., abs(x), abs(y))


def read(suite, keep):
    d = os.path.join(C, 'results', suite)
    lines = open(os.path.join(d, 'raw.txt')).read().splitlines()
    en = {(e['tag'], e['ds'], e['pl'], e['seed']): e for e in csv.DictReader(open(os.path.join(d, 'energy_recount.csv')))}
    out, shas = {}, set()
    for l in lines:
        if not l.startswith('RESULT'):
            continue
        r = dict(t.split('=', 1) for t in l.split()[1:] if '=' in t)
        if keep(r):
            e = en[(r['tag'], r['ds'], r['pl'], r['seed'])]
            out[(r['ds'], r['pl'], r['seed'])] = dict(mse=float(r['mse']), mae=float(r['mae']), es=float(r['syn_energy_uJ']),
                                                      ei=float(e['body_syn_int_uJ']) + float(e['head_AC_corr']) * 0.9e-6)
            shas.add(r['sha'])
    return out, sum(l.startswith('SKIP') for l in lines), shas


REF = {'ETTh1': 'spkmx_lif_muxlin_iand_lr0.00025_L1_e32_mlp4_wd0.1', 'ETTm1': 'spkmx_lif_muxlin_iand_lr0.0005_L2_pt16_e32_mlp0.5'}
M8, sk, sh1 = read(a.suite, lambda r: r['cond'] == 'spkmx_lif_muxlin_iand' and r['tag'].endswith('_T8'))
P8, _, sh2 = read(a.suite, lambda r: r['cond'] == 'spkmx_lif_msxdpca_iand' and r['tag'].endswith('_T8'))
shas = sh1 | sh2
ok(sk == 0 and len(M8) == 12 and len(P8) == 12, f'new suite: {len(M8)} + {len(P8)} rows, {sk} SKIP')
arms = {'muxlin IAND T8': M8, 'Pearson add + MUX IAND T8': P8,
        'muxlin IAND T4 (ref)': read('flif-v1-ett-spkmux-retune', lambda r: r['ds'] in REF and r['tag'] == REF[r['ds']] and r['pl'] in ('96', '720')
                                     and r['blocks'] == '1')[0],
        'Pearson add + MUX IAND T4 (stage 4)': read('flif-v1-ett-sxa-s4', lambda r: r['cond'] == 'spkmx_lif_msxdpca_iand')[0]}
ok(all(len(v) == 12 for v in arms.values()), 'every arm has 12 runs')
S = json.load(open(os.path.join(C, 'results', a.suite, 'summary.json')))
cells = [('ETTh1', '96'), ('ETTh1', '720'), ('ETTm1', '96'), ('ETTm1', '720')]
seeds = ['42', '7', '2026']
avg = lambda xs: sum(xs) / len(xs)
sm = lambda V, d, p, k: avg([V[(d, p, s)][k] for s in seeds])
for nm, V in arms.items():
    ok(close(avg([sm(V, d, p, 'mse') for d, p in cells]), S['mean4'][nm]['mse']) and close(avg([sm(V, d, p, 'es') for d, p in cells]), S['mean4'][nm]['syn_energy_uJ']),
       f'{nm}: 4-cell MSE / energy = summary')
for c in S['comparisons']:
    A, B = arms[c['A']], arms[c['B']]
    for met in ('mse', 'mae'):
        R = avg([avg([100 * (A[(d, p, s)][met] - B[(d, p, s)][met]) / B[(d, p, s)][met] for s in seeds]) for d, p in cells])
        lo = sum(A[(d, p, s)][met] < B[(d, p, s)][met] for d, p in cells for s in seeds)
        hi = sum(A[(d, p, s)][met] > B[(d, p, s)][met] for d, p in cells for s in seeds)
        v = 'better' if (R <= -1 and lo >= 8) else ('worse' if (R >= 1 and hi >= 8) else 'no clear difference')
        ok(close(R, c[met]['R_pct']) and lo == c[met]['pairs_lower'] and hi == c[met]['pairs_higher'] and v == c[met]['verdict'],
           f'{c["A"]} vs {c["B"]} {met}: R {R:+.4f} %, {lo} / {hi}, {v} = summary')
    ok(close(avg([sm(A, d, p, 'es') / sm(B, d, p, 'es') for d, p in cells]), c['energy_syops'])
       and close(avg([sm(A, d, p, 'ei') / sm(B, d, p, 'ei') for d, p in cells]), c['energy_int']), f'{c["A"]} vs {c["B"]} energy ratios = summary')
if a.sha:
    sa, sb = a.sha.split(',')
    ok(sorted(shas) == [sa], f'all new runs trained at {sa} ({sorted(shas)})')
    ch = subprocess.run(['git', 'diff', '--name-only', sa, sb, '--', '.'], cwd=os.path.dirname(C), capture_output=True, text=True).stdout.split()
    code = [x for x in ch if x.endswith(('.py', '.sh'))]
    ok(not code, f'commit {sb} changes no .py / .sh under NSMT/ since {sa} ({code})')
print('ALL CHECKS PASSED' if bad == 0 else f'{bad} FAIL')
sys.exit(0 if bad == 0 else 1)
