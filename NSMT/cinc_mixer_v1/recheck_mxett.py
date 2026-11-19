"""Independent recheck of summarize_pn.py --stage mxett (2026-11-19 evening, exp/f-lif-pop-v3; PROJECT_LOG mixer-only population ETT
pre-registration). Imports nothing from summarize_* or pass_ett: reads raw.txt and energy_recount.csv of each suite itself, recomputes
  B1 pop LIF mixer vs LIF, B3 pop LIF mixer vs pop LIF all, C3 pop LIF mixer+mean vs pop LIF mixer, C5 pop f-LIF mixer+mean vs pop f-LIF mixer
  (test MSE; r = 100 (A - B) / B per seed, R = mean over the 4 cells of the seed-mean r; better = R <= -1 and >= 8 of 12 lower, worse = mirror;
  energy A/B = mean over cells of seed-mean ratios, syops rule), and the adoption (4-cell mean test MSE and MAE both >= 0.001 below the LIF
  SpkMixer, syops energy A/B vs SF-de <= 1.05; lowest MSE among the passing arms, else LIF),
then compares them with the verdict table of results/flif-v1-ett-spkmux-mxpop/summary.md (printed precision) and with selected_structure.json
(full precision, 1e-9). syops energy: Es = total_fft_uJ - syn_energy_uJ + head_MAC x 4.6e-6 + body_syn_uJ (non-population rows: = total_fft_uJ).
Fail closed: missing / duplicate / non-finite rows, a recount that does not reproduce the logged MSE, SKIP lines, a row count != 48 -> exit 2.
usage (from cinc_mixer_v1/): python -B recheck_mxett.py [--sha <pre-reg commit>[,<later commit>...]]   (env PN_DRY=<folder>: dry runs)
"""
import collections
import csv
import json
import math
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
DRY = os.environ.get('PN_DRY')
NEW = ('flif-v1-ett-spkmux-mxpop', 'flif-v1-ett-sfde')
d_ = lambda s: os.path.join(DRY, s) if DRY and s in NEW else os.path.join(RES, s)
CELLS = (('ETTh2', '96'), ('ETTh2', '720'), ('ETTm2', '96'), ('ETTm2', '720'))
SEEDS = ('42', '7', '2026')
LRM = {'ETTh2': '0.0005', 'ETTm2': '0.0001'}
LYS = {'ETTh2': '1', 'ETTm2': '2'}
bad = []


def rows(p):
    out = []
    for l in open(p):
        if l.startswith('SKIP'):
            bad.append(f'SKIP line in {p}')
        if l.startswith('RESULT'):
            out.append(dict(t.split('=', 1) for t in l.split()[1:] if '=' in t))
    return out


def arm(suite, cond, tagf):
    p = d_(suite)
    rr = [r for r in rows(os.path.join(p, 'raw.txt')) if r.get('cond') == cond]
    en = collections.defaultdict(list)
    for e in csv.DictReader(open(os.path.join(p, 'energy_recount.csv'))):
        en[(e.get('tag', ''), e['ds'], e['pl'], e['seed'])].append(e)
    T = {}
    for ds, pl in CELLS:
        for s in SEEDS:
            m = [r for r in rr if r['ds'] == ds and r['pl'] == pl and r['seed'] == s and r['tag'] == tagf(ds)]
            e = en.get((tagf(ds), ds, pl, s), [])
            if len(m) != 1 or len(e) != 1:
                bad.append(f'missing/ambiguous {suite} {tagf(ds)} {ds} {pl} {s}: {len(m)} rows, {len(e)} energy rows'); continue
            r, e = m[0], e[0]
            tot, syn = float(r['total_fft_uJ']), float(r['syn_energy_uJ'])
            es = tot - syn + float(e['head_MAC']) * 4.6e-6 + float(e['body_syn_uJ'])
            v = dict(mse=float(r['mse']), mae=float(r['mae']), Es=es, Eo=float(e['total_fft_uJ_corr']))
            if not all(math.isfinite(x) for x in v.values()) or abs(float(e['mse_recomputed']) - v['mse']) > 1e-5:
                bad.append(f'bad row {suite} {tagf(ds)} {ds} {pl} {s}: {v}'); continue
            pop = cond.startswith('spkmx_pop_')
            if (not pop and abs(es - tot) > 1e-3 * syn) or (pop and es < tot - 1e-3 * syn):
                bad.append(f'syops energy {suite} {tagf(ds)} {ds} {pl} {s}: {es} vs logged {tot}')
            T[(ds, pl, s)] = v
    return T


def cmp(A, B):
    cell, lo, hi, er = [], 0, 0, []
    for ds, pl in CELLS:
        rs = [100 * (A[(ds, pl, s)]['mse'] - B[(ds, pl, s)]['mse']) / B[(ds, pl, s)]['mse'] for s in SEEDS]
        lo += sum(x < 0 for x in rs); hi += sum(x > 0 for x in rs); cell.append(sum(rs) / 3)
        er.append(sum(A[(ds, pl, s)]['Es'] for s in SEEDS) / sum(B[(ds, pl, s)]['Es'] for s in SEEDS))
    R = sum(cell) / 4
    v = 'better' if R <= -1 and lo >= 8 else 'worse' if R >= 1 and hi >= 8 else 'no clear difference'
    return R, lo, v, sum(er) / 4


def cm(A, k):
    return sum(sum(A[(ds, pl, s)][k] for s in SEEDS) / 3 for ds, pl in CELLS) / 4


if '--sha' in sys.argv:                                                       # rows of the new suite must come from these commits
    SHA = sys.argv[sys.argv.index('--sha') + 1].split(',')
    for c in SHA[1:]:
        d = subprocess.run(['git', 'diff', '--name-only', SHA[0], c, '--', '*.py', '*.sh'], capture_output=True, text=True, cwd=HERE)
        if d.returncode != 0 or d.stdout.strip():
            sys.exit(f'STOP: commit {c} changes code against {SHA[0]}: {d.stdout.strip() or d.stderr.strip()}')
        print(f'commit {c}: no .py / .sh change against {SHA[0]}')
    c = collections.Counter(r['sha'] for r in rows(os.path.join(d_('flif-v1-ett-spkmux-mxpop'), 'raw.txt')))
    print('flif-v1-ett-spkmux-mxpop sha:', dict(c))
    if not set(c) <= set(SHA):
        bad.append(f'sha {dict(c)} not in {SHA}')
n = len(rows(os.path.join(d_('flif-v1-ett-spkmux-mxpop'), 'raw.txt')))
if n != 48:
    bad.append(f'{n} RESULT rows in flif-v1-ett-spkmux-mxpop, want 48')
lrs = json.load(open(os.path.join(d_('flif-v1-ett-sfde'), 'selected_lr.json')))['selected']
T = {'LIF': arm('flif-v1-ett-spkmux-tune', 'spkmx_lif_muxlin_iand', lambda ds: f'spkmx_lif_muxlin_iand_lr{LRM[ds]}_L1'),
     'LIF+mean': arm('flif-v1-ett-spkmux-mx3', 'spkmx_lif_muxlin_iand', lambda ds: f'spkmx_lif_muxlin_iand_lr{LRM[ds]}_L1_mpmean'),
     'pop LIF all': arm('flif-v1-ett-spkmux-stft', 'spkmx_pop_lif_muxlin_iand', lambda ds: f'spkmx_pop_lif_muxlin_iand_lr{LRM[ds]}_L1'),
     'SF-de': arm('flif-v1-ett-sfde', 'spk_ssa', lambda ds: f'spk_ssa_lr{lrs[ds]}_L{LYS[ds]}_nalif')}
MX = {'pop LIF mixer': ('pop_lif', '_nsmixer'), 'pop LIF mixer+mean': ('pop_lif', '_mpmean_nsmixer'), 'pop f-LIF mixer': ('pop_flif', '_nsmixer'),
      'pop f-LIF mixer+mean': ('pop_flif', '_mpmean_nsmixer')}
for nm, (mx, suf) in MX.items():
    T[nm] = arm('flif-v1-ett-spkmux-mxpop', f'spkmx_{mx}_muxlin_iand', lambda ds, m=mx, s=suf: f'spkmx_{m}_muxlin_iand_lr{LRM[ds]}_L1{s}')
if bad:
    print('\n'.join(bad)); sys.exit(2)
Q = [('B1 pop LIF mixer vs LIF', 'pop LIF mixer', 'LIF'), ('B3 pop LIF mixer vs pop LIF all', 'pop LIF mixer', 'pop LIF all'),
     ('C3 pop LIF mixer+mean vs pop LIF mixer', 'pop LIF mixer+mean', 'pop LIF mixer'), ('C5 pop f-LIF mixer+mean vs pop f-LIF mixer', 'pop f-LIF mixer+mean', 'pop f-LIF mixer')]
summ = open(os.path.join(d_('flif-v1-ett-spkmux-mxpop'), 'summary.md')).read()
mis = 0
for name, a, b in Q:
    R, lo, v, er = cmp(T[a], T[b])
    m = re.search(r'^\| ' + re.escape(name) + r' \| ([+-]\d+\.\d+) \| (\d+)/12 \| ([a-z ]+) \| (\d+\.\d+) \|', summ, re.M)
    ok = bool(m) and m.group(1) == f'{R:+.2f}' and int(m.group(2)) == lo and m.group(3) == v and m.group(4) == f'{er:.2f}'
    mis += not ok
    print(f'{name}: R {R:+.3f} lower {lo}/12 {v} energy (syops) {er:.4f} -> summary {"ok" if ok else "MISMATCH " + (m.group(0) if m else "row not found")}')
lm = (cm(T['LIF'], 'mse'), cm(T['LIF'], 'mae'))
cand = []
sel = json.load(open(os.path.join(d_('flif-v1-ett-spkmux-mxpop'), 'selected_structure.json')))
for nm in MX:
    am = (cm(T[nm], 'mse'), cm(T[nm], 'mae'))
    er = sum(sum(T[nm][(ds, pl, s)]['Es'] for s in SEEDS) / sum(T['SF-de'][(ds, pl, s)]['Es'] for s in SEEDS) for ds, pl in CELLS) / 4
    ok = lm[0] - am[0] >= 0.001 - 1e-12 and lm[1] - am[1] >= 0.001 - 1e-12 and er <= 1.05 + 1e-12
    if ok:
        cand.append((am[0], er, nm))
    s = sel['scores'][nm]
    same = all(abs(x - y) < 1e-9 for x, y in ((s['mse'], am[0]), (s['mae'], am[1]), (s['energy_ratio_sfde'], er))) and s['passes'] == ok
    mis += not same
    print(f'{nm}: MSE / MAE {am[0]:.6f} / {am[1]:.6f} (LIF - arm {lm[0] - am[0]:+.6f} / {lm[1] - am[1]:+.6f}) energy vs SF-de {er:.4f} passes {ok} '
          f'-> selected_structure.json {"ok" if same else "MISMATCH"}')
cand.sort()
pick = cand[0][2] if cand else 'LIF'
if len(cand) > 1 and abs(cand[0][0] - cand[1][0]) < 1e-12:
    pick = min(cand[:2], key=lambda z: z[1])[2]
mis += pick != sel['selected']
print(f'adopted: {pick} -> selected_structure.json {sel["selected"]} {"ok" if pick == sel["selected"] else "MISMATCH"}')
for nm, X in T.items():
    print(f'{nm:22s} 4-cell mean MSE {cm(X, "mse"):.4f} MAE {cm(X, "mae"):.4f} energy syops {cm(X, "Es"):.2f} old {cm(X, "Eo"):.2f}')
print('ALL CHECKS PASSED' if mis == 0 else f'{mis} MISMATCH(ES)')
sys.exit(0 if mis == 0 else 2)
