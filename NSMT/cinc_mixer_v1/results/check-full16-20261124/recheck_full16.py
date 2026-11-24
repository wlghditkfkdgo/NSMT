"""Independent recheck of results/flif-v1-ett-spkmux-full16/summary.json (2026-11-24, exp/f-lif-pop-v3; PROJECT_LOG full16 result).
No training; imports nothing from this repository. Reads the RESULT lines directly:
  4 old cells  results/flif-v1-ett-spkmux-ettune/raw.txt  (first chain)  and  results/flif-v1-ett-spkmux-ettune2/raw.txt (extension),
               rows with the final tags (ETTh2 / ETTm2 x H96 / H720 x seeds 42 / 7 / 2026; the last line wins if a key repeats -- the
               summarizer's loader rejects repeats in its own suite, so a repeat is reported here)
  12 new cells results/flif-v1-ett-spkmux-full16/raw.txt
and recomputes: per model the 16-cell means of the seed means (MSE, MAE, syn_energy_uJ), the margin against each reference row of
refs16.json, the pass flags, the number of cells whose syops energy is below SpikF (spikf16/*.csv) and the verdict; plus per-dataset
means vs SpikF (report). Compares with summary.json (1e-9); prints PASS / FAIL lines and "ALL CHECKS PASSED".
usage (from cinc_mixer_v1/): python -B results/check-full16-20261124/recheck_full16.py > results/check-full16-20261124/recheck_full16.txt
"""
import csv
import json
import os
import sys

C = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = os.path.join(C, 'results')
DSS, PLS, SEEDS = ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'), (96, 192, 336, 720), (42, 7, 2026)
OLD = {('ETTh2', 96), ('ETTh2', 720), ('ETTm2', 96), ('ETTm2', 720)}
TAGS = {'first': {'h': 'spkmx_lif_muxlin_iand_lr0.0005_L1_pt16_wd0.1', 'm': 'spkmx_lif_muxlin_iand_lr0.0001_L1_pt16_mlp2_wd0.1'},
        'ext': {'h': 'spkmx_lif_muxlin_iand_lr0.0005_L1_pt24_wd0.005', 'm': 'spkmx_lif_muxlin_iand_lr0.0001_L1_pt24_e256_blk2_wd0.02'}}
OLDSUITE = {'first': 'flif-v1-ett-spkmux-ettune', 'ext': 'flif-v1-ett-spkmux-ettune2'}
fails = 0


def ok(c, m):
    global fails
    print(('PASS ' if c else 'FAIL ') + m)
    fails += 0 if c else 1


def rows(path):
    for l in open(path):
        if l.startswith('RESULT'):
            yield dict(t.split('=', 1) for t in l.split()[1:] if '=' in t)


V = {mk: {} for mk in TAGS}
rep = 0
for mk in TAGS:
    for r in rows(os.path.join(R, OLDSUITE[mk], 'raw.txt')):
        f = 'h' if r['ds'].startswith('ETTh') else 'm'
        k = (r['ds'], int(r['pl']), int(r['seed']))
        if r['tag'] == TAGS[mk][f] and (k[0], k[1]) in OLD:
            rep += k in V[mk]
            V[mk][k] = tuple(float(r[x]) for x in ('mse', 'mae', 'syn_energy_uJ'))
print(f'note: repeated keys among the old-cell rows (last line kept): {rep}')
skips = sum(1 for l in open(os.path.join(R, 'flif-v1-ett-spkmux-full16', 'raw.txt')) if l.startswith('SKIP'))
ok(skips == 0, f'no SKIP line in the new suite ({skips})')
n = 0
for r in rows(os.path.join(R, 'flif-v1-ett-spkmux-full16', 'raw.txt')):
    f = 'h' if r['ds'].startswith('ETTh') else 'm'
    mk = [m for m in TAGS if TAGS[m][f] == r['tag']]
    k = (r['ds'], int(r['pl']), int(r['seed']))
    if len(mk) == 1 and (k[0], k[1]) not in OLD and k not in V[mk[0]]:
        V[mk[0]][k] = tuple(float(r[x]) for x in ('mse', 'mae', 'syn_energy_uJ')); n += 1
ok(n == 72, f'72 new rows with the final tags, new cells only, no repeats ({n})')
for mk in TAGS:
    ok(len(V[mk]) == 48, f'{mk}: 48 rows = 16 cells x 3 seeds ({len(V[mk])})')
REF = json.load(open(os.path.join(R, 'ett-full16-20261124', 'refs16.json')))['models']
SP = {}
for ds in DSS:
    for r in csv.DictReader(open(os.path.join(R, 'ett-full16-20261124', 'spikf16', f'spikf_syops_{ds}.csv'))):
        SP[(r['ds'], int(r['pl']))] = float(r['energy_uJ'])
S = json.load(open(os.path.join(R, 'flif-v1-ett-spkmux-full16', 'summary.json')))
for mk in TAGS:
    cm = {(d, p): [sum(V[mk][(d, p, s)][j] for s in SEEDS) / 3 for j in range(3)] for d in DSS for p in PLS}
    om = sum(v[0] for v in cm.values()) / 16; oa = sum(v[1] for v in cm.values()) / 16
    M = S['models'][mk]
    ok(abs(M['mse16'] - om) < 1e-9 and abs(M['mae16'] - oa) < 1e-9, f'{mk}: 16-cell MSE / MAE {om:.6f} / {oa:.6f} = summary')
    acc = True
    for m, v in REF.items():
        rm = sum(v['rows'][f'{d} {p}'][0] for d in DSS for p in PLS) / 16
        ra = sum(v['rows'][f'{d} {p}'][1] for d in DSS for p in PLS) / 16
        ps = rm - om >= 0.001 - 1e-12 and ra - oa >= 0.001 - 1e-12
        g = M['rows'][m]
        ok(abs(g['margin_mse'] - (rm - om)) < 1e-9 and abs(g['margin_mae'] - (ra - oa)) < 1e-9 and g['passes'] == ps,
           f'{mk} vs {m}: margin {rm - om:+.6f} / {ra - oa:+.6f}, passes {ps} = summary')
        if v['role'] != 'own':
            acc = acc and ps
    below = sum(cm[c][2] < SP[c] for c in cm)
    verdict = 'PASS' if acc and below == 16 else 'NOT PASSED'
    ok(M['energy_cells_below_spikf'] == below and M['verdict'] == verdict, f'{mk}: energy below SpikF {below}/16, verdict {verdict} = summary')
    for ds in DSS:
        a = sum(cm[(ds, p)][0] for p in PLS) / 4; b = sum(REF['SpikF']['rows'][f'{ds} {p}'][0] for p in PLS) / 4
        a2 = sum(cm[(ds, p)][1] for p in PLS) / 4; b2 = sum(REF['SpikF']['rows'][f'{ds} {p}'][1] for p in PLS) / 4
        print(f'  report {mk} {ds}: mean of 4 horizons MSE {a:.4f} vs SpikF {b:.4f} ({a - b:+.4f}), MAE {a2:.4f} vs {b2:.4f} ({a2 - b2:+.4f})')
    ex = [c for c in cm if c[0] != 'ETTh1']
    om12 = sum(cm[c][0] for c in ex) / 12; oa12 = sum(cm[c][1] for c in ex) / 12
    sm12 = sum(REF['SpikF']['rows'][f'{d} {p}'][0] for d, p in ex) / 12; sa12 = sum(REF['SpikF']['rows'][f'{d} {p}'][1] for d, p in ex) / 12
    print(f'  report {mk} without ETTh1 (12 cells): MSE {om12:.4f} vs SpikF {sm12:.4f} ({sm12 - om12:+.4f}), MAE {oa12:.4f} vs {sa12:.4f} ({sa12 - oa12:+.4f})')
print('ALL CHECKS PASSED' if fails == 0 else f'{fails} CHECKS FAILED')
sys.exit(0 if fails == 0 else 1)
