"""Independent recheck of summarize_ettune.py (2026-11-20, exp/f-lif-pop-v3; PROJECT_LOG ETT tuning pre-registration). Imports nothing
from summarize_* or pass_ett. Reads the RESULT lines itself, rebuilds the selection chain from the data (its own base configuration
per stage, NOT the earlier selected_*.json), and compares with selected_<stage>.json (selected value, current value and every score,
1e-9); --stage final also recomputes the tuned / untuned / SF-de 4-cell means, the syops energy A/B vs SF-de and the user's pass criteria
(its own copy of the reference values, transcribed again from the 21:09 entry's tables) and compares them with final.json (1e-9 / equal).
Selection rule: score(v) = mean over H96 / H720 of seed-mean best validation L1 (field best_epoch_val_epochs) at v / at the current value;
lowest wins, ties (< 1e-12) keep the current value, otherwise the earlier candidate.
Fail closed (exit 2): SKIP lines, duplicates, missing runs, rows of the new suite whose tag disagrees with their fields.
usage (from cinc_mixer_v1/): python -B recheck_ettune.py --stage patch|dmodel|mlp|blocks|wd|final [--sha <commit>[,<commit>...]]
env ET_DRY=<folder>: the new suite under <folder> (dry runs).
"""
import argparse
import collections
import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
DRY = os.environ.get('ET_DRY')
NEWD = os.path.join(DRY or RES, 'flif-v1-ett-spkmux-ettune')
K = ('PATCH', 'EMB', 'MLPR', 'BLOCKS', 'WD')
F = dict(zip(K, ('patch', 'emb', 'mlpr', 'blocks', 'wd')))
S = dict(zip(K, ('_pt', '_e', '_mlp', '_blk', '_wd')))
D0 = {'ETTh2': dict(zip(K, ('8', '64', '0.5', '1', '0.06'))), 'ETTm2': dict(zip(K, ('8', '64', '1', '1', '0.06')))}
ST = [('patch', 'PATCH', ['4', '8', '16']), ('dmodel', 'EMB', ['32', '64', '128', '256']), ('mlp', 'MLPR', ['0.5', '1', '2', '4']),
      ('blocks', 'BLOCKS', ['1', '2', '3']), ('wd', 'WD', ['0', '0.01', '0.06', '0.1'])]
LRS = {'ETTh2': '0.0005', 'ETTm2': '0.0001'}
CELLS = [('ETTh2', '96'), ('ETTh2', '720'), ('ETTm2', '96'), ('ETTm2', '720')]
SEEDS = ['42', '7', '2026']
# reference 4 cells (ETTh2 H96, H720, ETTm2 H96, H720) MSE / MAE: SpikF and the 9 DNN rows (21:09 entry, transcribed again)
REF = {'SpikF': ([0.290, 0.416, 0.175, 0.405], [0.336, 0.436, 0.251, 0.397]),
       'iTransformer': ([0.297, 0.427, 0.180, 0.412], [0.349, 0.445, 0.264, 0.407]),
       'iTransformer, MAE loss': ([0.294, 0.423, 0.182, 0.411], [0.342, 0.439, 0.265, 0.403]),
       'RLinear': ([0.288, 0.420, 0.182, 0.407], [0.338, 0.440, 0.265, 0.398]),
       'PatchTST (re-run)': ([0.302, 0.431, 0.175, 0.402], [0.348, 0.446, 0.259, 0.400]),
       'PatchTST (other re-run)': ([0.308, 0.436, 0.183, 0.412], [0.355, 0.450, 0.270, 0.404]),
       'TimeMixer': ([0.289, 0.412, 0.175, 0.391], [0.341, 0.434, 0.258, 0.396]),
       'TimesNet': ([0.340, 0.462, 0.187, 0.408], [0.374, 0.468, 0.267, 0.403]),
       'FITS': ([0.295, 0.431, 0.183, 0.407], [0.350, 0.446, 0.266, 0.399]),
       'DLinear (re-run)': ([0.333, 0.831, 0.193, 0.554], [0.387, 0.657, 0.292, 0.522])}
bad = []


def rd(p):
    out = []
    for l in open(p):
        if l.startswith('SKIP'):
            bad.append(f'SKIP in {p}')
        elif l.startswith('RESULT'):
            out.append(dict(t.split('=', 1) for t in l.split()[1:] if '=' in t))
    return out


def tg(ds, c):
    return f'spkmx_lif_muxlin_iand_lr{LRS[ds]}_L1' + ''.join(S[k] + c[k] for k in K if c[k] != D0[ds][k])


def ck(c):
    return '|'.join(c[k] for k in K)


rows = {}                                                                    # (ds, pl, seed, cfg) -> row
for r in rd(os.path.join(RES, 'flif-v1-ett-spkmux-tune', 'raw.txt')):
    if r.get('cond') == 'spkmx_lif_muxlin_iand' and r['ds'] in D0 and r['tag'] == tg(r['ds'], D0[r['ds']]) and r['pl'] in ('96', '720') \
            and r['seed'] in SEEDS:
        k = (r['ds'], r['pl'], r['seed'], ck(D0[r['ds']]))
        if k in rows:
            bad.append(f'duplicate {k}')
        rows[k] = r
new = rd(os.path.join(NEWD, 'raw.txt')) if os.path.exists(os.path.join(NEWD, 'raw.txt')) else []
for r in new:
    c = {k: r.get(F[k], '?') for k in K}
    if r['tag'] != tg(r['ds'], c) or c == D0[r['ds']]:
        bad.append(f'new row tag / fields disagree or default: {r["tag"]} {c}')
    k = (r['ds'], r['pl'], r['seed'], ck(c))
    if k in rows:
        bad.append(f'duplicate {k}')
    rows[k] = r


def val(r):
    return float(r['best_epoch_val_epochs'].split(':')[1])


def sm(ds, pl, c, f):
    v = [rows.get((ds, pl, s, ck(c))) for s in SEEDS]
    if any(x is None for x in v):
        bad.append(f'missing {ds} H{pl} {tg(ds, c)}'); return float('nan')
    return sum(f(x) for x in v) / 3


ap = argparse.ArgumentParser()
ap.add_argument('--stage', required=True, choices=[s[0] for s in ST] + ['final'])
ap.add_argument('--sha', default='')
a = ap.parse_args()
if a.sha:                                                                    # later commits (log only) must not change code
    SHA = a.sha.split(',')
    for c in SHA[1:]:
        d = subprocess.run(['git', 'diff', '--name-only', SHA[0], c, '--', '*.py', '*.sh'], capture_output=True, text=True, cwd=HERE)
        if d.returncode != 0 or d.stdout.strip():
            bad.append(f'commit {c} changes code against {SHA[0]}: {d.stdout.strip() or d.stderr.strip()}')
        else:
            print(f'commit {c}: no .py / .sh change against {SHA[0]}')
    c = collections.Counter(r['sha'] for r in new)
    print('new-suite sha:', dict(c))
    if not set(c) <= set(a.sha.split(',')):
        bad.append(f'sha {dict(c)} not in {a.sha}')
upto = len(ST) if a.stage == 'final' else [s[0] for s in ST].index(a.stage) + 1
base = {ds: dict(D0[ds]) for ds in D0}
mis = 0
for name, knob, vals in ST[:upto]:
    js = json.load(open(os.path.join(NEWD, f'selected_{name}.json')))
    for ds in ('ETTh2', 'ETTm2'):
        cur = base[ds][knob]
        sc = {}
        for v in vals:
            c = dict(base[ds], **{knob: v})
            sc[v] = sum(sm(ds, pl, c, val) / sm(ds, pl, base[ds], val) for pl in ('96', '720')) / 2
        if bad:
            print('\n'.join(bad)); sys.exit(2)
        b = min(sc.values())
        win = cur if abs(sc[cur] - b) < 1e-12 else sorted(vals, key=lambda v: (sc[v], vals.index(v)))[0]
        same = js['selected'][ds] == win and js['current'][ds] == cur and all(abs(js['scores'][ds][v] - sc[v]) < 1e-9 for v in vals)
        mis += not same
        print(f'{name} {ds}: current {cur} -> {win} (scores {", ".join(f"{v}:{sc[v]:.6f}" for v in vals)}) -> selected_{name}.json '
              f'{"ok" if same else "MISMATCH " + str(js["selected"][ds])}')
        base[ds][knob] = win
if a.stage == 'final':
    lrs = json.load(open(os.path.join(RES, 'flif-v1-ett-sfde', 'selected_lr.json')))['selected']
    sf = {}
    for r in rd(os.path.join(RES, 'flif-v1-ett-sfde', 'raw.txt')):
        if r.get('cond') == 'spk_ssa' and r['ds'] in D0 and r['tag'] == f'spk_ssa_lr{lrs[r["ds"]]}_L{"1" if r["ds"] == "ETTh2" else "2"}_nalif' \
                and r['pl'] in ('96', '720') and r['seed'] in SEEDS:
            sf[(r['ds'], r['pl'], r['seed'])] = r
    if len(sf) != 12:
        bad.append(f'{len(sf)} SF-de rows')
    if bad:
        print('\n'.join(bad)); sys.exit(2)

    def m4(get):
        return sum(sum(get(ds, pl, s) for s in SEEDS) / 3 for ds, pl in CELLS) / 4

    def cfg_get(cfgs, f):
        return lambda ds, pl, s: float(rows[(ds, pl, s, ck(cfgs[ds]))][f])
    tm, ta = m4(cfg_get(base, 'mse')), m4(cfg_get(base, 'mae'))
    um, ua = m4(cfg_get(D0, 'mse')), m4(cfg_get(D0, 'mae'))
    fm, fa = m4(lambda ds, pl, s: float(sf[(ds, pl, s)]['mse'])), m4(lambda ds, pl, s: float(sf[(ds, pl, s)]['mae']))
    er = sum(sum(float(rows[(ds, pl, s, ck(base[ds]))]['total_fft_uJ']) for s in SEEDS) / sum(float(sf[(ds, pl, s)]['total_fft_uJ']) for s in SEEDS)
             for ds, pl in CELLS) / 4
    fj = json.load(open(os.path.join(NEWD, 'final.json')))
    chk = [('config', fj['config'] == base), ('tuned mse', abs(fj['tuned']['mse'] - tm) < 1e-9), ('tuned mae', abs(fj['tuned']['mae'] - ta) < 1e-9),
           ('untuned mse', abs(fj['untuned']['mse'] - um) < 1e-9), ('sfde mse', abs(fj['sfde']['mse'] - fm) < 1e-9)]
    ok1 = sum(REF['SpikF'][0]) / 4 - tm >= 0.001 - 1e-12 and sum(REF['SpikF'][1]) / 4 - ta >= 0.001 - 1e-12
    ok2 = fm - tm >= 0.001 - 1e-12 and fa - ta >= 0.001 - 1e-12 and er <= 1.05 + 1e-12
    dn = [sum(REF[n][0]) / 4 - tm >= 0.001 - 1e-12 and sum(REF[n][1]) / 4 - ta >= 0.001 - 1e-12 for n in REF if n != 'SpikF']
    pj = fj['pass']['tuned LIF SpkMixer']
    chk += [('(1) SNN', pj['(1) SNN'] == ok1), ('(2) Spikformer', pj['(2) Spikformer'] == ok2), ('(3) DNN', pj['(3) DNN'] == all(dn)),
            ('pass', pj['pass'] == (ok1 and ok2 and all(dn)))]
    for n, ok in chk:
        mis += not ok
        print(f'final {n}: {"ok" if ok else "MISMATCH"}')
    print(f'tuned 4-cell MSE / MAE {tm:.6f} / {ta:.6f}; untuned {um:.6f} / {ua:.6f}; SF-de {fm:.6f} / {fa:.6f}; energy A/B vs SF-de {er:.4f}; '
          f'(1) {ok1} (2) {ok2} (3) {sum(dn)}/9 -> {"PASS" if ok1 and ok2 and all(dn) else "NOT PASSED"}')
if bad:
    print('\n'.join(bad)); sys.exit(2)
print('ALL CHECKS PASSED' if mis == 0 else f'{mis} MISMATCH(ES)')
sys.exit(0 if mis == 0 else 2)
