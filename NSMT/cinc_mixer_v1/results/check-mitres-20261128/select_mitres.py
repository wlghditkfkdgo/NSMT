"""lr selection for the MIT-BIH residual test (2026-11-28, exp/f-lif-pop-v3; PROJECT_LOG pre-registration; user "1과 2를 진행해줘", item 1
"MIT-BIH SpkMixer의 energy 줄이기"). Validation only; the rule of the 11-01 MIT-BIH tuning (summarize_mitbih_tune.py):
  runs   muxlin, 2 mixer layers, residual IAND and MS (membrane shortcut), lr {0.0025, 0.005, 0.01, 0.02} x seeds 42 / 7 / 2026,
         val_fold 0 (StratifiedKFold fold 0 of the training file), suite flif-v1-mitbih-res
  score  the seed-mean last-epoch validation macro-F1 (val_f1_last); the highest wins; ties (< 1e-9) keep lr 0.02 (the lr of the
         ADD base); a missing run, a SKIP line, mixed commits or a non-finite score -> stop (exit 2), nothing written
  output results/flif-v1-mitbih-res/selected_lr.json (never overwritten with a different selection); --print_env prints the two
         lines "IAND=<lr>" / "MS=<lr>" for the driver
  report the IAND lr 0.02 validation rows of this suite against the 11-01 rows of flif-v1-mitbih-spkmux-tune (reproduction)
usage (from cinc_mixer_v1/): python -B results/check-mitres-20261128/select_mitres.py [--root results] [--print_env]
"""
import argparse
import json
import math
import os
import sys

C = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser()
ap.add_argument('--root', default=os.path.join(C, 'results'))
ap.add_argument('--print_env', action='store_true')
a = ap.parse_args()
LRS = ('0.0025', '0.005', '0.01', '0.02')
SEEDS = ('42', '7', '2026')
RES = {'IAND': 'spkmx_lif_muxlin_iand', 'MS': 'spkmx_lif_muxlin_ms'}


def stop(m):
    print(f'STOP: {m}')
    sys.exit(2)


p = os.path.join(a.root, 'flif-v1-mitbih-res', 'raw.txt')
if not os.path.exists(p):
    stop(f'{p} missing')
lines = open(p).read().splitlines()
if any(l.startswith('SKIP') for l in lines):
    stop('SKIP line in flif-v1-mitbih-res')
rows = [dict(t.split('=', 1) for t in l.split()[1:] if '=' in t) for l in lines if l.startswith('RESULT')]
val = [r for r in rows if r.get('val_fold') == '0']
if len({r['sha'] for r in val}) != 1:
    stop(f'mixed commits {sorted({r["sha"] for r in val})}')
S, sel = {}, {}
for k, cond in RES.items():
    S[k] = {}
    for lr in LRS:
        rr = [r for r in val if r['cond'] == cond and r['lr'] == lr]
        if sorted(r['seed'] for r in rr) != sorted(SEEDS):
            stop(f'{k} lr {lr}: seeds {sorted(r["seed"] for r in rr)}')
        v = [100. * float(r['val_f1_last']) for r in rr]
        if not all(math.isfinite(x) for x in v):
            stop(f'{k} lr {lr}: non-finite validation macro-F1')
        S[k][lr] = sum(v) / 3
    best = max(S[k].values())
    win = [lr for lr in LRS if best - S[k][lr] < 1e-9]
    sel[k] = '0.02' if '0.02' in win else win[0]
out = dict(stage='MIT-BIH residual lr (validation only)', selected=sel, scores=S)
sp = os.path.join(a.root, 'flif-v1-mitbih-res', 'selected_lr.json')
if os.path.exists(sp) and json.load(open(sp))['selected'] != sel:
    stop(f'{sp} holds a different selection {json.load(open(sp))["selected"]} vs {sel}')
json.dump(out, open(sp, 'w'), indent=1)
if a.print_env:
    print(f'IAND={sel["IAND"]}')
    print(f'MS={sel["MS"]}')
    sys.exit(0)
for k in RES:
    print(f'{k}: selected lr {sel[k]}; seed-mean last-epoch validation macro-F1 ' + ', '.join(f'{lr} {S[k][lr]:.3f}' for lr in LRS))
old = os.path.join(a.root, 'flif-v1-mitbih-spkmux-tune', 'raw.txt')
if os.path.exists(old):
    o = {r['seed']: r for r in (dict(t.split('=', 1) for t in l.split()[1:] if '=' in t) for l in open(old) if l.startswith('RESULT'))
         if r.get('cond') == 'spkmx_lif_muxlin_iand' and r.get('lr') == '0.02' and r.get('val_fold') == '0'}
    n = {r['seed']: r for r in val if r['cond'] == RES['IAND'] and r['lr'] == '0.02'}
    if sorted(o) == sorted(SEEDS):
        same = [o[s]['val_f1_last'] == n[s]['val_f1_last'] for s in SEEDS]
        print('report: IAND lr 0.02 validation macro-F1 vs the 11-01 rows (seed 42 / 7 / 2026): ' +
              ' / '.join(f'{n[s]["val_f1_last"]} vs {o[s]["val_f1_last"]}' for s in SEEDS) + f' -> identical {sum(same)} / 3')
