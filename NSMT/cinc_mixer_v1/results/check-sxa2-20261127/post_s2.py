"""Post hoc analysis of stage 2 of the fusion redesign (2026-11-27, exp/f-lif-pop-v3; written after training, NOT pre-registered).
  Q1 each stage-2 arm and the stage-1 selected arm against the IAND one-block reference (stage-1 verdict rule)
  Q2 per-cell MSE change of the MUX arms against their no-MUX arm and against muxlin (where the MUX gain comes from)
  Q3 epochs (best : run) per arm
usage (from cinc_mixer_v1/): python -B results/check-sxa2-20261127/post_s2.py
"""
import os

C = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CELLS = [('ETTh1', '96'), ('ETTh1', '720'), ('ETTm1', '96'), ('ETTm1', '720')]
SEEDS = ('42', '7', '2026')
avg = lambda x: sum(x) / len(x)


def rows(p):
    return [dict(t.split('=', 1) for t in l.split()[1:] if '=' in t) for l in open(p) if l.startswith('RESULT')]


D = {}
for su in ('flif-v1-ett-sxa-s1', 'flif-v1-ett-sxa-s2'):
    for r in rows(os.path.join(C, 'results', su, 'raw.txt')):
        D[(f'{r["cond"][10:-4]} {"shared" if r["tag"].endswith("_ws") else "indep."}', r['ds'], r['pl'], r['seed'])] = r
REF = {'ETTh1': 'spkmx_lif_muxlin_iand_lr0.00025_L1_e32_mlp4_wd0.1', 'ETTm1': 'spkmx_lif_muxlin_iand_lr0.0005_L2_pt16_e32_mlp0.5'}
for r in rows(os.path.join(C, 'results', 'flif-v1-ett-spkmux-retune', 'raw.txt')):
    if r['ds'] in REF and r['tag'] == REF[r['ds']] and r['pl'] in ('96', '720') and r['blocks'] == '1':
        D[('IAND ref', r['ds'], r['pl'], r['seed'])] = r


def comp(A, B, met='mse'):
    rc = [avg([100 * (float(D[(A, d, p, s)][met]) - float(D[(B, d, p, s)][met])) / float(D[(B, d, p, s)][met]) for s in SEEDS]) for d, p in CELLS]
    lo = sum(float(D[(A, d, p, s)][met]) < float(D[(B, d, p, s)][met]) for d, p in CELLS for s in SEEDS)
    hi = sum(float(D[(A, d, p, s)][met]) > float(D[(B, d, p, s)][met]) for d, p in CELLS for s in SEEDS)
    R = avg(rc)
    v = 'better' if R <= -1 and lo >= 8 else ('worse' if R >= 1 and hi >= 8 else 'no clear difference')
    return R, lo, hi, v, rc


print('Q1 vs the IAND one-block reference (test MSE; post hoc)')
for a in ('sxdpc indep.', 'sxdot indep.', 'msxdpc indep.', 'msxdot indep.'):
    R, lo, hi, v, rc = comp(a, 'IAND ref')
    Ra = comp(a, 'IAND ref', 'mae')
    print(f'  {a:14s} MSE R {R:+.2f} % ({lo} lower / {hi} higher) {v:20s} MAE R {Ra[0]:+.2f} % {Ra[3]:20s} per cell ' + ' '.join(f'{x:+.2f}' for x in rc))
print('Q2 per-cell MSE change (seed-mean %, ETTh1 96, 720, ETTm1 96, 720)')
for A, B in (('msxdpc indep.', 'sxdpc indep.'), ('msxdot indep.', 'sxdot indep.'), ('msxdpc indep.', 'muxlin indep.'), ('msxdot indep.', 'muxlin indep.')):
    R, lo, hi, v, rc = comp(A, B)
    print(f'  {A} vs {B}: ' + ' '.join(f'{x:+.2f}' for x in rc) + f'  (R {R:+.2f}, {lo}/{hi})')
print('Q3 best epoch : epochs run (means over 12 runs), runs at the cap')
for a in ('muxlin indep.', 'sxdpc indep.', 'sxdot indep.', 'msxdpc indep.', 'msxdot indep.', 'IAND ref'):
    be, er, cap = [], [], 0
    for d, p in CELLS:
        for s in SEEDS:
            b, _, n = D[(a, d, p, s)]['best_epoch_val_epochs'].split(':')
            be.append(int(b)); er.append(int(n)); cap += int(n) >= (100 if d == 'ETTh1' else 50)
    print(f'  {a:14s} best {avg(be):5.1f}  run {avg(er):5.1f}  at cap {cap}')
