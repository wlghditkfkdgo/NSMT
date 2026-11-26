"""Post hoc analysis of stage 1 of the fusion redesign (2026-11-27, exp/f-lif-pop-v3; written after training, NOT pre-registered).
  Q1 every stage-1 arm (all ADD) against the IAND one-block reference with the stage-1 verdict rule (only muxlin indep. vs IAND was
     pre-registered)
  Q2 integer-rule energy of the IAND reference from its own energy_recount.csv (flif-v1-ett-spkmux-retune), so that ADD and IAND can
     be compared under both rules
  Q3 direction of the five sharing comparisons per cell (seed-mean MSE, shared - independent)
  Q4 epochs: best epoch / epochs run per arm (from best_epoch_val_epochs), and runs that reached the epoch cap
usage (from cinc_mixer_v1/): python -B results/check-sxa-20261127/post_s1.py
"""
import csv
import json
import os

C = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
S = json.load(open(os.path.join(C, 'results', 'flif-v1-ett-sxa-s1', 'summary.json')))
CELLS = [('ETTh1', '96'), ('ETTh1', '720'), ('ETTm1', '96'), ('ETTm1', '720')]
SEEDS = ('42', '7', '2026')
avg = lambda x: sum(x) / len(x)


def rows(path):
    return [dict(t.split('=', 1) for t in l.split()[1:] if '=' in t) for l in open(path) if l.startswith('RESULT')]


D = {}
for r in rows(os.path.join(C, 'results', 'flif-v1-ett-sxa-s1', 'raw.txt')):
    arm = f'{r["cond"][10:-4]} {"shared" if r["tag"].endswith("_ws") else "indep."}'
    D[(arm, r['ds'], r['pl'], r['seed'])] = r
REF = {'ETTh1': 'spkmx_lif_muxlin_iand_lr0.00025_L1_e32_mlp4_wd0.1', 'ETTm1': 'spkmx_lif_muxlin_iand_lr0.0005_L2_pt16_e32_mlp0.5'}
for r in rows(os.path.join(C, 'results', 'flif-v1-ett-spkmux-retune', 'raw.txt')):
    if r['ds'] in REF and r['tag'] == REF[r['ds']] and r['pl'] in ('96', '720') and r['blocks'] == '1':
        D[('IAND ref', r['ds'], r['pl'], r['seed'])] = r
arms = [a for a in S['mean4'] if a != 'IAND ref']
print('Q1 each ADD arm vs the IAND reference (test MSE; stage-1 rule; post hoc)')
for a in arms:
    rc, lo, hi = [], 0, 0
    for d, p in CELLS:
        rc.append(avg([100 * (float(D[(a, d, p, s)]['mse']) - float(D[('IAND ref', d, p, s)]['mse'])) / float(D[('IAND ref', d, p, s)]['mse']) for s in SEEDS]))
        lo += sum(float(D[(a, d, p, s)]['mse']) < float(D[('IAND ref', d, p, s)]['mse']) for s in SEEDS)
        hi += sum(float(D[(a, d, p, s)]['mse']) > float(D[('IAND ref', d, p, s)]['mse']) for s in SEEDS)
    R = avg(rc)
    v = 'better' if R <= -1 and lo >= 8 else ('worse' if R >= 1 and hi >= 8 else 'no clear difference')
    print(f'  {a:14s} R {R:+.2f} %  lower {lo:2d} / higher {hi:2d}  {v:20s} per cell ' + ' '.join(f'{x:+.2f}' for x in rc))
print('Q2 integer-rule energy (body_syn_int_uJ + head_AC_corr x 0.9 pJ), 4-cell mean of seed means')
en = {(e['tag'], e['ds'], e['pl'], e['seed']): e for e in csv.DictReader(open(os.path.join(C, 'results', 'flif-v1-ett-spkmux-retune', 'energy_recount.csv')))}
vals = []
for d, p in CELLS:
    vals.append(avg([float(en[(REF[d], d, p, s)]['body_syn_int_uJ']) + float(en[(REF[d], d, p, s)]['head_AC_corr']) * 0.9e-6 for s in SEEDS]))
print(f'  IAND ref: {avg(vals):.2f} uJ (cells ' + ' '.join(f'{x:.2f}' for x in vals) + ')')
for a in ('muxlin indep.', 'sxdpc indep.', 'sxppc shared'):
    print(f'  {a}: {S["mean4"][a]["E_int"]:.2f} uJ; syops {S["mean4"][a]["syn_energy_uJ"]:.2f} uJ (IAND ref syops {S["mean4"]["IAND ref"]["syn_energy_uJ"]:.2f})')
print('Q3 sharing: seed-mean MSE shared - independent per cell (ETTh1 96, 720, ETTm1 96, 720)')
for c in ('muxlin', 'sxd', 'sxdpc', 'sxp', 'sxppc'):
    diff = [S['cells'][f'{c} shared'][f'{d} {p}']['mse'] - S['cells'][f'{c} indep.'][f'{d} {p}']['mse'] for d, p in CELLS]
    print(f'  {c:7s} ' + ' '.join(f'{x:+.4f}' for x in diff) + f'   cells lower {sum(x < 0 for x in diff)}/4')
print('Q4 best epoch : epochs run (mean over 12 runs) and runs that reached the epoch cap (ETTh1 100, ETTm1 50)')
for a in arms + ['IAND ref']:
    be, er, cap = [], [], 0
    for d, p in CELLS:
        for s in SEEDS:
            b, _, n = D[(a, d, p, s)]['best_epoch_val_epochs'].split(':')
            be.append(int(b)); er.append(int(n))
            cap += int(n) >= (100 if d == 'ETTh1' else 50)
    print(f'  {a:14s} best {avg(be):5.1f}  run {avg(er):5.1f}  at cap {cap}')
