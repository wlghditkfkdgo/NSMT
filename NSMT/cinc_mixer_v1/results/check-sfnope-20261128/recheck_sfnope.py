"""Independent recheck of summary_sfnope.json (2026-11-28, exp/f-lif-pop-v3; written while the pool ran, run after "driver end").
No repo import and no code shared with summarize_sfnope.py: reads the raw.txt files directly, recomputes every MIT-BIH d / mean / verdict
(M1-M4) and ETT R / pair counts / verdict (E1-E3, MSE and MAE) and the energy ratios, and compares them with the summary; also the order
checks from the order JSONs. Prints PASS / FAIL lines and ALL CHECKS PASSED.
usage (from cinc_mixer_v1/): python -B results/check-sfnope-20261128/recheck_sfnope.py
"""
import json
import math
import os

C = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
H = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(os.path.join(H, 'summary_sfnope.json')))
bad = 0


def ok(c, m):
    global bad
    bad += not c
    print(('PASS ' if c else 'FAIL ') + m)


def res(suite):
    out = []
    for l in open(os.path.join(C, 'results', suite, 'raw.txt')):
        if l.startswith('RESULT'):
            out.append(dict(t.split('=', 1) for t in l.split()[1:] if '=' in t))
    return out


seeds = ['42', '7', '2026']
mt = {'SF-de PE off (new)': ('flif-v1-mitbih-sfde-nope', 'spk_ssa_lr0.01_L2_nalif_pe0'), 'SF-de PE on': ('flif-v1-mitbih-sfde', 'spk_ssa_lr0.01_L2_nalif'),
      'SpkMixer LIF memory': ('flif-v1-mitbih-spkmux-tune', 'spkmx_lif_muxlin_add_lr0.02_L2'),
      'SpkMixer memoryless': ('flif-v1-mitbih-memoff', 'spkmx_lif_muxlin_add_lr0.02_L2_tmoff')}
M = {}
for k, (su, tg) in mt.items():
    rr = [r for r in res(su) if r['tag'] == tg]
    ok(sorted(r['seed'] for r in rr) == sorted(seeds), f'MIT-BIH {k}: three seeds, no duplicate')
    M[k] = {r['seed']: r for r in rr}
new = res('flif-v1-mitbih-sfde-nope')
ok(len(new) == 3 and len({r['sha'] for r in new}) == 1, f'MIT-BIH new suite: 3 rows, one commit {sorted({r["sha"] for r in new})}')
ok(not any(l.startswith('SKIP') for l in open(os.path.join(C, 'results', 'flif-v1-mitbih-sfde-nope', 'raw.txt'))), 'MIT-BIH new suite: no SKIP')
for c in S['mitbih']['comparisons']:
    A, B = M[c['A']], M[c['B']]
    d = [100 * (float(A[s]['f1']) - float(B[s]['f1'])) for s in seeds]
    m = sum(d) / 3
    v = 'A better' if m >= 0.5 and sum(x > 0 for x in d) >= 2 else ('A worse' if m <= -0.5 and sum(x < 0 for x in d) >= 2 else 'no clear difference')
    er = sum(float(A[s]['syn_energy_uJ']) for s in seeds) / sum(float(B[s]['syn_energy_uJ']) for s in seeds)
    ok(all(abs(x - y) < 1e-9 for x, y in zip(d, c['d_pp'])) and abs(m - c['mean_pp']) < 1e-9 and v == c['verdict'] and abs(er - c['energy_ratio']) < 1e-9,
       f'MIT-BIH {c["kind"]}: d {[round(x, 3) for x in d]}, mean {m:+.3f} %p, {v}, energy {er:.3f} = summary')
om = json.load(open(os.path.join(H, 'order_sfdenope.json')))
ok(all(abs(x[v]['drop_f1_pp']) < 0.05 for x in om['runs'] for v in ('mixer', 'full')),
   'MIT-BIH PE-off order-blind: ' + ', '.join(f'{x["seed"]} SSA {x["mixer"]["drop_f1_pp"]:+.4f} / full {x["full"]["drop_f1_pp"]:+.4f}' for x in om['runs']))
ok(all(abs(float(M['SF-de PE off (new)'][x['seed']]['f1']) * 100 - x['f1']) < 1e-4 for x in om['runs']), 'MIT-BIH order eval: unshuffled macro-F1 = logged')

cells = [(d, p) for d in ('ETTh1', 'ETTm1') for p in ('96', '720')]
REF = {'ETTh1': 'spkmx_lif_muxlin_iand_lr0.00025_L1_e32_mlp4_wd0.1', 'ETTm1': 'spkmx_lif_muxlin_iand_lr0.0005_L2_pt16_e32_mlp0.5'}
SF = {'ETTh1': 'spk_ssa_lr0.0005_L1_nalif', 'ETTm1': 'spk_ssa_lr0.0001_L2_pt16_nalif'}
et = {'SF-de PE off (new)': ('flif-v1-ett-sfde-nope', lambda r: r['tag'] == SF.get(r['ds'], '') + '_pe0'),
      'SF-de PE on': ('flif-v1-ett-sfde-h1m1', lambda r: r['tag'] == SF.get(r['ds'], '')),
      'muxlin IAND LIF memory': ('flif-v1-ett-spkmux-retune', lambda r: r['tag'] == REF.get(r['ds'], '') and r.get('blocks') == '1'),
      'muxlin IAND memoryless': ('flif-v1-ett-memoff', lambda r: r['tag'] == REF.get(r['ds'], '') + '_tmoff')}
E = {}
for k, (su, f) in et.items():
    rr = [r for r in res(su) if r['ds'] in ('ETTh1', 'ETTm1') and r['pl'] in ('96', '720') and f(r)]
    keys = [(r['ds'], r['pl'], r['seed']) for r in rr]
    ok(len(keys) == 12 and len(set(keys)) == 12, f'ETT {k}: 12 distinct runs')
    E[k] = {(r['ds'], r['pl'], r['seed']): r for r in rr}
new = res('flif-v1-ett-sfde-nope')
ok(len(new) == 12 and len({r['sha'] for r in new}) == 1, f'ETT new suite: 12 rows, one commit {sorted({r["sha"] for r in new})}')
ok(not any(l.startswith('SKIP') for l in open(os.path.join(C, 'results', 'flif-v1-ett-sfde-nope', 'raw.txt'))), 'ETT new suite: no SKIP')
for c in S['ett']['comparisons']:
    A, B = E[c['A']], E[c['B']]
    for met in ('mse', 'mae'):
        rc = [sum(100 * (float(A[(d, p, s)][met]) - float(B[(d, p, s)][met])) / float(B[(d, p, s)][met]) for s in seeds) / 3 for d, p in cells]
        R = sum(rc) / 4
        lo = sum(float(A[(d, p, s)][met]) < float(B[(d, p, s)][met]) for d, p in cells for s in seeds)
        hi = sum(float(A[(d, p, s)][met]) > float(B[(d, p, s)][met]) for d, p in cells for s in seeds)
        v = 'A better' if R <= -1 and lo >= 8 else ('A worse' if R >= 1 and hi >= 8 else 'no clear difference')
        sm = c[met]
        ok(abs(R - sm['R_pct']) < 1e-9 and lo == sm['lower'] and hi == sm['higher'] and v == sm['verdict'],
           f'ETT {c["kind"]} {met}: R {R:+.3f} %, lower {lo} / higher {hi}, {v} = summary')
    er = sum(sum(float(A[(d, p, s)]['syn_energy_uJ']) for s in seeds) / sum(float(B[(d, p, s)]['syn_energy_uJ']) for s in seeds) for d, p in cells) / 4
    ok(abs(er - c['energy_ratio']) < 1e-9, f'ETT {c["kind"]} energy ratio {er:.3f} = summary')
oe = json.load(open(os.path.join(H, 'order_sfde1n.json')))
ok(all(math.isfinite(x['mixer']['d_mse_pct']) and abs(x['mixer']['d_mse_pct']) < 0.05 for x in oe['runs']),
   f'ETT PE-off SSA shuffle ~0: max |change| {max(abs(x["mixer"]["d_mse_pct"]) for x in oe["runs"]):.4f} %')
ok(all(abs(x['mse'] - float(E['SF-de PE off (new)'][(x['ds'], x['pl'], x['seed'])]['mse'])) <= 1e-5 * x['mse'] for x in oe['runs']), 'ETT order eval: unshuffled MSE = logged')
print('ALL CHECKS PASSED' if bad == 0 else f'{bad} FAIL')
