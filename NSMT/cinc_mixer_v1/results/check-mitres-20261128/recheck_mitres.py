"""Independent recheck of summary_mitres.json and summary_tshuf.json (2026-11-28, exp/f-lif-pop-v3; written while the pool ran, run after
"driver end"). No repo import and no code shared with the summarizers: reads raw.txt files directly and recomputes the lr selection, every
d / mean / verdict / energy ratio / adoption-bar reading, the candidates and the recommendation (residual test), and S1-S4 with the order
and interaction shares (order-only control). Prints PASS / FAIL and ALL CHECKS PASSED.
usage (from cinc_mixer_v1/): python -B results/check-mitres-20261128/recheck_mitres.py
"""
import json
import os

C = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = json.load(open(os.path.join(C, 'results', 'check-mitres-20261128', 'summary_mitres.json')))
T = json.load(open(os.path.join(C, 'results', 'check-tshuf-20261128', 'summary_tshuf.json')))
seeds = ['42', '7', '2026']
bad = 0


def ok(c, m):
    global bad
    bad += not c
    print(('PASS ' if c else 'FAIL ') + m)


def res(suite):
    return [dict(t.split('=', 1) for t in l.split()[1:] if '=' in t) for l in open(os.path.join(C, 'results', suite, 'raw.txt')) if l.startswith('RESULT')]


def by_seed(rr):
    d = {}
    for r in rr:
        d.setdefault(r['seed'], []).append(r)
    return {s: v[0] for s, v in d.items() if len(v) == 1}


rows = res('flif-v1-mitbih-res')
ok(len(rows) == 30 and len({r['sha'] for r in rows}) == 1, f'res suite: {len(rows)} rows, commits {sorted({r["sha"] for r in rows})}')
sel = {}
for k, cond in (('IAND', 'spkmx_lif_muxlin_iand'), ('MS', 'spkmx_lif_muxlin_ms')):
    sc = {}
    for lr in ('0.0025', '0.005', '0.01', '0.02'):
        v = [float(r['val_f1_last']) * 100 for r in rows if r['cond'] == cond and r['lr'] == lr and r['val_fold'] == '0']
        sc[lr] = sum(v) / len(v) if len(v) == 3 else float('nan')
    best = max(sc.values())
    w = [lr for lr in sc if best - sc[lr] < 1e-9]
    sel[k] = '0.02' if '0.02' in w else w[0]
    ok(all(abs(sc[lr] - R['val_scores'][k][lr]) < 1e-9 for lr in sc) and sel[k] == R['selected'][k],
       f'{k}: validation scores ' + ', '.join(f'{lr} {sc[lr]:.3f}' for lr in sc) + f' -> lr {sel[k]} = summary')
A = {'IAND (new)': by_seed([r for r in rows if r['cond'] == 'spkmx_lif_muxlin_iand' and r['val_fold'] == '-1' and r['lr'] == sel['IAND']]),
     'MS (new)': by_seed([r for r in rows if r['cond'] == 'spkmx_lif_muxlin_ms' and r['val_fold'] == '-1' and r['lr'] == sel['MS']]),
     'ADD base': by_seed([r for r in res('flif-v1-mitbih-spkmux-tune') if r.get('tag') == 'spkmx_lif_muxlin_add_lr0.02_L2']),
     'SF-de PE on': by_seed([r for r in res('flif-v1-mitbih-sfde') if r.get('tag') == 'spk_ssa_lr0.01_L2_nalif']),
     'SF-de PE off': by_seed([r for r in res('flif-v1-mitbih-sfde-nope') if r.get('tag') == 'spk_ssa_lr0.01_L2_nalif_pe0']),
     'SpkMixer base': by_seed([r for r in res('flif-v1-mitbih-spkmux-tune') if r.get('tag') == 'spkmx_lif_muxlin_add_lr0.02_L2']),
     'SpkMixer memoryless': by_seed([r for r in res('flif-v1-mitbih-memoff') if r.get('tag') == 'spkmx_lif_muxlin_add_lr0.02_L2_tmoff']),
     'SpkMixer order-shuffled (new)': by_seed([r for r in res('flif-v1-mitbih-tshuf') if r.get('tag') == 'spkmx_lif_muxlin_add_lr0.02_L2_tshuf'])}
for k, v in A.items():
    ok(sorted(v) == sorted(seeds), f'{k}: three seeds, one row each')


def comp(x, y):
    d = [100 * (float(A[x][s]['f1']) - float(A[y][s]['f1'])) for s in seeds]
    m = sum(d) / 3
    v = 'A better' if m >= 0.5 and sum(t > 0 for t in d) >= 2 else ('A worse' if m <= -0.5 and sum(t < 0 for t in d) >= 2 else 'no clear difference')
    er = sum(float(A[x][s]['syn_energy_uJ']) for s in seeds) / sum(float(A[y][s]['syn_energy_uJ']) for s in seeds)
    return d, m, v, er


for c in R['comparisons']:
    d, m, v, er = comp(c['A'], c['B'])
    ok(all(abs(a - b) < 1e-9 for a, b in zip(d, c['d_pp'])) and abs(m - c['mean_pp']) < 1e-9 and v == c['verdict'] and abs(er - c['energy_ratio']) < 1e-9
       and (m >= 0.1 and er <= 1.5) == c['adoption_bar'], f'residual {c["kind"]} {c["A"]} vs {c["B"]}: {m:+.3f} %p, {v}, energy {er:.3f} = summary')
cd = {c['kind']: comp(c['A'], c['B']) for c in R['comparisons']}
cand = [k for k, rk, sk in (('IAND', 'R1', 'R3'), ('MS', 'R2', 'R4')) if cd[rk][2] != 'A worse' and cd[sk][3] <= 1.5]
rec = 'IAND' if 'IAND' in cand else ('MS' if 'MS' in cand else 'none (keep ADD)')
ok(cand == R['candidates'] and rec == R['recommendation'], f'candidates {cand}, recommendation {rec} = summary')
for c in T['comparisons']:
    d, m, v, er = comp(c['A'], c['B'])
    ok(all(abs(a - b) < 1e-9 for a, b in zip(d, c['d_pp'])) and abs(m - c['mean_pp']) < 1e-9 and v == c['verdict'] and abs(er - c['energy_ratio']) < 1e-9,
       f'order control {c["kind"]} vs {c["B"]}: {m:+.3f} %p, {v}, energy {er:.3f} = summary')
osh = [100 * (float(A['SpkMixer base'][s]['f1']) - float(A['SpkMixer order-shuffled (new)'][s]['f1'])) for s in seeds]
ish = [100 * (float(A['SpkMixer order-shuffled (new)'][s]['f1']) - float(A['SpkMixer memoryless'][s]['f1'])) for s in seeds]
ok(all(abs(a - b) < 1e-9 for a, b in zip(osh, T['order_share_pp'])) and all(abs(a - b) < 1e-9 for a, b in zip(ish, T['interaction_share_pp'])),
   f'order share {sum(osh) / 3:+.3f} %p, interaction share {sum(ish) / 3:+.3f} %p = summary')
ok(all(abs(x['default'] - 100 * float(A['SpkMixer order-shuffled (new)'][x['seed']]['f1'])) < 1e-4 for x in T['spread']), 'spread: default reading = logged')
print('ALL CHECKS PASSED' if bad == 0 else f'{bad} FAIL')
