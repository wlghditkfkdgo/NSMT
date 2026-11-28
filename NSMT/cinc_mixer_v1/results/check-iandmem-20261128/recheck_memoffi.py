"""Independent recheck of summary_memoffi.json (2026-11-28, exp/f-lif-pop-v3; written before training). No import of the summarizer:
reads raw.txt files directly and recomputes U1-U5, the decomposition and the order-free check. Prints PASS / FAIL and ALL CHECKS PASSED.
usage (from cinc_mixer_v1/): python -B results/check-iandmem-20261128/recheck_memoffi.py [--new <dir>] [--dir <summary dir>] [--order <json>]
"""
import argparse
import json
import os

C = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RS = os.path.join(C, 'results')
H = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--new', default=os.path.join(RS, 'flif-v1-mitbih-memoff-iand'))
ap.add_argument('--dir', default=H)
ap.add_argument('--order', default=os.path.join(H, 'order_iandoff.json'))
a = ap.parse_args()
S = json.load(open(os.path.join(a.dir, 'summary_memoffi.json')))
seeds = ['42', '7', '2026']
bad = 0


def ok(c, m):
    global bad
    bad += not c
    print(('PASS ' if c else 'FAIL ') + m)


def get(path, tag):
    v = {}
    for l in open(os.path.join(path, 'raw.txt')):
        if not l.startswith('RESULT'):
            continue
        r = dict(t.split('=', 1) for t in l.split()[1:] if '=' in t)
        if r.get('tag') == tag:
            assert r['seed'] not in v
            v[r['seed']] = (float(r['f1']), float(r['syn_energy_uJ']), r['sha'])
    assert sorted(v) == sorted(seeds), (path, tag)
    return v


M = {'SpkMixer IAND memoryless (new)': get(a.new, 'spkmx_lif_muxlin_iand_lr0.01_L2_tmoff'),
     'SpkMixer IAND base': get(os.path.join(RS, 'flif-v1-mitbih-res'), 'spkmx_lif_muxlin_iand_lr0.01_L2'),
     'SpkMixer IAND order-shuffled': get(os.path.join(RS, 'flif-v1-mitbih-tshuf-iand'), 'spkmx_lif_muxlin_iand_lr0.01_L2_tshuf'),
     'SpkMixer ADD memoryless': get(os.path.join(RS, 'flif-v1-mitbih-memoff'), 'spkmx_lif_muxlin_add_lr0.02_L2_tmoff'),
     'SF-de PE off': get(os.path.join(RS, 'flif-v1-mitbih-sfde-nope'), 'spk_ssa_lr0.01_L2_nalif_pe0'),
     'SF-de PE on': get(os.path.join(RS, 'flif-v1-mitbih-sfde'), 'spk_ssa_lr0.01_L2_nalif')}
new = M['SpkMixer IAND memoryless (new)']
ok(len({v[2] for v in new.values()}) == 1 and S['sha'] == new['42'][2], f'new suite: 3 seeds, one commit {new["42"][2]}')
for c in S['comparisons']:
    d = [100 * (M[c['A']][s][0] - M[c['B']][s][0]) for s in seeds]
    m = sum(d) / 3
    v = 'A better' if m >= 0.5 and sum(t > 0 for t in d) >= 2 else ('A worse' if m <= -0.5 and sum(t < 0 for t in d) >= 2 else 'no clear difference')
    er = sum(M[c['A']][s][1] for s in seeds) / sum(M[c['B']][s][1] for s in seeds)
    ok(all(abs(x - y) < 1e-9 for x, y in zip(d, c['d_pp'])) and abs(m - c['mean_pp']) < 1e-9 and v == c['verdict'] and abs(er - c['energy_ratio']) < 1e-9,
       f'{c["kind"]} {c["A"]} vs {c["B"]}: {m:+.3f} %p, {v}, energy {er:.3f} = summary')
b, sh, o = M['SpkMixer IAND base'], M['SpkMixer IAND order-shuffled'], new
tot = [100 * (b[s][0] - o[s][0]) for s in seeds]; osh = [100 * (b[s][0] - sh[s][0]) for s in seeds]; ish = [100 * (sh[s][0] - o[s][0]) for s in seeds]
ok(all(abs(x - y) < 1e-9 for x, y in zip(tot, S['memoryless_drop_pp'])) and all(abs(x - y) < 1e-9 for x, y in zip(osh, S['order_share_pp']))
   and all(abs(x - y) < 1e-9 for x, y in zip(ish, S['interaction_share_pp'])) and all(abs(t - (x + y)) < 1e-9 for t, x, y in zip(tot, osh, ish))
   and abs(sum(osh) / sum(tot) - S['order_fraction_of_mean_drop']) < 1e-9,
   f'decomposition: drop {sum(tot) / 3:+.3f} = order {sum(osh) / 3:+.3f} + interaction {sum(ish) / 3:+.3f} %p, order fraction {sum(osh) / sum(tot):.3f} = summary')
oe = json.load(open(a.order))
ok(all(abs(x['mixer']['drop_f1_pp']) < 0.05 and abs(x['full']['drop_f1_pp']) < 0.05 and abs(x['f1'] - 100 * new[x['seed']][0]) < 1e-4 for x in oe['runs'])
   and len(oe['runs']) == 3, 'order-free check: every evaluation-shuffle drop |d| < 0.05 %p and unshuffled reading = logged')
print('ALL CHECKS PASSED' if bad == 0 else f'{bad} FAIL')
