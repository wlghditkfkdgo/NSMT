"""Independent recheck of summary_adiand.json and summary_tshufi.json (2026-11-28, exp/f-lif-pop-v3; written before training). No import
of the summarizers and no shared code: reads raw.txt files directly and recomputes every R / pairs / verdict / ΔAUC-PR / energy ratio /
adoption-bar reading and the recommendation (AD), and T1-T6 with the order shares (MIT-BIH). Prints PASS / FAIL and ALL CHECKS PASSED.
usage (from cinc_mixer_v1/): python -B results/check-iand-20261128/recheck_iand.py [--new_ad <dir>] [--new_mb <dir>] [--repro <dir>] [--dir <summary dir>]
"""
import argparse
import json
import os

C = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RS = os.path.join(C, 'results')
ap = argparse.ArgumentParser()
ap.add_argument('--new_ad', default=os.path.join(RS, 'flif-v1-ad-iand'))
ap.add_argument('--new_mb', default=os.path.join(RS, 'flif-v1-mitbih-tshuf-iand'))
ap.add_argument('--repro', default=os.path.join(RS, 'check-adiand-repro'))
ap.add_argument('--dir', default=os.path.dirname(os.path.abspath(__file__)))
a = ap.parse_args()
SAD = json.load(open(os.path.join(a.dir, 'summary_adiand.json')))
SMB = json.load(open(os.path.join(a.dir, 'summary_tshufi.json')))
seeds = ['42', '7', '2026']
bad = 0


def ok(c, m):
    global bad
    bad += not c
    print(('PASS ' if c else 'FAIL ') + m)


def res(path):
    return [dict(t.split('=', 1) for t in l.split()[1:] if '=' in t) for l in open(os.path.join(path, 'raw.txt')) if l.startswith('RESULT')]


def avg(v):
    return sum(v) / len(v)


# ---- AD
nad = res(a.new_ad)
ok(len(nad) == 6 and len({r['sha'] for r in nad}) == 1 and {r['ds'] for r in nad} == {'MSL', 'SMAP'}, f'AD new suite: {len(nad)} rows, commits {sorted({r["sha"] for r in nad})}')
ene = res(os.path.join(RS, 'flif-v1-ad-energy'))
sfd = res(os.path.join(RS, 'flif-v1-ad-sfde'))
sfl = {'MSL': '0.002', 'SMAP': '0.002', 'PSM': '0.0005'}


def get(rr, ds, tag):
    v = {}
    for r in rr:
        if r['ds'] == ds and r['tag'] == tag and r['holdout'] == '0':
            assert r['seed'] not in v, (ds, tag, r['seed'])
            v[r['seed']] = r
    assert sorted(v) == sorted(seeds), (ds, tag, sorted(v))
    return v


X = {'IAND-all (new MSL / SMAP)': {'MSL': get(nad, 'MSL', 'spkmx_lif_muxlin_iand_lr0.002_L1'), 'SMAP': get(nad, 'SMAP', 'spkmx_lif_muxlin_iand_lr0.002_L1'),
                                   'PSM': get(ene, 'PSM', 'spkmx_lif_muxlin_iand_lr0.002_L1')},
     'G5 (MSL / SMAP OR, PSM IAND)': {'MSL': get(ene, 'MSL', 'spkmx_lif_muxlin_or_lr0.002_L1'), 'SMAP': get(ene, 'SMAP', 'spkmx_lif_muxlin_or_lr0.002_L1'),
                                      'PSM': get(ene, 'PSM', 'spkmx_lif_muxlin_iand_lr0.002_L1')},
     'SF-de(tuned)': {ds: get(sfd, ds, f'spk_ssa_lr{sfl[ds]}_L2_nalif') for ds in ('MSL', 'SMAP', 'PSM')}}
for c in SAD['comparisons']:
    A, B, dss = X[c['A']], X[c['B']], c['datasets']
    rs = [[100 * (float(A[ds][s]['aucpr']) - float(B[ds][s]['aucpr'])) / float(B[ds][s]['aucpr']) for s in seeds] for ds in dss]
    R = avg([avg(x) for x in rs]); flat = [t for x in rs for t in x]
    need = 2 * len(flat) // 3
    v = 'A better' if R >= 2 and sum(t > 0 for t in flat) >= need else ('A worse' if R <= -2 and sum(t < 0 for t in flat) >= need else 'no clear difference')
    D = avg([avg([float(A[ds][s]['aucpr']) for s in seeds]) - avg([float(B[ds][s]['aucpr']) for s in seeds]) for ds in dss])
    E = avg([sum(float(A[ds][s]['syn_energy_uJ']) for s in seeds) / sum(float(B[ds][s]['syn_energy_uJ']) for s in seeds) for ds in dss])
    ET = avg([sum(float(A[ds][s]['total_fft_uJ']) for s in seeds) / sum(float(B[ds][s]['total_fft_uJ']) for s in seeds) for ds in dss])
    ok(abs(R - c['R']) < 1e-9 and sum(t > 0 for t in flat) == c['hi'] and v == c['verdict'] and abs(D - c['d_aucpr']) < 1e-12
       and abs(E - c['energy_syn']) < 1e-9 and abs(ET - c['energy_total']) < 1e-9 and (D >= 0.001 and E <= 1.5) == c['adoption_bar'],
       f'AD {c["kind"]} {c["A"]} vs {c["B"]}: R {R:+.3f} %, {sum(t > 0 for t in flat)}/{len(flat)}, {v}, ΔAUC-PR {D:+.4f}, energy {E:.4f} ({ET:.4f}) = summary')
cj = {c['kind']: c for c in SAD['comparisons']}
rec = 'IAND for AD' if cj['J2']['verdict'] != 'A worse' and cj['J1']['adoption_bar'] else 'keep G5'
ok(rec == SAD['recommendation'], f'AD recommendation {rec} = summary')
rp = res(a.repro)
old = [r for r in ene if r['ds'] == 'PSM' and r['tag'] == 'spkmx_lif_muxlin_iand_lr0.002_L1' and r['holdout'] == '0' and r['seed'] == '42']
same = len(rp) == 1 and len(old) == 1 and all(rp[0][k] == old[0][k] for k in ('val_mse_best', 'aucpr', 'syn_energy_uJ', 'best_epoch', 'epochs_run'))
ok(same == SAD['reproduction']['ok'], f'AD reproduction reading ({"same" if same else "not the same"}) = summary (report)')

# ---- MIT-BIH


def one(path, tag):
    v = {}
    for r in res(path):
        if r.get('tag') == tag:
            assert r['seed'] not in v
            v[r['seed']] = float(r['f1']), float(r['syn_energy_uJ'])
    assert sorted(v) == sorted(seeds), (path, tag)
    return v


nmb = res(a.new_mb)
ok(len(nmb) == 3 and len({r['sha'] for r in nmb}) == 1, f'MIT-BIH new suite: {len(nmb)} rows, commits {sorted({r["sha"] for r in nmb})}')
M = {'SpkMixer IAND order-shuffled (new)': one(a.new_mb, 'spkmx_lif_muxlin_iand_lr0.01_L2_tshuf'),
     'SpkMixer IAND base': one(os.path.join(RS, 'flif-v1-mitbih-res'), 'spkmx_lif_muxlin_iand_lr0.01_L2'),
     'SpkMixer ADD base': one(os.path.join(RS, 'flif-v1-mitbih-spkmux-tune'), 'spkmx_lif_muxlin_add_lr0.02_L2'),
     'SpkMixer ADD order-shuffled': one(os.path.join(RS, 'flif-v1-mitbih-tshuf'), 'spkmx_lif_muxlin_add_lr0.02_L2_tshuf'),
     'SpkMixer ADD memoryless': one(os.path.join(RS, 'flif-v1-mitbih-memoff'), 'spkmx_lif_muxlin_add_lr0.02_L2_tmoff'),
     'SF-de PE off': one(os.path.join(RS, 'flif-v1-mitbih-sfde-nope'), 'spk_ssa_lr0.01_L2_nalif_pe0'),
     'SF-de PE on': one(os.path.join(RS, 'flif-v1-mitbih-sfde'), 'spk_ssa_lr0.01_L2_nalif')}


def vd(d):
    m = avg(d)
    return m, 'A better' if m >= 0.5 and sum(t > 0 for t in d) >= 2 else ('A worse' if m <= -0.5 and sum(t < 0 for t in d) >= 2 else 'no clear difference')


osI = [100 * (M['SpkMixer IAND base'][s][0] - M['SpkMixer IAND order-shuffled (new)'][s][0]) for s in seeds]
osA = [100 * (M['SpkMixer ADD base'][s][0] - M['SpkMixer ADD order-shuffled'][s][0]) for s in seeds]
for c in SMB['comparisons']:
    if c['kind'] == 'T2':
        d = [x - y for x, y in zip(osI, osA)]
        er = None
    else:
        d = [100 * (M[c['A']][s][0] - M[c['B']][s][0]) for s in seeds]
        er = sum(M[c['A']][s][1] for s in seeds) / sum(M[c['B']][s][1] for s in seeds)
    m, v = vd(d)
    ok(all(abs(x - y) < 1e-9 for x, y in zip(d, c['d_pp'])) and abs(m - c['mean_pp']) < 1e-9 and v == c['verdict'] and (er is None or abs(er - c['energy_ratio']) < 1e-9),
       f'MIT-BIH {c["kind"]} {c["A"]} vs {c["B"]}: {m:+.3f} %p, {v}' + ('' if er is None else f', energy {er:.3f}') + ' = summary')
ok(all(abs(x - y) < 1e-9 for x, y in zip(osI, SMB['order_share_iand_pp'])) and all(abs(x - y) < 1e-9 for x, y in zip(osA, SMB['order_share_add_pp'])),
   f'order share IAND {avg(osI):+.3f} %p, ADD {avg(osA):+.3f} %p = summary')
ok(all(abs(x['default'] - 100 * M['SpkMixer IAND order-shuffled (new)'][x['seed']][0]) < 1e-4 for x in SMB['spread']), 'spread: default reading = logged')
print('ALL CHECKS PASSED' if bad == 0 else f'{bad} FAIL')
