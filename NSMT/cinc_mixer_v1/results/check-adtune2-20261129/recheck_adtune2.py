"""Independent recheck of the AD tuning (2026-11-29, exp/f-lif-pop-v3; written before training). No import of adtune.py or the
summarizer: re-implements the tag names, walks the seven stages from the start setting with its own selection rule (highest seed-mean
test AUC-PR, ties keep the current value), compares every selection with selected_<stage>.json, and recomputes the final arms and the
comparisons of summary_adtune2.json. Prints PASS / FAIL and ALL CHECKS PASSED.
usage (from cinc_mixer_v1/): python -B results/check-adtune2-20261129/recheck_adtune2.py   (ADT_ROOT / ADT_OUT as the summarizer)
"""
import json
import os

H = os.path.dirname(os.path.abspath(__file__))
RT = os.environ.get('ADT_ROOT') or os.path.join(os.path.dirname(os.path.dirname(H)), 'results')
OUT = os.environ.get('ADT_OUT') or H
SU = os.path.join(RT, 'flif-v1-ad-tune2')
seeds = ['42', '7', '2026']
bad = 0


def ok(c, m):
    global bad
    bad += not c
    print(('PASS ' if c else 'FAIL ') + m)


rows = {}
for su in ('flif-v1-ad-energy', 'flif-v1-ad-iand', 'flif-v1-ad-tune2'):
    for l in open(os.path.join(RT, su, 'raw.txt')):
        if l.startswith('SKIP') and su == 'flif-v1-ad-tune2':
            ok(False, 'SKIP line in the tuning suite')
        if not l.startswith('RESULT'):
            continue
        r = dict(t.split('=', 1) for t in l.split()[1:] if '=' in t)
        if r['holdout'] != '0' or not r['cond'].startswith('spkmx_lif_muxlin_'):
            continue
        k = (r['ds'], r['tag'], r['seed'])
        assert k not in rows, k
        rows[k] = r


def name(c):
    s = 'spkmx_lif_muxlin_%s_lr%s_L%d' % (c['res'], c['lr'], c['ml'])
    for key, dflt, pre in (('bl', 1, '_b'), ('emb', 64, '_e'), ('pt', 8, '_p'), ('mlp', '1', '_mlp'), ('norm', 'window', '_nm'), ('ep', 10, '_ep')):
        if c[key] != dflt:
            s += pre + str(c[key])
    return s


def score(ds, c):
    v = [float(rows[(ds, name(c), s)]['aucpr']) for s in seeds]
    return sum(v) / 3


stages = [('norm', ['norm'], [['window'], ['mean'], ['none']]), ('epochs', ['ep'], [[1], [3], [10], [30]]), ('width', ['emb'], [[32], [64], [128]]),
          ('depth', ['bl', 'ml'], [[1, 1], [1, 2], [2, 1]]), ('patch', ['pt', 'mlp'], [[8, '1'], [4, '1'], [20, '1'], [8, '2']]),
          ('lr', ['lr'], [['0.001'], ['0.002'], ['0.004']]), ('res', ['res'], [['or'], ['iand'], ['add']])]
cfg = {ds: dict(res={'MSL': 'or', 'SMAP': 'or', 'PSM': 'iand'}[ds], lr='0.002', ml=1, bl=1, emb=64, pt=8, mlp='1', norm='window', ep=10) for ds in ('MSL', 'SMAP', 'PSM')}
start = {ds: dict(c) for ds, c in cfg.items()}
for st, keys, vals in stages:
    js = json.load(open(os.path.join(SU, f'selected_{st}.json')))['selected']
    mine = {}
    for ds in cfg:
        cur = [cfg[ds][k] for k in keys]
        sc = []
        for v in vals:
            c = dict(cfg[ds]); c.update(zip(keys, v)); sc.append(score(ds, c))
        best = max(sc)
        pick = cur if best - sc[vals.index(cur)] < 1e-9 else vals[sc.index(best)]
        mine[ds] = pick
        cfg[ds].update(zip(keys, pick))
    ok(all(mine[ds] == js[ds] for ds in cfg), f'stage {st}: selection {mine} = selected_{st}.json')
S = json.load(open(os.path.join(OUT, 'summary_adtune2.json')))
ok(all(S['final_config'][ds]['tag'] == name(cfg[ds]) for ds in cfg), 'final tags = ' + ', '.join(name(cfg[ds]) for ds in cfg))
sfl = json.load(open(os.path.join(RT, 'flif-v1-ad-sfde', 'selected_lr.json')))['selected']
sf = {}
for l in open(os.path.join(RT, 'flif-v1-ad-sfde', 'raw.txt')):
    if l.startswith('RESULT'):
        r = dict(t.split('=', 1) for t in l.split()[1:] if '=' in t)
        if r['holdout'] == '0' and r['tag'] == f'spk_ssa_lr{sfl[r["ds"]]}_L2_nalif':
            sf[(r['ds'], r['seed'])] = r
arm = {'SpkMixer tuned (test-selected)': lambda ds, s: rows[(ds, name(cfg[ds]), s)], 'SpkMixer start (11-19 G5)': lambda ds, s: rows[(ds, name(start[ds]), s)],
       'SF-de (11-18, lr only)': lambda ds, s: sf[(ds, s)]}
for c in S['comparisons']:
    A, B, k = arm[c['A']], arm[c['B']], c['metric']
    rd, flat, dd, es = [], [], [], []
    for ds in ('MSL', 'SMAP', 'PSM'):
        r_ = [100 * (float(A(ds, s)[k]) - float(B(ds, s)[k])) / float(B(ds, s)[k]) for s in seeds]
        rd.append(sum(r_) / 3); flat += r_
        dd.append(sum(float(A(ds, s)[k]) for s in seeds) / 3 - sum(float(B(ds, s)[k]) for s in seeds) / 3)
        es.append(sum(float(A(ds, s)['syn_energy_uJ']) for s in seeds) / sum(float(B(ds, s)['syn_energy_uJ']) for s in seeds))
    R = sum(rd) / 3; hi = sum(t > 0 for t in flat); lo = sum(t < 0 for t in flat)
    v = 'A better' if R >= 2 and hi >= 6 else ('A worse' if R <= -2 and lo >= 6 else 'no clear difference')
    ok(abs(R - c['R']) < 1e-9 and hi == c['hi'] and v == c['verdict'] and abs(sum(dd) / 3 - c['d_mean']) < 1e-12 and abs(sum(es) / 3 - c['energy']) < 1e-9,
       f"{k} {c['A']} vs {c['B']}: R {R:+.3f} %, {hi}/9, {v}, energy {sum(es) / 3:.3f} = summary")
for n, f in arm.items():
    for ds in ('MSL', 'SMAP', 'PSM'):
        for k in ('pa_f1', 'raw_f1', 'aucpr'):
            m = sum(float(f(ds, s)[k]) for s in seeds) / 3
            if abs(m - S['arms'][n][ds][k]) > 1e-12:
                ok(False, f'{n} {ds} {k} mean {m} != summary')
ok(True, 'arm means (P / R / F1, AUC-PR) checked against the summary (no FAIL above = equal)')
print('ALL CHECKS PASSED' if bad == 0 else f'{bad} FAIL')
