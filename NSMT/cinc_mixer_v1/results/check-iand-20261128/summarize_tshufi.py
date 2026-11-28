"""MIT-BIH order-only control on the IAND base (2026-11-28, exp/f-lif-pop-v3; PROJECT_LOG pre-registration; user "동시에 3번 순서 대조
다시하기" after the 21:28 decision that the MIT-BIH base is IAND lr 0.01; rules fixed before training).
  shufI  LIF SpkMixer muxlin IAND lr 0.01, 2 mixer layers, --token_shuffle (random token order per sample in training and evaluation)
         suite flif-v1-mitbih-tshuf-iand, tag spkmx_lif_muxlin_iand_lr0.01_L2_tshuf, seeds 42 / 7 / 2026 (3 new runs)
References (paired seeds; logged rows): IAND base (flif-v1-mitbih-res, tag spkmx_lif_muxlin_iand_lr0.01_L2, whole training file),
ADD base (flif-v1-mitbih-spkmux-tune spkmx_lif_muxlin_add_lr0.02_L2), ADD shuffled (flif-v1-mitbih-tshuf), ADD memoryless
(flif-v1-mitbih-memoff ..._tmoff), SF-de PE off (flif-v1-mitbih-sfde-nope), SF-de PE on (flif-v1-mitbih-sfde).
Comparisons (test macro-F1; d = 100 (F1_A - F1_B) per seed in %p; 'A better' if mean >= +0.5 and >= 2 of 3 > 0, 'A worse' mirror, else
'no clear difference'; energy = logged syops syn_energy_uJ, ratio of seed means):
  T1  shufI vs IAND base             'A worse' = token order contributes on the IAND base; order share (IAND) = base - shufI
  T2  order share IAND vs order share ADD, d = (IAND base - shufI) - (ADD base - ADD shuffled): 'A better' = IAND uses order more
  T3  shufI vs SF-de PE off (report)  T4  shufI vs SF-de PE on (report)
  T5  shufI vs ADD shuffled (report: residual under shuffling)  T6  shufI vs ADD memoryless (report; cross-residual, NOT the IAND
      interaction share: an IAND memoryless model was not run)
Reports: tshuf_spread_iand.json (other evaluation orders, natural order) and order_iand.json (evaluation-time token shuffle of the IAND
base, no training; drop next to the ADD base's order_mux.json).
Fails closed (exit 2, nothing written): SKIP line or a row of another tag, not 3 seeds, mixed commits, non-finite values, a missing
energy recount row or recomputed macro-F1 != logged, a report json missing or its unshuffled / default reading != logged.
usage (from cinc_mixer_v1/): python -B results/check-iand-20261128/summarize_tshufi.py [--new <suite dir>] [--spread <json>] [--order <json>] [--out_dir ...]
"""
import argparse
import csv
import json
import math
import os
import sys

C = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
RS = os.path.join(C, 'results')
ap = argparse.ArgumentParser()
ap.add_argument('--new', default=os.path.join(RS, 'flif-v1-mitbih-tshuf-iand'))
ap.add_argument('--spread', default=os.path.join(HERE, 'tshuf_spread_iand.json'))
ap.add_argument('--order', default=os.path.join(HERE, 'order_iand.json'))
ap.add_argument('--out_dir', default=HERE)
a = ap.parse_args()
SEEDS = ('42', '7', '2026')
KEYS = ('f1', 'overall_acc', 'f1_0', 'f1_1', 'f1_2', 'f1_3', 'f1_4', 'syn_energy_uJ')
TAG = 'spkmx_lif_muxlin_iand_lr0.01_L2_tshuf'
mean = lambda x: sum(x) / len(x)


def stop(m):
    print(f'STOP: {m}')
    sys.exit(2)


def fin(r, k):
    try:
        v = float(r[k])
    except (KeyError, ValueError, TypeError):
        stop(f'{k} missing ({r.get("tag")} seed {r.get("seed")})')
    if not math.isfinite(v):
        stop(f'{k} not finite ({r.get("tag")} seed {r.get("seed")})')
    return v


def load(path, tag, new=False):
    p = os.path.join(path, 'raw.txt')
    if not os.path.exists(p):
        stop(f'{p} missing')
    L = open(p).read().splitlines()
    rr = [dict(t.split('=', 1) for t in l.split()[1:] if '=' in t) for l in L if l.startswith('RESULT')]
    if new:
        if any(l.startswith('SKIP') for l in L):
            stop(f'{path}: SKIP line')
        if any(r.get('tag') != tag for r in rr):
            stop(f'{path}: unexpected rows')
        if len({r['sha'] for r in rr}) != 1:
            stop(f'{path}: mixed training commits {sorted({r["sha"] for r in rr})}')
        ep = os.path.join(path, 'energy_recount.csv')
        if not os.path.exists(ep):
            stop(f'{path}: energy_recount.csv missing')
        en = {(e['tag'], e['seed']): e for e in csv.DictReader(open(ep))}
        for r in rr:
            e = en.get((r['tag'], r['seed']))
            if e is None or abs(fin(e, 'f1_recomputed') - fin(r, 'f1')) > 1e-6:
                stop(f'{path}: energy recount missing or recomputed macro-F1 != logged (seed {r["seed"]})')
    d = {}
    for r in rr:
        if r.get('tag') != tag:
            continue
        if r['seed'] in d:
            stop(f'duplicate {path} {tag} seed {r["seed"]}')
        d[r['seed']] = {k: fin(r, k) for k in KEYS}
    if sorted(d) != sorted(SEEDS):
        stop(f'{path} {tag}: seeds {sorted(d)}')
    return d, (rr[0]['sha'] if new else None)


SH, sha = load(a.new, TAG, new=True)
N = 'SpkMixer IAND order-shuffled (new)'
A = {N: SH, 'SpkMixer IAND base': load(os.path.join(RS, 'flif-v1-mitbih-res'), 'spkmx_lif_muxlin_iand_lr0.01_L2')[0],
     'SpkMixer ADD base': load(os.path.join(RS, 'flif-v1-mitbih-spkmux-tune'), 'spkmx_lif_muxlin_add_lr0.02_L2')[0],
     'SpkMixer ADD order-shuffled': load(os.path.join(RS, 'flif-v1-mitbih-tshuf'), 'spkmx_lif_muxlin_add_lr0.02_L2_tshuf')[0],
     'SpkMixer ADD memoryless': load(os.path.join(RS, 'flif-v1-mitbih-memoff'), 'spkmx_lif_muxlin_add_lr0.02_L2_tmoff')[0],
     'SF-de PE off': load(os.path.join(RS, 'flif-v1-mitbih-sfde-nope'), 'spk_ssa_lr0.01_L2_nalif_pe0')[0],
     'SF-de PE on': load(os.path.join(RS, 'flif-v1-mitbih-sfde'), 'spk_ssa_lr0.01_L2_nalif')[0]}
for p in (a.spread, a.order):
    if not os.path.exists(p):
        stop(f'{p} missing')
SP, OE = json.load(open(a.spread)), json.load(open(a.order))
if SP.get('tag') != TAG or sorted(x['seed'] for x in SP['runs']) != sorted(SEEDS) or \
        any(abs(x['default'] - 100. * SH[x['seed']]['f1']) > 1e-4 for x in SP['runs']):
    stop('spread json: wrong runs or default reading != logged')
if OE.get('tag') != 'spkmx_lif_muxlin_iand_lr0.01_L2' or sorted(x['seed'] for x in OE['runs']) != sorted(SEEDS) or \
        any(abs(x['f1'] - 100. * A['SpkMixer IAND base'][x['seed']]['f1']) > 1e-4 for x in OE['runs']):
    stop('order json: wrong runs or unshuffled reading != logged')
OM = json.load(open(os.path.join(RS, 'check-mitorder-20261128', 'order_mux.json')))


def verdict(d):
    m = mean(d)
    return m, ('A better' if m >= 0.5 and sum(t > 0 for t in d) >= 2 else ('A worse' if m <= -0.5 and sum(t < 0 for t in d) >= 2 else 'no clear difference'))


def comp(x, y):
    d = [100. * (A[x][s]['f1'] - A[y][s]['f1']) for s in SEEDS]
    m, v = verdict(d)
    return dict(A=x, B=y, d_pp=d, mean_pp=m, verdict=v, energy_ratio=mean([A[x][s]['syn_energy_uJ'] for s in SEEDS]) / mean([A[y][s]['syn_energy_uJ'] for s in SEEDS]))


comps = [dict(kind='T1', **comp(N, 'SpkMixer IAND base'))]
osI = [100. * (A['SpkMixer IAND base'][s]['f1'] - SH[s]['f1']) for s in SEEDS]
osA = [100. * (A['SpkMixer ADD base'][s]['f1'] - A['SpkMixer ADD order-shuffled'][s]['f1']) for s in SEEDS]
d2 = [x - y for x, y in zip(osI, osA)]
m2, v2 = verdict(d2)
comps.append(dict(kind='T2', A='order share IAND', B='order share ADD', d_pp=d2, mean_pp=m2, verdict=v2, energy_ratio=float('nan')))
for k, y in (('T3', 'SF-de PE off'), ('T4', 'SF-de PE on'), ('T5', 'SpkMixer ADD order-shuffled'), ('T6', 'SpkMixer ADD memoryless')):
    comps.append(dict(kind=k + ' (report)', **comp(N, y)))
ev = {x['seed']: x for x in OE['runs']}
evA = {x['seed']: x for x in OM['runs']}
evrow = dict(iand_mixer_drop=[ev[s]['mixer']['drop_f1_pp'] for s in SEEDS], iand_full_drop=[ev[s]['full']['drop_f1_pp'] for s in SEEDS],
             add_mixer_drop=[evA[s]['mixer']['drop_f1_pp'] for s in SEEDS], add_full_drop=[evA[s]['full']['drop_f1_pp'] for s in SEEDS])
out = dict(sha=sha, arms={n: {k: mean([v[s][k] for s in SEEDS]) for k in KEYS} for n, v in A.items()}, comparisons=comps,
           order_share_iand_pp=osI, order_share_add_pp=osA, spread=SP['runs'], eval_shuffle=evrow)
os.makedirs(a.out_dir, exist_ok=True)
json.dump(out, open(os.path.join(a.out_dir, 'summary_tshufi.json'), 'w'), indent=1)
L = ['# MIT-BIH order-only control on the IAND base: SpkMixer IAND trained and tested with a random token order per sample', '',
     f'Rules: summarize_tshufi.py docstring. 3 new runs, commit `{sha}`.', '',
     '| arm | macro-F1 % | accuracy % | F1 N / S / V / F / Q % | energy µJ (syops) |', '|---|--:|--:|---|--:|']
for n, m in out['arms'].items():
    L.append(f'| {n} | {100 * m["f1"]:.2f} | {100 * m["overall_acc"]:.2f} | ' + ' / '.join(f'{100 * m[f"f1_{i}"]:.2f}' for i in range(5)) + f' | {m["syn_energy_uJ"]:.2f} |')
L += ['', '| kind | A | B | d %p (42 / 7 / 2026) | mean %p | verdict | energy A/B |', '|---|---|---|---|--:|---|--:|']
for c in comps:
    L.append(f'| {c["kind"]} | {c["A"]} | {c["B"]} | ' + ' / '.join(f'{t:+.2f}' for t in c['d_pp']) + f' | {c["mean_pp"]:+.2f} | **{c["verdict"]}** | '
             + ('—' if math.isnan(c['energy_ratio']) else f'{c["energy_ratio"]:.3f}') + ' |')
L += ['', 'Order share (macro-F1 %p, seed 42 / 7 / 2026): IAND base - IAND shuffled ' + ' / '.join(f'{t:+.2f}' for t in osI) + f' (mean {mean(osI):+.2f}); '
      'ADD base - ADD shuffled ' + ' / '.join(f'{t:+.2f}' for t in osA) + f' (mean {mean(osA):+.2f})', '',
      'Evaluation-time token shuffle of the trained base (no training; drop in macro-F1 %p, seed 42 / 7 / 2026, mean of 3 permutation seeds): '
      'IAND mixer input ' + ' / '.join(f'{t:.2f}' for t in evrow['iand_mixer_drop']) + f' (mean {mean(evrow["iand_mixer_drop"]):.2f}), patch tokens '
      + ' / '.join(f'{t:.2f}' for t in evrow['iand_full_drop']) + f' (mean {mean(evrow["iand_full_drop"]):.2f}); ADD mixer input '
      + ' / '.join(f'{t:.2f}' for t in evrow['add_mixer_drop']) + f' (mean {mean(evrow["add_mixer_drop"]):.2f})', '',
      'Evaluation-order spread of the IAND shuffled model (report): ' + '; '.join(
          f'seed {x["seed"]}: default {x["default"]:.2f}, eval seed 1 {x["seed1"]:.2f}, seed 2 {x["seed2"]:.2f}, natural order {x["natural"]:.2f} '
          f'(spread {x["spread"]:.2f} %p)' for x in SP['runs'])]
open(os.path.join(a.out_dir, 'summary_tshufi.md'), 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
