"""Summary of the MIT-BIH order-only control (2026-11-28, exp/f-lif-pop-v3; PROJECT_LOG pre-registration; user "1과 2를 진행해줘", item 2
"SpkMixer 안에서 순서의 몫만 떼는 대조"; rules fixed before training).
  shuf  LIF SpkMixer muxlin ADD lr 0.02, 2 mixer layers, --token_shuffle (random token order per sample in training and evaluation: the
        token-axis neurons still pass information from token to token, but the order is noise; the token-mean head is order-blind)
        suite flif-v1-mitbih-tshuf, tag spkmx_lif_muxlin_add_lr0.02_L2_tshuf, seeds 42 / 7 / 2026 (3 new runs)
References (paired seeds): base = the same model without the shuffle (flif-v1-mitbih-spkmux-tune spkmx_lif_muxlin_add_lr0.02_L2: order +
token interaction), memoryless (flif-v1-mitbih-memoff ..._tmoff: neither), SF-de PE off (flif-v1-mitbih-sfde-nope: attention without
order), SF-de PE on (report).
Comparisons (A vs B, test macro-F1): S1 shuf vs base ('A worse' = token order contributes: order share = base - shuf), S2 shuf vs memoryless
('A better' = token interaction without order contributes: interaction share = shuf - memoryless), S3 shuf vs SF-de PE off (report),
S4 shuf vs SF-de PE on (report). d = 100 (F1_A - F1_B) per seed (%p); 'A better' if mean >= +0.5 and >= 2 of 3 > 0, 'A worse' mirror,
else 'no clear difference'. Energy = logged syops, ratio of seed means.
Report: tshuf_spread.json (other evaluation orders, natural order).
Fails closed (exit 2, nothing written): SKIP line or a row of another tag, not 3 seeds, mixed commits, non-finite values, a missing energy
recount row or recomputed macro-F1 != logged, tshuf_spread.json missing or its default reading != logged.
usage (from cinc_mixer_v1/): python -B results/check-tshuf-20261128/summarize_tshuf.py [--root results] [--spread <json>] [--out_dir ...]
"""
import argparse
import csv
import json
import math
import os
import sys

C = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--root', default=os.path.join(C, 'results'))
ap.add_argument('--spread', default=os.path.join(HERE, 'tshuf_spread.json'))
ap.add_argument('--out_dir', default=HERE)
a = ap.parse_args()
SEEDS = ('42', '7', '2026')
KEYS = ('f1', 'overall_acc', 'f1_0', 'f1_1', 'f1_2', 'f1_3', 'f1_4', 'syn_energy_uJ')
TAG = 'spkmx_lif_muxlin_add_lr0.02_L2_tshuf'
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


def load(suite, tag, new=False):
    p = os.path.join(a.root, suite, 'raw.txt')
    if not os.path.exists(p):
        stop(f'{p} missing')
    L = open(p).read().splitlines()
    rr = [dict(t.split('=', 1) for t in l.split()[1:] if '=' in t) for l in L if l.startswith('RESULT')]
    if new:
        if any(l.startswith('SKIP') for l in L):
            stop(f'{suite}: SKIP line')
        if any(r.get('tag') != tag for r in rr):
            stop(f'{suite}: unexpected rows')
        if len({r['sha'] for r in rr}) != 1:
            stop(f'{suite}: mixed training commits {sorted({r["sha"] for r in rr})}')
        en = {(e['tag'], e['seed']): e for e in csv.DictReader(open(os.path.join(a.root, suite, 'energy_recount.csv')))} \
            if os.path.exists(os.path.join(a.root, suite, 'energy_recount.csv')) else stop(f'{suite}: energy_recount.csv missing')
        for r in rr:
            e = en.get((r['tag'], r['seed']))
            if e is None or abs(fin(e, 'f1_recomputed') - fin(r, 'f1')) > 1e-6:
                stop(f'{suite}: energy recount missing or recomputed macro-F1 != logged (seed {r["seed"]})')
    d = {}
    for r in rr:
        if r.get('tag') != tag:
            continue
        if r['seed'] in d:
            stop(f'duplicate {suite} {tag} seed {r["seed"]}')
        d[r['seed']] = {k: fin(r, k) for k in KEYS}
    if sorted(d) != sorted(SEEDS):
        stop(f'{suite} {tag}: seeds {sorted(d)}')
    return d, (rr[0]['sha'] if new else None)


SH, sha = load('flif-v1-mitbih-tshuf', TAG, new=True)
A = {'SpkMixer order-shuffled (new)': SH, 'SpkMixer base': load('flif-v1-mitbih-spkmux-tune', 'spkmx_lif_muxlin_add_lr0.02_L2')[0],
     'SpkMixer memoryless': load('flif-v1-mitbih-memoff', 'spkmx_lif_muxlin_add_lr0.02_L2_tmoff')[0],
     'SF-de PE off': load('flif-v1-mitbih-sfde-nope', 'spk_ssa_lr0.01_L2_nalif_pe0')[0], 'SF-de PE on': load('flif-v1-mitbih-sfde', 'spk_ssa_lr0.01_L2_nalif')[0]}
if not os.path.exists(a.spread):
    stop(f'{a.spread} missing')
SP = json.load(open(a.spread))
if SP.get('tag') != TAG or sorted(x['seed'] for x in SP['runs']) != sorted(SEEDS) or \
        any(abs(x['default'] - 100. * SH[x['seed']]['f1']) > 1e-4 for x in SP['runs']):
    stop('tshuf_spread.json: wrong runs or default reading != logged')


def comp(x, y):
    d = [100. * (A[x][s]['f1'] - A[y][s]['f1']) for s in SEEDS]
    m = mean(d)
    v = 'A better' if m >= 0.5 and sum(t > 0 for t in d) >= 2 else ('A worse' if m <= -0.5 and sum(t < 0 for t in d) >= 2 else 'no clear difference')
    return dict(A=x, B=y, d_pp=d, mean_pp=m, verdict=v, energy_ratio=mean([A[x][s]['syn_energy_uJ'] for s in SEEDS]) / mean([A[y][s]['syn_energy_uJ'] for s in SEEDS]))


N = 'SpkMixer order-shuffled (new)'
CS = [('S1', N, 'SpkMixer base'), ('S2', N, 'SpkMixer memoryless'), ('S3', N, 'SF-de PE off'), ('S4', N, 'SF-de PE on')]
comps = [dict(kind=k, **comp(x, y)) for k, x, y in CS]
order_share = [100. * (A['SpkMixer base'][s]['f1'] - SH[s]['f1']) for s in SEEDS]
inter_share = [100. * (SH[s]['f1'] - A['SpkMixer memoryless'][s]['f1']) for s in SEEDS]
out = dict(sha=sha, arms={n: {k: mean([v[s][k] for s in SEEDS]) for k in KEYS} for n, v in A.items()}, comparisons=comps,
           order_share_pp=order_share, interaction_share_pp=inter_share, spread=SP['runs'])
os.makedirs(a.out_dir, exist_ok=True)
json.dump(out, open(os.path.join(a.out_dir, 'summary_tshuf.json'), 'w'), indent=1)
L = ['# MIT-BIH order-only control: SpkMixer trained and tested with a random token order per sample', '',
     f'Rules: summarize_tshuf.py docstring. 3 new runs, commit `{sha}`.', '',
     '| arm | macro-F1 % | accuracy % | F1 N / S / V / F / Q % | energy µJ (syops) |', '|---|--:|--:|---|--:|']
for n, m in out['arms'].items():
    L.append(f'| {n} | {100 * m["f1"]:.2f} | {100 * m["overall_acc"]:.2f} | ' + ' / '.join(f'{100 * m[f"f1_{i}"]:.2f}' for i in range(5)) + f' | {m["syn_energy_uJ"]:.2f} |')
L += ['', '| kind | A | B | d %p (42 / 7 / 2026) | mean %p | verdict | energy A/B |', '|---|---|---|---|--:|---|--:|']
for c in comps:
    L.append(f'| {c["kind"]} | {c["A"]} | {c["B"]} | ' + ' / '.join(f'{t:+.2f}' for t in c['d_pp']) + f' | {c["mean_pp"]:+.2f} | **{c["verdict"]}** | {c["energy_ratio"]:.3f} |')
L += ['', 'Decomposition (macro-F1 %p, seed 42 / 7 / 2026): token order = base - shuffled ' + ' / '.join(f'{t:+.2f}' for t in order_share) +
      f' (mean {mean(order_share):+.2f}); token interaction without order = shuffled - memoryless ' + ' / '.join(f'{t:+.2f}' for t in inter_share) +
      f' (mean {mean(inter_share):+.2f})', '',
      'Evaluation-order spread (report): ' + '; '.join(f'seed {x["seed"]}: default {x["default"]:.2f}, eval seed 1 {x["seed1"]:.2f}, seed 2 {x["seed2"]:.2f}, '
                                                   f'natural order {x["natural"]:.2f} (spread {x["spread"]:.2f} %p)' for x in SP['runs'])]
open(os.path.join(a.out_dir, 'summary_tshuf.md'), 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
