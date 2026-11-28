"""MIT-BIH memoryless control on the IAND base (2026-11-28, exp/f-lif-pop-v3; PROJECT_LOG pre-registration; user "3. 기억 없는 실험을
MIT-BIH + IAND에 대해서 진행해줘"; rules fixed before training).
  offI  LIF SpkMixer muxlin IAND lr 0.01, 2 mixer layers, --token_memory off (every token-axis mixer neuron resets after each token)
        suite flif-v1-mitbih-memoff-iand, tag spkmx_lif_muxlin_iand_lr0.01_L2_tmoff, seeds 42 / 7 / 2026 (3 new runs)
References (paired seeds; logged rows): IAND base (flif-v1-mitbih-res spkmx_lif_muxlin_iand_lr0.01_L2), IAND shuffled
(flif-v1-mitbih-tshuf-iand ..._tshuf), ADD memoryless (flif-v1-mitbih-memoff ..._tmoff), SF-de PE off / on.
Comparisons (test macro-F1; d = 100 (F1_A - F1_B) per seed in %p; 'A better' if mean >= +0.5 and >= 2 of 3 > 0, 'A worse' mirror, else
'no clear difference'; energy = logged syops syn_energy_uJ, ratio of seed means):
  U1  offI vs IAND base           'A worse' = the token-axis neuron dynamics contribute on the IAND base (as the 14:03 ADD control)
  U2  IAND shuffled vs offI       'A better' = token interaction without order contributes; interaction share = shuffled - offI
  decomposition per seed: base - offI = (base - shuffled: order share) + (shuffled - offI: interaction share); order fraction reported
  U3  offI vs ADD memoryless (report)   U4  offI vs SF-de PE off (report)   U5  offI vs SF-de PE on (report)
Check (fail closed): the evaluation-time token shuffle of offI (order_iandoff.json) must drop macro-F1 by |d| < 0.05 %p on every seed
(no token-axis flow -> the model is order-free), as order_muxoff.json for ADD.
Fails closed (exit 2, nothing written): SKIP line or a row of another tag, not 3 seeds, mixed commits, non-finite values, a missing energy
recount row or recomputed macro-F1 != logged, the order json missing / its unshuffled reading != logged / a drop >= 0.05 %p.
usage (from cinc_mixer_v1/): python -B results/check-iandmem-20261128/summarize_memoffi.py [--new <suite dir>] [--order <json>] [--out_dir ...]
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
ap.add_argument('--new', default=os.path.join(RS, 'flif-v1-mitbih-memoff-iand'))
ap.add_argument('--order', default=os.path.join(HERE, 'order_iandoff.json'))
ap.add_argument('--out_dir', default=HERE)
a = ap.parse_args()
SEEDS = ('42', '7', '2026')
KEYS = ('f1', 'overall_acc', 'f1_0', 'f1_1', 'f1_2', 'f1_3', 'f1_4', 'syn_energy_uJ')
TAG = 'spkmx_lif_muxlin_iand_lr0.01_L2_tmoff'
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


OFF, sha = load(a.new, TAG, new=True)
N = 'SpkMixer IAND memoryless (new)'
A = {N: OFF, 'SpkMixer IAND base': load(os.path.join(RS, 'flif-v1-mitbih-res'), 'spkmx_lif_muxlin_iand_lr0.01_L2')[0],
     'SpkMixer IAND order-shuffled': load(os.path.join(RS, 'flif-v1-mitbih-tshuf-iand'), 'spkmx_lif_muxlin_iand_lr0.01_L2_tshuf')[0],
     'SpkMixer ADD memoryless': load(os.path.join(RS, 'flif-v1-mitbih-memoff'), 'spkmx_lif_muxlin_add_lr0.02_L2_tmoff')[0],
     'SF-de PE off': load(os.path.join(RS, 'flif-v1-mitbih-sfde-nope'), 'spk_ssa_lr0.01_L2_nalif_pe0')[0],
     'SF-de PE on': load(os.path.join(RS, 'flif-v1-mitbih-sfde'), 'spk_ssa_lr0.01_L2_nalif')[0]}
if not os.path.exists(a.order):
    stop(f'{a.order} missing')
OE = json.load(open(a.order))
if OE.get('tag') != TAG or sorted(x['seed'] for x in OE['runs']) != sorted(SEEDS) or \
        any(abs(x['f1'] - 100. * OFF[x['seed']]['f1']) > 1e-4 for x in OE['runs']):
    stop('order json: wrong runs or unshuffled reading != logged')
drops = {x['seed']: (x['mixer']['drop_f1_pp'], x['full']['drop_f1_pp']) for x in OE['runs']}
if not all(math.isfinite(v) and abs(v) < 0.05 for t in drops.values() for v in t):
    stop(f'memoryless IAND model is not order-free under the evaluation shuffle: {drops}')


def comp(x, y):
    d = [100. * (A[x][s]['f1'] - A[y][s]['f1']) for s in SEEDS]
    m = mean(d)
    v = 'A better' if m >= 0.5 and sum(t > 0 for t in d) >= 2 else ('A worse' if m <= -0.5 and sum(t < 0 for t in d) >= 2 else 'no clear difference')
    return dict(A=x, B=y, d_pp=d, mean_pp=m, verdict=v, energy_ratio=mean([A[x][s]['syn_energy_uJ'] for s in SEEDS]) / mean([A[y][s]['syn_energy_uJ'] for s in SEEDS]))


comps = [dict(kind='U1', **comp(N, 'SpkMixer IAND base')), dict(kind='U2', **comp('SpkMixer IAND order-shuffled', N)),
         dict(kind='U3 (report)', **comp(N, 'SpkMixer ADD memoryless')), dict(kind='U4 (report)', **comp(N, 'SF-de PE off')),
         dict(kind='U5 (report)', **comp(N, 'SF-de PE on'))]
tot = [100. * (A['SpkMixer IAND base'][s]['f1'] - OFF[s]['f1']) for s in SEEDS]
osh = [100. * (A['SpkMixer IAND base'][s]['f1'] - A['SpkMixer IAND order-shuffled'][s]['f1']) for s in SEEDS]
ish = [100. * (A['SpkMixer IAND order-shuffled'][s]['f1'] - OFF[s]['f1']) for s in SEEDS]
frac = mean(osh) / mean(tot) if abs(mean(tot)) > 1e-12 else float('nan')
out = dict(sha=sha, arms={n: {k: mean([v[s][k] for s in SEEDS]) for k in KEYS} for n, v in A.items()}, comparisons=comps,
           memoryless_drop_pp=tot, order_share_pp=osh, interaction_share_pp=ish, order_fraction_of_mean_drop=frac,
           order_check_drops_pp={s: list(v) for s, v in drops.items()})
os.makedirs(a.out_dir, exist_ok=True)
json.dump(out, open(os.path.join(a.out_dir, 'summary_memoffi.json'), 'w'), indent=1)
L = ['# MIT-BIH memoryless control on the IAND base (SpkMixer IAND, token-axis mixer neurons reset after each token)', '',
     f'Rules: summarize_memoffi.py docstring. 3 new runs, commit `{sha}`.', '',
     '| arm | macro-F1 % | accuracy % | F1 N / S / V / F / Q % | energy µJ (syops) |', '|---|--:|--:|---|--:|']
for n, m in out['arms'].items():
    L.append(f'| {n} | {100 * m["f1"]:.2f} | {100 * m["overall_acc"]:.2f} | ' + ' / '.join(f'{100 * m[f"f1_{i}"]:.2f}' for i in range(5)) + f' | {m["syn_energy_uJ"]:.2f} |')
L += ['', '| kind | A | B | d %p (42 / 7 / 2026) | mean %p | verdict | energy A/B |', '|---|---|---|---|--:|---|--:|']
for c in comps:
    L.append(f'| {c["kind"]} | {c["A"]} | {c["B"]} | ' + ' / '.join(f'{t:+.2f}' for t in c['d_pp']) + f' | {c["mean_pp"]:+.2f} | **{c["verdict"]}** | {c["energy_ratio"]:.3f} |')
L += ['', 'Decomposition (macro-F1 %p, seed 42 / 7 / 2026): base - memoryless ' + ' / '.join(f'{t:+.2f}' for t in tot) + f' (mean {mean(tot):+.2f}) = '
      'order share (base - shuffled) ' + ' / '.join(f'{t:+.2f}' for t in osh) + f' (mean {mean(osh):+.2f}) + interaction share (shuffled - memoryless) '
      + ' / '.join(f'{t:+.2f}' for t in ish) + f' (mean {mean(ish):+.2f}); order fraction of the mean drop {frac:.3f}', '',
      'Check: evaluation-time token shuffle of the memoryless IAND model, drop in macro-F1 %p (mixer input / patch tokens): ' +
      '; '.join(f'seed {s} {v[0]:+.4f} / {v[1]:+.4f}' for s, v in sorted(drops.items())) + ' (|drop| < 0.05 required)']
open(os.path.join(a.out_dir, 'summary_memoffi.md'), 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
