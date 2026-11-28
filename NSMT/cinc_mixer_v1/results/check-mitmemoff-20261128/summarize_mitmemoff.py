"""Summary of the MIT-BIH memoryless control (2026-11-28, exp/f-lif-pop-v3; PROJECT_LOG pre-registration of 13:5x; rules fixed before
training). New: LIF SpkMixer muxlin ADD lr 0.02 mixer layers 2 with token_memory off (suite flif-v1-mitbih-memoff, 3 seeds); reference:
the same model with LIF memory (flif-v1-mitbih-spkmux-tune tag spkmx_lif_muxlin_add_lr0.02_L2, same seeds and initial weights).
d = macro-F1(memoryless) - macro-F1(reference) in %p per seed; 'worse' (the token-axis neuron dynamics contribute) if mean d <= -0.5 %p
and d < 0 for >= 2 of 3 seeds; 'better' if >= +0.5 %p and > 0 for >= 2 of 3; else 'no clear difference'. Accuracy and per-class F1,
syops energy (logged syn_energy_uJ) and the integer-rule total (energy_recount.csv) reported. Check: the order-shuffle evaluation of
the memoryless model (order_muxoff.json) must show |mixer-shuffle drop| < 0.05 %p (no token-axis flow). Fails closed on a SKIP line,
not 3 rows, mixed commits, missing / non-finite values.
usage (from cinc_mixer_v1/): python -B results/check-mitmemoff-20261128/summarize_mitmemoff.py [--suite flif-v1-mitbih-memoff]
"""
import argparse
import csv
import json
import math
import os
import sys

C = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser(); ap.add_argument('--suite', default='flif-v1-mitbih-memoff'); ap.add_argument('--order_json', default=None)
a = ap.parse_args()


def stop(m):
    print(f'STOP: {m}'); sys.exit(2)


def rows(suite, tag):
    lines = open(os.path.join(C, 'results', suite, 'raw.txt')).read().splitlines()
    if any(l.startswith('SKIP') for l in lines) and suite == a.suite:
        stop('SKIP line')
    rr = [dict(t.split('=', 1) for t in l.split()[1:] if '=' in t) for l in lines if l.startswith('RESULT')]
    rr = [r for r in rr if r.get('tag') == tag]
    if sorted(r['seed'] for r in rr) != ['2026', '42', '7']:
        stop(f'{suite} {tag}: seeds {[r["seed"] for r in rr]}')
    for r in rr:
        for k in ('f1', 'overall_acc', 'syn_energy_uJ', 'f1_0', 'f1_1', 'f1_2', 'f1_3', 'f1_4'):
            if not math.isfinite(float(r[k])):
                stop(f'{k} not finite')
    return {r['seed']: r for r in rr}


N = rows(a.suite, 'spkmx_lif_muxlin_add_lr0.02_L2_tmoff')
if len({r['sha'] for r in N.values()}) != 1:
    stop('mixed training commits')
B = rows('flif-v1-mitbih-spkmux-tune', 'spkmx_lif_muxlin_add_lr0.02_L2')
S = ('42', '7', '2026')
d = [100 * (float(N[s]['f1']) - float(B[s]['f1'])) for s in S]
m = sum(d) / 3
verdict = 'worse' if m <= -0.5 and sum(x < 0 for x in d) >= 2 else ('better' if m >= 0.5 and sum(x > 0 for x in d) >= 2 else 'no clear difference')
ec = os.path.join(C, 'results', a.suite, 'energy_recount.csv')
if not os.path.exists(ec):
    stop('energy_recount.csv missing')
E = {r['seed']: r for r in csv.DictReader(open(ec))}
oj = a.order_json or os.path.join(C, 'results', 'check-mitorder-20261128', 'order_muxoff.json')
if not os.path.exists(oj):
    stop('order_muxoff.json missing')
mo = [r['mixer']['drop_f1_pp'] for r in json.load(open(oj))['runs']]
if len(mo) != 3 or not all(abs(x) < 0.05 for x in mo):
    stop(f'memoryless mixer shuffle drops {mo} (want |drop| < 0.05 %p)')
avg = lambda x: sum(x) / len(x)
out = dict(sha=sorted({r['sha'] for r in N.values()}), d_pp=d, mean_d_pp=m, verdict=verdict, memoryless_mixer_shuffle_drop_pp=mo,
           f1_new=avg([100 * float(N[s]['f1']) for s in S]), f1_ref=avg([100 * float(B[s]['f1']) for s in S]),
           acc_new=avg([100 * float(N[s]['overall_acc']) for s in S]), acc_ref=avg([100 * float(B[s]['overall_acc']) for s in S]),
           class_f1_new=[avg([100 * float(N[s][f'f1_{c}']) for s in S]) for c in range(5)], class_f1_ref=[avg([100 * float(B[s][f'f1_{c}']) for s in S]) for c in range(5)],
           syn_new=avg([float(N[s]['syn_energy_uJ']) for s in S]), syn_ref=avg([float(B[s]['syn_energy_uJ']) for s in S]),
           total_int_new=avg([float(E[str(int(s))]['total_fft_uJ_corr']) if str(int(s)) in E else float('nan') for s in S]))
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'summary_mitmemoff.json'), 'w'), indent=1)
L = ['# MIT-BIH memoryless control (SpkMixer muxlin ADD, token_memory off vs LIF memory; 3 seeds)', '',
     f'- macro-F1: memoryless {out["f1_new"]:.2f} vs LIF memory {out["f1_ref"]:.2f}; per seed 42 / 7 / 2026: ' + ' / '.join(f'{x:+.2f}' for x in d)
     + f' %p; mean {m:+.2f} %p -> **{verdict}**',
     f'- accuracy: {out["acc_new"]:.2f} vs {out["acc_ref"]:.2f} %',
     '- class F1 N / S / V / F / Q: memoryless ' + ' / '.join(f'{x:.2f}' for x in out['class_f1_new']) + '; LIF memory ' + ' / '.join(f'{x:.2f}' for x in out['class_f1_ref']),
     f'- syops energy: {out["syn_new"]:.2f} vs {out["syn_ref"]:.2f} uJ (ratio {out["syn_new"] / out["syn_ref"]:.3f})',
     f'- check: mixer-shuffle drop of the memoryless model ' + ' / '.join(f'{x:+.4f}' for x in mo) + ' %p (|drop| < 0.05 required)']
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'summary_mitmemoff.md'), 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
