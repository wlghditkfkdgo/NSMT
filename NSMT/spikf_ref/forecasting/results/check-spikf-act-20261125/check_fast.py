"""Equivalence of --fast_eval with the released evaluation (2026-11-26, exp/f-lif-pop-v3; PROJECT_LOG SpikF-LIF pre-registration).
ETTh1 H96 seed 42, 2 epochs, both head activations:
  released protocol  suite check-fast-spikf-20261126, FAST=0 (validation / test at batch 1 every epoch, anomaly mode on, syops at batch 1)
  fast (final form)  suite check-fast3-spikf-20261126, FAST=1 EVBS=256 SYBS=1 (batched validation / in-loop test / final test, anomaly
                     mode off, neurons reset before syops)
Checks per activation: every per-epoch train loss equal (7 printed digits, the training steps are the same), per-epoch validation and
test losses within 1e-6, the tested epoch equal, final test MSE / MAE within 1e-6, syops total ops equal, energy within 1e-4 relative.
Reported, not checked: the two earlier fast forms (check-fast-spikf-20261126 FAST=1: no in-loop test pass -> a different shuffle from
epoch 2; and its syops count after a batched test without reset -> 8 % more ops; SYBS=64 crashed).
usage (from spikf_ref/forecasting/): python -B results/check-spikf-act-20261125/check_fast.py
"""
import json
import os
import re
import sys

F = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
fails = 0


def check(label, c, detail=''):
    global fails
    fails += not c
    print(f'{"PASS" if c else "FAIL"} {label}{": " + detail if detail else ""}')


def rows(suite):
    out = {}
    for l in open(os.path.join(F, 'results', suite, 'raw.txt')):
        if l.startswith('RESULT'):
            r = dict(t.split('=', 1) for t in l.split()[1:] if '=' in t)
            out[(r['act'], r['fast'])] = r
    return out


def epochs(stdout):
    return [tuple(float(x) for x in m) for m in re.findall(r'^Epoch: \d+, Steps: \d+ \| Train Loss: (\S+) valid Loss: (\S+) Test Loss: (\S+)',
                                                           open(stdout).read(), re.M)]


# The released-protocol runs wrote result.json but no RESULT line: run_spikf.sh was edited (comment lines) while they ran, and bash
# lost its place in the script after python ended (2026-11-26 00:12, my mistake). The check therefore reads the run folders.
L = os.path.join(F, 'log')
B = rows('check-fast3-spikf-20261126')
for act in ('gelu', 'lif'):
    pa = os.path.join(L, 'check-fast-spikf-20261126', 'ETTh1', 'p96', f'{act}_seed42', 'result.json')
    b = B.get((act, '1_fe256_sy1'))
    if not os.path.exists(pa) or b is None:
        check(f'{act}: both runs present', False)
        continue
    ja, jb = json.load(open(pa)), json.load(open(os.path.join(b['dir'], 'result.json')))
    check(f'{act}: released-protocol run used the released path (no --fast_eval in its stdout name, epochs 2)', 'fe256' not in pa and len(ja['hist']) == 2)
    ha, hb = ja['hist'], jb['hist']
    check(f'{act}: train loss per epoch equal', [h[1] for h in ha] == [h[1] for h in hb], f'{[h[1] for h in ha]} vs {[h[1] for h in hb]}')
    dv = max(abs(x[2] - y[2]) for x, y in zip(ha, hb))
    dt = max(abs(x[3] - y[3]) for x, y in zip(ha, hb))
    check(f'{act}: validation / in-loop test loss per epoch within 1e-6', len(ha) == len(hb) and dv <= 1e-6 and dt <= 1e-6, f'max |diff| {dv:.2g} / {dt:.2g}')
    check(f'{act}: tested epoch equal', ja['best_epoch'] == jb['best_epoch'], f'{ja["best_epoch"]} vs {jb["best_epoch"]}')
    for m in ('mse', 'mae'):
        check(f'{act}: final test {m} within 1e-6', abs(ja[m] - jb[m]) <= 1e-6, f'{ja[m]:.9f} vs {jb[m]:.9f}')
    check(f'{act}: syops total ops equal', ja['total_op'] == jb['total_op'], f'{ja["total_op"]} vs {jb["total_op"]}')
    check(f'{act}: energy within 1e-4 relative', abs(ja['energy_uJ'] - jb['energy_uJ']) <= 1e-4 * ja['energy_uJ'],
          f'{ja["energy_uJ"]:.4f} vs {jb["energy_uJ"]:.4f} µJ')
    print(f'  {act}: fast run {b["time_s"]} s (released-protocol runs: about 22-24 min, from the stdout file times)')
print('ALL PASS' if fails == 0 else f'{fails} FAIL')
sys.exit(0 if fails == 0 else 1)
