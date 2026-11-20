"""MODI 2026-11-20 POST HOC (not pre-registered) reading of the aux-target memory check (aux_check.py): written after the
pre-registered summary showed that the no-memory control reconstructs its OWN token better than the memory model in all 6 cells,
so the own-token calibration g(0) of the pre-registered rule is negative and ADDS to d. This script reports, per task / neuron /
target, what the verdict rests on without that calibration:
  raw gap         e(control, s) - e(memory, s)                     (no own-token calibration)
  g(0)            e(control, 0) - e(memory, 0)                     (the calibration term of the pre-registered rule)
  linear margin   e_lin(token i alone, s) - e(memory, s)           (how much the memory model beats a ridge fit on token i alone)
  control - lin   e(no memory, s) - e_lin(token i alone, s)        (does the no-memory control reach the linear no-memory bound?)
  rho             Neocortex firing rate of the no-memory run at s and at s = 0
ETT: per cell (ETTh2 / ETTm2) and the mean. Report only: the pre-registered verdict (results/flif-v1-auxck.md) is not changed.
usage (classification folder): python aux_check_posthoc.py   -> results/flif-v1-auxck-posthoc.md
"""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, 'results', 'flif-v1-auxck')
R = [json.loads(l) for l in open(os.path.join(D, 'rows.jsonl'))]
BL = json.load(open(os.path.join(D, 'baselines.json')))
if len(R) != 72 or len(BL) != 24:
    raise RuntimeError(f'{len(R)} rows, {len(BL)} baselines')
for r in R + BL:
    if not all(isinstance(r[k], (int, float)) and math.isfinite(r[k]) for k in ('e', 'E', 'Z')):
        raise RuntimeError(f'non-finite {r}')


def one(L, key):
    if len(L) != 1:
        raise RuntimeError(f'{key}: {len(L)} matches')
    return L[0]


def e(task, ds, n, v, s, k='e'):
    return one([r[k] for r in R if r['task'] == task and r['ds'] == ds and r['neuron'] == n and r['variant'] == v and r['s'] == s],
               (task, ds, n, v, s))


def lin(task, ds, s):
    return one([b['e'] for b in BL if b['task'] == task and b['ds'] == ds and b['s'] == s and b['inputs'] == 'current'], (task, ds, s))


CELLS = {'ett': ['ETTh2', 'ETTm2'], 'mb': [None]}
NAME = {'lif': 'pop LIF', 'flif': 'pop f-LIF'}
TN = {2: 'past, 2 back (s = 2)', 4: 'past, 4 back (s = 4)', -2: 'next patch (s = -2)'}


def fmt(L):
    m = sum(L) / len(L)
    return (' / '.join(f'{x:+.3f}' for x in L) + f' → {m:+.3f}') if len(L) > 1 else f'{m:+.3f}'


out = ['# Aux-target memory check — POST HOC reading (not pre-registered; report only)', '', __doc__.strip().split('\n\n')[0], '',
       '| task | neuron | target | raw gap, no memory | raw gap, shuffled | g(0), no memory | linear margin of memory | '
       'no memory − linear | rho no memory (s / s = 0) |', '|---|---|---|---|---|---|---|---|---|']
for task in ('ett', 'mb'):
    for n in ('lif', 'flif'):
        for s in (2, 4, -2):
            c = CELLS[task]
            cols = [fmt([e(task, ds, n, 'nomem', s) - e(task, ds, n, 'mem', s) for ds in c]),
                    fmt([e(task, ds, n, 'shuf', s) - e(task, ds, n, 'mem', s) for ds in c]),
                    fmt([e(task, ds, n, 'nomem', 0) - e(task, ds, n, 'mem', 0) for ds in c]),
                    fmt([lin(task, ds, s) - e(task, ds, n, 'mem', s) for ds in c]),
                    fmt([e(task, ds, n, 'nomem', s) - lin(task, ds, s) for ds in c]),
                    ' ; '.join(f"{e(task, ds, n, 'nomem', s, 'rho_neo'):.5f} / {e(task, ds, n, 'nomem', 0, 'rho_neo'):.5f}" for ds in c)]
            out.append(f'| {"ETT" if task == "ett" else "MIT-BIH"} | {NAME[n]} | {TN[s]} | ' + ' | '.join(cols) + ' |')
open(os.path.join(HERE, 'results', 'flif-v1-auxck-posthoc.md'), 'w').write('\n'.join(out) + '\n')
print('\n'.join(out))
