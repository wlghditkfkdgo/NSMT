"""MODI 2026-10-29: whole-model neuron swap (Neo-mixer --neuron_scope all) vs the neocortex-only swap on ETT, with the
pre-registered rule (docs/MODI_PROJECT_LOG.md 2026-10-29). Missing / non-finite runs make a verdict 'incomplete'.

Per neuron n and cell (ds, pl), paired by seed: R = (MSE mixall_n - MSE mix_n) / MSE mix_n (mix_n = flif-v1-gen-ett, the neuron in
the mixer stacks only). 'better' = mean R <= -1 % and R < 0 in 3 of 3 seeds; 'worse' = mean R >= +1 % and R > 0 in 3 of 3;
otherwise 'no clear difference'. Reported: R against Spikformer-style (spk_ssa) and the energy ratios (total_fft_uJ).
usage: python summarize_nall.py [--suite flif-v1-ett-nall] [--base flif-v1-gen-ett] [--cells ETTh2:96]
"""
import argparse
import math
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = (42, 7, 2026)
NEURONS = ('lif', 'flif', 'pop_lif', 'pop_flif')
NL = {'lif': 'LIF', 'flif': 'f-LIF', 'pop_lif': 'population LIF', 'pop_flif': 'population f-LIF'}


def parse(path):
    rows = []
    if not os.path.exists(path):
        return rows
    for l in open(path):
        if l.startswith('RESULT'):
            rows.append(dict(re.findall(r'(\w+)=(\S+)', l)))
    return rows


def per_seed(rows, ds, pl, cond, key):
    out = []
    for s in SEEDS:
        v = [r.get(key) for r in rows if r.get('ds') == ds and r.get('pl') == str(pl) and r.get('cond') == cond and r.get('seed') == str(s)]
        try:
            x = float(v[-1]) if v else None
        except ValueError:
            x = None
        out.append(x if x is not None and math.isfinite(x) else None)
    return out


def rel(a, b):
    if any(x is None for x in a + b) or any(y == 0 for y in b):
        return None
    return [(x - y) / y for x, y in zip(a, b)]


def verdict(r):
    if r is None:
        return 'incomplete'
    m = np.mean(r)
    if m <= -0.01 and all(x < 0 for x in r):
        return 'better'
    if m >= 0.01 and all(x > 0 for x in r):
        return 'worse'
    return 'no clear difference'


def ms(v, nd=4):
    return 'incomplete' if any(x is None for x in v) else f'{np.mean(v):.{nd}f} (' + ' / '.join(f'{x:.{nd}f}' for x in v) + ')'


def pct(r):
    return '—' if r is None else f'{np.mean(r) * 100:+.2f} % (' + ' / '.join(f'{x * 100:+.2f}' for x in r) + ')'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--suite', default='flif-v1-ett-nall')
    ap.add_argument('--base', default='flif-v1-gen-ett')
    ap.add_argument('--cells', default='ETTh2:96')
    a = ap.parse_args()
    new = parse(os.path.join(HERE, 'results', a.suite, 'raw.txt'))
    base = parse(os.path.join(HERE, 'results', a.base, 'raw.txt'))
    L = [f'# Whole-model neuron swap `{a.suite}` vs neocortex-only swap `{a.base}` — summary', '',
         'Test MSE, best-validation weights; mean (seed 42 / 7 / 2026). R = relative MSE difference per seed. Energy = total_fft_uJ.', '']
    for cell in a.cells.split(','):
        ds, pl = cell.split(':')
        L += [f'## {ds}, horizon {pl}', '',
              '| neuron | MSE, neocortex only | MSE, every layer | R every − neocortex only | verdict | R every − Spikformer-style | energy every / neocortex only | energy every / Spikformer-style |',
              '|---|---|---|---|---|---|---|---|']
        ssa = per_seed(base, ds, pl, 'spk_ssa', 'mse'); e_ssa = per_seed(base, ds, pl, 'spk_ssa', 'total_fft_uJ')
        for n in NEURONS:
            mo, ma = per_seed(base, ds, pl, f'mix_{n}_bidir', 'mse'), per_seed(new, ds, pl, f'mixall_{n}_bidir', 'mse')
            eo, ea = per_seed(base, ds, pl, f'mix_{n}_bidir', 'total_fft_uJ'), per_seed(new, ds, pl, f'mixall_{n}_bidir', 'total_fft_uJ')
            r = rel(ma, mo)
            ro = rel(ea, eo); rs = rel(ea, e_ssa)
            L.append(f"| {NL[n]} | {ms(mo)} | {ms(ma)} | {pct(r)} | {verdict(r)} | {pct(rel(ma, ssa))} | "
                     f"{'—' if ro is None else f'{np.mean(ro) + 1:.2f}'} | {'—' if rs is None else f'{np.mean(rs) + 1:.2f}'} |")
        L += ['', f"Spikformer-style MSE {ms(ssa)}, energy {ms(e_ssa, 2)} µJ", '']
    os.makedirs(os.path.join(HERE, 'results', a.suite), exist_ok=True)
    open(os.path.join(HERE, 'results', a.suite, 'summary.md'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
