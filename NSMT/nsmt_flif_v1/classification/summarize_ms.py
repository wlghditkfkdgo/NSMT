"""MODI 2026-11-08: summary of the membrane shortcut test (docs/MODI_PROJECT_LOG.md 2026-11-08, pre-registered rules).

Variants of [query pop LIF NSMT T=4] (nsmt4q_pop_lif) and the query pop f-LIF (nsmt4q_pop_flif):
  base = IAND hippocampus x (1 - f(x)), no residual between Neocortex layers (flif-v1-ett-query / -m2, flif-v1-mitbih-query)
  msh  = --res_mode ms (hippocampus membrane shortcut), msn = --neo_res ms (Neocortex), mshn = both (flif-v1-ett-ms, flif-v1-mitbih-ms)
Comparisons, per neuron: M1 msh - base (request 2), M2 msn - base (request 3 alone), M3 mshn - msh (request 3 on top of request 2);
  report only: mshn - base.
ETT (ETTh2 / ETTm2 x H96 / H720, seeds 42 / 7 / 2026, best-validation weights): r = 100 (A / B - 1) of test MSE per cell and seed;
  better: mean of the 12 r <= -1 % and >= 8/12 lower; worse: the mirror; per cell +-1 % and 2/3 seeds (compare_query_pop_lif_paper.compare).
MIT-BIH (last-epoch weights, lr 0.01, full training file): delta = A - B test macro-F1 (%p); better: mean >= +0.5 and >= 2/3 seeds > 0;
  worse: the mirror.
Conclusion of a comparison: better (worse) for both neurons -> better (worse); for one neuron only -> neuron-dependent (weak); else no clear
difference. A missing, duplicated or non-finite value raises (fail-closed).
Report only: test MAE, mean validation L1 (ETT), firing rates rho_neo / rho_hippo, energy (ETT: total with the spike-driven head recount,
energy_recount.csv, the standard since 10-28; MIT-BIH: logged total_fft) and, separately, the membrane additions syops does not count
(0.9 pJ each: hippocampus 2 T_h N D per block and row, Neocortex N D per layer and row; the base's IAND products are not counted either),
parameters, and every model vs Spikformer-style.
usage (classification folder): python summarize_ms.py [--out FILE]
"""
import argparse
import os

import numpy as np

from compare_query_pop_lif_paper import CELLS, CL, COLS, FC, HERE, SEEDS, by_seed, cmp_row, compare, ett_recounted, num, results

NEURONS = ('pop_lif', 'pop_flif')
NAMES = {'pop_lif': 'query pop LIF', 'pop_flif': 'query pop f-LIF'}
VARS = ('base', 'msh', 'msn', 'mshn')
VNAME = {'base': 'base (IAND, no Neocortex residual)', 'msh': 'msh (hippocampus MS)', 'msn': 'msn (Neocortex MS)', 'mshn': 'mshn (both)'}
COMPS = (('M1', 'msh', 'base', 'request 2: hippocampus MS'), ('M2', 'msn', 'base', 'request 3 alone: Neocortex MS'),
         ('M3', 'mshn', 'msh', 'request 3 on top of request 2'), ('report', 'mshn', 'base', 'both vs base (report only)'))
GEOM = {'ETTh2': dict(N=24, D=64, Th=4, Lt=1, C=7), 'ETTm2': dict(N=24, D=64, Th=4, Lt=2, C=7), 'MITBIH': dict(N=11, D=384, Th=4, Lt=2, C=1)}


def cond(n, v):
    return f'nsmt4q_{n}' if v == 'base' else f'nsmt4q_{n}_{v}'


def ms_adds_uJ(ds, v):
    """membrane additions per window (ETT, C rows) or beat (MIT-BIH) that syops does not count, at 0.9 pJ each (1 hippocampus block)."""
    g = GEOM[ds]
    h = 2 * g['Th'] * g['N'] * g['D'] * g['C'] if v in ('msh', 'mshn') else 0
    n = g['Lt'] * g['N'] * g['D'] * g['C'] if v in ('msn', 'mshn') else 0
    return (h + n) * 0.9e-6


def ett(n, v):
    suites = ('flif-v1-ett-query', 'flif-v1-ett-query-m2') if v == 'base' else ('flif-v1-ett-ms',)
    out = ett_recounted(cond(n, v), suites)
    for ds, pl in CELLS:                                               # rates (ett_recounted keeps mse / mae / syn / tot / par / val)
        rows = [r for s in suites for r in results(os.path.join(FC, s, 'raw.txt'), ds=ds, pl=pl, cond=cond(n, v))]
        for sd, d in by_seed(rows, f'{cond(n, v)} {ds} {pl}').items():
            out[ds, pl, sd].update(rn=num(d, 'rho_neo'), rh=num(d, 'rho_hippo'))
    return out


def mb(n, v):
    path = os.path.join(CL, 'flif-v1-mitbih-query' if v == 'base' else 'flif-v1-mitbih-ms', 'raw.txt')
    sel = dict(lr='0.01', val='0')
    rows = results(path, cond=cond(n, v), **sel)
    return {s: dict(f1=100 * num(d, 'f1'), acc=100 * num(d, 'overall_acc'), tot=num(d, 'total_fft_uJ'), rn=num(d, 'rho_neo'),
                    rh=num(d, 'rho_hippo'), par=int(d['params']), f1s=[100 * num(d, f'f1_{i}') for i in range(5)])
            for s, d in by_seed(rows, f'{cond(n, v)} MIT-BIH').items()}


def mb_verdict(a, b):
    d = [a[s]['f1'] - b[s]['f1'] for s in SEEDS]; m = float(np.mean(d))
    v = 'better' if m >= 0.5 and sum(x > 0 for x in d) >= 2 else 'worse' if m <= -0.5 and sum(x < 0 for x in d) >= 2 else 'no clear difference'
    return d, m, v


def conclusion(vs):
    nb = sum(x == 'better' for x in vs); nw = sum(x == 'worse' for x in vs)
    if nb == len(vs):
        return 'better'
    if nw == len(vs):
        return 'worse'
    if nb or nw:
        return 'neuron-dependent (weak)'
    return 'no clear difference'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'flif-v1-ms-summary.md'))
    a = ap.parse_args()
    E = {(n, v): ett(n, v) for n in NEURONS for v in VARS}
    M = {(n, v): mb(n, v) for n in NEURONS for v in VARS}
    sp_e = ett_recounted('spk_ssa', ('flif-v1-gen-ett',))
    sp_m = {s: dict(f1=100 * num(d, 'f1'), tot=num(d, 'total_fft_uJ'))
            for s, d in by_seed(results(os.path.join(CL, 'flif-v1-gen-mitbih', 'raw.txt'), cond='spk_ssa'), 'spk_ssa MIT-BIH').items()}
    L = ['# Membrane shortcut test (query population LIF / f-LIF) — summary', '',
         'See the module docstring and docs/MODI_PROJECT_LOG.md 2026-11-08 for the pre-registered rules.', '',
         '## ETT — test MSE (mean ± sd of seeds 42 / 7 / 2026, best-validation weights)', '',
         '| model | variant | ' + COLS + ' | mean validation L1 |', '|---|---|---|---|---|---|---|']
    for n in NEURONS:
        for v in VARS:
            t = E[n, v]
            cells = ' | '.join(f"{np.mean([t[ds, pl, s]['mse'] for s in SEEDS]):.4f} ± {np.std([t[ds, pl, s]['mse'] for s in SEEDS], ddof=1):.4f}"
                               for ds, pl in CELLS)
            L.append(f"| {NAMES[n]} | {VNAME[v]} | {cells} | {np.mean([t[k]['val'] for k in t]):.4f} |")
    L += ['', '## ETT — verdicts (r = A / B − 1 of test MSE, %)', '',
          '| comparison | ' + COLS + ' | R % | pairs A lower | verdict | energy A/B |', '|---|---|---|---|---|---|---|---|---|']
    CE = {}
    for tag, x, y, what in COMPS:
        for n in NEURONS:
            c = compare(E[n, x], E[n, y]); CE[tag, n] = c['v']
            L.append(cmp_row(f'{tag} {NAMES[n]}: {x} − {y} ({what})', c))
    L += ['', '## MIT-BIH — test macro-F1 (%, last-epoch weights, lr 0.01)', '',
          '| model | variant | seed 42 | seed 7 | seed 2026 | mean | accuracy | F1 N / S / V / F / Q |', '|---|---|---|---|---|---|---|---|']
    for n in NEURONS:
        for v in VARS:
            t = M[n, v]
            L.append(f"| {NAMES[n]} | {VNAME[v]} | " + ' | '.join(f"{t[s]['f1']:.2f}" for s in SEEDS) + f" | {np.mean([t[s]['f1'] for s in SEEDS]):.2f}"
                     f" | {np.mean([t[s]['acc'] for s in SEEDS]):.2f} | " + ' / '.join(f"{np.mean([t[s]['f1s'][k] for s in SEEDS]):.1f}" for k in range(5)) + ' |')
    L += ['', '## MIT-BIH — verdicts (Δ = A − B, %p)', '', '| comparison | per seed (42 / 7 / 2026) | mean | > 0 | verdict | energy A/B |', '|---|---|---|---|---|---|']
    CM = {}
    for tag, x, y, what in COMPS:
        for n in NEURONS:
            d, m, v = mb_verdict(M[n, x], M[n, y]); CM[tag, n] = v
            er = np.mean([M[n, x][s]['tot'] for s in SEEDS]) / np.mean([M[n, y][s]['tot'] for s in SEEDS])
            L.append(f"| {tag} {NAMES[n]}: {x} − {y} ({what}) | " + ' / '.join(f'{z:+.2f}' for z in d) + f" | {m:+.2f} | {sum(z > 0 for z in d)}/3 | {v} | {er:.2f} |")
    L += ['', '## conclusions (rule of the pre-registration)', '', '| comparison | task | pop LIF | pop f-LIF | conclusion |', '|---|---|---|---|---|']
    for tag, x, y, what in COMPS[:3]:
        for task, C in (('ETT', CE), ('MIT-BIH', CM)):
            L.append(f"| {tag} {x} − {y} ({what}) | {task} | {C[tag, 'pop_lif']} | {C[tag, 'pop_flif']} | {conclusion([C[tag, n] for n in NEURONS])} |")
    L += ['', '## report only', '', '### firing rates and energy (mean of seeds; ETT: mean over the 4 cells)', '',
          '| model | variant | ETT rho_neo | ETT rho_hippo | ETT energy uJ (head recount) | ETT MS additions uJ | MIT-BIH rho_neo | MIT-BIH rho_hippo | MIT-BIH energy uJ (logged) | MIT-BIH MS additions uJ | params ETTh2 H96 / MIT-BIH |',
          '|---|---|---|---|---|---|---|---|---|---|---|']
    for n in NEURONS:
        for v in VARS:
            t, m = E[n, v], M[n, v]
            ea = np.mean([ms_adds_uJ(ds, v) for ds, pl in CELLS])
            L.append(f"| {NAMES[n]} | {v} | {np.mean([t[k]['rn'] for k in t]):.4f} | {np.mean([t[k]['rh'] for k in t]):.4f} | {np.mean([t[k]['tot'] for k in t]):.2f}"
                     f" | {ea:.4f} | {np.mean([m[s]['rn'] for s in SEEDS]):.4f} | {np.mean([m[s]['rh'] for s in SEEDS]):.4f} | {np.mean([m[s]['tot'] for s in SEEDS]):.2f}"
                     f" | {ms_adds_uJ('MITBIH', v):.4f} | {t['ETTh2', 96, 42]['par']} / {m[42]['par']} |")
    L += ['', '### test MAE (mean of seeds)', '', '| model | variant | ' + COLS + ' |', '|---|---|---|---|---|---|']
    for n in NEURONS:
        for v in VARS:
            L.append(f"| {NAMES[n]} | {v} | " + ' | '.join(f"{np.mean([E[n, v][ds, pl, s]['mae'] for s in SEEDS]):.4f}" for ds, pl in CELLS) + ' |')
    L += ['', '### every model vs Spikformer-style (flif-v1-gen-ett / flif-v1-gen-mitbih; same rules)', '',
          '| model | ETT: ' + COLS + ' | ETT R % | lower | ETT verdict | ETT energy ratio | MIT-BIH Δ %p | MIT-BIH verdict |', '|---|---|---|---|---|---|---|---|---|---|---|']
    for n in NEURONS:
        for v in VARS:
            c = compare(E[n, v], sp_e); d, m, vv = mb_verdict(M[n, v], sp_m)
            cells = ' | '.join(f"{c['cv'][x][0]:+.2f}" for x in CELLS)
            L.append(f"| {NAMES[n]} {v} | {cells} | {c['R']:+.2f} | {c['low']}/12 | {c['v']} | {c['en']:.2f} | {m:+.2f} | {vv} |")
    open(a.out, 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
