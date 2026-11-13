"""MODI 2026-11-13: summary of the STFT test (docs/MODI_PROJECT_LOG.md, pre-registered rules; user request 11-13).

Base model = the Neocortex-MS query population nsmt4q_pop_<lif|flif>_msn (flif-v1-ett-ms, flif-v1-mitbih-ms). The Neocortex input
is a short-time Fourier transform whose frames are the tokens (window = patch, hop = stride, periodic Hann), its bins grouped into
4 bands, one band per member of the population Neocortex embedding. Variants (flif-v1-ett-stft, flif-v1-mitbih-stft):
  stft      = (na): population with a soma, member time constants 4 / 8 / 16 / 32, lowest band -> slowest member
  fire_stft = (ga): no soma, the 4 members are independent spiking neurons with time constant 2, a unit spikes if any member does
  fire_same = control of (ga): the same soma-less structure, every member reads the unsplit token through its own Linear
Earlier runs used for report-only comparisons (flif-v1-ett-band, flif-v1-mitbih-band): freq (whole-window DCT bands), same.
Comparisons, per neuron: S1 stft - msn, S2 fire_stft - msn, S3 fire_stft - stft ((ga) vs (na)), S4 fire_stft - fire_same (is it the
frequency input); report only: stft - freq (STFT vs DCT), stft - same, fire_same - msn.
Rules = summarize_ms.py (ETT: 12 pairs, mean r <= -1 % and >= 8/12 lower = better, mirror = worse, per cell +-1 % and 2/3;
MIT-BIH: mean delta >= +0.5 %p and >= 2/3 seeds > 0 = better, mirror = worse; conclusion over the two neurons). A missing, duplicated
or non-finite value raises (fail-closed).
Report only: MAE, mean validation L1, firing rates, energy (ETT: head recount; MIT-BIH: logged), parameters, vs Spikformer-style, and
the costs syops does not count: membrane additions (0.9 pJ each, 4 member streams) and the STFT (dense upper estimate: per frame P
window products + P^2 MAC at 4.6 pJ; frames = tokens). The OR over the members of the soma-less variants is a logic operation and not costed.
usage (classification folder): python summarize_stft.py [--out FILE]
"""
import argparse
import os

import numpy as np

from compare_query_pop_lif_paper import CELLS, CL, COLS, FC, HERE, SEEDS, by_seed, cmp_row, compare, ett_recounted, num, results
from summarize_ms import GEOM, NAMES, NEURONS, conclusion, mb_verdict

VARS = ('msn', 'stft', 'fire_stft', 'fire_same', 'freq', 'same')
SUITE = {'msn': 'ms', 'stft': 'stft', 'fire_stft': 'stft', 'fire_same': 'stft', 'freq': 'band', 'same': 'band'}
VNAME = {'msn': 'msn (base)', 'stft': 'stft = (na): soma, tau 4-32, STFT bands', 'fire_stft': 'fire_stft = (ga): no soma, tau 2, STFT bands',
         'fire_same': 'fire_same: no soma, tau 2, unsplit x4', 'freq': 'freq (11-13): soma, tau 4-32, DCT bands', 'same': 'same (11-13): soma, tau 4-32, unsplit x4'}
COMPS = (('S1', 'stft', 'msn', '(na) vs the base'), ('S2', 'fire_stft', 'msn', '(ga) vs the base'),
         ('S3', 'fire_stft', 'stft', '(ga) vs (na)'), ('S4', 'fire_stft', 'fire_same', '(ga): STFT bands vs the unsplit signal'),
         ('report', 'stft', 'freq', 'STFT vs whole-window DCT, same structure (report only)'),
         ('report', 'stft', 'same', 'STFT bands vs unsplit x4, soma structure (report only)'),
         ('report', 'fire_same', 'msn', 'soma-less tau-2 structure alone (report only)'))
PATCH = {'ETTh2': 8, 'ETTm2': 8, 'MITBIH': 32}


def cond(n, v):
    return f'nsmt4q_{n}_msn' if v == 'msn' else f'nsmt4q_{n}_msn_{v}'


def uncounted_uJ(ds, v):
    """(membrane additions, input transform) per window (ETT, C rows) or beat (MIT-BIH), not counted by syops."""
    g = GEOM[ds]
    adds = g['Lt'] * g['N'] * g['D'] * g['C'] * (1 if v == 'msn' else 4) * 0.9e-6
    if v in ('stft', 'fire_stft'):
        tr = g['N'] * (PATCH[ds] ** 2 + PATCH[ds]) * g['C'] * 4.6e-6
    elif v == 'freq':
        tr = 2 * {'ETTh2': 100, 'ETTm2': 100, 'MITBIH': 203}[ds] ** 2 * g['C'] * 4.6e-6
    else:
        tr = 0.
    return adds, tr


def ett(n, v):
    suites = (f'flif-v1-ett-{SUITE[v]}',)
    out = ett_recounted(cond(n, v), suites)
    for ds, pl in CELLS:
        rows = [r for s in suites for r in results(os.path.join(FC, s, 'raw.txt'), ds=ds, pl=pl, cond=cond(n, v))]
        for sd, d in by_seed(rows, f'{cond(n, v)} {ds} {pl}').items():
            out[ds, pl, sd].update(rn=num(d, 'rho_neo'), rh=num(d, 'rho_hippo'))
    return out


def mb(n, v):
    rows = results(os.path.join(CL, f'flif-v1-mitbih-{SUITE[v]}', 'raw.txt'), cond=cond(n, v), lr='0.01', val='0')
    return {s: dict(f1=100 * num(d, 'f1'), acc=100 * num(d, 'overall_acc'), tot=num(d, 'total_fft_uJ'), rn=num(d, 'rho_neo'),
                    rh=num(d, 'rho_hippo'), par=int(d['params']), f1s=[100 * num(d, f'f1_{i}') for i in range(5)])
            for s, d in by_seed(rows, f'{cond(n, v)} MIT-BIH').items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'flif-v1-stft-summary.md'))
    a = ap.parse_args()
    E = {(n, v): ett(n, v) for n in NEURONS for v in VARS}
    M = {(n, v): mb(n, v) for n in NEURONS for v in VARS}
    sp_e = ett_recounted('spk_ssa', ('flif-v1-gen-ett',))
    sp_m = {s: dict(f1=100 * num(d, 'f1'), tot=num(d, 'total_fft_uJ'))
            for s, d in by_seed(results(os.path.join(CL, 'flif-v1-gen-mitbih', 'raw.txt'), cond='spk_ssa'), 'spk_ssa MIT-BIH').items()}
    L = ['# STFT test (Neocortex-MS query population LIF / f-LIF; (ga) soma-less tau 2 vs (na) soma tau 4-32) — summary', '',
         'See the module docstring and docs/MODI_PROJECT_LOG.md for the pre-registered rules.', '',
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
            c = compare(E[n, x], E[n, y]); CE[tag, x, y, n] = c['v']
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
            d, m, v = mb_verdict(M[n, x], M[n, y]); CM[tag, x, y, n] = v
            er = np.mean([M[n, x][s]['tot'] for s in SEEDS]) / np.mean([M[n, y][s]['tot'] for s in SEEDS])
            L.append(f"| {tag} {NAMES[n]}: {x} − {y} ({what}) | " + ' / '.join(f'{z:+.2f}' for z in d) + f" | {m:+.2f} | {sum(z > 0 for z in d)}/3 | {v} | {er:.2f} |")
    L += ['', '## conclusions (rule of the pre-registration)', '', '| comparison | task | pop LIF | pop f-LIF | conclusion |', '|---|---|---|---|---|']
    for tag, x, y, what in COMPS[:4]:
        for task, C in (('ETT', CE), ('MIT-BIH', CM)):
            L.append(f"| {tag} {x} − {y} ({what}) | {task} | {C[tag, x, y, 'pop_lif']} | {C[tag, x, y, 'pop_flif']} | {conclusion([C[tag, x, y, n] for n in NEURONS])} |")
    L += ['', '## report only', '', '### firing rates, energy and uncounted costs (mean of seeds; ETT: mean over the 4 cells)', '',
          '| model | variant | ETT rho_neo | ETT rho_hippo | ETT energy uJ (head recount) | ETT MS additions / transform uJ | MIT-BIH rho_neo | MIT-BIH rho_hippo | MIT-BIH energy uJ (logged) | MIT-BIH MS additions / transform uJ | params ETTh2 H96 / MIT-BIH |',
          '|---|---|---|---|---|---|---|---|---|---|---|']
    for n in NEURONS:
        for v in VARS:
            t, m = E[n, v], M[n, v]
            ea = np.mean([uncounted_uJ(ds, v)[0] for ds, pl in CELLS]); ed = np.mean([uncounted_uJ(ds, v)[1] for ds, pl in CELLS])
            ma, md = uncounted_uJ('MITBIH', v)
            L.append(f"| {NAMES[n]} | {v} | {np.mean([t[k]['rn'] for k in t]):.4f} | {np.mean([t[k]['rh'] for k in t]):.4f} | {np.mean([t[k]['tot'] for k in t]):.2f}"
                     f" | {ea:.3f} / {ed:.3f} | {np.mean([m[s]['rn'] for s in SEEDS]):.4f} | {np.mean([m[s]['rh'] for s in SEEDS]):.4f} | {np.mean([m[s]['tot'] for s in SEEDS]):.2f}"
                     f" | {ma:.3f} / {md:.3f} | {t['ETTh2', 96, 42]['par']} / {m[42]['par']} |")
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
