"""MODI 2026-11-16: summary of the input-length test (docs/MODI_PROJECT_LOG.md, pre-registered rules; user decision 11-16).

Question: does the gap between the base model (Neocortex-MS query population, nsmt4q_pop_<lif|flif>_msn) and Spikformer-style
(spk_ssa) grow when the input window, and with it the number of tokens the T = N Neocortex reads, grows?  Input length L = 96 (24
tokens, every earlier run), 336 (84 tokens), 720 (180 tokens); ETTh2 / ETTm2 x H96 / H720, seeds 42 / 7 / 2026; the validation and
test targets are the same at every length (check_len.py c60c). Controls of the base model: nomem (--neo_memoryless: every token goes
through the Neocortex alone, no state crosses tokens) and nshuf (--neo_shuffle: the Neocortex reads the tokens in a shuffled order).
Suites: L = 96: flif-v1-gen-ett (spk_ssa), flif-v1-ett-ms (msn), flif-v1-ett-len96ctl (controls); L = 336 / 720: flif-v1-ett-len<L>.
Comparisons (rules = summarize_ms.py / compare(): r = A / B - 1 of test MSE over the 12 cell x seed pairs; better = mean <= -1 % and
>= 8/12 lower, worse = mirror; per cell +-1 % and 2/3; conclusion over the two neurons):
  L1 base - Spikformer-style at each length                         (the main comparison)
  L2 change of that gap with the length: g = r_L - r_96 per pair (%p); grew = mean <= -1 and >= 8/12 negative, shrank = mirror
  L3 base - nshuf (order), base - nomem (memory), at each length; report only: nshuf - nomem
  L4 (report only) every model against itself at the shorter length (same targets): does a longer input help it at all
Report only: MAE, mean validation L1, firing rates, energy (head recount) and its ratio to Spikformer-style, parameters, run time.
A missing, duplicated or non-finite value raises (fail-closed); rows of the new suites must carry the expected seq_len.
usage (classification folder): python summarize_len.py [--lens 336 | 336,720] [--out FILE]
"""
import argparse
import os

import numpy as np

from compare_query_pop_lif_paper import CELLS, COLS, FC, HERE, SEEDS, by_seed, cmp_row, compare, ett_recounted, num, results
from summarize_ms import NAMES, NEURONS, conclusion

VARS = ('msn', 'nomem', 'nshuf')
VNAME = {'msn': 'base (msn)', 'nomem': 'nomem (no state across tokens)', 'nshuf': 'nshuf (shuffled token order)'}
TOKENS = {96: 24, 336: 84, 720: 180}


def cond(n, v):
    return 'spk_ssa' if n == 'spk' else (f'nsmt4q_{n}_msn' if v == 'msn' else f'nsmt4q_{n}_msn_{v}')


def suite(n, v, L):
    if L != 96:
        return f'flif-v1-ett-len{L}'
    return 'flif-v1-gen-ett' if n == 'spk' else ('flif-v1-ett-ms' if v == 'msn' else 'flif-v1-ett-len96ctl')


def load(n, v, L):
    c, s = cond(n, v), suite(n, v, L)
    out = ett_recounted(c, (s,))
    new = s.startswith('flif-v1-ett-len')
    for ds, pl in CELLS:
        rows = results(os.path.join(FC, s, 'raw.txt'), ds=ds, pl=pl, cond=c)
        for sd, d in by_seed(rows, f'{c} {ds} {pl} L={L}').items():
            if new and int(d.get('seq_len', -1)) != L:
                raise RuntimeError(f'{s} {c} {ds} {pl} seed {sd}: seq_len {d.get("seq_len")} != {L}')
            if (not new) and 'seq_len' in d and int(d['seq_len']) != 96:
                raise RuntimeError(f'{s} {c}: unexpected seq_len {d["seq_len"]}')
            out[ds, pl, sd].update(rh=num(d, 'rho_hippo'), time=num(d, 'time_s'))
            out[ds, pl, sd]['rn'] = num(d, 'rho_neo') if n != 'spk' else float('nan')
    return out


def gap_change(cL, c96):
    """g = r_L - r_96 per cell and seed (%p of the base-vs-Spikformer-style ratio); the 12-pair verdict."""
    per = {c: [cL['per'][c][i] - c96['per'][c][i] for i in range(len(SEEDS))] for c in CELLS}
    allg = [x for c in CELLS for x in per[c]]; G = float(np.mean(allg)); neg = sum(x < 0 for x in allg); pos = sum(x > 0 for x in allg)
    v = 'grew' if G <= -1 and neg >= 8 else 'shrank' if G >= 1 and pos >= 8 else 'no clear change'
    return dict(per=per, G=G, neg=neg, v=v)


def gap_conclusion(vs):
    if all(x == 'grew' for x in vs):
        return 'grew'
    if all(x == 'shrank' for x in vs):
        return 'shrank'
    if any(x != 'no clear change' for x in vs):
        return 'neuron-dependent (weak)'
    return 'no clear change'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lens', default='336')
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    new = tuple(int(x) for x in a.lens.split(','))
    lens = (96,) + new
    out = a.out or os.path.join(HERE, 'results', f'flif-v1-len{new[-1]}-summary.md')
    SP = {L: load('spk', 'msn', L) for L in lens}
    E = {(n, v, L): load(n, v, L) for n in NEURONS for v in VARS for L in lens}
    models = [('Spikformer-style', lambda L: SP[L])] + [(f'{NAMES[n]} {VNAME[v]}', (lambda L, n=n, v=v: E[n, v, L])) for n in NEURONS for v in VARS]
    L_ = ['# Input-length test (Neocortex-MS query population LIF / f-LIF vs Spikformer-style; input 96 / ' + ' / '.join(map(str, new)) + ') — summary',
          '', 'See the module docstring and docs/MODI_PROJECT_LOG.md for the pre-registered rules. Tokens: '
          + ', '.join(f'L {L} = {TOKENS[L]}' for L in lens) + '.', '',
          '## ETT — test MSE (mean ± sd of seeds 42 / 7 / 2026, best-validation weights)', '',
          '| model | input | ' + COLS + ' | mean validation L1 |', '|---|---|---|---|---|---|---|']
    for name, get in models:
        for L in lens:
            t = get(L)
            cells = ' | '.join(f"{np.mean([t[ds, pl, s]['mse'] for s in SEEDS]):.4f} ± {np.std([t[ds, pl, s]['mse'] for s in SEEDS], ddof=1):.4f}"
                               for ds, pl in CELLS)
            L_.append(f"| {name} | {L} | {cells} | {np.mean([t[k]['val'] for k in t]):.4f} |")
    head = ['| comparison | ' + COLS + ' | R % | pairs A lower | verdict | energy A/B |', '|---|---|---|---|---|---|---|---|---|']
    L_ += ['', '## L1 — base − Spikformer-style at each input length (r = A / B − 1 of test MSE, %)', ''] + head
    C1 = {}
    for n in NEURONS:
        for L in lens:
            C1[n, L] = compare(E[n, 'msn', L], SP[L])
            L_.append(cmp_row(f'L1 {NAMES[n]} − Spikformer-style, input {L}', C1[n, L]))
    L_ += ['', '## L2 — change of the gap with the input length (g = r_L − r_shorter per cell and seed, %p; negative = the base gains more)', '',
           '| comparison | ' + COLS + ' | mean g | pairs negative | verdict |', '|---|---|---|---|---|---|---|---|']
    C2 = {}
    pairs = [(L, 96) for L in new] + ([(new[-1], new[0])] if len(new) > 1 else [])
    for n in NEURONS:
        for L, L0 in pairs:
            g = gap_change(C1[n, L], C1[n, L0]); C2[n, L, L0] = g['v']
            L_.append(f"| L2 {NAMES[n]}: gap at {L} − gap at {L0} | " + ' | '.join(f"{np.mean(g['per'][c]):+.2f}" for c in CELLS)
                      + f" | {g['G']:+.2f} | {g['neg']}/12 | {g['v']} |")
    L_ += ['', '## L3 — controls of the base model at each input length (r = A / B − 1 of test MSE, %)', ''] + head
    C3 = {}
    for x, y, what, tag in (('msn', 'nshuf', 'order', 'L3'), ('msn', 'nomem', 'memory', 'L3'), ('nshuf', 'nomem', 'report only', 'report')):
        for n in NEURONS:
            for L in lens:
                c = compare(E[n, x, L], E[n, y, L]); C3[x, y, n, L] = c['v']
                L_.append(cmp_row(f'{tag} {NAMES[n]}: {x} − {y} ({what}), input {L}', c))
    L_ += ['', '## L4 (report only) — every model against itself at the shorter input (same test targets; r = A / B − 1, %)', ''] + head
    for name, get in models:
        for L, L0 in pairs:
            L_.append(cmp_row(f'{name}: input {L} − input {L0}', compare(get(L), get(L0))))
    L_ += ['', '## conclusions (rule of the pre-registration)', '', '| comparison | pop LIF | pop f-LIF | conclusion |', '|---|---|---|---|']
    for L in new:
        L_.append(f"| L1 base − Spikformer-style, input {L} | {C1['pop_lif', L]['v']} | {C1['pop_flif', L]['v']} | {conclusion([C1[n, L]['v'] for n in NEURONS])} |")
    for L, L0 in pairs:
        L_.append(f"| L2 gap at {L} − gap at {L0} | {C2['pop_lif', L, L0]} | {C2['pop_flif', L, L0]} | {gap_conclusion([C2[n, L, L0] for n in NEURONS])} |")
    for x, y, what in (('msn', 'nshuf', 'order'), ('msn', 'nomem', 'memory')):
        for L in lens:
            L_.append(f"| L3 {x} − {y} ({what}), input {L} | {C3[x, y, 'pop_lif', L]} | {C3[x, y, 'pop_flif', L]} | {conclusion([C3[x, y, n, L] for n in NEURONS])} |")
    L_ += ['', '## report only', '', '### firing rates, energy, parameters, run time (mean of seeds)', '',
           '| model | input | rho_neo (4 cells) | rho_neo mean | rho_hippo mean | energy uJ (head recount, mean of 4 cells) | energy / Spikformer-style (mean of per-cell ratios) | params ETTh2 H96 / ETTm2 H720 | run time s (min–max) |',
           '|---|---|---|---|---|---|---|---|---|']
    for name, get in models:
        for L in lens:
            t = get(L)
            rn = [np.mean([t[ds, pl, s]['rn'] for s in SEEDS]) for ds, pl in CELLS]
            er = np.mean([np.mean([t[ds, pl, s]['tot'] for s in SEEDS]) / np.mean([SP[L][ds, pl, s]['tot'] for s in SEEDS]) for ds, pl in CELLS])
            tm = [t[k]['time'] for k in t]
            L_.append(f"| {name} | {L} | " + (' / '.join(f'{x:.4f}' for x in rn) if not np.isnan(rn[0]) else '—') + ' | '
                      + (f'{np.mean(rn):.4f}' if not np.isnan(rn[0]) else '—') + f" | {np.mean([t[k]['rh'] for k in t]):.4f} | {np.mean([t[k]['tot'] for k in t]):.2f}"
                      f" | {er:.2f} | {t['ETTh2', 96, 42]['par']} / {t['ETTm2', 720, 42]['par']} | {min(tm):.0f}–{max(tm):.0f} |")
    L_ += ['', '### energy by cell (uJ, head recount, mean of seeds)', '', '| model | input | ' + COLS + ' |', '|---|---|---|---|---|---|']
    for name, get in models:
        for L in lens:
            L_.append(f"| {name} | {L} | " + ' | '.join(f"{np.mean([get(L)[ds, pl, s]['tot'] for s in SEEDS]):.2f}" for ds, pl in CELLS) + ' |')
    L_ += ['', '### test MAE (mean of seeds)', '', '| model | input | ' + COLS + ' |', '|---|---|---|---|---|---|']
    for name, get in models:
        for L in lens:
            L_.append(f"| {name} | {L} | " + ' | '.join(f"{np.mean([get(L)[ds, pl, s]['mae'] for s in SEEDS]):.4f}" for ds, pl in CELLS) + ' |')
    open(out, 'w').write('\n'.join(L_) + '\n')
    print(out)


if __name__ == '__main__':
    main()
