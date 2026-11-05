"""MODI 2026-11-05 (no training): what the original NSMT paper can claim, measured on the runs we have (user question 11-05;
docs/MODI_PROJECT_LOG.md 2026-11-05). Everything is recomputed from existing RESULT lines; rules as in compare_query_pop_lif_paper.py.
  ETT, same stopping rule (patience 10, best-validation weights): NSMT vs Spikformer-style, NSMT vs hippo-only (no neocortex, T = 24,
      flif-v1-s2p10), hippo-only vs Spikformer-style
  ETT, paper rule (patience 2, last weights, flif-v1-main): NSMT vs hippo-only
  NSMT vs the linear baseline of 10-18 (window normalisation + Linear(96 -> H), results/flif-v1-s4/summary.md, another training script)
  MIT-BIH: NSMT reproduction vs Spikformer-style, hippo-only, no_neo
  Table 7 of the paper recomputed from its own energy columns (the text says 40-55 %, the "Improvement" column is relative to the smaller value)
usage (classification folder): python check_nsmt_claims.py [--out FILE]
"""
import argparse
import os

import numpy as np

from compare_query_pop_lif_paper import CELLS, CL, COLS, FC, HERE, SEEDS, by_seed, cell_vals, cmp_row, compare, ett_recounted, mitbih, results, row

LINEAR_BEST = {('ETTh2', 96): 0.291515, ('ETTh2', 720): 0.420805, ('ETTm2', 96): 0.182854, ('ETTm2', 720): 0.407828}   # as in summarize_head_h2h6.py
TABLE7 = {'ETTh2': (53.32, 37.36, 42.74), 'ETTm1': (67.65, 44.28, 52.79), 'ETTm2': (62.49, 41.36, 51.10)}   # w/o align, w/ align, printed "Improvement"


def ett_cond(suite, cond, keys):
    out = {}
    for ds, pl in CELLS:
        rows = results(os.path.join(FC, suite, 'raw.txt'), cond=cond, dataset=f'{ds}_p{pl}')
        for sd, d in by_seed(rows, f'{suite} {cond} {ds} {pl}').items():
            out[ds, pl, sd] = row(d, *keys, 'params')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'flif-v1-nsmt-claims-check.md'))
    a = ap.parse_args()
    best = ('mse_best', 'mae_best', 'syn_energy_best', 'total_fft_uJ_best'); last = ('mse', 'mae', 'syn_energy', 'total_fft_uJ')
    b, h = ett_cond('flif-v1-s2p10', 'base', best), ett_cond('flif-v1-s2p10', 'hippo_only', best)
    bp, hp = ett_cond('flif-v1-main', 'base', last), ett_cond('flif-v1-main', 'hippo_only', last)
    sp = ett_recounted('spk_ssa', ('flif-v1-gen-ett',))
    L = ['# What the original NSMT paper can claim, on the runs we have', '', 'See the module docstring. r % of test MSE, seeds 42 / 7 / 2026.', '',
         '## ETT', '', '| A vs B | ' + COLS + ' | R % | pairs A lower | verdict | energy A/B |', '|---|---|---|---|---|---|---|---|---|']
    for name, x, y in (('NSMT vs Spikformer-style (same rule)', b, sp), ('NSMT vs hippo-only T = 24 (same rule)', b, h), ('hippo-only T = 24 vs Spikformer-style (same rule)', h, sp),
                       ('NSMT vs hippo-only T = 24 (paper rule)', bp, hp)):
        L.append(cmp_row(name, compare(x, y)))
    L += ['', '## NSMT vs the linear baseline (r % of test MSE, mean of 3 seeds)', '', '| rule | ' + COLS + ' | mean |', '|---|---|---|---|---|---|']
    for name, x in (('same rule', b), ('paper rule', bp)):
        r = [100 * (float(np.mean(cell_vals(x, *c, 'mse'))) / LINEAR_BEST[c] - 1) for c in CELLS]
        L.append(f'| {name} | ' + ' | '.join(f'{v:+.2f}' for v in r) + f' | {np.mean(r):+.2f} |')
    mb = mitbih(os.path.join(CL, 'flif-v1-cls', 'raw.txt'), 'base')
    L += ['', '## MIT-BIH (macro-F1 %p, NSMT reproduction − B, 42 / 7 / 2026)', '', '| B | per seed | mean | verdict | energy NSMT / B |', '|---|---|---|---|---|']
    for name, t in (('Spikformer-style', mitbih(os.path.join(CL, 'flif-v1-gen-mitbih', 'raw.txt'), 'spk_ssa')), ('hippo-only', mitbih(os.path.join(CL, 'flif-v1-cls', 'raw.txt'), 'hippo_only')),
                    ('no_neo (no neocortex, no cross-attention)', mitbih(os.path.join(CL, 'flif-v1-cls', 'raw.txt'), 'no_neo'))):
        d = [mb[s]['f1'] - t[s]['f1'] for s in SEEDS]; m = float(np.mean(d))
        v = 'better' if m >= 0.5 and sum(x > 0 for x in d) >= 2 else 'worse' if m <= -0.5 and sum(x < 0 for x in d) >= 2 else 'no clear difference'
        L.append(f'| {name} | ' + ' / '.join(f'{x:+.2f}' for x in d) + f" | {m:+.2f} | {v} | {np.mean([mb[s]['tot'] for s in SEEDS]) / np.mean([t[s]['tot'] for s in SEEDS]):.2f} |")
    L += ['', '## Table 7 of the paper, recomputed from its own energy columns', '', '| dataset | w/o align uJ | w/ align uJ | w/ / w/o | real reduction % | (w/o − w/) / w/ % | printed "Improvement" % |', '|---|---|---|---|---|---|---|']
    for ds, (e, f, p) in TABLE7.items():
        L.append(f'| {ds} | {e:.2f} | {f:.2f} | {f / e:.3f} | {100 * (1 - f / e):.1f} | {100 * (e - f) / f:.2f} | {p:.2f} |')
    L.append('')
    with open(a.out, 'w') as f:
        f.write('\n'.join(L))
    print('\n'.join(L))


if __name__ == '__main__':
    main()
