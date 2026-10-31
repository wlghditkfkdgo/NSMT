"""MODI 2026-10-31: summary and pre-registered verdicts of the CinC follow-up (docs/MODI_PROJECT_LOG.md 2026-10-31; user decisions
after the query batch and tuning stage 3): (A) the query form (nsmt4q_pop_lif / nsmt4q_pop_flif, Neocortex as the attention query)
on CinC folds 1-4, (B) its population memoryless controls on fold 0, (C) the Neo-mixer with LIF in every layer, spikeDE reset
default vs zero, on fold 0 (the tuning-stage-3 positive setting). All CinC, 90 epochs, best-validation weights, seeds 42 / 7 / 2026.
Sources: query fold 0 flif-v1-rhythm-query-cinc, folds 1-4 flif-v1-rhythm-query-cv-cinc; Spikformer-style @30 fold 0
flif-v1-rhythm-cinc (stage 9), folds 1-4 flif-v1-rhythm-folds-cinc (its better budget on fold 0: 65.60 @30 vs 64.89 @90); Spikformer-
style @90 fold 0 flif-v1-rhythm-budget-cinc; follow-up runs (B, C) flif-v1-rhythm-followup-cinc.
Rules (fail-closed; a missing / duplicated / non-finite value -> incomplete):
  per comparison on one fold: better = mean A - B >= +0.5 %p and A > B in >= 2 of 3 seeds, worse = mirror.
  A (main) 5 folds: the 15 (fold, seed) pairs of query - Spikformer-style @30; better = pooled mean >= +0.5 %p and A > B in >= 10 of
      15 pairs, worse = mirror; reported with the 95 % t interval (14 degrees of freedom) and the per-fold verdicts.
  B: nsmt4q_pop_n - nsmt4q_pop_n_nomem (main: T = N memory for the population neurons); report: nsmt4q_pop_n_nomem - nsmt4q_lif_nomem.
  C: mixall_lif_bidir_rzero - mixall_lif_bidir (main); report: each - Spikformer-style @90 / @30, and - the mixer-only Neo-mixer LIF @90.
usage: python summarize_followup.py   ->  results/flif-v1-followup-summary.md
"""
import math
import os

import numpy as np

from summarize_rhythm import parse

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = (42, 7, 2026)
T14 = 2.1447866879                                         # t_{0.975, 14}


def fin(x):
    return isinstance(x, float) and math.isfinite(x)


def rows(suite, keep, fold=None):
    out = []
    for r in parse(os.path.join(HERE, 'results', suite, 'raw.txt')):
        if r.get('data') == 'CINC2017' and int(r.get('crop', 0)) == 0 and keep(r['cond']) and (fold is None or int(r.get('fold', 0)) == fold):
            out.append(r)
    return out


def index(rr, tag):
    by = {}
    for r in rr:
        by.setdefault((tag, int(r.get('fold', 0)), int(r['seed'])), []).append(r)
    return by


def get(by, tag, fold, seed):
    v = by.get((tag, fold, seed), [])
    return v[0] if len(v) == 1 and fin(v[0]['score']) else None


def verdict(d, n_need):
    m = float(np.mean(d))
    return m, ('better' if m >= 0.5 and sum(z > 0 for z in d) >= n_need else 'worse' if m <= -0.5 and sum(z < 0 for z in d) >= n_need else 'no clear difference')


def main():
    by = {}
    def add(tag, suite, cond, fold=None):
        for k, v in index(rows(suite, lambda c: c == cond, fold), tag).items():
            by.setdefault(k, []).extend(v)
    for n in ('pop_lif', 'pop_flif'):
        add(f'q_{n}', 'flif-v1-rhythm-query-cinc', f'nsmt4q_{n}', 0)
        add(f'q_{n}', 'flif-v1-rhythm-query-cv-cinc', f'nsmt4q_{n}')
        add(f'qnomem_{n}', 'flif-v1-rhythm-followup-cinc', f'nsmt4q_{n}_nomem', 0)
    add('q_lif_nomem', 'flif-v1-rhythm-query-cinc', 'nsmt4q_lif_nomem', 0)
    add('spk30', 'flif-v1-rhythm-cinc', 'spk_ssa', 0)
    add('spk30', 'flif-v1-rhythm-folds-cinc', 'spk_ssa')
    add('spk90', 'flif-v1-rhythm-budget-cinc', 'spk_ssa', 0)
    add('mix90', 'flif-v1-rhythm-budget-cinc', 'mix_lif_bidir', 0)
    add('mixall', 'flif-v1-rhythm-followup-cinc', 'mixall_lif_bidir', 0)
    add('mixall_rzero', 'flif-v1-rhythm-followup-cinc', 'mixall_lif_bidir_rzero', 0)
    L = ['# CinC follow-up (query folds 1-4, population memoryless controls, Neo-mixer reset) — summary', '',
         'See the module docstring for sources and rules. Test CinC score (%), best-validation weights.', '']
    # A: 5-fold query vs Spikformer-style @30
    L += ['## A. Query form vs Spikformer-style @30 over folds 0-4 (15 fold x seed pairs)', '',
          '| model | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 | 5-fold mean | pooled A − B | 95 % t interval | A > B pairs | verdict |',
          '|---|---|---|---|---|---|---|---|---|---|---|']
    def foldmean(tag, f):
        v = [get(by, tag, f, s) for s in SEEDS]
        return None if any(x is None for x in v) else 100 * float(np.mean([x['score'] for x in v]))
    sm = [foldmean('spk30', f) for f in range(5)]
    L.append('| Spikformer-style @30 | ' + ' | '.join('incomplete' if m is None else f'{m:.2f}' for m in sm)
             + (f' | {np.mean(sm):.2f} | | | | |' if all(m is not None for m in sm) else ' | incomplete | | | | |'))
    for n in ('pop_lif', 'pop_flif'):
        tag = f'q_{n}'; fm = [foldmean(tag, f) for f in range(5)]
        pairs = [(get(by, tag, f, s), get(by, 'spk30', f, s)) for f in range(5) for s in SEEDS]
        if any(a is None or b is None for a, b in pairs):
            L.append(f'| query {n} | ' + ' | '.join('incomplete' if m is None else f'{m:.2f}' for m in fm) + ' | incomplete | | | | incomplete |'); continue
        d = [100 * (a['score'] - b['score']) for a, b in pairs]; m, v = verdict(d, 10)
        h = T14 * float(np.std(d, ddof=1)) / math.sqrt(len(d))
        L.append(f'| query {n} | ' + ' | '.join(f'{x:.2f}' for x in fm) + f' | {np.mean(fm):.2f} | {m:+.2f} | [{m - h:+.2f}, {m + h:+.2f}] | '
                 f'{sum(z > 0 for z in d)}/15 | {v} |')
    L += ['', '### per-fold verdicts (query − Spikformer-style @30; >= +0.5 %p and 2 of 3 seeds)', '', '| model | ' + ' | '.join(f'fold {f}' for f in range(5)) + ' |',
          '|---|' + '---|' * 5]
    for n in ('pop_lif', 'pop_flif'):
        cells = []
        for f in range(5):
            pr = [(get(by, f'q_{n}', f, s), get(by, 'spk30', f, s)) for s in SEEDS]
            if any(a is None or b is None for a, b in pr):
                cells.append('incomplete'); continue
            m, v = verdict([100 * (a['score'] - b['score']) for a, b in pr], 2); cells.append(f'{m:+.2f} {v}')
        L.append(f'| query {n} | ' + ' | '.join(cells) + ' |')
    # B, C: fold-0 paired comparisons
    def fold0(title, Q):
        out = ['', title, '', '| comparison | A (mean) | B (mean) | A − B per seed (%p) | mean | verdict | energy A/B |', '|---|---|---|---|---|---|---|']
        for q, x, y in Q:
            a = [get(by, x, 0, s) for s in SEEDS]; b = [get(by, y, 0, s) for s in SEEDS]
            if any(r is None for r in a + b):
                out.append(f'| {q} | | | | | incomplete | |'); continue
            d = [100 * (p['score'] - o['score']) for p, o in zip(a, b)]; m, v = verdict(d, 2)
            er = float(np.mean([p['total_fft_uJ'] for p in a]) / np.mean([o['total_fft_uJ'] for o in b]))
            out.append(f"| {q} | {100 * np.mean([p['score'] for p in a]):.2f} | {100 * np.mean([o['score'] for o in b]):.2f} | "
                       f"{' / '.join(f'{z:+.2f}' for z in d)} | {m:+.2f} | {v} | {er:.2f} |")
        return out
    L += fold0('## B. T = N memory for the population neurons in the query form (fold 0)',
               [('B (main) query population LIF − its memoryless control', 'q_pop_lif', 'qnomem_pop_lif'),
                ('B (main) query population f-LIF − its memoryless control', 'q_pop_flif', 'qnomem_pop_flif'),
                ('B (report) memoryless population LIF − memoryless LIF', 'qnomem_pop_lif', 'q_lif_nomem'),
                ('B (report) memoryless population f-LIF − memoryless LIF', 'qnomem_pop_flif', 'q_lif_nomem')])
    L += fold0('## C. Neo-mixer with LIF in every layer: spikeDE reset zero vs default (fold 0)',
               [('C (main) zero − default reset', 'mixall_rzero', 'mixall'),
                ('C (report) default reset − Spikformer-style @90', 'mixall', 'spk90'), ('C (report) zero reset − Spikformer-style @90', 'mixall_rzero', 'spk90'),
                ('C (report) default reset − Spikformer-style @30', 'mixall', 'spk30'), ('C (report) zero reset − Spikformer-style @30', 'mixall_rzero', 'spk30'),
                ('C (report) zero reset − mixer-only Neo-mixer LIF @90', 'mixall_rzero', 'mix90')])
    out = os.path.join(HERE, 'results', 'flif-v1-followup-summary.md')
    open(out, 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    main()
