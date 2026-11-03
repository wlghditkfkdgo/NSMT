"""MODI 2026-11-02: the T = N memory of the query form with population neurons on CinC folds 0-4, split into its order and
order-free parts (docs/MODI_PROJECT_LOG.md 2026-11-02; user decision "1. 나머지 fold, 2. 순서의 영향"). CinC, 90 epochs, lr 1e-3,
best-validation weights, seeds 42 / 7 / 2026; n in {pop_lif, pop_flif}.
Models: ordered = nsmt4q_pop_n (Neocortex runs the tokens in time order; fold 0 flif-v1-rhythm-query-cinc, folds 1-4
flif-v1-rhythm-query-cv-cinc); memoryless = nsmt4q_pop_n_nomem (every token alone, 1 step; fold 0 flif-v1-rhythm-followup-cinc,
folds 1-4 flif-v1-rhythm-order-cinc); shuffled = nsmt4q_pop_n_nshuf (--neo_shuffle: the Neocortex runs the tokens in a shuffled
order -- per-sample random in training, one fixed permutation in evaluation -- and its outputs go back to the original token
positions; the hippocampus keeps the true order; folds 0-4 flif-v1-rhythm-order-cinc). Spikformer-style @30 (report): fold 0
flif-v1-rhythm-cinc, folds 1-4 flif-v1-rhythm-folds-cinc.
Rules (fail-closed; a missing / duplicated / non-finite value -> incomplete), on the 15 (fold, seed) pairs of A - B:
  better = mean >= +0.5 %p and A > B in >= 10 of 15 pairs, worse = mirror, else no clear difference; 95 % t interval (14 df);
  per-fold verdicts: mean >= +0.5 %p and 2 of 3 seeds.
  A (main) memory share  = ordered - memoryless   (better = 'memory helps')
  B (main) order share   = ordered - shuffled     (better = 'order helps'; worse = 'shuffled is better')
  C order-free share     = shuffled - memoryless  (better = 'state without order helps')
  report: shuffled / memoryless - Spikformer-style @30, energy A/B.
usage: python summarize_order.py   ->  results/flif-v1-order-summary.md
"""
import math
import os

import numpy as np

from summarize_rhythm import parse

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = (42, 7, 2026)
FOLDS = range(5)
T14 = 2.1447866879                                         # t_{0.975, 14}


def fin(x):
    return isinstance(x, float) and math.isfinite(x)


class Table:
    def __init__(self):
        self.by = {}

    def add(self, tag, suite, cond, folds):
        p = os.path.join(HERE, 'results', suite, 'raw.txt')
        if not os.path.exists(p):
            return
        for r in parse(p):
            f = int(r.get('fold', 0))
            if r.get('data') == 'CINC2017' and int(r.get('crop', 0)) == 0 and r['cond'] == cond and f in folds:
                self.by.setdefault((tag, f, int(r['seed'])), []).append(r)

    def get(self, tag, f, s):
        v = self.by.get((tag, f, s), [])
        return v[0] if len(v) == 1 and fin(v[0]['score']) else None


def verdict(d, need):
    m = float(np.mean(d))
    return m, ('better' if m >= 0.5 and sum(z > 0 for z in d) >= need else 'worse' if m <= -0.5 and sum(z < 0 for z in d) >= need else 'no clear difference')


def main():
    T = Table()
    for n in ('pop_lif', 'pop_flif'):
        T.add(f'ord_{n}', 'flif-v1-rhythm-query-cinc', f'nsmt4q_{n}', {0})
        T.add(f'ord_{n}', 'flif-v1-rhythm-query-cv-cinc', f'nsmt4q_{n}', {1, 2, 3, 4})
        T.add(f'nom_{n}', 'flif-v1-rhythm-followup-cinc', f'nsmt4q_{n}_nomem', {0})
        T.add(f'nom_{n}', 'flif-v1-rhythm-order-cinc', f'nsmt4q_{n}_nomem', {1, 2, 3, 4})
        T.add(f'shf_{n}', 'flif-v1-rhythm-order-cinc', f'nsmt4q_{n}_nshuf', set(FOLDS))
    T.add('spk30', 'flif-v1-rhythm-cinc', 'spk_ssa', {0})
    T.add('spk30', 'flif-v1-rhythm-folds-cinc', 'spk_ssa', {1, 2, 3, 4})
    names = {'ord': 'ordered', 'nom': 'memoryless', 'shf': 'Neocortex-shuffled'}
    L = ['# Query form, population neurons: T = N memory and token order on CinC folds 0-4 — summary', '',
         'See the module docstring for sources and rules. Test CinC score (%), best-validation weights.', '',
         '## fold means (seeds 42 / 7 / 2026)', '', '| model | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 | 5-fold mean | energy µJ |', '|---|---|---|---|---|---|---|---|']
    tags = [('spk30', 'Spikformer-style @30')] + [(f'{k}_{n}', f'query {n} {v}') for n in ('pop_lif', 'pop_flif') for k, v in names.items()]
    for tag, name in tags:
        fm, en = [], []
        for f in FOLDS:
            v = [T.get(tag, f, s) for s in SEEDS]
            fm.append(None if any(x is None for x in v) else 100 * float(np.mean([x['score'] for x in v])))
            if all(x is not None for x in v):
                en += [x['total_fft_uJ'] for x in v]
        L.append(f'| {name} | ' + ' | '.join('incomplete' if m is None else f'{m:.2f}' for m in fm)
                 + (f' | {np.mean(fm):.2f}' if all(m is not None for m in fm) else ' | incomplete') + (f' | {np.mean(en):.1f} |' if en else ' | |'))
    def block(title, rows):
        out = ['', title, '', '| comparison | pooled A − B (%p) | 95 % t interval | A > B pairs | verdict | per fold (mean, verdict) | energy A/B |',
               '|---|---|---|---|---|---|---|']
        for q, a, b in rows:
            pairs = [(T.get(a, f, s), T.get(b, f, s)) for f in FOLDS for s in SEEDS]
            if any(x is None or y is None for x, y in pairs):
                done = sum(x is not None and y is not None for x, y in pairs)
                out.append(f'| {q} | incomplete ({done}/15 pairs) | | | incomplete | | |'); continue
            d = [100 * (x['score'] - y['score']) for x, y in pairs]; m, v = verdict(d, 10)
            h = T14 * float(np.std(d, ddof=1)) / math.sqrt(len(d))
            pf = []
            for i, f in enumerate(FOLDS):
                mf, vf = verdict(d[3 * i:3 * i + 3], 2); pf.append(f'f{f} {mf:+.2f} {vf}')
            er = float(np.mean([x['total_fft_uJ'] for x, _ in pairs]) / np.mean([y['total_fft_uJ'] for _, y in pairs]))
            out.append(f'| {q} | {m:+.2f} | [{m - h:+.2f}, {m + h:+.2f}] | {sum(z > 0 for z in d)}/15 | {v} | {"; ".join(pf)} | {er:.2f} |')
        return out
    L += block('## A (main) memory share = ordered − memoryless', [(f'A {n}', f'ord_{n}', f'nom_{n}') for n in ('pop_lif', 'pop_flif')])
    L += block('## B (main) order share = ordered − Neocortex-shuffled', [(f'B {n}', f'ord_{n}', f'shf_{n}') for n in ('pop_lif', 'pop_flif')])
    L += block('## C order-free share = Neocortex-shuffled − memoryless', [(f'C {n}', f'shf_{n}', f'nom_{n}') for n in ('pop_lif', 'pop_flif')])
    L += block('## report: − Spikformer-style @30', [(f'{names[k]} {n} − Spikformer-style @30', f'{k}_{n}', 'spk30')
                                                      for n in ('pop_lif', 'pop_flif') for k in ('shf', 'nom')])
    # MODI 2026-11-03: report-only block added after the results (not pre-registered; the sections above are unchanged)
    cls = ['macro_f1', 'f1_0', 'f1_1', 'f1_2', 'f1_3']
    def cell(tag, k):
        v = [T.get(tag, f, s) for f in FOLDS for s in SEEDS]
        v = [x.get(k) for x in v if x is not None]
        return float(np.mean(v)) if len(v) == 15 and all(fin(z) for z in v) else None
    L += ['', '## report only (added after the results): per-class test F1 (%) and firing rates, 5-fold × 3-seed means', '',
          'CinC score = mean F1 of N / AF / Other (Noisy excluded); macro-F1 includes Noisy. rho = mean firing rate.', '',
          '| model | macro-F1 | F1 N | F1 AF | F1 Other | F1 Noisy | rho_neo | rho_hippo |', '|---|---|---|---|---|---|---|---|']
    for tag, name in tags:
        c = [cell(tag, k) for k in cls] + [cell(tag, 'rho_neo'), cell(tag, 'rho_hippo')]
        L.append(f'| {name} | ' + ' | '.join('—' if z is None else (f'{100 * z:.2f}' if i < 5 else f'{z:.4f}') for i, z in enumerate(c)) + ' |')
    L += ['', '| difference (%p, 15-pair mean) | macro-F1 | F1 N | F1 AF | F1 Other | F1 Noisy |', '|---|---|---|---|---|---|']
    for q, a, b in [(f'{q} {n}', f'{x}_{n}', f'{y}_{n}') for n in ('pop_lif', 'pop_flif')
                    for q, x, y in (('A', 'ord', 'nom'), ('B', 'ord', 'shf'), ('C', 'shf', 'nom'))]:
        c = [cell(a, k) for k in cls], [cell(b, k) for k in cls]
        L.append(f'| {q} | ' + ' | '.join('—' if x is None or y is None else f'{100 * (x - y):+.2f}' for x, y in zip(*c)) + ' |')
    out = os.path.join(HERE, 'results', 'flif-v1-order-summary.md')
    open(out, 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    main()
