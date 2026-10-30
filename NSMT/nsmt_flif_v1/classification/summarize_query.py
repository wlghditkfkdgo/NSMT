"""MODI 2026-10-30: summary and pre-registered verdicts of the query batch (docs/MODI_PROJECT_LOG.md 2026-10-30) -- the dual-path
model with the Neocortex as the attention QUERY (nsmt4q_<n>: queries = the T = N neocortex output laid out [T_h, B, N, D], keys /
values = the hippocampus tokens; --hippo_xattn_mode query), next to [NSMT hippocampus T=4] (nsmt4_<n>, keys / values replaced by
the neocortex), the B form (spkneo_<n>, keys / values appended) and the Spikformer-style baseline. Same settings as the dual-path
batch (aux loss off, --neo_grad 1.0, neocortex-only neuron swap); n in {lif, flif, pop_lif, pop_flif}; seeds 42 / 7 / 2026.
Sources: CinC fold 0 @90 -- nsmt4q flif-v1-rhythm-query-cinc, nsmt4 flif-v1-rhythm-dual-cinc, spkneo flif-v1-rhythm-addneo90-cinc,
  spk_ssa flif-v1-rhythm-budget-cinc. ETT (user scope for quick checks, 10-30: ETTh2 x H96 / H720) -- nsmt4q flif-v1-ett-query,
  nsmt4 / spkneo flif-v1-ett-dual, spk_ssa flif-v1-gen-ett.
Rules (fail-closed; a missing / duplicated / non-finite value -> incomplete):
  CinC: better = mean A - B >= +0.5 %p and A > B in >= 2 of 3 seeds, worse = mirror; criterion (2) = better, or mean >= -0.5 %p with
      energy <= 0.5x.
  ETT 2 cells: r = 100 (A - B) / B per seed; R = mean of the 2 cell means; better = R <= -1 % and r < 0 in >= 4 of the 6 seed pairs,
      worse = mirror; criterion (2) = better, or R <= +1 % with energy <= 0.5x (T_h-averaged heads recounted with --energy_fix).
Comparisons: Q1 (main) nsmt4q_n - nsmt4_n (query vs replace); Q2 nsmt4q_n - spkneo_n (query vs append); Q3 nsmt4q_n - spk_ssa
  (criterion 2); Q4 T = N memory: nsmt4q_lif - nsmt4q_lif_nomem (main), other neurons - nsmt4q_lif_nomem (report); Q5 (report)
  f-LIF / population LIF / population f-LIF - LIF inside nsmt4q; Q1 (report) the memoryless controls nsmt4q - nsmt4.
usage: python summarize_query.py [--energy_fix]   ->  results/flif-v1-query-summary[-energyfix].md
"""
import argparse
import csv
import os

import numpy as np

from summarize_dual import FC, HERE, NL, NS, SEEDS, fin, index, paired_table, recounted
from summarize_rhythm import parse

CELLS = [('ETTh2', 96), ('ETTh2', 720)]                    # user scope for the quick checks (10-30)


def questions():
    Q = [(f'Q1 query vs replace, {NL[n]} (main)', f'nsmt4q_{n}', f'nsmt4_{n}', False) for n in NS]
    Q += [('Q1 (report) query vs replace, memoryless LIF', 'nsmt4q_lif_nomem', 'nsmt4_lif_nomem', False)]
    Q += [(f'Q2 query vs append (B form), {NL[n]}', f'nsmt4q_{n}', f'spkneo_{n}', False) for n in NS]
    Q += [(f'Q3 query vs Spikformer-style, {NL[n]} (criterion 2)', f'nsmt4q_{n}', 'spk_ssa', True) for n in NS]
    Q += [('Q4 T = N memory vs memoryless, query LIF (main)', 'nsmt4q_lif', 'nsmt4q_lif_nomem', False)]
    Q += [(f'Q4 (report) {NL[n]} vs memoryless LIF, query', f'nsmt4q_{n}', 'nsmt4q_lif_nomem', False) for n in NS[1:]]
    Q += [(f'Q5 (report) {NL[n]} vs LIF, query', f'nsmt4q_{n}', 'nsmt4q_lif', False) for n in NS[1:]]
    return Q


def cinc():
    def f0(s, keep):
        return [r for r in parse(os.path.join(HERE, 'results', s, 'raw.txt'))
                if r.get('data') == 'CINC2017' and int(r.get('crop', 0)) == 0 and int(r.get('fold', 0)) == 0 and keep(r['cond'])]
    rows = f0('flif-v1-rhythm-query-cinc', lambda c: c.startswith('nsmt4q_'))
    rows += f0('flif-v1-rhythm-dual-cinc', lambda c: c.startswith('nsmt4_'))
    rows += f0('flif-v1-rhythm-addneo90-cinc', lambda c: c.startswith('spkneo_'))
    rows += f0('flif-v1-rhythm-budget-cinc', lambda c: c == 'spk_ssa')
    return paired_table('## CinC 2017 fold 0, 90 epochs (test score %, best-validation weights)', rows, 'score', questions())


def ett(energy_fix):
    rows = []
    for s, keep in (('flif-v1-ett-query', lambda c: c.startswith('nsmt4q_')), ('flif-v1-ett-dual', lambda c: c.startswith(('nsmt4_', 'spkneo_'))),
                    ('flif-v1-gen-ett', lambda c: c == 'spk_ssa')):
        rr = [r for r in parse(os.path.join(FC, 'results', s, 'raw.txt')) if keep(r['cond']) and (r['ds'], int(r['pl'])) in CELLS]
        if energy_fix:
            p = os.path.join(FC, 'results', s, 'energy_recount.csv')
            fix = {(f['ds'], int(f['pl']), f['cond'], int(f['seed'])): float(f['total_fft_uJ_corr']) for f in csv.DictReader(open(p))} if os.path.exists(p) else {}
            for r in rr:
                if recounted(r['cond']):
                    r['total_fft_uJ'] = fix.get((r['ds'], int(r['pl']), r['cond'], int(r['seed'])), float('nan'))   # missing -> incomplete
        rows += rr
    get = index(rows, lambda r: (r['ds'], int(r['pl']), r['cond'], int(r['seed'])))
    L = ['## ETT forecasting, ETTh2 x H96 / H720 (test MSE; r = 100 (A − B) / B, %)' + (' — T_h-averaged head energy recounted' if energy_fix else ''), '',
         '| question | A | B | cells b / . / w | R | pairs A < B | verdict | energy A/B | criterion (2) |', '|---|---|---|---|---|---|---|---|---|']
    for q, x, y, crit in questions():
        cm, cv, pr, ers, ok = [], [], [], [], True
        for d, p in CELLS:
            ra = [get((d, p, x, s)) for s in SEEDS]; rb = [get((d, p, y, s)) for s in SEEDS]
            if any(r is None for r in ra + rb) or not all(fin(r['mse']) and fin(r['total_fft_uJ']) for r in ra + rb) or any(r['mse'] <= 0 for r in rb):
                ok = False; break
            r_ = [100 * (a['mse'] - b['mse']) / b['mse'] for a, b in zip(ra, rb)]
            m = float(np.mean(r_)); cm.append(m); pr += r_
            cv.append('b' if m <= -1 and sum(z < 0 for z in r_) >= 2 else 'w' if m >= 1 and sum(z > 0 for z in r_) >= 2 else '.')
            ers.append(np.mean([a['total_fft_uJ'] for a in ra]) / np.mean([b['total_fft_uJ'] for b in rb]))
        if not ok:
            L.append(f'| {q} | {x} | {y} | | | | incomplete | | |'); continue
        R = float(np.mean(cm)); need = 2 * len(pr) // 3; lo, hi = sum(z < 0 for z in pr), sum(z > 0 for z in pr); er = float(np.mean(ers))
        v = 'better' if R <= -1 and lo >= need else 'worse' if R >= 1 and hi >= need else 'no clear difference'
        c2 = ('met' if v == 'better' or (R <= 1 and er <= 0.5) else 'not met') if crit else ''
        L.append(f"| {q} | {x} | {y} | {cv.count('b')} / {cv.count('.')} / {cv.count('w')} | {R:+.2f} | {lo}/{len(pr)} | {v} | {er:.2f} | {c2} |")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--energy_fix', action='store_true')
    a = ap.parse_args()
    L = ['# Query batch (Neocortex as the attention query) — summary', '',
         'See the module docstring for names, sources and rules. Exploratory (one CinC fold, 3 seeds).', '']
    L += cinc() + [''] + ett(a.energy_fix)
    out = os.path.join(HERE, 'results', 'flif-v1-query-summary' + ('-energyfix' if a.energy_fix else '') + '.md')
    open(out, 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
