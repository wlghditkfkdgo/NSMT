"""MODI 2026-10-29: summary and pre-registered verdicts of the dual-path batch (docs/MODI_PROJECT_LOG.md 2026-10-29) -- the
[NSMT hippocampus T=4] model back in the comparisons (user, 10-29) next to the B form, on CinC, MIT-BIH and ETT.

Models (dual path = hippocampus T_h = 4 + T = N neocortex; the neuron condition swaps ONLY the neocortex neuron, user rule of
10-29 17:1x; B-form settings for both: aux loss off, --neo_grad 1.0):
  nsmt4_<n>   [NSMT hippocampus T=4]: the hippocampus attention's keys / values are REPLACED by the neocortex tokens (mode replace)
  spkneo_<n>  B form: keys / values = [own tokens; neocortex tokens] (mode cat)
  *_lif_nomem the same model with a memoryless LIF neocortex (every token alone, 1 step)
  spk_ssa     Spikformer-style baseline (spikingjelly LIF); mixall_<n>_bidir = Neo-mixer with n in every layer (report)
  n in {lif, flif, pop_lif, pop_flif}; seeds 42 / 7 / 2026.
Sources: CinC fold 0 (best-validation weights, test score): nsmt4_* @90 flif-v1-rhythm-dual-cinc, spkneo_* @90
  flif-v1-rhythm-addneo90-cinc, spk_ssa @90 flif-v1-rhythm-budget-cinc, old nsmt_*_t4 @30 (aux 0.1, neo_grad 0.5)
  flif-v1-rhythm-fix-cinc. MIT-BIH (last-epoch weights, test macro-F1): flif-v1-mitbih-dual, spk_ssa flif-v1-gen-mitbih,
  mixall flif-v1-mitbih-nall. ETT (best-validation weights, test MSE; ETTh2 + ETTm2 x H96 / H720, user decision 12 of 10-14):
  flif-v1-ett-dual, spk_ssa flif-v1-gen-ett, mixall flif-v1-ett-nall.
Rules (= the generalization batch; paired by seed; a missing / duplicated / non-finite value -> incomplete):
  CinC, MIT-BIH: better = mean A - B >= +0.5 %p and A > B in >= 2 of 3 seeds, worse = mirror; criterion (2) = better, or
      mean >= -0.5 %p with energy <= 0.5x.
  ETT: r = 100 (A - B) / B per seed; cell better = mean r <= -1 % and r < 0 in >= 2 of 3 seeds; overall R = mean of the 4 cell
      means; better = R <= -1 % and r < 0 in >= 8 of 12 seed pairs, worse = mirror; criterion (2) = better, or R <= +1 % with
      energy <= 0.5x (mean over cells of the cell-mean total_fft_uJ ratio; T_h-averaged heads recounted with --energy_fix).
Comparisons: R1 (main) nsmt4_n - spkneo_n (replace vs append); R2 nsmt4_n - spk_ssa and spkneo_n - spk_ssa (criterion 2;
  CinC spkneo = the earlier D1); R3 memory nsmt4_lif - nsmt4_lif_nomem and spkneo_lif - spkneo_lif_nomem (CinC spkneo = the
  earlier D2); R4 (report) within each family f-LIF / population LIF / population f-LIF - LIF; R5 (report, MIT-BIH / ETT)
  nsmt4_n - mixall_n and spkneo_n - mixall_n; R6 (report, CinC) nsmt4 @90 - the old nsmt_*_t4 @30 (settings and budget differ).
usage: python summarize_dual.py [--energy_fix]   ->  results/flif-v1-dual-summary[-energyfix].md
"""
import argparse
import csv
import math
import os

import numpy as np

from summarize_rhythm import parse

HERE = os.path.dirname(os.path.abspath(__file__))
FC = os.path.join(HERE, '..', 'forecasting')
SEEDS = (42, 7, 2026)
NS = ('lif', 'flif', 'pop_lif', 'pop_flif')
NL = {'lif': 'LIF', 'flif': 'f-LIF', 'pop_lif': 'pop LIF', 'pop_flif': 'pop f-LIF'}
CELLS = [('ETTh2', 96), ('ETTh2', 720), ('ETTm2', 96), ('ETTm2', 720)]   # user decision 12 (10-14)


def fin(x):
    return isinstance(x, float) and math.isfinite(x)


def recounted(cond):                                    # = energy_recount_ett.recount (T_h-averaged forecast heads)
    return cond == 'spk_ssa' or cond.startswith(('spkall_', 'spkneo_', 'nsmt4_', 'nsmt4q_'))   # MODI 2026-10-30: + nsmt4q_


def questions(task):
    Q = [(f'R1 replace vs append, {NL[n]} (main)', f'nsmt4_{n}', f'spkneo_{n}', False) for n in NS]
    Q += [('R1 (report) replace vs append, memoryless LIF', 'nsmt4_lif_nomem', 'spkneo_lif_nomem', False)]
    Q += [(f'R2 [NSMT T=4] vs Spikformer-style, {NL[n]} (criterion 2)', f'nsmt4_{n}', 'spk_ssa', True) for n in NS]
    Q += [(f'R2 B form vs Spikformer-style, {NL[n]} (criterion 2)' + (' [= earlier D1]' if task == 'cinc' else ''),
           f'spkneo_{n}', 'spk_ssa', True) for n in NS]
    Q += [('R3 T = N memory vs memoryless, [NSMT T=4] LIF', 'nsmt4_lif', 'nsmt4_lif_nomem', False),
          ('R3 T = N memory vs memoryless, B form LIF' + (' [= earlier D2]' if task == 'cinc' else ''), 'spkneo_lif', 'spkneo_lif_nomem', False)]
    Q += [(f'R3 (report) {NL[n]} vs memoryless LIF, {fam}', f'{p}_{n}', f'{p}_lif_nomem', False)
          for p, fam in (('nsmt4', '[NSMT T=4]'), ('spkneo', 'B form')) for n in NS[1:]]
    Q += [(f'R4 (report) {NL[n]} vs LIF, {fam}', f'{p}_{n}', f'{p}_lif', False) for p, fam in (('nsmt4', '[NSMT T=4]'), ('spkneo', 'B form')) for n in NS[1:]]
    if task != 'cinc':
        Q += [(f'R5 (report) {fam} vs Neo-mixer every layer, {NL[n]}', f'{p}_{n}', f'mixall_{n}_bidir', False)
              for p, fam in (('nsmt4', '[NSMT T=4]'), ('spkneo', 'B form')) for n in NS]
    else:
        Q += [('R6 (report) [NSMT T=4] @90 new settings vs @30 old settings, LIF', 'nsmt4_lif', 'nsmt_lif_t4', False),
              ('R6 (report) [NSMT T=4] @90 new settings vs @30 old settings, pop f-LIF', 'nsmt4_pop_flif', 'nsmt_pop_t4', False)]
    return Q


def index(rows, key):
    by = {}
    for r in rows:
        by.setdefault(key(r), []).append(r)
    return lambda k: by[k][0] if len(by.get(k, [])) == 1 else None      # duplicates -> missing (fail-closed)


def paired_table(title, rows, metric, qs):
    get = index(rows, lambda r: (r['cond'], int(r['seed'])))
    L = [title, '', '| question | A | B | A (mean) | B (mean) | A − B per seed (%p) | mean | verdict | energy A/B | criterion (2) |',
         '|---|---|---|---|---|---|---|---|---|---|']
    for q, x, y, crit in qs:
        a = [get((x, s)) for s in SEEDS]; b = [get((y, s)) for s in SEEDS]
        if any(r is None for r in a + b) or not all(fin(r[metric]) and fin(r['total_fft_uJ']) for r in a + b):
            L.append(f'| {q} | {x} | {y} | | | | | incomplete | | |'); continue
        d = [100 * (p[metric] - o[metric]) for p, o in zip(a, b)]; m = float(np.mean(d))
        v = 'better' if m >= 0.5 and sum(z > 0 for z in d) >= 2 else 'worse' if m <= -0.5 and sum(z < 0 for z in d) >= 2 else 'no clear difference'
        er = float(np.mean([p['total_fft_uJ'] for p in a]) / np.mean([o['total_fft_uJ'] for o in b]))
        c2 = ('met' if v == 'better' or (m >= -0.5 and er <= 0.5) else 'not met') if crit else ''
        L.append(f"| {q} | {x} | {y} | {100 * np.mean([p[metric] for p in a]):.2f} | {100 * np.mean([o[metric] for o in b]):.2f} | "
                 f"{' / '.join(f'{z:+.2f}' for z in d)} | {m:+.2f} | {v} | {er:.2f} | {c2} |")
    return L


def cinc():
    def f0(s, keep=None):
        rr = [r for r in parse(os.path.join(HERE, 'results', s, 'raw.txt'))
              if r.get('data') == 'CINC2017' and int(r.get('crop', 0)) == 0 and int(r.get('fold', 0)) == 0]
        return [r for r in rr if keep is None or keep(r['cond'])]
    rows = f0('flif-v1-rhythm-dual-cinc', lambda c: c.startswith('nsmt4_'))
    rows += f0('flif-v1-rhythm-addneo90-cinc', lambda c: c.startswith('spkneo_'))
    rows += f0('flif-v1-rhythm-budget-cinc', lambda c: c == 'spk_ssa')
    rows += f0('flif-v1-rhythm-fix-cinc', lambda c: c in ('nsmt_lif_t4', 'nsmt_pop_t4'))
    return paired_table('## CinC 2017 fold 0 (test score %, best-validation weights; dual-path models and Spikformer-style at 90 epochs)',
                        rows, 'score', questions('cinc'))


def mitbih():
    rows = []
    for s, keep in (('flif-v1-mitbih-dual', lambda c: c.startswith(('nsmt4_', 'spkneo_'))), ('flif-v1-gen-mitbih', lambda c: c == 'spk_ssa'),
                    ('flif-v1-mitbih-nall', lambda c: c.startswith('mixall_'))):
        rows += [r for r in parse(os.path.join(HERE, 'results', s, 'raw.txt')) if r.get('dataset') == 'MITBIH' and keep(r['cond'])]
    return paired_table('## MIT-BIH (test macro-F1 %, last-epoch weights)', rows, 'f1', questions('mitbih'))


def ett(energy_fix):
    rows = []
    for s, keep in (('flif-v1-ett-dual', lambda c: c.startswith(('nsmt4_', 'spkneo_'))), ('flif-v1-gen-ett', lambda c: c == 'spk_ssa'),
                    ('flif-v1-ett-nall', lambda c: c.startswith('mixall_'))):
        rr = [r for r in parse(os.path.join(FC, 'results', s, 'raw.txt')) if keep(r['cond'])]
        if energy_fix:
            p = os.path.join(FC, 'results', s, 'energy_recount.csv')
            fix = {(f['ds'], int(f['pl']), f['cond'], int(f['seed'])): float(f['total_fft_uJ_corr']) for f in csv.DictReader(open(p))} if os.path.exists(p) else {}
            for r in rr:
                if recounted(r['cond']):
                    r['total_fft_uJ'] = fix.get((r['ds'], int(r['pl']), r['cond'], int(r['seed'])), float('nan'))   # missing -> incomplete
        rows += rr
    get = index(rows, lambda r: (r['ds'], int(r['pl']), r['cond'], int(r['seed'])))
    L = ['## ETT forecasting, ETTh2 + ETTm2 x H96 / H720 (test MSE; r = 100 (A − B) / B, %)' + (' — T_h-averaged head energy recounted' if energy_fix else ''), '',
         '| question | A | B | cells b / . / w | R | pairs A < B | verdict | energy A/B | criterion (2) |', '|---|---|---|---|---|---|---|---|---|']
    detail = ['', '### Per-cell mean r (%)', '', '| A vs B | ' + ' | '.join(f'{d} H{p}' for d, p in CELLS) + ' |', '|---|' + '---|' * len(CELLS)]
    for q, x, y, crit in questions('ett'):
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
        detail.append(f'| {x} − {y} | ' + ' | '.join(f'{m:+.1f} {c}' for m, c in zip(cm, cv)) + ' |')
    return L + detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--energy_fix', action='store_true')
    a = ap.parse_args()
    L = ['# Dual-path batch ([NSMT hippocampus T=4] and B form) — summary', '',
         'See the module docstring for names, sources and rules. Exploratory (one CinC fold, 3 seeds).', '']
    L += cinc() + [''] + mitbih() + [''] + ett(a.energy_fix)
    out = os.path.join(HERE, 'results', 'flif-v1-dual-summary' + ('-energyfix' if a.energy_fix else '') + '.md')
    open(out, 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
