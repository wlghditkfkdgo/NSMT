"""MODI 2026-10-29: summary and pre-registered verdicts of the whole-model neuron swap batch (docs/MODI_PROJECT_LOG.md 2026-10-29).

Condition names: mix_<n>_bidir = Neo-mixer OR, neuron n in the mixer only (spikingjelly LIF elsewhere; earlier suites);
mixlb_<n>_bidir = n in the mixer, spikeDE LIF elsewhere (control); mixall_<n>_bidir = n in every layer; spk_ssa = Spikformer-style
(spikingjelly LIF); spkall_<n> = Spikformer-style with n in every layer; spkneo_<n> = B form, n in the neocortex only (earlier);
spkneoall_<n> = B form, n in every layer; *_nomem = the memoryless-neocortex control.  n in {lif, flif, pop_lif, pop_flif}.

Rules = the generalization batch (summarize_gen.py), paired by seed (42, 7, 2026); missing / non-finite -> incomplete:
  ETT (test MSE, best-validation weights): r = 100 (A - B) / B per seed; cell better = mean r <= -1 % and r < 0 in >= 2/3 seeds;
      overall R = mean of the cell means; better = R <= -1 % and r < 0 in >= 2/3 of the seed pairs (8 / 12 for the 4 cells,
      6 / 9 for 3), worse = mirror; criterion (2) = better, or R <= +1 % with energy <= 0.5x (mean over cells of cell-mean
      total_fft_uJ ratios; Spikformer-style energies from energy_recount.csv when --energy_fix).
      Quick cell ETTh2 H96 (C1, C2, Q3): better = mean r <= -1 % and r < 0 in 3/3 seeds, worse = mirror (11:03 rule).
  MIT-BIH (test macro-F1, last-epoch weights) and CinC fold 0 (test score, best-validation weights, 90 epochs): better = mean
      A - B >= +0.5 %p and A > B in >= 2 of 3 seeds, worse = mirror; criterion (2) = better, or mean >= -0.5 %p with energy <= 0.5x.
Comparisons: W1 mixall_n - mix_n_bidir (whole vs neocortex-only), W2 mixall_n - spkall_n (both whole; criterion 2),
W3 spkall_n - spk_ssa (report), W4 mixall_flif - mixall_lif and spkall_flif - spkall_lif (ETT: confirmatory on the 3 cells without
ETTh2 H96), W5 mixall_n - spk_ssa (report; criterion 2), C1 mixlb_n - mix_n_bidir (implementation; LIF: mixall_lif - mix_lif_bidir),
C2 mixall_n - mixlb_n (neuron type outside the mixer); CinC: D1w spkneoall_n - spkneoall_n_nomem (main), D2w spkneoall_n - spkall_n,
D3w (report) spkneoall_n - spkneo_n, spkall_n - spk_ssa, spkneoall_n_nomem - spkneo_lif_nomem.
ETT cells (correction 2026-10-29 13:3x, user decision 12 of 10-14): ETTh2 and ETTm2 x H96 and H720 only (4 cells), not 16.
Design change 2026-10-29 17:1x (user): a dual-path model (hippocampus + neocortex, the B form) swaps ONLY the neocortex neuron
(= the earlier spkneo_<n> runs), the hippocampus-free Neo-mixer swaps every layer, and the Spikformer-style baseline keeps
spikingjelly LIF. Dropped: W2, W3, the Spikformer-style W4 line and the whole CinC part (D1w-D3w); the 8 running spkneoall_* runs
were stopped and their lines held. W5 (Neo-mixer every layer vs the spikingjelly Spikformer-style baseline, criterion 2) is now the
baseline comparison (same rule; it was 'report' while W2 existed). Q3 (ETTh2 H96, done) stays as a report.
usage: python summarize_nall2.py [--energy_fix]   ->  results/flif-v1-nall-summary.md
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
DSS = ('ETTh2', 'ETTm2')                                       # user decision 12 (10-14): ETTh2 + ETTm2, H96 / H720
PLS = (96, 720)
CELLS = [(d, p) for d in DSS for p in PLS]
CELLS15 = [c for c in CELLS if c != ('ETTh2', 96)]              # W4 confirmatory cells (3; name kept)


def fin(x):
    return isinstance(x, float) and math.isfinite(x)


def mlb(n):                                                    # the control of n (LIF: the whole-model LIF is the control itself)
    return 'mixall_lif_bidir' if n == 'lif' else f'mixlb_{n}_bidir'


def pairs_q():
    W = [(f'W1 every layer vs neocortex only, {NL[n]}', f'mixall_{n}_bidir', f'mix_{n}_bidir', False) for n in NS]
    W += [('W4 fractional order, Neo-mixer every layer', 'mixall_flif_bidir', 'mixall_lif_bidir', False)]
    W += [(f'W5 Neo-mixer every layer vs Spikformer-style (spikingjelly LIF), {NL[n]} (criterion 2)', f'mixall_{n}_bidir', 'spk_ssa', True) for n in NS]
    C = [(f'C1 implementation: spikeDE LIF outside the mixer, {NL[n]}', mlb(n), f'mix_{n}_bidir', False) for n in NS]
    C += [(f'C2 neuron type outside the mixer, {NL[n]}', f'mixall_{n}_bidir', mlb(n), False) for n in NS[1:]]
    return W, C


def ett(energy_fix):
    rows = []
    for s in ('flif-v1-gen-ett', 'flif-v1-ett-nall'):
        rr = parse(os.path.join(FC, 'results', s, 'raw.txt'))
        if energy_fix:
            p = os.path.join(FC, 'results', s, 'energy_recount.csv')
            fix = {(f['ds'], int(f['pl']), f['cond'], int(f['seed'])): float(f['total_fft_uJ_corr']) for f in csv.DictReader(open(p))} if os.path.exists(p) else {}
            for r in rr:
                if r['cond'].startswith('spk'):
                    r['total_fft_uJ'] = fix.get((r['ds'], int(r['pl']), r['cond'], int(r['seed'])), float('nan'))
        rows += rr
    by = {}
    for r in rows:
        by.setdefault((r['ds'], int(r['pl']), r['cond'], int(r['seed'])), []).append(r)
    get = lambda d, p, c, s: by[(d, p, c, s)][0] if len(by.get((d, p, c, s), [])) == 1 else None

    def cmp(x, y, cells, quick=False):
        cm, cv, pr, ers = [], [], [], []
        for d, p in cells:
            ra = [get(d, p, x, s) for s in SEEDS]; rb = [get(d, p, y, s) for s in SEEDS]
            if any(r is None for r in ra + rb) or not all(fin(r['mse']) and fin(r['total_fft_uJ']) for r in ra + rb):
                return None
            r_ = [100 * (a['mse'] - b['mse']) / b['mse'] for a, b in zip(ra, rb)]
            m = float(np.mean(r_)); cm.append(m); pr += r_
            cv.append('b' if m <= -1 and sum(z < 0 for z in r_) >= 2 else 'w' if m >= 1 and sum(z > 0 for z in r_) >= 2 else '.')
            ers.append(np.mean([a['total_fft_uJ'] for a in ra]) / np.mean([b['total_fft_uJ'] for b in rb]))
        R = float(np.mean(cm)); need = len(pr) if quick else int(math.ceil(2 * len(pr) / 3))
        lo, hi = sum(z < 0 for z in pr), sum(z > 0 for z in pr)
        v = 'better' if R <= -1 and lo >= need else 'worse' if R >= 1 and hi >= need else 'no clear difference'
        return dict(R=R, lo=lo, n=len(pr), v=v, er=float(np.mean(ers)), cells=f"{cv.count('b')} / {cv.count('.')} / {cv.count('w')}", pr=pr)

    W, C = pairs_q()
    L = ['## ETT forecasting (test MSE; r = 100 (A − B) / B, %)' + (' — Spikformer-style energy recounted' if energy_fix else ''), '',
         '| question | A | B | cells b / . / w | R | pairs A < B | verdict | energy A/B | criterion (2) |', '|---|---|---|---|---|---|---|---|---|']
    for q, x, y, crit in W:
        cells = CELLS15 if q.startswith('W4') else CELLS          # W4: the 3 cells without ETTh2 H96
        o = cmp(x, y, cells)
        if o is None:
            L.append(f'| {q}{" (3 cells)" if q.startswith("W4") else ""} | {x} | {y} | | | | incomplete | | |'); continue
        c2 = ('met' if o['v'] == 'better' or (o['R'] <= 1 and o['er'] <= 0.5) else 'not met') if crit else ''
        L.append(f"| {q}{' (3 cells)' if q.startswith('W4') else ''} | {x} | {y} | {o['cells']} | {o['R']:+.2f} | {o['lo']}/{o['n']} | {o['v']} | {o['er']:.2f} | {c2} |")
    L += ['', '### ETTh2 H96 (quick cell; better / worse need 3/3 seeds; Q3 = report since 10-29 17:1x)', '',
          '| question | A | B | r per seed | R | verdict | energy A/B |', '|---|---|---|---|---|---|---|']
    for q, x, y, _ in C + [(f'Q3 Spikformer-style every layer vs spikingjelly, {NL[n]}', f'spkall_{n}', 'spk_ssa', False) for n in NS]:
        o = cmp(x, y, [('ETTh2', 96)], quick=True)
        L.append(f'| {q} | {x} | {y} | | | incomplete | |' if o is None else
                 f"| {q} | {x} | {y} | {' / '.join(f'{z:+.2f}' for z in o['pr'])} | {o['R']:+.2f} | {o['v']} | {o['er']:.2f} |")
    return L


def paired(rows, key, x, y, crit):
    by = {}
    for r in rows:
        by.setdefault((r['cond'], int(r['seed'])), []).append(r)
    get = lambda c, s: by[(c, s)][0] if len(by.get((c, s), [])) == 1 else None
    a = [get(x, s) for s in SEEDS]; b = [get(y, s) for s in SEEDS]
    if any(r is None for r in a + b) or not all(fin(r[key]) and fin(r['total_fft_uJ']) for r in a + b):
        return None
    d = [100 * (p[key] - q[key]) for p, q in zip(a, b)]; m = float(np.mean(d))
    v = 'better' if m >= 0.5 and sum(z > 0 for z in d) >= 2 else 'worse' if m <= -0.5 and sum(z < 0 for z in d) >= 2 else 'no clear difference'
    er = float(np.mean([p['total_fft_uJ'] for p in a]) / np.mean([q['total_fft_uJ'] for q in b]))
    c2 = ('met' if v == 'better' or (m >= -0.5 and er <= 0.5) else 'not met') if crit else ''
    return dict(d=d, m=m, v=v, er=er, c2=c2, a=float(np.mean([100 * p[key] for p in a])), b=float(np.mean([100 * q[key] for q in b])))


def table(title, rows, key, qs):
    L = [title, '', '| question | A | B | A (mean) | B (mean) | A − B per seed (%p) | mean | verdict | energy A/B | criterion (2) |',
         '|---|---|---|---|---|---|---|---|---|---|']
    for q, x, y, crit in qs:
        o = paired(rows, key, x, y, crit)
        L.append(f'| {q} | {x} | {y} | | | | | incomplete | | |' if o is None else
                 f"| {q} | {x} | {y} | {o['a']:.2f} | {o['b']:.2f} | {' / '.join(f'{z:+.2f}' for z in o['d'])} | {o['m']:+.2f} | {o['v']} | {o['er']:.2f} | {o['c2']} |")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--energy_fix', action='store_true')
    a = ap.parse_args()
    L = ['# Whole-model neuron swap batch — summary', '', 'See the module docstring for names and rules. Exploratory (one CinC fold, 3 seeds).', '']
    L += ett(a.energy_fix)
    W, C = pairs_q()
    mit = [r for s in ('flif-v1-gen-mitbih', 'flif-v1-mitbih-nall') for r in parse(os.path.join(HERE, 'results', s, 'raw.txt')) if r.get('dataset') == 'MITBIH']
    L += [''] + table('## MIT-BIH (test macro-F1 %, last-epoch weights)', mit, 'f1', W + C)
    L += ['', '## CinC 2017', '', 'Dropped on 2026-10-29 17:1x (user): the B form is a dual-path model, so its neuron conditions swap only '
          'the neocortex -- the existing spkneo_<n> runs (flif-v1-rhythm-addneo90-cinc). The 8 B form every-layer runs were stopped '
          'at epoch 34-44 of 90 (SKIP rc=143 in flif-v1-rhythm-nall-cinc/raw.txt) and are not used.']
    out = os.path.join(HERE, 'results', 'flif-v1-nall-summary' + ('-energyfix' if a.energy_fix else '') + '.md')
    open(out, 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
