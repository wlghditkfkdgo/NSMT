"""MODI 2026-10-27: summary and pre-registered verdicts of the generalization batch -- the CinC models (Spikformer-style SSA, the
Neo-mixer with OR fusion, the Neo-mixer with charge-difference fusion 'linmx', x LIF / f-LIF / population LIF / population f-LIF) on
ETT forecasting (nsmt_flif_v1/forecasting/scripts/run_ett_mixer.sh) and MIT-BIH classification (scripts/run_mitbih2.sh).

Comparisons (A - B, paired by seed): G1 linmx LIF - Spikformer-style (primary; criterion 2), G2 the other linmx neurons -
Spikformer-style (criterion 2), G3 linmx - OR for each neuron (fusion), G4 OR population LIF - Spikformer-style (criterion 2),
G5 OR for the other neurons - Spikformer-style (criterion 2, report).
MIT-BIH: test macro-F1 of the last-epoch weights (ECG framework); better = mean >= +0.5 %p and A > B in >= 2 of 3 seeds, worse =
mirror; criterion 2 = better, or mean >= -0.5 %p with energy <= 0.5x.
ETT: test MSE of the best-validation weights; per seed r = 100 (MSE_A - MSE_B) / MSE_B. Cell (dataset x horizon): better = mean r
<= -1 % and r < 0 in >= 2 of 3 seeds, worse = mirror. Overall: R = mean of the 16 cell means; better = R <= -1 % and r < 0 in >= 2/3
of the 48 seed pairs, worse = mirror; criterion 2 = better, or R <= +1 % with energy <= 0.5x (energy = mean over cells of the ratio
of cell-mean total_fft_uJ). A missing seed / cell or a non-finite value makes the comparison incomplete (fail-closed).
MODI 2026-10-29 --cells4: the same rules on the user's ETT scope (decision 12 of 2026-10-14: ETTh2 + ETTm2 x H96 / H720), R over
the 4 cells and >= 8 of the 12 seed pairs -> summary4[_energyfix].md (post hoc recomputation, no new training).
usage: python summarize_gen.py --ett --suite flif-v1-gen-ett-cinc   |   python summarize_gen.py --mitbih --suite flif-v1-gen-mitbih
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
NAME = {'lif': 'spikeDE LIF', 'flif': 'f-LIF', 'pop_lif': 'population LIF (psi 0.5)', 'pop_flif': 'population f-LIF (psi 1)'}
LAB = {'spk_ssa': 'Spikformer-style SSA (T=4)', **{f'mix_{n}_bidir': f'Neo-mixer OR, {NAME[n]}' for n in NS},
       **{f'mixlinmx_{n}': f'Neo-mixer charges + |H_f - H_r|, {NAME[n]}' for n in NS}}
ORDER = ['spk_ssa'] + [f'mixlinmx_{n}' for n in NS] + [f'mix_{n}_bidir' for n in NS]
DSS = ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2')
PLS = (96, 192, 336, 720)
CELLS4 = (('ETTh2', 96), ('ETTh2', 720), ('ETTm2', 96), ('ETTm2', 720))   # user decision 12 (2026-10-14)
Q = ([('G1 linmx LIF vs Spikformer (primary, criterion 2)', 'mixlinmx_lif', 'spk_ssa')]
     + [('G2 linmx vs Spikformer (criterion 2)', f'mixlinmx_{n}', 'spk_ssa') for n in NS[1:]]
     + [('G3 linmx vs OR (fusion)', f'mixlinmx_{n}', f'mix_{n}_bidir') for n in NS]
     + [('G4 OR population LIF vs Spikformer (criterion 2)', 'mix_pop_lif_bidir', 'spk_ssa')]
     + [('G5 (report) OR vs Spikformer (criterion 2)', f'mix_{n}_bidir', 'spk_ssa') for n in ('lif', 'flif', 'pop_flif')])


def finite(*v):
    return all(isinstance(x, float) and math.isfinite(x) for x in v)


def ms(v, sc=1., nd=4):
    v = np.asarray(list(v), dtype=float)
    if len(v) == 0 or not np.all(np.isfinite(v)):
        return 'n/a'
    return f'{sc * v.mean():.{nd}f} ± {sc * v.std(ddof=1) if len(v) > 1 else 0:.{nd}f}'


def mitbih(a):
    R = os.path.join(HERE, 'results', a.suite)
    rows = [r for r in parse(os.path.join(R, 'raw.txt')) if r.get('dataset') == 'MITBIH']
    by = {}
    for r in rows:
        by.setdefault((r['cond'], int(r['seed'])), []).append(r)
    dup = [k for k, v in by.items() if len(v) > 1]
    get = lambda c, s: by[(c, s)][0] if len(by.get((c, s), [])) == 1 else None
    L = [f'# Generalization batch, MIT-BIH `{a.suite}` — summary', '',
         'ECG framework (30 epochs, last-epoch weights on the 30 % test file); macro-F1 / accuracy mean ± sd over seeds (42, 7, 2026) (%).'
         + (f' **Duplicate RESULT rows (treated as missing): {dup}**' if dup else ''), '',
         '| condition | n | macro-F1 | accuracy | F1 S | F1 V | F1 F | F1 Q | total µJ (FFT) | neocortex firing | params |', '|---|---|---|---|---|---|---|---|---|---|---|']
    energy = {}
    for c in ORDER:
        rr = [get(c, s) for s in SEEDS if get(c, s) is not None]
        if not rr:
            L.append(f'| {LAB[c]} | 0 | n/a | | | | | | | | |'); continue
        energy[c] = float(np.mean([r['total_fft_uJ'] for r in rr]))
        L.append(f"| {LAB[c]} | {len(rr)} | {ms([r['f1'] for r in rr], 100, 2)} | {ms([r['overall_acc'] for r in rr], 100, 2)} | "
                 + ' | '.join(f"{100 * np.mean([r[f'f1_{i}'] for r in rr]):.2f}" for i in (1, 2, 3, 4))
                 + f" | {energy[c]:.2f} | {ms([r['rho_neo'] for r in rr if isinstance(r.get('rho_neo'), float)], 1, 4)} | {int(rr[0]['params'])} |")
    L += ['', '## Pre-registered comparisons (macro-F1, %p, paired by seed)', '',
          '| question | A | B | A − B per seed | mean | verdict | energy A/B | criterion (2) |', '|---|---|---|---|---|---|---|---|']
    for q, x, y in Q:
        da = [get(x, s) for s in SEEDS]; db = [get(y, s) for s in SEEDS]
        if any(r is None for r in da + db) or not all(finite(r['f1']) for r in da + db):
            L.append(f'| {q} | {LAB[x]} | {LAB[y]} | | | incomplete | | |'); continue
        d = [100 * (p['f1'] - q_['f1']) for p, q_ in zip(da, db)]; m = float(np.mean(d))
        v = 'better' if m >= 0.5 and sum(z > 0 for z in d) >= 2 else 'worse' if m <= -0.5 and sum(z < 0 for z in d) >= 2 else 'no clear difference'
        er = energy[x] / energy[y]
        crit = ''
        if 'criterion 2' in q:
            crit = 'met' if v == 'better' or (m >= -0.5 and er <= 0.5) else 'not met'
        L.append(f"| {q} | {LAB[x]} | {LAB[y]} | {' / '.join(f'{z:+.2f}' for z in d)} | {m:+.2f} | {v} | {er:.2f} | {crit} |")
    out = os.path.join(R, 'summary.md'); open(out, 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


def ett(a):
    R = os.path.join(FC, 'results', a.suite)
    rows = parse(os.path.join(R, 'raw.txt'))
    if a.energy_fix:                                    # MODI 2026-10-28: energy_recount_ett.py (Spikformer-style head as spike ACs)
        fix = {(f['ds'], int(f['pl']), f['cond'], int(f['seed'])): float(f['total_fft_uJ_corr'])
               for f in csv.DictReader(open(os.path.join(R, 'energy_recount.csv')))}
        for r in rows:
            r['total_fft_uJ'] = fix.get((r['ds'], int(r['pl']), r['cond'], int(r['seed'])), float('nan'))   # missing -> incomplete
    by = {}
    for r in rows:
        by.setdefault((r['ds'], int(r['pl']), r['cond'], int(r['seed'])), []).append(r)
    dup = [k for k, v in by.items() if len(v) > 1]
    get = lambda ds, pl, c, s: by[(ds, pl, c, s)][0] if len(by.get((ds, pl, c, s), [])) == 1 else None
    cells = list(CELLS4) if a.cells4 else [(ds, pl) for ds in DSS for pl in PLS]
    L = [f'# Generalization batch, ETT forecasting `{a.suite}` — summary' + (' (corrected head energy)' if a.energy_fix else '')
         + (', user scope ETTh2 + ETTm2 x H96 / H720 (4 cells)' if a.cells4 else ''), '',
         'Best-validation weights (early stopping on the validation L1, patience 10) on the test set; MSE / MAE on the scaled data, '
         'mean over seeds (42, 7, 2026).' + (f' **Duplicate RESULT rows (treated as missing): {dup}**' if dup else ''), '']
    if a.energy_fix:
        L += ['**Post-hoc energy correction (2026-10-28, not pre-registered):** total µJ of the Spikformer-style runs from '
              '`energy_recount.csv` (`energy_recount_ett.py`: the flat head counted as spike-driven ACs on each of the T_h binary '
              'steps instead of MACs on their average; same model function, MSE unchanged). Neo-mixer energies unchanged. The '
              'pre-registered numbers are in `summary.md`.', '']
    if a.cells4:
        L += ['**Post-hoc recomputation (2026-10-29, not pre-registered):** the pre-registered rules applied to the 4 cells of the '
              "user's ETT scope (decision 12 of 2026-10-14) instead of the 16-cell grid; overall verdict needs >= 8 of the 12 seed "
              'pairs. Same runs, no new training. The 16-cell numbers are in `summary.md` / `summary_energyfix.md`.', '']
    for metric in ('mse', 'mae'):
        L += [f'## Test {metric.upper()} (mean over seeds; n < 3 marked)', '',
              '| condition | ' + ' | '.join(f'{ds} H{pl}' for ds, pl in cells) + ' |', '|---|' + '---|' * len(cells)]
        for c in ORDER:
            cs = []
            for ds, pl in cells:
                v = [get(ds, pl, c, s)[metric] for s in SEEDS if get(ds, pl, c, s) is not None]
                cs.append('' if not v else f'{np.mean(v):.4f}' + ('' if len(v) == 3 else f' (n={len(v)})'))
            L.append(f'| {LAB[c]} | ' + ' | '.join(cs) + ' |')
        L.append('')
    energy = {}
    L += [f'## Energy and size (mean over the {len(cells)} cells and seeds)', '', '| condition | runs | total µJ (FFT) per window | neocortex firing | epochs run | params (H96 / H720) |',
          '|---|---|---|---|---|---|']
    for c in ORDER:
        rr = [get(ds, pl, c, s) for ds, pl in cells for s in SEEDS if get(ds, pl, c, s) is not None]
        for ds, pl in cells:
            v = [get(ds, pl, c, s)['total_fft_uJ'] for s in SEEDS if get(ds, pl, c, s) is not None]
            if v:
                energy[(c, ds, pl)] = float(np.mean(v))
        if not rr:
            L.append(f'| {LAB[c]} | 0 | | | | |'); continue
        ep = [float(str(r.get('best_epoch_val_epochs', 'nan:nan:nan')).split(':')[-1]) for r in rr]
        p96 = [int(r['params']) for r in rr if int(r['pl']) == 96]; p720 = [int(r['params']) for r in rr if int(r['pl']) == 720]
        L.append(f"| {LAB[c]} | {len(rr)} | {np.mean([r['total_fft_uJ'] for r in rr]):.2f} | "
                 f"{ms([r['rho_neo'] for r in rr if isinstance(r.get('rho_neo'), float)], 1, 4)} | {np.nanmean(ep):.1f} | "
                 f"{(min(p96) if p96 else 'n/a')}–{(max(p96) if p96 else '')} / {(min(p720) if p720 else 'n/a')}–{(max(p720) if p720 else '')} |")
    L += ['', '## Pre-registered comparisons (test MSE; r = 100 (A − B) / B per seed, %)', '',
          '| question | A | B | cells better / none / worse | R = mean of cell means | seed pairs A < B | verdict | energy A/B | criterion (2) |',
          '|---|---|---|---|---|---|---|---|---|']
    detail = ['', '## Per-cell mean r (%) and cell verdict (b = better, w = worse, . = no clear difference)', '',
              '| A vs B | ' + ' | '.join(f'{ds} H{pl}' for ds, pl in cells) + ' |', '|---|' + '---|' * len(cells)]
    for q, x, y in Q:
        cm, cv, pairs, ers, ok = [], [], [], [], True
        for ds, pl in cells:
            ra = [get(ds, pl, x, s) for s in SEEDS]; rb = [get(ds, pl, y, s) for s in SEEDS]
            if any(r is None for r in ra + rb) or not all(finite(r['mse'], r['total_fft_uJ']) for r in ra + rb) or any(r['mse'] <= 0 for r in rb):
                ok = False; break
            r_ = [100 * (p['mse'] - b['mse']) / b['mse'] for p, b in zip(ra, rb)]
            m = float(np.mean(r_)); cm.append(m); pairs += r_
            cv.append('b' if m <= -1 and sum(z < 0 for z in r_) >= 2 else 'w' if m >= 1 and sum(z > 0 for z in r_) >= 2 else '.')
            ers.append(energy[(x, ds, pl)] / energy[(y, ds, pl)])
        if not ok:
            L.append(f'| {q} | {LAB[x]} | {LAB[y]} | | | | incomplete | | |'); continue
        Rm = float(np.mean(cm)); nlow = sum(z < 0 for z in pairs); nhigh = sum(z > 0 for z in pairs); er = float(np.mean(ers))
        need = 2 * len(pairs) // 3                      # 2/3 of the seed pairs: 32 of 48 (16 cells), 8 of 12 (4 cells)
        v = 'better' if Rm <= -1 and nlow >= need else 'worse' if Rm >= 1 and nhigh >= need else 'no clear difference'
        crit = ''
        if 'criterion 2' in q:
            crit = 'met' if v == 'better' or (Rm <= 1 and er <= 0.5) else 'not met'
        L.append(f"| {q} | {LAB[x]} | {LAB[y]} | {cv.count('b')} / {cv.count('.')} / {cv.count('w')} | {Rm:+.2f} | {nlow}/{len(pairs)} | {v} | {er:.2f} | {crit} |")
        detail.append(f'| {x} − {y} | ' + ' | '.join(f'{m:+.1f} {c}' for m, c in zip(cm, cv)) + ' |')
    L += detail
    out = os.path.join(R, ('summary4' if a.cells4 else 'summary') + ('_energyfix' if a.energy_fix else '') + '.md')
    open(out, 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ett', action='store_true')
    ap.add_argument('--mitbih', action='store_true')
    ap.add_argument('--suite', required=True)
    ap.add_argument('--energy_fix', action='store_true', help='ETT: use energy_recount.csv (corrected head energy) -> summary_energyfix.md')
    ap.add_argument('--cells4', action='store_true', help='ETT: only ETTh2 + ETTm2 x H96 / H720 (user decision 12) -> summary4*.md')
    a = ap.parse_args()
    if a.ett:
        ett(a)
    elif a.mitbih:
        mitbih(a)
    else:
        ap.error('choose --ett or --mitbih')
