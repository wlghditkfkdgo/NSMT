"""Verdicts of the ETT check of muxlin (2026-10-31, PROJECT_LOG, exp/f-lif-pop-v3). Cells = ETTh2 + ETTm2 x H96 / H720 (user decision
12 of 10-14), seeds 42 / 7 / 2026, test MSE with the best-validation weights.

Criterion copied from exp/nsmt-flif-neo summarize_gen.py (blob 0ce13f7b, lines 151-169; MODI_PROJECT_LOG 3906):
    r = 100 (MSE_A - MSE_B) / MSE_B per seed (paired); a cell is better if its mean r <= -1 % and >= 2 of 3 seeds r < 0 (mirror worse)
    R = mean over the 4 cells of the cell mean r; better if R <= -1 % and r < 0 in >= 8 of the 12 seed pairs; worse mirror;
    else no clear difference. Criterion (2) = better, or R <= +1 % with energy ratio <= 0.5 (mean over cells of the cell-mean
    energy ratio). Energy = total_fft_uJ_corr (forecast head recounted as AC, energy_recount_ett.py) for every condition.
Pre-registered: E1 muxlin vs Spikformer-style (the decision: worse -> fold 1-4 / tuning on hold; better -> plan them; no clear
difference -> report, user decides), E2 muxlin vs sdlin, E3 sdlin vs Spikformer-style. MAE-based R reported only.
Spikformer-style rows = results/<suite>/baseline_spk_ssa_raw.txt + baseline_spk_ssa_energy.csv (copies, see their headers).
Writes results/<suite>/summary.md.
"""
import argparse
import csv
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from summarize_rhythm import parse                                        # noqa: E402

CELLS = (('ETTh2', 96), ('ETTh2', 720), ('ETTm2', 96), ('ETTm2', 720))
SEEDS = (42, 7, 2026)
REF, MUX, SDL = 'spk_ssa', 'spkmx_lif_muxlin_or', 'spkmx_lif_sdlin_or'
LAB = {REF: 'Spikformer-style', MUX: 'muxlin (s_d ? L : F ^ R)', SDL: 'sdlin (L)'}
Q = (('E1 muxlin vs Spikformer-style (criterion 2)', MUX, REF), ('E2 muxlin vs sdlin', MUX, SDL),
     ('E3 sdlin vs Spikformer-style (criterion 2)', SDL, REF))


def key(r):
    return (r['ds'], int(r['pl']), r['cond'], int(r['seed']))


def load(res):
    rows = [r for r in parse(os.path.join(res, 'raw.txt')) if r['cond'] in (MUX, SDL)]
    rows += [r for r in parse(os.path.join(res, 'baseline_spk_ssa_raw.txt')) if r['cond'] == REF]
    corr = {}
    for fn in ('energy_recount.csv', 'baseline_spk_ssa_energy.csv'):
        for e in csv.DictReader(open(os.path.join(res, fn))):
            corr[(e['ds'], int(e['pl']), e['cond'], int(e['seed']))] = float(e['total_fft_uJ_corr'])
    D = {}
    for r in rows:
        k = key(r)
        if k in D:
            raise SystemExit(f'duplicate {k}')
        if k not in corr:
            raise SystemExit(f'no corrected energy for {k}')
        r['E'] = corr[k]
        D[k] = r
    return D


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--suite', default='flif-v1-ett-spkmux'); a = ap.parse_args()
    res = os.path.join(HERE, 'results', a.suite)
    D = load(res)
    get = lambda ds, pl, c, s: D.get((ds, pl, c, s))
    L = [f'# ETT check of muxlin `{a.suite}` (LIF, residual OR; ETTh2 + ETTm2 x H96 / H720, seeds {SEEDS})', '',
         'Test MSE / MAE on the scaled data, best-validation weights. Energy = forecast-head-corrected total µJ per window (7 channels).', '']
    for metric in ('mse', 'mae'):
        L += [f'## Test {metric.upper()} (mean ± sd over seeds; n < 3 marked)', '', '| condition | ' + ' | '.join(f'{d} H{p}' for d, p in CELLS) + ' |',
              '|---|' + '---:|' * len(CELLS)]
        for c in (REF, SDL, MUX):
            cs = []
            for ds, pl in CELLS:
                v = [get(ds, pl, c, s)[metric] for s in SEEDS if get(ds, pl, c, s) is not None]
                cs.append('—' if not v else f'{np.mean(v):.4f} ± {np.std(v):.4f}' + ('' if len(v) == 3 else f' (n={len(v)})'))
            L.append(f'| {LAB[c]} | ' + ' | '.join(cs) + ' |')
        L.append('')
    L += ['## Energy (corrected µJ per window), epochs run, parameters', '', '| condition | ' + ' | '.join(f'{d} H{p}' for d, p in CELLS) + ' | epochs run | params H96 / H720 |',
          '|---|' + '---:|' * len(CELLS) + '---:|---|']
    energy = {}
    for c in (REF, SDL, MUX):
        cs = []
        for ds, pl in CELLS:
            v = [get(ds, pl, c, s)['E'] for s in SEEDS if get(ds, pl, c, s) is not None]
            if v:
                energy[(c, ds, pl)] = float(np.mean(v))
            cs.append('—' if not v else f'{np.mean(v):.2f}')
        rr = [get(ds, pl, c, s) for ds, pl in CELLS for s in SEEDS if get(ds, pl, c, s) is not None]
        ep = [float(str(r.get('best_epoch_val_epochs', 'nan:nan:nan')).split(':')[-1]) for r in rr]
        p96 = sorted({int(r['params']) for r in rr if int(r['pl']) == 96}); p720 = sorted({int(r['params']) for r in rr if int(r['pl']) == 720})
        L.append(f'| {LAB[c]} | ' + ' | '.join(cs) + f" | {np.mean(ep) if ep else float('nan'):.1f} | {p96} / {p720} |")
    L += ['', '## Pre-registered comparisons (test MSE; r = 100 (A − B) / B per seed, %)', '',
          '| question | cells better / none / worse | R | seed pairs A < B | verdict | energy A/B | criterion (2) | R on MAE (report) |',
          '|---|---|---:|---|---|---:|---|---:|']
    detail = ['', '## Per-cell mean r (%), per seed (42, 7, 2026), cell verdict (b better, w worse, . no clear difference)', '',
              '| A − B | ' + ' | '.join(f'{d} H{p}' for d, p in CELLS) + ' |', '|---|' + '---|' * len(CELLS)]
    verdicts = {}
    for q, x, y in Q:
        cm, cv, pairs, ers, cm_mae, cells_txt, ok = [], [], [], [], [], [], True
        for ds, pl in CELLS:
            ra = [get(ds, pl, x, s) for s in SEEDS]; rb = [get(ds, pl, y, s) for s in SEEDS]
            if any(r is None for r in ra + rb) or not all(math.isfinite(r['mse']) and math.isfinite(r['E']) for r in ra + rb) or any(r['mse'] <= 0 for r in rb):
                ok = False; break
            r_ = [100 * (p['mse'] - b['mse']) / b['mse'] for p, b in zip(ra, rb)]
            m = float(np.mean(r_)); cm.append(m); pairs += r_
            cm_mae.append(float(np.mean([100 * (p['mae'] - b['mae']) / b['mae'] for p, b in zip(ra, rb)])))
            cv.append('b' if m <= -1 and sum(z < 0 for z in r_) >= 2 else 'w' if m >= 1 and sum(z > 0 for z in r_) >= 2 else '.')
            ers.append(energy[(x, ds, pl)] / energy[(y, ds, pl)])
            cells_txt.append(f'{m:+.2f} ({", ".join(f"{z:+.2f}" for z in r_)}) {cv[-1]}')
        if not ok:
            L.append(f'| {q} | | | | incomplete | | | |'); verdicts[q] = 'incomplete'; continue
        Rm = float(np.mean(cm)); nlow = sum(z < 0 for z in pairs); nhigh = sum(z > 0 for z in pairs); er = float(np.mean(ers))
        need = 2 * len(pairs) // 3
        v = 'better' if Rm <= -1 and nlow >= need else 'worse' if Rm >= 1 and nhigh >= need else 'no clear difference'
        crit = ('met' if v == 'better' or (Rm <= 1 and er <= 0.5) else 'not met') if 'criterion 2' in q else ''
        verdicts[q] = v
        L.append(f"| {q} | {cv.count('b')} / {cv.count('.')} / {cv.count('w')} | {Rm:+.2f} | {nlow}/{len(pairs)} | {v} | {er:.2f} | {crit} | {float(np.mean(cm_mae)):+.2f} |")
        detail.append(f'| {x} − {y} | ' + ' | '.join(cells_txt) + ' |')
    L += detail
    e1 = verdicts[Q[0][0]]
    gate = {'worse': 'E1 worse -> fold 1-4 and tuning of muxlin ON HOLD (user rule 10-31)',
            'better': 'E1 better -> plan fold 1-4 and tuning, for the user\'s approval',
            'no clear difference': 'E1 no clear difference -> report; the user decides on fold 1-4 / tuning',
            'incomplete': 'E1 incomplete -> no decision'}[e1]
    n_new = sum(1 for k in D if k[2] in (MUX, SDL)); skips = sum(1 for l in open(os.path.join(res, 'raw.txt')) if l.startswith('SKIP'))
    L += ['', f'**Decision rule:** {gate}.', '', f'RESULT rows (new): {n_new}; SKIP lines: {skips}; baseline rows: {sum(1 for k in D if k[2] == REF)}.']
    open(os.path.join(res, 'summary.md'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
