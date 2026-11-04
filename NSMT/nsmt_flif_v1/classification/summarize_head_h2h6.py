"""MODI 2026-11-04: summary of the head test H2 / H6 (docs/MODI_PROJECT_LOG.md 2026-11-04, pre-registered rules).

H2 (MIT-BIH): nsmt4q_pop_<lif|flif>_hattn (results/flif-v1-mitbih-h2) vs the same model with the mean head (results/flif-v1-mitbih-query,
   lr 0.01, full training file), paired by seed. Delta = test macro-F1 difference (%p). better: mean >= +0.5 and >= 2/3 seeds > 0;
   worse: the mirror; otherwise no clear difference.
H6 (ETT): nsmt4q_pop_<lif|flif>_h6 (forecasting/results/flif-v1-ett-h6) vs the flat head (flif-v1-ett-query for ETTh2,
   flif-v1-ett-query-m2 for ETTm2), paired by cell and seed: r = test MSE ratio - 1 over 4 cells x 3 seeds = 12 pairs per model.
   better: mean r <= -1 % and >= 8/12 pairs < 0; worse: mean r >= +1 % and >= 8/12 > 0; per cell: mean <= -1 % (>= +1 %) and 2/3 seeds.
A missing, duplicated or non-finite value makes the comparison "incomplete" (fail-closed).
Report only: per-class F1, energy (MIT-BIH: logged totals; ETT: H6 logged total vs the baseline's spike-driven recount
energy_recount.csv and vs its logged total), parameters, |q| of H2, H6 / the 10-18 linear baseline - 1 (what the body adds to a
linear forecaster; that baseline was trained by another script), the use of the body term (H6 / its own shortcut alone - 1; the split
between the two terms is not unique, so this is not the body's value) and Spikformer-style.
usage: python summarize_head_h2h6.py [--h2_suite S] [--h6_suite S] [--out FILE]
"""
import argparse
import csv
import math
import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
FC = os.path.join(HERE, '..', 'forecasting', 'results')
SEEDS = (42, 7, 2026)
MODELS = ('nsmt4q_pop_lif', 'nsmt4q_pop_flif')
NAMES = {'nsmt4q_pop_lif': 'query pop LIF', 'nsmt4q_pop_flif': 'query pop f-LIF'}
CELLS = (('ETTh2', 96), ('ETTh2', 720), ('ETTm2', 96), ('ETTm2', 720))
# results/flif-v1-s4/summary.md "test MSE" table (best-validation weights), as in summarize_head_probe.py
LINEAR_BEST = {('ETTh2', 96): 0.291515, ('ETTh2', 720): 0.420805, ('ETTm2', 96): 0.182854, ('ETTm2', 720): 0.407828}


def results(path, **sel):
    """RESULT lines of a raw.txt as dicts, filtered by key = value (string compare)."""
    out = []
    if not os.path.exists(path):
        return out
    for ln in open(path):
        if not ln.startswith('RESULT'):
            continue
        d = dict(t.split('=', 1) for t in ln.split()[1:] if '=' in t)
        if all(d.get(k) == str(v) for k, v in sel.items()):
            out.append(d)
    return out


def one(rows, key):
    """The finite float of `key` if exactly one row, else None."""
    if len(rows) != 1 or key not in rows[0]:
        return None
    v = float(rows[0][key])
    return v if math.isfinite(v) else None


def verdict(vals, thr, need, lower_better):
    if vals is None or any(v is None for v in vals):
        return 'incomplete'
    m = float(np.mean(vals)); good = sum(v < 0 for v in vals) if lower_better else sum(v > 0 for v in vals)
    bad = sum(v > 0 for v in vals) if lower_better else sum(v < 0 for v in vals)
    if (m <= -thr if lower_better else m >= thr) and good >= need:
        return 'better'
    if (m >= thr if lower_better else m <= -thr) and bad >= need:
        return 'worse'
    return 'no clear difference'


def fmt(v, f='{:.2f}'):
    return 'incomplete' if v is None else f.format(v)


def h2(L, suite):
    raw_h = os.path.join(HERE, 'results', suite, 'raw.txt')
    raw_b = os.path.join(HERE, 'results', 'flif-v1-mitbih-query', 'raw.txt')
    raw_s = os.path.join(HERE, 'results', 'flif-v1-gen-mitbih', 'raw.txt')
    L += ['## H2 — attention pooling in front of the head (MIT-BIH, test macro-F1 %, last-epoch weights, full training file, lr 0.01)', '',
          '| model | head | seed 42 | seed 7 | seed 2026 | mean | energy uJ (logged) | params |', '|---|---|---|---|---|---|---|---|']
    V, conc = {}, {}
    for c in MODELS:
        rb = {s: results(raw_b, cond=c, seed=s, lr='0.01', val='0') for s in SEEDS}
        rh = {s: results(raw_h, cond=c + '_hattn', seed=s) for s in SEEDS}
        fb = [one(rb[s], 'f1') for s in SEEDS]; fh = [one(rh[s], 'f1') for s in SEEDS]
        for nm, f, r in (('mean (baseline)', fb, rb), ('H2 attention pooling', fh, rh)):
            e = [one(r[s], 'total_fft_uJ') for s in SEEDS]; p = [one(r[s], 'params') for s in SEEDS]
            ok = all(v is not None for v in f)
            L.append(f"| {NAMES[c]} | {nm} | " + ' | '.join(fmt(None if v is None else 100 * v) for v in f) +
                     f" | {fmt(100 * np.mean(f)) if ok else 'incomplete'} | {fmt(np.mean(e)) if all(v is not None for v in e) else 'incomplete'}"
                     f" | {int(p[0]) if p[0] is not None else 'incomplete'} |")
        V[c] = [None if (a is None or b is None) else 100 * (b - a) for a, b in zip(fb, fh)]
    L += ['', '### verdict: H2 − mean head (%p, paired by seed)', '', '| model | per seed (42 / 7 / 2026) | mean | > 0 | verdict |', '|---|---|---|---|---|']
    for c in MODELS:
        d = V[c]; v = verdict(d, 0.5, 2, lower_better=False); conc[c] = v
        ok = all(x is not None for x in d)
        L.append(f"| {NAMES[c]} | " + ' / '.join(fmt(x, '{:+.2f}') for x in d) + f" | {fmt(np.mean(d), '{:+.2f}') if ok else 'incomplete'}"
                 f" | {sum(x > 0 for x in d) if ok else ''}/3 | {v} |")
    L += ['', '### report only', '', 'Per-class test F1 (%, mean of 3 seeds; N / S / V / F / Q):', '', '| model | head | N | S | V | F | Q |', '|---|---|---|---|---|---|---|']
    for c in MODELS:
        for nm, raw, cc, extra in (('mean', raw_b, c, dict(lr='0.01', val='0')), ('H2', raw_h, c + '_hattn', {})):
            vals = [[one(results(raw, cond=cc, seed=s, **extra), f'f1_{k}') for s in SEEDS] for k in range(5)]
            L.append(f"| {NAMES[c]} | {nm} | " + ' | '.join(fmt(100 * np.mean(v)) if all(x is not None for x in v) else 'incomplete' for v in vals) + ' |')
    fs = [one(results(raw_s, cond='spk_ssa', seed=s), 'f1') for s in SEEDS]
    if all(v is not None for v in fs):
        def vs(c):
            h = [one(results(raw_h, cond=c + '_hattn', seed=s), 'f1') for s in SEEDS]
            return 'incomplete' if any(v is None for v in h) else f"{100 * (np.mean(h) - np.mean(fs)):+.2f}"
        L += ['', f"Spikformer-style (original head, flif-v1-gen-mitbih): mean {100 * np.mean(fs):.2f}; H2 − Spikformer-style: " +
              ', '.join(f"{NAMES[c]} {vs(c)}" for c in MODELS) + ' %p (body and head both differ: user decision 3)']
    qn = []
    for c in MODELS:
        for s in SEEDS:
            r = results(raw_h, cond=c + '_hattn', seed=s)
            p = os.path.join(r[0]['dir'], 'model_state', 'best+model.pt') if len(r) == 1 else None
            if p and os.path.exists(p):
                w = torch.load(p, map_location='cpu').get('head_score.weight')
                qn.append(f"{NAMES[c]} seed {s} {float(w.norm()):.3f}" if w is not None else f"{NAMES[c]} seed {s} missing")
    L += ['', 'Learned query |q| (starts at 0; the scores are q . x / sqrt(384)): ' + ('; '.join(qn) if qn else 'incomplete')]
    return conc


def h6(L, suite):
    raw_h = os.path.join(FC, suite, 'raw.txt')
    base = {'ETTh2': os.path.join(FC, 'flif-v1-ett-query'), 'ETTm2': os.path.join(FC, 'flif-v1-ett-query-m2')}
    rec = {}
    for ds, d in base.items():
        p = os.path.join(d, 'energy_recount.csv')
        if os.path.exists(p):
            for r in csv.DictReader(open(p)):
                rec.setdefault((r['ds'], int(r['pl']), r['cond'], int(r['seed'])), []).append(float(r['total_fft_uJ_corr']))
    L += ['## H6 — linear shortcut + small head (ETT, test MSE on scaled data, best-validation weights)', '']
    R, conc = {c: [] for c in MODELS}, {}
    cellv = {}
    for ds, pl in CELLS:
        L += [f'### {ds} H{pl}', '', '| model | head | seed 42 | seed 7 | seed 2026 | mean | val L1 (mean) | params |', '|---|---|---|---|---|---|---|---|']
        for c in MODELS:
            rb = {s: results(os.path.join(base[ds], 'raw.txt'), ds=ds, pl=pl, cond=c, seed=s) for s in SEEDS}
            rh = {s: results(raw_h, ds=ds, pl=pl, cond=c + '_h6', seed=s) for s in SEEDS}
            mb = [one(rb[s], 'mse') for s in SEEDS]; mh = [one(rh[s], 'mse') for s in SEEDS]; ml = [one(rh[s], 'mse_lin') for s in SEEDS]
            for nm, m, r in (('flat (baseline)', mb, rb), ('H6', mh, rh), ('H6 shortcut alone (report)', ml, rh)):
                vl = [one(r[s], 'val_loss_best') for s in SEEDS]; p = one(r[SEEDS[0]], 'params')
                ok = all(v is not None for v in m)
                L.append(f"| {NAMES[c]} | {nm} | " + ' | '.join(fmt(v, '{:.6f}') for v in m) + f" | {fmt(np.mean(m), '{:.6f}') if ok else 'incomplete'}"
                         f" | {fmt(np.mean(vl), '{:.6f}') if (nm != 'H6 shortcut alone (report)' and all(v is not None for v in vl)) else ''}"
                         f" | {int(p) if (p is not None and nm != 'H6 shortcut alone (report)') else ''} |")
            r = [None if (a is None or b is None) else 100 * (b / a - 1) for a, b in zip(mb, mh)]
            R[c] += r; cellv[(c, ds, pl)] = r
            share = [None if (a is None or b is None) else 100 * (a / b - 1) for a, b in zip(mh, ml)]
            eb = [rec.get((ds, pl, c, s)) for s in SEEDS]; eb = [v[0] if v and len(v) == 1 else None for v in eb]
            el = [one(rb[s], 'total_fft_uJ') for s in SEEDS]; eh = [one(rh[s], 'total_fft_uJ') for s in SEEDS]
            okv = lambda v: all(x is not None for x in v)
            cellv[(c, ds, pl, 'rep')] = (share, eb, el, eh, mh)
        L.append('')
    L += ['### verdict: H6 / flat − 1 (%, 12 pairs per model)', '', '| model | pooled mean | < 0 | verdict | per cell (mean, verdict) |', '|---|---|---|---|---|']
    for c in MODELS:
        v = verdict(R[c], 1.0, 8, lower_better=True); conc[c] = v
        ok = all(x is not None for x in R[c])
        cells = '; '.join(f"{ds} H{pl} {fmt(np.mean(cellv[(c, ds, pl)]), '{:+.2f}') if all(x is not None for x in cellv[(c, ds, pl)]) else 'incomplete'} "
                          f"{verdict(cellv[(c, ds, pl)], 1.0, 2, lower_better=True)}" for ds, pl in CELLS)
        L.append(f"| {NAMES[c]} | {fmt(np.mean(R[c]), '{:+.2f}') if ok else 'incomplete'} | {sum(x < 0 for x in R[c]) if ok else ''}/12 | {v} | {cells} |")
    L += ['', '### report only (mean of 3 seeds per cell)', '',
          '| model | cell | H6 / linear baseline 10-18 − 1 (%) | linear baseline 10-18 | body term use: H6 / shortcut alone − 1 (%) | energy uJ: flat recount (spike-driven head) | flat logged | H6 logged |',
          '|---|---|---|---|---|---|---|---|']
    for c in MODELS:
        for ds, pl in CELLS:
            share, eb, el, eh, mh = cellv[(c, ds, pl, 'rep')]
            g = lambda v, f='{:.2f}': fmt(np.mean(v), f) if all(x is not None for x in v) else 'incomplete'
            lin = [None if x is None else 100 * (x / LINEAR_BEST[(ds, pl)] - 1) for x in mh]
            L.append(f"| {NAMES[c]} | {ds} H{pl} | {g(lin, '{:+.2f}')} | {LINEAR_BEST[(ds, pl)]:.4f} | {g(share, '{:+.2f}')} | {g(eb)} | {g(el)} | {g(eh)} |")
    sp = os.path.join(FC, 'flif-v1-gen-ett', 'raw.txt')
    row = []
    for ds, pl in CELLS:
        v = [one(results(sp, ds=ds, pl=pl, cond='spk_ssa', seed=s), 'mse') for s in SEEDS]
        row.append(fmt(np.mean(v), '{:.4f}') if all(x is not None for x in v) else 'incomplete')
    L += ['', 'Spikformer-style test MSE (original flat head, flif-v1-gen-ett; ETTh2 H96 / H720 / ETTm2 H96 / H720): ' + ' / '.join(row) +
          ' (body and head both differ from H6: user decision 3)']
    return conc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h2_suite', default='flif-v1-mitbih-h2')
    ap.add_argument('--h6_suite', default='flif-v1-ett-h6')
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'flif-v1-head-h2h6-summary.md'))
    a = ap.parse_args()
    L = ['# Head test H2 / H6 (query population, new heads on our models only) — summary', '',
         'See the module docstring and docs/MODI_PROJECT_LOG.md 2026-11-04 for the pre-registered rules.', '']
    c2 = h2(L, a.h2_suite); L.append('')
    c6 = h6(L, a.h6_suite)

    def conc(v):
        if 'incomplete' in v.values():
            return 'incomplete'
        n = sum(x == 'better' for x in v.values()); w = sum(x == 'worse' for x in v.values())
        if n == 2:
            return 'better for both models'
        if w == 2:
            return 'worse for both models'
        if n == 1 or w == 1:
            return 'neuron-dependent (weak)'
        return 'no clear difference'
    L += ['', '## conclusions (rule of the pre-registration)', '', '| head | pop LIF | pop f-LIF | conclusion |', '|---|---|---|---|',
          f"| H2 (MIT-BIH) | {c2['nsmt4q_pop_lif']} | {c2['nsmt4q_pop_flif']} | {conc(c2)} |",
          f"| H6 (ETT) | {c6['nsmt4q_pop_lif']} | {c6['nsmt4q_pop_flif']} | {conc(c6)} |"]
    open(a.out, 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
