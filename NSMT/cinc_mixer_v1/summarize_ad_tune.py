"""Anomaly-detection tuning of the fusion MUX (2026-11-02, PROJECT_LOG AD pre-registration, exp/f-lif-pop-v3). User decisions
11-02: the order of the other tasks (lr -> mixer layers per direction -> residual), AUC-PR / raw F1 / point-adjusted F1 all
reported with the comparisons on AUC-PR, the three small datasets first (MSL, SMAP, PSM).

Tuning runs (holdout=1, suite flif-v1-ad-spkmux-tune): trained on the first 80 % of the training file, validated on the last 20 %
(no test pass). Selection per condition and dataset: score = seed-mean best validation MSE of the candidate / that of the default
(lower is better); ties (< 1e-12) keep the default; a missing run -> no selection; selected_<stage>.json is never overwritten
with a different selection.
    lr      {1e-4, 2.5e-4, 5e-4*, 1e-3, 2e-3}   muxlin, sdlin (OR, 2 mixer layers), Spikformer-style
    layers  {1, 2*, 3}                          muxlin, sdlin at the selected lr
    res     {OR*, IAND, ADD}                    muxlin, sdlin at the selected lr and layers                       (* = default)
Final (--stage final): the tuned settings trained on the whole training file (holdout=0, TSLib), test AUC-PR of the
best-validation weights. Per seed r = 100 (AUC-PR_A - AUC-PR_B) / AUC-PR_B; R = mean over the 3 datasets of the seed-mean r;
better = R >= +2 % and >= 6 of 9 pairs higher, worse = R <= -2 % and >= 6 of 9 lower; criterion (2) = better or (R >= -2 % and
energy <= 0.5x); energy = total_fft_uJ_corr (head recounted as AC, ADD under the integer rule). Fail closed: a final row with a
non-finite AUC-PR or energy_check != ok makes the comparison incomplete.
A1 muxlin(tuned) vs Spikformer-style(tuned) (criterion 2), A2 muxlin(tuned) vs sdlin(tuned), A3 sdlin(tuned) vs
Spikformer-style(tuned) (criterion 2), A4 muxlin(tuned) vs muxlin(default). Report only: AUC-ROC, raw F1, point-adjusted F1.
usage: python summarize_ad_tune.py --stage lr|layers|res|final
"""
import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from summarize_rhythm import parse                                        # noqa: E402

TUNE = os.path.join(HERE, 'results', 'flif-v1-ad-spkmux-tune')
DSS, SEEDS = ('MSL', 'SMAP', 'PSM'), (42, 7, 2026)
LR0, L0, R0 = 0.0005, 2, 'or'
GRID = {'lr': (0.0001, 0.00025, 0.0005, 0.001, 0.002), 'layers': (1, 2, 3), 'res': ('or', 'iand', 'add')}
MUX, SDL, REF = 'muxlin', 'sdlin', 'spk_ssa'
CONDKEY = {MUX: 'spkmx_lif_muxlin_or', SDL: 'spkmx_lif_sdlin_or', REF: 'spk_ssa'}
THR = 2.0


def split(cond):
    if cond == REF:
        return REF, '-'
    p = cond.split('_')
    return p[2], p[3]


def load():
    D = {}
    for r in parse(os.path.join(TUNE, 'raw.txt')):
        base, res = split(r['cond'])
        k = (base, res, r['ds'], float(r['lr']), int(r['layers']), int(r['holdout']), int(r['seed']))
        if k in D:
            raise SystemExit(f'duplicate run {k}')
        D[k] = r
    return D


def seeds(D, b, ds, cfg, ho):
    v = [D.get((b, cfg[0], ds, float(cfg[1]), int(cfg[2]), ho, s)) for s in SEEDS]
    return None if any(x is None for x in v) else v


def write_sel(stage, sel, scores):
    p = os.path.join(TUNE, f'selected_{stage}.json')
    if os.path.exists(p):
        old = json.load(open(p))
        if old['selected'] != sel:
            raise SystemExit(f'{p} exists with a different selection {old["selected"]} (new {sel}); not overwritten')
        return
    json.dump({'stage': stage, 'selected': sel, 'scores': scores}, open(p, 'x'), indent=1)


def ok_rows(rr):
    return rr is not None and all(math.isfinite(float(x['aucpr'])) and x.get('energy_check') == 'ok' for x in rr)


def compare(A, B):
    """A, B: ds -> 3 final rows. Relative AUC-PR differences."""
    dm, pairs, ers, txt = [], [], [], []
    for ds in DSS:
        ra, rb = A(ds), B(ds)
        if not (ok_rows(ra) and ok_rows(rb)):
            return None
        r_ = [100 * (a['aucpr'] - b['aucpr']) / b['aucpr'] for a, b in zip(ra, rb)]
        m = float(np.mean(r_)); dm.append(m); pairs += r_
        ers.append(float(np.mean([a['total_fft_uJ_corr'] for a in ra])) / float(np.mean([b['total_fft_uJ_corr'] for b in rb])))
        cv = 'b' if m >= THR and sum(z > 0 for z in r_) >= 2 else 'w' if m <= -THR and sum(z < 0 for z in r_) >= 2 else '.'
        txt.append(f'{ds} {m:+.2f} ({", ".join(f"{z:+.2f}" for z in r_)}) {cv}')
    R = float(np.mean(dm)); need = 2 * len(pairs) // 3
    hi, lo = sum(z > 0 for z in pairs), sum(z < 0 for z in pairs)
    v = 'better' if R >= THR and hi >= need else 'worse' if R <= -THR and lo >= need else 'no clear difference'
    er = float(np.mean(ers))
    return dict(R=R, hi=hi, n=len(pairs), verdict=v, energy=er, crit='met' if v == 'better' or (R >= -THR and er <= 0.5) else 'not met',
                cells=txt)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--stage', choices=['lr', 'layers', 'res', 'final'], required=True); a = ap.parse_args()
    D = load()
    sel = {s: json.load(open(os.path.join(TUNE, f'selected_{s}.json')))['selected'] for s in ('lr', 'layers', 'res')
           if os.path.exists(os.path.join(TUNE, f'selected_{s}.json'))}
    L = [f'# Anomaly-detection tuning of the fusion MUX — stage {a.stage}', '']
    if a.stage != 'final':
        bases = (MUX, SDL, REF) if a.stage == 'lr' else (MUX, SDL)
        dflt = {'lr': LR0, 'layers': L0, 'res': R0}[a.stage]
        out, scores, complete = {}, {}, True
        L += ['Selection score = seed-mean best validation MSE / that of the default (holdout: first 80 % trained, last 20 % '
              'validated; lower is better). No test values are computed in these runs.', '',
              '| condition | dataset | candidate | val MSE per seed (42, 7, 2026) | mean | score | best epoch (mean) | selected |',
              '|---|---|---|---|---:|---:|---:|---|']
        for b in bases:
            ck = CONDKEY[b]; out[ck], scores[ck] = {}, {}
            for ds in DSS:
                cfg = {'lr': lambda v: ('-' if b == REF else R0, v, L0), 'layers': lambda v: (R0, sel['lr'][ck][ds], v),
                       'res': lambda v: (v, sel['lr'][ck][ds], sel['layers'][ck][ds])}[a.stage]
                ref = seeds(D, b, ds, cfg(dflt), 1)
                m0 = float(np.mean([x['val_mse_best'] for x in ref])) if ref else None
                sc = {}
                for v in GRID[a.stage]:
                    rr = seeds(D, b, ds, cfg(v), 1)
                    if rr is None or m0 is None or not all(math.isfinite(float(x['val_mse_best'])) for x in rr):
                        complete = False
                        L.append(f'| {b} | {ds} | {v} | incomplete | | | | |'); continue
                    m = float(np.mean([x['val_mse_best'] for x in rr])); sc[v] = m / m0
                    per = ', '.join('%.5f' % x['val_mse_best'] for x in rr)
                    be = float(np.mean([x['best_epoch'] for x in rr]))
                    L.append(f"| {b} | {ds} | {v}{' (default)' if v == dflt else ''} | {per} | {m:.5f} | {sc[v]:.4f} | {be:.1f} | |")
                if len(sc) == len(GRID[a.stage]):
                    best = min(sc.values())
                    win = dflt if abs(sc[dflt] - best) < 1e-12 else min(sc, key=lambda v: sc[v])
                    out[ck][ds] = win
                    L.append(f'| {b} | {ds} | **selected: {win}** | | | | | yes |')
                scores[ck][ds] = {str(k): v for k, v in sc.items()}
        if complete:
            write_sel(a.stage, out, scores)
            L += ['', f'**Selection written:** selected_{a.stage}.json = {json.dumps(out)}']
        else:
            L += ['', '**Incomplete:** no selection written.']
    else:
        def tuned(b):
            ck = CONDKEY[b]
            if b == REF:
                return lambda ds: seeds(D, b, ds, ('-', sel['lr'][ck][ds], L0), 0)
            return lambda ds: seeds(D, b, ds, (sel['res'][ck][ds], sel['lr'][ck][ds], sel['layers'][ck][ds]), 0)
        mux_default = lambda ds: seeds(D, MUX, ds, (R0, LR0, L0), 0)
        settings = {CONDKEY[b]: {ds: dict(lr=sel['lr'][CONDKEY[b]][ds], layers=(L0 if b == REF else sel['layers'][CONDKEY[b]][ds]),
                                          res=('-' if b == REF else sel['res'][CONDKEY[b]][ds])) for ds in DSS} for b in (MUX, SDL, REF)}
        L += ['Tuned settings: ' + json.dumps(settings), '',
              '| question | R (AUC-PR, %) | pairs higher | verdict | energy A/B | criterion (2) |', '|---|---:|---|---|---:|---|']
        det = ['', '| comparison | per-dataset mean r (seeds 42, 7, 2026) |', '|---|---|']
        for q, A, B, crit in (('A1 muxlin(tuned) vs Spikformer-style(tuned)', tuned(MUX), tuned(REF), True),
                              ('A2 muxlin(tuned) vs sdlin(tuned)', tuned(MUX), tuned(SDL), False),
                              ('A3 sdlin(tuned) vs Spikformer-style(tuned)', tuned(SDL), tuned(REF), True),
                              ('A4 muxlin(tuned) vs muxlin(default)', tuned(MUX), mux_default, False)):
            c = compare(A, B)
            if c is None:
                L.append(f'| {q} | | | incomplete | | |'); continue
            L.append(f"| {q} | {c['R']:+.2f} | {c['hi']}/{c['n']} | {c['verdict']} | {c['energy']:.2f} | {c['crit'] if crit else ''} |")
            det.append(f"| {q.split(' ')[0]} | {' ; '.join(c['cells'])} |")
        L += det
        for ds in DSS:
            L += ['', f'## {ds} (seed mean ± std; report)', '',
                  '| condition | AUC-PR | AUC-ROC | raw F1 | point-adjusted F1 | energy µJ | best epoch |', '|---|---:|---:|---:|---:|---:|---:|']
            for name, f in (('Spikformer-style (tuned)', tuned(REF)), ('sdlin (tuned)', tuned(SDL)), ('muxlin (tuned)', tuned(MUX)),
                            ('muxlin (default)', mux_default)):
                rr = f(ds)
                if rr is None:
                    L.append(f'| {name} | — | | | | | |'); continue
                ms = lambda k: f"{np.mean([100 * x[k] for x in rr]):.2f} ± {np.std([100 * x[k] for x in rr]):.2f}"
                L.append(f"| {name} | {ms('aucpr')} | {ms('aucroc')} | {ms('raw_f1')} | {ms('pa_f1')} | "
                         f"{np.mean([x['total_fft_uJ_corr'] for x in rr]):.2f} | {np.mean([x['best_epoch'] for x in rr]):.1f} |")
    p = os.path.join(TUNE, f'summary_{a.stage}.md')
    os.makedirs(TUNE, exist_ok=True)
    open(p, 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    main()
