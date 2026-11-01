"""ETT tuning of the fusion MUX (2026-11-01, PROJECT_LOG, exp/f-lif-pop-v3). Cells ETTh2 + ETTm2 x H96 / H720 (user decision 12),
seeds 42 / 7 / 2026; conditions muxlin and sdlin (SpkMixer LIF) and the Spikformer-style model (lr stage only).

Stages (chained; user decisions 11-01: select on all 4 cells, lower lr values added, the Spikformer-style model tuned on the same
lr grid, residual IAND and ADD against OR, sdlin tuned too):
    lr      {1e-4, 2.5e-4, 5e-4*, 1e-3, 2e-3}       muxlin, sdlin (OR residual, Table 13 layers), Spikformer-style
    layers  {1, 2, 3} (ETTh2 1*, ETTm2 2*)          muxlin, sdlin at their selected lr
    res     {OR*, IAND, ADD}                         muxlin, sdlin at their selected lr and layers           (* = default)
Selection (validation only; the rule of the other session's summarize_tune.py, applied per condition AND dataset because Table 13
defaults differ per dataset): score(v) = mean over H96 / H720 of [seed-mean best validation L1 at v] / [seed-mean at the default];
the lowest score wins, ties (< 1e-12) keep the default; any missing run -> no selection. Writes selected_<stage>.json (refuses to
overwrite a different selection).
Verdict 1 (report): selected vs default on test MSE per dataset (2 cells, 6 pairs: better if R <= -1 % and >= 4 pairs lower).
Final (--stage final, pre-registered): T1 muxlin(tuned) vs Spikformer-style(tuned, criterion (2)), T2 muxlin(tuned) vs
sdlin(tuned), T3 sdlin(tuned) vs Spikformer-style(tuned, criterion (2)), T4 muxlin(tuned) vs muxlin(default); 4 cells,
better if R <= -1 % and >= 8 of 12 pairs lower (mirror worse), criterion (2) = better or (R <= +1 % and energy <= 0.5x);
energy = total_fft_uJ_corr (energy_recount_ett.py; integer rule for ADD).
Runs: results/flif-v1-ett-spkmux-tune/raw.txt (tag / lr / layers), the 10-31 defaults results/flif-v1-ett-spkmux/raw.txt and the
reused Spikformer-style rows results/flif-v1-ett-spkmux/baseline_spk_ssa_raw.txt, with their energy_recount files.
usage: python summarize_ett_tune.py --stage lr|layers|res|final
"""
import argparse
import csv
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from summarize_rhythm import parse                                        # noqa: E402

TUNE = os.path.join(HERE, 'results', 'flif-v1-ett-spkmux-tune')
BASE = os.path.join(HERE, 'results', 'flif-v1-ett-spkmux')
CELLS = (('ETTh2', 96), ('ETTh2', 720), ('ETTm2', 96), ('ETTm2', 720))
DSS, PLS, SEEDS = ('ETTh2', 'ETTm2'), (96, 720), (42, 7, 2026)
L0 = {'ETTh2': 1, 'ETTm2': 2}
LR0 = 0.0005
LRS = (0.0001, 0.00025, 0.0005, 0.001, 0.002)
LAYERS = (1, 2, 3)
RES = ('or', 'iand', 'add')
MUX, SDL, REF = 'muxlin', 'sdlin', 'spk_ssa'
CONDKEY = {MUX: 'spkmx_lif_muxlin_or', SDL: 'spkmx_lif_sdlin_or', REF: 'spk_ssa'}


def split(cond):
    """cond -> (base, residual): spkmx_lif_<comb>_<cf> -> (comb, cf); spk_ssa -> ('spk_ssa', '-')."""
    if cond == REF:
        return REF, '-'
    p = cond.split('_')
    return p[2], p[3]


def load():
    energy = {}
    for path in (os.path.join(TUNE, 'energy_recount.csv'), os.path.join(BASE, 'energy_recount.csv'), os.path.join(BASE, 'baseline_spk_ssa_energy.csv')):
        if os.path.exists(path):
            for e in csv.DictReader(open(path)):
                energy[(e['cond'], e.get('tag', '') or '', e['ds'], int(e['pl']), int(e['seed']))] = float(e['total_fft_uJ_corr'])
    D = {}
    srcs = [(os.path.join(TUNE, 'raw.txt'), True), (os.path.join(BASE, 'raw.txt'), False), (os.path.join(BASE, 'baseline_spk_ssa_raw.txt'), False)]
    for path, tuned in srcs:
        for r in parse(path):
            if r['cond'] not in ('spkmx_lif_muxlin_or', 'spkmx_lif_sdlin_or', 'spkmx_lif_muxlin_iand', 'spkmx_lif_sdlin_iand',
                                 'spkmx_lif_muxlin_add', 'spkmx_lif_sdlin_add', REF):
                continue
            base, res = split(r['cond'])
            lr = float(r['lr']) if tuned else LR0
            lay = int(r['layers']) if tuned else L0[r['ds']]
            k = (base, res, r['ds'], int(r['pl']), lr, lay, int(r['seed']))
            if k in D:
                raise SystemExit(f'duplicate run {k}')
            r['val'] = float(str(r['best_epoch_val_epochs']).split(':')[1])
            r['E'] = energy.get((r['cond'], str(r.get('tag', '')) if tuned else '', r['ds'], int(r['pl']), int(r['seed'])), float('nan'))
            D[k] = r
    return D


def cell_mean(D, base, res, ds, pl, lr, lay, key):
    v = [D.get((base, res, ds, pl, lr, lay, s)) for s in SEEDS]
    if any(x is None for x in v):
        return None
    return float(np.mean([x[key] for x in v]))


def select(D, base, ds, cands, cfg):
    """cfg(v) -> (res, lr, layers); cands[0..] include the default cfg_default = cfg(default)."""
    scores = {}
    dflt = cands['default']
    for v in cands['values']:
        num = [cell_mean(D, base, *cfg(v)[:1], ds, pl, *cfg(v)[1:], 'val') for pl in PLS]
        den = [cell_mean(D, base, *cfg(dflt)[:1], ds, pl, *cfg(dflt)[1:], 'val') for pl in PLS]
        if any(x is None for x in num + den):
            return None, scores
        scores[v] = float(np.mean([a / b for a, b in zip(num, den)]))
    best = min(scores.values())
    win = dflt if abs(scores[dflt] - best) < 1e-12 else min(scores, key=lambda v: (scores[v], v != dflt))
    return win, scores


def compare(D, A, B, cells):
    """A, B: functions (ds, pl) -> list of 3 rows (or None). Returns verdict dict over the given cells."""
    cm, pairs, ers, ok, txt, cmae = [], [], [], True, [], []
    for ds, pl in cells:
        ra, rb = A(ds, pl), B(ds, pl)
        if ra is None or rb is None or any(x is None for x in ra + rb):
            return None
        r_ = [100 * (a['mse'] - b['mse']) / b['mse'] for a, b in zip(ra, rb)]
        m = float(np.mean(r_)); cm.append(m); pairs += r_
        cmae.append(float(np.mean([100 * (a['mae'] - b['mae']) / b['mae'] for a, b in zip(ra, rb)])))
        ea = float(np.mean([a['E'] for a in ra])); eb = float(np.mean([b['E'] for b in rb]))
        ers.append(ea / eb if math.isfinite(ea) and math.isfinite(eb) else float('nan'))
        cv = 'b' if m <= -1 and sum(z < 0 for z in r_) >= 2 else 'w' if m >= 1 and sum(z > 0 for z in r_) >= 2 else '.'
        txt.append(f'{ds} H{pl} {m:+.2f} ({", ".join(f"{z:+.2f}" for z in r_)}) {cv}')
    R = float(np.mean(cm)); need = 2 * len(pairs) // 3
    lo, hi = sum(z < 0 for z in pairs), sum(z > 0 for z in pairs)
    v = 'better' if R <= -1 and lo >= need else 'worse' if R >= 1 and hi >= need else 'no clear difference'
    er = float(np.mean(ers))
    return dict(R=R, lo=lo, n=len(pairs), verdict=v, energy=er, crit='met' if v == 'better' or (R <= 1 and er <= 0.5) else 'not met',
                R_mae=float(np.mean(cmae)), cells=txt)


def rows(D, base, res, lr, lay):
    """per-dataset settings: lr / lay are dicts ds -> value (or scalars)."""
    def f(ds, pl):
        l_ = lr[ds] if isinstance(lr, dict) else lr
        y = lay[ds] if isinstance(lay, dict) else lay
        r_ = res[ds] if isinstance(res, dict) else res
        v = [D.get((base, r_, ds, pl, l_, y, s)) for s in SEEDS]
        return None if any(x is None for x in v) else v
    return f


def write_sel(stage, sel, scores):
    p = os.path.join(TUNE, f'selected_{stage}.json')
    new = {'stage': stage, 'selected': sel, 'scores': scores}
    if os.path.exists(p):
        old = json.load(open(p))
        if old['selected'] != sel:
            raise SystemExit(f'{p} exists with a different selection {old["selected"]} (new {sel}); not overwritten')
        return
    json.dump(new, open(p, 'x'), indent=1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--stage', choices=['lr', 'layers', 'res', 'final'], required=True); a = ap.parse_args()
    D = load()
    L = [f'# ETT tuning of the fusion MUX — stage {a.stage}', '', 'Validation-only selection per condition and dataset; test MSE for the verdicts. '
         'r = 100 (MSE_A − MSE_B) / MSE_B per seed.', '']
    sel_lr = json.load(open(os.path.join(TUNE, 'selected_lr.json')))['selected'] if a.stage != 'lr' else None
    sel_lay = json.load(open(os.path.join(TUNE, 'selected_layers.json')))['selected'] if a.stage in ('res', 'final') else None
    if a.stage in ('lr', 'layers', 'res'):
        bases = (MUX, SDL, REF) if a.stage == 'lr' else (MUX, SDL)
        sel, scores, complete = {}, {}, True
        L += ['| condition | dataset | candidate | score (val ratio) | val L1 H96 / H720 | test MSE H96 / H720 | selected |', '|---|---|---|---:|---|---|---|']
        for b in bases:
            ck = CONDKEY[b]; sel[ck], scores[ck] = {}, {}
            for ds in DSS:
                if a.stage == 'lr':
                    cands = {'values': LRS, 'default': LR0}; cfg = lambda v, ds=ds, b=b: ('-' if b == REF else 'or', v, L0[ds])
                elif a.stage == 'layers':
                    cands = {'values': LAYERS, 'default': L0[ds]}; cfg = lambda v, ds=ds, ck=ck: ('or', sel_lr[ck][ds], v)
                else:
                    cands = {'values': RES, 'default': 'or'}; cfg = lambda v, ds=ds, ck=ck: (v, sel_lr[ck][ds], sel_lay[ck][ds])
                win, sc = select(D, b, ds, cands, cfg)
                scores[ck][ds] = {str(k): v for k, v in sc.items()}
                if win is None:
                    complete = False
                sel[ck][ds] = win
                for v in cands['values']:
                    res_, lr_, lay_ = cfg(v)
                    vals = [cell_mean(D, b, res_, ds, pl, lr_, lay_, 'val') for pl in PLS]
                    mses = [cell_mean(D, b, res_, ds, pl, lr_, lay_, 'mse') for pl in PLS]
                    f = lambda x: '—' if x is None else f'{x:.4f}'
                    L.append(f"| {b} | {ds} | {v}{' (default)' if v == cands['default'] else ''} | {sc.get(v, float('nan')):.4f} | "
                             f"{f(vals[0])} / {f(vals[1])} | {f(mses[0])} / {f(mses[1])} | {'**yes**' if v == win else ''} |")
        L += ['', '## Verdict 1: selected vs default (test MSE, per dataset, 6 pairs)', '', '| condition | dataset | selected | R | pairs lower | verdict |', '|---|---|---|---:|---|---|']
        for b in bases:
            ck = CONDKEY[b]
            for ds in DSS:
                if sel[ck][ds] is None:
                    L.append(f'| {b} | {ds} | incomplete | | | |'); continue
                dflt = LR0 if a.stage == 'lr' else (L0[ds] if a.stage == 'layers' else 'or')
                cfg_ = {'lr': lambda v: ('-' if b == REF else 'or', v, L0[ds]), 'layers': lambda v: ('or', sel_lr[ck][ds], v),
                        'res': lambda v: (v, sel_lr[ck][ds], sel_lay[ck][ds])}[a.stage]
                ra, rb = cfg_(sel[ck][ds]), cfg_(dflt)
                c = compare(D, rows(D, b, ra[0], ra[1], ra[2]), rows(D, b, rb[0], rb[1], rb[2]), [(ds, 96), (ds, 720)])
                L.append(f"| {b} | {ds} | {sel[ck][ds]} | {c['R']:+.2f} | {c['lo']}/{c['n']} | {c['verdict']} |" if c else f'| {b} | {ds} | {sel[ck][ds]} | | | incomplete |')
        if complete:
            write_sel(a.stage, sel, scores)
            L += ['', f'**Selection written:** selected_{a.stage}.json = {json.dumps(sel)}']
        else:
            L += ['', '**Incomplete:** no selection written.']
    else:
        sel_res = json.load(open(os.path.join(TUNE, 'selected_res.json')))['selected']
        def tuned(b):
            ck = CONDKEY[b]
            return rows(D, b, {ds: sel_res[ck][ds] for ds in DSS}, {ds: sel_lr[ck][ds] for ds in DSS}, {ds: sel_lay[ck][ds] for ds in DSS})
        ref = rows(D, REF, '-', {ds: sel_lr[CONDKEY[REF]][ds] for ds in DSS}, L0)
        mux_default = rows(D, MUX, 'or', LR0, L0)
        Q = (('T1 muxlin(tuned) vs Spikformer-style(tuned) (criterion 2)', tuned(MUX), ref, True),
             ('T2 muxlin(tuned) vs sdlin(tuned)', tuned(MUX), tuned(SDL), False),
             ('T3 sdlin(tuned) vs Spikformer-style(tuned) (criterion 2)', tuned(SDL), ref, True),
             ('T4 muxlin(tuned) vs muxlin(default)', tuned(MUX), mux_default, False))
        L += ['Tuned settings: ' + json.dumps({CONDKEY[b]: {ds: dict(lr=sel_lr[CONDKEY[b]][ds], layers=(sel_lay[CONDKEY[b]][ds] if b != REF else L0[ds]),
                                                                  res=(sel_res[CONDKEY[b]][ds] if b != REF else '-')) for ds in DSS} for b in (MUX, SDL, REF)}), '',
              '| question | R | pairs lower | verdict | energy A/B | criterion (2) | R on MAE |', '|---|---:|---|---|---:|---|---:|']
        det = ['', '| comparison | per-cell mean r (seeds 42, 7, 2026) |', '|---|---|']
        for q, A, B, crit in Q:
            c = compare(D, A, B, CELLS)
            if c is None:
                L.append(f'| {q} | | | incomplete | | | |'); continue
            L.append(f"| {q} | {c['R']:+.2f} | {c['lo']}/{c['n']} | {c['verdict']} | {c['energy']:.2f} | {c['crit'] if crit else ''} | {c['R_mae']:+.2f} |")
            det.append(f"| {q.split(' ')[0]} | {' ; '.join(c['cells'])} |")
        L += det
        L += ['', '## Test MSE of the tuned settings (seed mean)', '', '| condition | ' + ' | '.join(f'{d} H{p}' for d, p in CELLS) + ' |', '|---|' + '---:|' * 4]
        for name, f in (('Spikformer-style (tuned)', ref), ('sdlin (tuned)', tuned(SDL)), ('muxlin (tuned)', tuned(MUX)), ('muxlin (default)', mux_default)):
            cs = []
            for ds, pl in CELLS:
                v = f(ds, pl)
                cs.append('—' if v is None else f"{np.mean([x['mse'] for x in v]):.4f} (best epoch {np.mean([float(str(x['best_epoch_val_epochs']).split(':')[0]) for x in v]):.1f})")
            L.append(f'| {name} | ' + ' | '.join(cs) + ' |')
    out = os.path.join(TUNE, f'summary_{a.stage}.md')
    open(out, 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    main()
