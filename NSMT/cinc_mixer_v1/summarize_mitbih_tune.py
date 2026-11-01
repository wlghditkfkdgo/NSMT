"""MIT-BIH tuning of the fusion MUX (2026-11-01, PROJECT_LOG, exp/f-lif-pop-v3). User decisions 11-01: select on a validation split,
retrain on the whole training file, tune the Spikformer-style baseline too, the ETT order (lr -> mixer layers -> residual); sdlin too.

Tuning runs (suite flif-v1-mitbih-spkmux-tune, val_fold 0): the training file split by create_loader's StratifiedKFold (5 folds,
random_state 42), fold 0 = validation; selection score = the last-epoch validation macro-F1 (val_last.csv; the protocol tests the
last-epoch weights). Their test values are never used here.
    lr      {0.0025, 0.005, 0.01*, 0.02}     muxlin, sdlin (OR, 2 mixer layers), Spikformer-style
    layers  {1, 2*, 3}                       muxlin, sdlin at the selected lr
    res     {OR*, IAND, ADD}                 muxlin, sdlin at the selected lr and layers                    (* = default)
Selection: the highest seed-mean validation macro-F1; ties (< 1e-9) keep the default; a missing run -> no selection;
selected_<stage>.json is never overwritten with a different selection.
Final (--stage final): the tuned settings retrained on the whole training file (same suite, val_fold -1; a setting equal to the
default reuses the 11-01 runs / the reused Spikformer-style rows of results/flif-v1-mitbih-spkmux), test macro-F1 of the last
epoch. MIT-BIH rule: per-seed differences (%p), better = mean >= +0.5 and >= 2 of 3 seeds, worse mirror; criterion (2) = better or
(mean >= -0.5 and energy <= 0.5x); energy = total_fft_uJ_corr of energy_recount_mitbih.py (integer rule for ADD) when present,
else the logged total_fft_uJ. TM1 muxlin(tuned) vs Spikformer-style(tuned), TM2 muxlin(tuned) vs sdlin(tuned), TM3 sdlin(tuned) vs
Spikformer-style(tuned), TM4 muxlin(tuned) vs muxlin(default).
usage: python summarize_mitbih_tune.py --stage lr|layers|res|final
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from summarize_rhythm import parse                                        # noqa: E402

TUNE = os.path.join(HERE, 'results', 'flif-v1-mitbih-spkmux-tune')
BASE = os.path.join(HERE, 'results', 'flif-v1-mitbih-spkmux')
SEEDS = (42, 7, 2026)
LR0, L0, R0 = 0.01, 2, 'or'
GRID = {'lr': (0.0025, 0.005, 0.01, 0.02), 'layers': (1, 2, 3), 'res': ('or', 'iand', 'add')}
MUX, SDL, REF = 'muxlin', 'sdlin', 'spk_ssa'
CONDKEY = {MUX: 'spkmx_lif_muxlin_or', SDL: 'spkmx_lif_sdlin_or', REF: 'spk_ssa'}


def split(cond):
    if cond == REF:
        return REF, '-'
    p = cond.split('_')
    return p[2], p[3]


def load():
    energy = {}
    for path in (os.path.join(TUNE, 'energy_recount.csv'), os.path.join(BASE, 'energy_recount.csv')):
        if os.path.exists(path):
            for e in csv.DictReader(open(path)):
                energy[(e['cond'], e.get('tag', ''), int(e['seed']), int(e['val_fold']))] = float(e['total_fft_uJ_corr'])
    D = {}
    for path, tuned in ((os.path.join(TUNE, 'raw.txt'), True), (os.path.join(BASE, 'raw.txt'), False), (os.path.join(BASE, 'baseline_spk_ssa_raw.txt'), False)):
        for r in parse(path):
            base, res = split(r['cond'])
            lr = float(r['lr']) if tuned else LR0
            lay = int(r['layers']) if tuned else L0
            vf = int(r['val_fold']) if tuned else -1
            k = (base, res, lr, lay, vf, int(r['seed']))
            if k in D:
                raise SystemExit(f'duplicate run {k}')
            r['E'] = energy.get((r['cond'], str(r.get('tag', '')) if tuned else '', int(r['seed']), vf), r['total_fft_uJ'])
            D[k] = r
    return D


def stage_cfg(stage, b, sel):
    """value -> (res, lr, layers) of the stage for condition b given earlier selections."""
    ck = CONDKEY[b]
    if stage == 'lr':
        return lambda v: ('-' if b == REF else R0, v, L0)
    if stage == 'layers':
        return lambda v: (R0, sel['lr'][ck], v)
    return lambda v: (v, sel['lr'][ck], sel['layers'][ck])


def seeds(D, b, cfg, vf):
    v = [D.get((b, cfg[0], cfg[1], cfg[2], vf, s)) for s in SEEDS]
    return None if any(x is None for x in v) else v


def write_sel(stage, sel, scores):
    p = os.path.join(TUNE, f'selected_{stage}.json')
    if os.path.exists(p):
        old = json.load(open(p))
        if old['selected'] != sel:
            raise SystemExit(f'{p} exists with a different selection {old["selected"]} (new {sel}); not overwritten')
        return
    json.dump({'stage': stage, 'selected': sel, 'scores': scores}, open(p, 'x'), indent=1)


def tuned_cfg(b, sel):
    ck = CONDKEY[b]
    if b == REF:
        return ('-', sel['lr'][ck], L0)
    return (sel['res'][ck], sel['lr'][ck], sel['layers'][ck])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--stage', choices=['lr', 'layers', 'res', 'final'], required=True); a = ap.parse_args()
    D = load()
    sel = {s: json.load(open(os.path.join(TUNE, f'selected_{s}.json')))['selected'] for s in ('lr', 'layers', 'res')
           if os.path.exists(os.path.join(TUNE, f'selected_{s}.json'))}
    L = [f'# MIT-BIH tuning of the fusion MUX — stage {a.stage}', '']
    if a.stage != 'final':
        bases = (MUX, SDL, REF) if a.stage == 'lr' else (MUX, SDL)
        out, scores, complete = {}, {}, True
        dflt = {'lr': LR0, 'layers': L0, 'res': R0}[a.stage]
        L += ['Selection score = seed-mean last-epoch validation macro-F1 (%; validation = fold 0 of 5 of the training file).', '',
              '| condition | candidate | validation macro-F1 per seed (42, 7, 2026) | mean | selected |', '|---|---|---|---:|---|']
        for b in bases:
            cfg = stage_cfg(a.stage, b, sel)
            sc = {}
            for v in GRID[a.stage]:
                rr = seeds(D, b, cfg(v), 0)
                if rr is None:
                    complete = False
                    L.append(f'| {b} | {v} | incomplete | | |'); continue
                sc[v] = float(np.mean([100 * x['val_f1_last'] for x in rr]))
                per = ', '.join('%.2f' % (100 * x['val_f1_last']) for x in rr)
                L.append(f"| {b} | {v}{' (default)' if v == dflt else ''} | {per} | {sc[v]:.2f} | |")
            if len(sc) == len(GRID[a.stage]):
                best = max(sc.values())
                win = dflt if abs(sc[dflt] - best) < 1e-9 else max(sc, key=lambda v: sc[v])
                out[CONDKEY[b]] = win
                L.append(f'| {b} | **selected: {win}** | | | yes |')
            scores[CONDKEY[b]] = {str(k): v for k, v in sc.items()}
        if complete:
            write_sel(a.stage, out, scores)
            L += ['', f'**Selection written:** selected_{a.stage}.json = {json.dumps(out)}']
        else:
            L += ['', '**Incomplete:** no selection written.']
    else:
        full = {}
        for b in (MUX, SDL, REF):
            full[b] = seeds(D, b, tuned_cfg(b, sel), -1)
        mux_default = seeds(D, MUX, (R0, LR0, L0), -1)
        L += ['Tuned settings (res, lr, mixer layers): ' + json.dumps({b: tuned_cfg(b, sel) for b in (MUX, SDL, REF)}), '',
              '| question | mean %p | per seed (42, 7, 2026) | verdict | energy A/B | criterion (2) |', '|---|---:|---|---|---:|---|']
        for q, A, B, crit in (('TM1 muxlin(tuned) vs Spikformer-style(tuned)', full[MUX], full[REF], True),
                              ('TM2 muxlin(tuned) vs sdlin(tuned)', full[MUX], full[SDL], False),
                              ('TM3 sdlin(tuned) vs Spikformer-style(tuned)', full[SDL], full[REF], True),
                              ('TM4 muxlin(tuned) vs muxlin(default)', full[MUX], mux_default, False)):
            if A is None or B is None:
                L.append(f'| {q} | incomplete | | incomplete | | |'); continue
            d = [100 * (x['f1'] - y['f1']) for x, y in zip(A, B)]
            m = float(np.mean(d))
            v = 'better' if m >= 0.5 and sum(z > 0 for z in d) >= 2 else 'worse' if m <= -0.5 and sum(z < 0 for z in d) >= 2 else 'no clear difference'
            er = float(np.mean([x['E'] for x in A])) / float(np.mean([y['E'] for y in B]))
            c2 = ('met' if v == 'better' or (m >= -0.5 and er <= 0.5) else 'not met') if crit else ''
            L.append(f"| {q} | {m:+.2f} | {', '.join(f'{z:+.2f}' for z in d)} | {v} | {er:.2f} | {c2} |")
        L += ['', '| condition | macro-F1 % | F1 N / S / V / F / Q | energy µJ |', '|---|---:|---|---:|']
        for name, rr in (('Spikformer-style (tuned)', full[REF]), ('sdlin (tuned)', full[SDL]), ('muxlin (tuned)', full[MUX]), ('muxlin (default)', mux_default)):
            if rr is None:
                L.append(f'| {name} | — | | |'); continue
            L.append(f"| {name} | {np.mean([100 * x['f1'] for x in rr]):.2f} ± {np.std([100 * x['f1'] for x in rr]):.2f} | "
                     + ' / '.join(f"{np.mean([100 * x[f'f1_{i}'] for x in rr]):.2f}" for i in range(5)) + f" | {np.mean([x['E'] for x in rr]):.2f} |")
    out_p = os.path.join(TUNE, f'summary_{a.stage}.md')
    open(out_p, 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    main()
