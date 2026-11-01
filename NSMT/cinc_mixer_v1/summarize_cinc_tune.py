"""CinC tuning of the fusion MUX (2026-11-01, PROJECT_LOG, exp/f-lif-pop-v3; user: "CinC의 muxlin도 튜닝을 진행하자"). CinC 2017 fold 0,
seeds 42 / 7 / 2026; the existing validation split (683 records) selects, the test score (mean F1 of N / AF / Other) of the
best-validation weights is reported (the earlier CinC protocol). Same order as ETT; sdlin and the Spikformer-style model (lr) too.
    lr      {2.5e-4, 5e-4, 1e-3*, 2e-3}     muxlin, sdlin (OR, 2 mixer layers), Spikformer-style
    layers  {1, 2*, 3}                       muxlin, sdlin at the selected lr
    res     {OR*, IAND, ADD}                 muxlin, sdlin at the selected lr and layers                    (* = default)
Default runs are reused: muxlin = flif-v1-rhythm-spk-mux-cinc (10-31), sdlin = flif-v1-rhythm-spk-cinc (B1, incl. IAND / ADD),
Spikformer-style = flif-v1-rhythm-cinc (copy of the other session's runs, commit 9094bfaf9). New runs: flif-v1-rhythm-spk-tune-cinc.
Selection: the highest seed-mean best-validation score; ties (< 1e-9) keep the default; a missing run -> no selection;
selected_<stage>.json is never overwritten with a different selection.
Final (--stage final, pre-registered): TC1 muxlin(tuned) vs Spikformer-style(tuned) (criterion 2), TC2 muxlin(tuned) vs sdlin(tuned),
TC3 sdlin(tuned) vs Spikformer-style(tuned) (criterion 2), TC4 muxlin(tuned) vs muxlin(default). CinC rule: per-seed test-score
differences (%p), better = mean >= +0.5 and >= 2 of 3 seeds, worse mirror; criterion (2) = better or (mean >= -0.5 and energy <= 0.5x);
energy = total_fft_int_uJ (integer rule) when present, else total_fft_uJ (the copied Spikformer-style rows; equal for 0/1 models).
usage: python summarize_cinc_tune.py --stage lr|layers|res|final
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

R = os.path.join(HERE, 'results')
TUNE = os.path.join(R, 'flif-v1-rhythm-spk-tune-cinc')
SRC = [(os.path.join(TUNE, 'raw.txt'), True), (os.path.join(R, 'flif-v1-rhythm-spk-mux-cinc', 'raw.txt'), False),
       (os.path.join(R, 'flif-v1-rhythm-spk-cinc', 'raw.txt'), False), (os.path.join(R, 'flif-v1-rhythm-cinc', 'raw.txt'), False)]
SEEDS = (42, 7, 2026)
LR0, L0, R0 = 0.001, 2, 'or'
GRID = {'lr': (0.00025, 0.0005, 0.001, 0.002), 'layers': (1, 2, 3), 'res': ('or', 'iand', 'add')}
MUX, SDL, REF = 'muxlin', 'sdlin', 'spk_ssa'
CONDKEY = {MUX: 'spk_lif_muxlin_or', SDL: 'spk_lif_sdlin_or', REF: 'spk_ssa'}


def split(cond):
    """spk_lif_<comb>_<cf> -> (comb, cf); spk_ssa -> ('spk_ssa', '-'); anything else -> None."""
    if cond == REF:
        return REF, '-'
    p = cond.split('_')
    if len(p) == 4 and p[0] == 'spk' and p[1] == 'lif' and p[2] in (MUX, SDL):
        return p[2], p[3]
    return None


def load():
    D = {}
    for path, tuned in SRC:
        for r in parse(path):
            sp = split(r['cond'])
            if sp is None or int(r.get('fold', 0)) != 0 or int(r.get('crop', 0)) != 0:
                continue
            if not tuned and float(r.get('lr', LR0)) != LR0:
                continue
            base, res = sp
            lay = int(r['layers']) if tuned else L0
            k = (base, res, float(r['lr']), lay, int(r['seed']))
            if k in D:
                raise SystemExit(f'duplicate run {k} ({path})')
            r['val'] = float(str(r['best_epoch_val']).split(':')[1])
            r['E'] = float(r['total_fft_int_uJ']) if 'total_fft_int_uJ' in r else float(r['total_fft_uJ'])
            D[k] = r
    return D


def seeds(D, b, cfg):
    v = [D.get((b, cfg[0], cfg[1], cfg[2], s)) for s in SEEDS]
    return None if any(x is None for x in v) else v


def write_sel(stage, sel, scores):
    p = os.path.join(TUNE, f'selected_{stage}.json')
    if os.path.exists(p):
        old = json.load(open(p))
        if old['selected'] != sel:
            raise SystemExit(f'{p} exists with a different selection {old["selected"]} (new {sel}); not overwritten')
        return
    json.dump({'stage': stage, 'selected': sel, 'scores': scores}, open(p, 'x'), indent=1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--stage', choices=['lr', 'layers', 'res', 'final'], required=True); a = ap.parse_args()
    os.makedirs(TUNE, exist_ok=True)
    D = load()
    sel = {s: json.load(open(os.path.join(TUNE, f'selected_{s}.json')))['selected'] for s in ('lr', 'layers', 'res')
           if os.path.exists(os.path.join(TUNE, f'selected_{s}.json'))}
    L = [f'# CinC tuning of the fusion MUX — stage {a.stage}', '']
    if a.stage != 'final':
        bases = (MUX, SDL, REF) if a.stage == 'lr' else (MUX, SDL)
        dflt = {'lr': LR0, 'layers': L0, 'res': R0}[a.stage]
        out, scores, complete = {}, {}, True
        L += ['Selection score = seed-mean best-validation CinC score (%).', '',
              '| condition | candidate | validation per seed (42, 7, 2026) | mean | test mean (report) | selected |', '|---|---|---|---:|---:|---|']
        for b in bases:
            ck = CONDKEY[b]
            cfg = {'lr': lambda v: ('-' if b == REF else R0, v, L0), 'layers': lambda v: (R0, sel['lr'][ck], v),
                   'res': lambda v: (v, sel['lr'][ck], sel['layers'][ck])}[a.stage]
            sc = {}
            for v in GRID[a.stage]:
                rr = seeds(D, b, cfg(v))
                if rr is None:
                    complete = False
                    L.append(f'| {b} | {v} | incomplete | | | |'); continue
                sc[v] = float(np.mean([100 * x['val'] for x in rr]))
                per = ', '.join('%.2f' % (100 * x['val']) for x in rr)
                L.append(f"| {b} | {v}{' (default)' if v == dflt else ''} | {per} | {sc[v]:.2f} | "
                         f"{np.mean([100 * x['score'] for x in rr]):.2f} | |")
            if len(sc) == len(GRID[a.stage]):
                best = max(sc.values())
                win = dflt if abs(sc[dflt] - best) < 1e-9 else max(sc, key=lambda v: sc[v])
                out[ck] = win
                L.append(f'| {b} | **selected: {win}** | | | | yes |')
            scores[ck] = {str(k): v for k, v in sc.items()}
        if complete:
            write_sel(a.stage, out, scores)
            L += ['', f'**Selection written:** selected_{a.stage}.json = {json.dumps(out)}']
        else:
            L += ['', '**Incomplete:** no selection written.']
    else:
        def tuned(b):
            ck = CONDKEY[b]
            return ('-', sel['lr'][ck], L0) if b == REF else (sel['res'][ck], sel['lr'][ck], sel['layers'][ck])
        full = {b: seeds(D, b, tuned(b)) for b in (MUX, SDL, REF)}
        mux_default = seeds(D, MUX, (R0, LR0, L0))
        L += ['Tuned settings (res, lr, mixer layers): ' + json.dumps({b: tuned(b) for b in (MUX, SDL, REF)}), '',
              '| question | mean %p | per seed (42, 7, 2026) | verdict | energy A/B | criterion (2) |', '|---|---:|---|---|---:|---|']
        for q, A, B, crit in (('TC1 muxlin(tuned) vs Spikformer-style(tuned)', full[MUX], full[REF], True),
                              ('TC2 muxlin(tuned) vs sdlin(tuned)', full[MUX], full[SDL], False),
                              ('TC3 sdlin(tuned) vs Spikformer-style(tuned)', full[SDL], full[REF], True),
                              ('TC4 muxlin(tuned) vs muxlin(default)', full[MUX], mux_default, False)):
            if A is None or B is None:
                L.append(f'| {q} | incomplete | | incomplete | | |'); continue
            d = [100 * (x['score'] - y['score']) for x, y in zip(A, B)]
            m = float(np.mean(d))
            v = 'better' if m >= 0.5 and sum(z > 0 for z in d) >= 2 else 'worse' if m <= -0.5 and sum(z < 0 for z in d) >= 2 else 'no clear difference'
            er = float(np.mean([x['E'] for x in A])) / float(np.mean([y['E'] for y in B]))
            c2 = ('met' if v == 'better' or (m >= -0.5 and er <= 0.5) else 'not met') if crit else ''
            L.append(f"| {q} | {m:+.2f} | {', '.join(f'{z:+.2f}' for z in d)} | {v} | {er:.2f} | {c2} |")
        L += ['', '| condition | validation % | test score % | macro-F1 % | AF / Other / Noisy F1 | energy µJ |', '|---|---:|---:|---:|---|---:|']
        for name, rr in (('Spikformer-style (tuned)', full[REF]), ('sdlin (tuned)', full[SDL]), ('muxlin (tuned)', full[MUX]), ('muxlin (default)', mux_default)):
            if rr is None:
                L.append(f'| {name} | — | | | | |'); continue
            L.append(f"| {name} | {np.mean([100 * x['val'] for x in rr]):.2f} | {np.mean([100 * x['score'] for x in rr]):.2f} ± {np.std([100 * x['score'] for x in rr]):.2f} | "
                     f"{np.mean([100 * x['macro_f1'] for x in rr]):.2f} | " + ' / '.join(f"{np.mean([100 * x[f'f1_{i}'] for x in rr]):.2f}" for i in (1, 2, 3))
                     + f" | {np.mean([x['E'] for x in rr]):.1f} |")
    out_p = os.path.join(TUNE, f'summary_{a.stage}.md')
    open(out_p, 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    main()
