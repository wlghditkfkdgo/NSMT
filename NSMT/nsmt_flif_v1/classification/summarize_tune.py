"""MODI 2026-10-30: summary, selection and pre-registered verdicts of the ETT-first tuning stages (docs/MODI_PROJECT_LOG.md
2026-10-30), ETT ETTh2 x H96 / H720 (user scope for quick checks), seeds 42 / 7 / 2026, runner forecasting/scripts/run_ett_tune.sh.

The 17 compared settings: spk_ssa (Spikformer-style), mixall_<n>_bidir (Neo-mixer, n in every layer), spkneo_<n> (B form),
nsmt4_<n> ([NSMT hippocampus T=4], keys / values replaced), nsmt4q_<n> (neocortex as the query); n in {lif, flif, pop_lif, pop_flif}.
Default-setting runs (ETTh2: lr 0.0005, layers 1, reset spikede, alpha 0.5) come from the earlier suites (flif-v1-gen-ett,
flif-v1-ett-nall, flif-v1-ett-dual, flif-v1-ett-query); the other settings from flif-v1-ett-tune (RESULT fields lr / layers / reset /
alpha).

Stage lr: lr in {0.00025, 0.0005, 0.001, 0.002} for all 17 settings; every value is reported side by side (validation L1 and test MSE).
Stage layers (MODI 2026-10-31): --time_layers in {1, 2, 3, 4} (Neo-mixer layers per direction) for the 4 Neo-mixer settings only, each
at its stage-lr selection (selected_lr.json); comparisons: each Neo-mixer setting - spk_ssa (at its own base) and W4 f-LIF - LIF.
Stage reset (MODI 2026-10-31): --neo_reset in {spikede (default), subtract, zero} for the 16 settings with spikeDE neurons (population:
spikede vs zero only, see stage_values), each at its lr / layers selections; comparisons as stage lr + W4; the overall mean relative
validation per mode is reported (one mode for every spikeDE neuron).
Selection (validation only): per setting, the value with the lowest mean over the 2 cells of (cell-mean validation L1 at the value /
cell-mean validation L1 at the default); every value needs 3 seeds x 2 cells, else the selection is incomplete.
Verdict 1: selected - default on the test MSE (2 cells: r = 100 (A - B) / B per seed, R = mean of the cell means; better = R <= -1 %
and r < 0 in >= 4 of 6 pairs, worse = mirror).
Verdict 2: the comparisons each setting - spk_ssa (criterion 2: better, or R <= +1 % with energy <= 0.5x), R1 nsmt4 - spkneo,
Q1 nsmt4q - nsmt4, Q2 nsmt4q - spkneo, both at the defaults and with every setting at its own selected value; 'changed' marks a
verdict that differs.
Positive (pre-registered CinC extension rule): any setting whose selected value is better than its default (verdict 1), or any
changed verdict 2.
usage: python summarize_tune.py --stage lr [--energy_fix]  ->  results/flif-v1-tune-lr-summary[-energyfix].md and
       forecasting/results/flif-v1-ett-tune/selected_lr.json
"""
import argparse
import csv
import json
import math
import os

import numpy as np

from summarize_rhythm import parse

HERE = os.path.dirname(os.path.abspath(__file__))
FC = os.path.join(HERE, '..', 'forecasting')
SEEDS = (42, 7, 2026)
CELLS = [('ETTh2', 96), ('ETTh2', 720)]
NS = ('lif', 'flif', 'pop_lif', 'pop_flif')
CONDS = ['spk_ssa'] + [f'mixall_{n}_bidir' for n in NS] + [f'{p}_{n}' for p in ('spkneo', 'nsmt4', 'nsmt4q') for n in NS]
DEFAULT = {'lr': 0.0005, 'layers': 1, 'reset': 'spikede', 'alpha': 0.5}
STAGES = {'lr': [0.00025, 0.0005, 0.001, 0.002], 'layers': [1, 2, 3, 4], 'reset': ['spikede', 'subtract', 'zero']}
# MODI 2026-10-31: each stage runs on the earlier stages' per-setting selections (selected_<stage>.json) and varies only its own
# settings; the other settings stay at their base (selected) values and enter the comparisons at that base.
PREV = {'lr': [], 'layers': ['lr'], 'reset': ['lr', 'layers']}
STAGE_CONDS = {'lr': CONDS, 'layers': [f'mixall_{n}_bidir' for n in NS], 'reset': [c for c in CONDS if c != 'spk_ssa']}


def stage_values(stage, cond):
    """values of the stage knob for one setting. Stage reset (MODI 2026-10-31): a population neuron does not reset its members and
    its soma already subtracts theta, so 'subtract' is the default model there -> only spikede (default) vs zero."""
    if stage == 'reset' and 'pop_' in cond:
        return ['spikede', 'zero']
    return STAGES[stage]
OLD = {'flif-v1-gen-ett': lambda c: c == 'spk_ssa', 'flif-v1-ett-nall': lambda c: c.startswith('mixall_'),
       'flif-v1-ett-dual': lambda c: c.startswith(('spkneo_', 'nsmt4_')) and not c.endswith('_nomem'),
       'flif-v1-ett-query': lambda c: c.startswith('nsmt4q_') and not c.endswith('_nomem')}


def fin(x):
    return isinstance(x, float) and math.isfinite(x)


def same(k, x, ref=None):
    """RESULT value x equals ref (default: the DEFAULT of knob k); parse() turns numbers into floats; alpha '-' = not an f-LIF
    setting. MODI 2026-10-31 fix: the first version compared strings ('1.0' != '1') and dropped every tuning run."""
    ref = DEFAULT[k] if ref is None else ref
    if k == 'alpha' and x == '-':
        return True
    try:
        return float(x) == float(ref)
    except (TypeError, ValueError):
        return str(x) == str(ref)


def recounted(cond):                                    # = energy_recount_ett.recount
    return cond == 'spk_ssa' or cond.startswith(('spkall_', 'spkneo_', 'nsmt4_', 'nsmt4q_'))


def base_settings(stage):
    """per setting: DEFAULT with the earlier stages' selected values (selected_<stage>.json; selection is validation-only)."""
    base = {c: dict(DEFAULT) for c in CONDS}
    for p_ in PREV[stage]:
        sel = json.load(open(os.path.join(FC, 'results', 'flif-v1-ett-tune', f'selected_{p_}.json')))['selected']
        for c in CONDS:
            if sel.get(c) is None:
                raise RuntimeError(f'stage {p_}: no selection for {c}')
            base[c][p_] = sel[c]
    return base


def load(stage, energy_fix):
    """rows keyed (cond, value, ds, pl, seed). A run enters at value v of the stage knob when every other knob equals the setting's
    base (DEFAULT + earlier selections); the old suites' runs have the DEFAULT settings."""
    base = base_settings(stage)
    out = {}
    def add(r, v, suite):
        if (r['ds'], int(r['pl'])) not in CELLS:
            return
        out.setdefault((r['cond'], v, r['ds'], int(r['pl']), int(r['seed'])), []).append(r)
    def fixed(suite):
        p = os.path.join(FC, 'results', suite, 'energy_recount.csv')
        return {(f['ds'], int(f['pl']), f['cond'], f.get('tag', ''), int(f['seed'])): float(f['total_fft_uJ_corr'])
                for f in csv.DictReader(open(p))} if os.path.exists(p) else {}
    for suite, keep in OLD.items():
        fx = fixed(suite)
        for r in parse(os.path.join(FC, 'results', suite, 'raw.txt')):
            if keep(r['cond']) and all(same(k, DEFAULT[k], base[r['cond']][k]) for k in DEFAULT if k != stage):
                if energy_fix and recounted(r['cond']):
                    r['total_fft_uJ'] = fx.get((r['ds'], int(r['pl']), r['cond'], '', int(r['seed'])), float('nan'))
                add(r, DEFAULT[stage], suite)
    fx = fixed('flif-v1-ett-tune')
    for r in parse(os.path.join(FC, 'results', 'flif-v1-ett-tune', 'raw.txt')):
        if r['cond'] in CONDS and all(same(k, r.get(k), base[r['cond']][k]) for k in DEFAULT if k != stage):
            if energy_fix and recounted(r['cond']):
                r['total_fft_uJ'] = fx.get((r['ds'], int(r['pl']), r['cond'], r.get('tag', ''), int(r['seed'])), float('nan'))
            v = r[stage] if stage == 'reset' else float(r[stage])
            add(r, v, 'flif-v1-ett-tune')
    return lambda c, v, d, p, s: out[(c, v, d, p, s)][0] if len(out.get((c, v, d, p, s), [])) == 1 else None


def cellmean(get, c, v, key):
    vals = []
    for d, p in CELLS:
        rr = [get(c, v, d, p, s) for s in SEEDS]
        if any(r is None for r in rr) or not all(fin(r[key]) for r in rr):
            return None
        vals.append(float(np.mean([r[key] for r in rr])))
    return vals


def compare(get, x, vx, y, vy, crit=False):
    cm, pr, ers = [], [], []
    for d, p in CELLS:
        ra = [get(x, vx, d, p, s) for s in SEEDS]; rb = [get(y, vy, d, p, s) for s in SEEDS]
        if any(r is None for r in ra + rb) or not all(fin(r['mse']) and fin(r['total_fft_uJ']) for r in ra + rb):
            return None
        r_ = [100 * (a['mse'] - b['mse']) / b['mse'] for a, b in zip(ra, rb)]
        cm.append(float(np.mean(r_))); pr += r_
        ers.append(np.mean([a['total_fft_uJ'] for a in ra]) / np.mean([b['total_fft_uJ'] for b in rb]))
    R = float(np.mean(cm)); need = 2 * len(pr) // 3; lo, hi = sum(z < 0 for z in pr), sum(z > 0 for z in pr); er = float(np.mean(ers))
    v = 'better' if R <= -1 and lo >= need else 'worse' if R >= 1 and hi >= need else 'no clear difference'
    c2 = ('met' if v == 'better' or (R <= 1 and er <= 0.5) else 'not met') if crit else ''
    return dict(R=R, lo=lo, n=len(pr), v=v, er=er, c2=c2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=sorted(STAGES))
    ap.add_argument('--energy_fix', action='store_true')
    a = ap.parse_args()
    st, vals, dflt = a.stage, STAGES[a.stage], DEFAULT[a.stage]
    get = load(st, a.energy_fix)
    L = [f'# Tuning stage `{st}` (ETT ETTh2 x H96 / H720, seeds 42 / 7 / 2026)' + (' — T_h-averaged head energy recounted' if a.energy_fix else ''), '',
         'See the module docstring for sources, selection and rules. Validation = best validation L1 (selection); test = test MSE of '
         'the best-validation weights. Default value marked *.', '']
    for key, name in (('val_loss_best', 'validation L1'), ('mse', 'test MSE')):
        L += [f'## {name}: H96 / H720 cell means over seeds', '', '| setting | ' + ' | '.join(f'{v}{"*" if v == dflt else ""}' for v in vals) + ' |',
              '|---|' + '---|' * len(vals)]
        for c in STAGE_CONDS[st]:
            cells = [cellmean(get, c, v, key) if v in stage_values(st, c) else 'n/a' for v in vals]
            L.append(f'| {c} | ' + ' | '.join(m if m == 'n/a' else 'incomplete' if m is None else f'{m[0]:.4f} / {m[1]:.4f}' for m in cells) + ' |')
        L.append('')
    sel, relv, L1 = {}, {}, ['## Selection (validation) and verdict 1 (selected − default, test MSE)', '',
                   '| setting | relative validation per value | selected | R selected − default | pairs | verdict 1 | energy sel/default |',
                   '|---|---|---|---|---|---|---|']
    for c in STAGE_CONDS[st]:
        base = cellmean(get, c, dflt, 'val_loss_best'); rel = {}
        for v in stage_values(st, c):
            m = cellmean(get, c, v, 'val_loss_best')
            rel[v] = None if (m is None or base is None) else float(np.mean([x / b for x, b in zip(m, base)]))
        relv[c] = rel
        if any(x is None for x in rel.values()):
            sel[c] = None; L1.append(f'| {c} | incomplete | incomplete | | | | |'); continue
        s_ = min(stage_values(st, c), key=lambda v: rel[v]); sel[c] = s_
        o = compare(get, c, s_, c, dflt) if s_ != dflt else None
        L1.append(f"| {c} | {' / '.join(f'{v}: {rel[v]:.4f}' for v in stage_values(st, c))} | {s_} | "
                  + (f"{o['R']:+.2f} | {o['lo']}/{o['n']} | {o['v']} | {o['er']:.2f} |" if o else '(default) | | default selected | |'))
    varied = set(STAGE_CONDS[st])
    sel.update({c: dflt for c in CONDS if c not in varied})          # settings this stage does not vary stay at the default value
    if st == 'reset':                                                # one mode for every spikeDE neuron (report, not a verdict)
        for grp, cs in (('LIF / f-LIF settings', [c for c in STAGE_CONDS[st] if 'pop_' not in c]), ('population settings', [c for c in STAGE_CONDS[st] if 'pop_' in c])):
            per = {v: [relv[c][v] for c in cs if relv.get(c) and relv[c].get(v) is not None] for v in stage_values(st, cs[0])}
            L1.append(f"| (report) overall, {grp} | " + ' / '.join(f'{v}: {np.mean(x):.4f} (n={len(x)})' if x else f'{v}: incomplete' for v, x in per.items()) + ' | | | | | |')
    Q = [(f'{c} vs spk_ssa', c, 'spk_ssa', True) for c in STAGE_CONDS[st] if c != 'spk_ssa']
    if st in ('lr', 'reset'):
        Q += [(f'R1 replace vs append, {n}', f'nsmt4_{n}', f'spkneo_{n}', False) for n in NS]
        Q += [(f'Q1 query vs replace, {n}', f'nsmt4q_{n}', f'nsmt4_{n}', False) for n in NS]
        Q += [(f'Q2 query vs append, {n}', f'nsmt4q_{n}', f'spkneo_{n}', False) for n in NS]
    if st != 'lr':                                                   # + fractional order inside the Neo-mixer
        Q += [('W4 f-LIF vs LIF, Neo-mixer every layer', 'mixall_flif_bidir', 'mixall_lif_bidir', False)]
    L2 = ['', '## Verdict 2: the comparisons at the defaults and at each setting\'s selected value', '',
          '| comparison | R / verdict at defaults | criterion (2) | R / verdict at selected values | criterion (2) | changed |', '|---|---|---|---|---|---|']
    positive, changed_any = False, False
    for q, x, y, crit in Q:
        o0 = compare(get, x, dflt, y, dflt, crit)
        o1 = None if (sel.get(x) is None or sel.get(y) is None) else compare(get, x, sel[x], y, sel[y], crit)
        fmt = lambda o: ('incomplete', '') if o is None else (f"{o['R']:+.2f} ({o['lo']}/{o['n']}) {o['v']}, E {o['er']:.2f}", o['c2'])
        ch = '' if (o0 is None or o1 is None) else ('**yes**' if (o0['v'], o0['c2']) != (o1['v'], o1['c2']) else 'no')
        changed_any |= ch == '**yes**'
        L2.append(f'| {q} | {fmt(o0)[0]} | {fmt(o0)[1]} | {fmt(o1)[0]} | {fmt(o1)[1]} | {ch} |')
    better_any = any('| better |' in l for l in L1)
    complete = all(sel.get(c) is not None for c in STAGE_CONDS[st])
    positive = complete and (better_any or changed_any)
    L += L1 + L2 + ['', f'**Positive (CinC extension rule):** {"incomplete" if not complete else positive} '
                    f'(any verdict 1 better: {better_any}; any verdict 2 changed: {changed_any})']
    out = os.path.join(HERE, 'results', f'flif-v1-tune-{st}-summary' + ('-energyfix' if a.energy_fix else '') + '.md')
    open(out, 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
    os.makedirs(os.path.join(FC, 'results', 'flif-v1-ett-tune'), exist_ok=True)
    json.dump({'stage': st, 'default': dflt, 'selected': sel, 'complete': complete, 'positive': positive},
              open(os.path.join(FC, 'results', 'flif-v1-ett-tune', f'selected_{st}' + ('_energyfix' if a.energy_fix else '') + '.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
