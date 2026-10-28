"""MODI 2026-10-28: summary of the shuffle-retrain suite (flif-v1-rhythm-shuf-cinc) with the pre-registered rules
(docs/MODI_PROJECT_LOG.md, 2026-10-28 pre-registration). Missing / non-finite runs make the affected verdict 'incomplete'.

S1 per model m (paired by seed; normal order = flif-v1-rhythm-budget-cinc, 90 epochs):
   d = normal-order score - shuffle-trained score (both on their own test order: normal / the fixed shuffle).
   'order helps' = mean d >= +0.5 %p and d > 0 in >= 2 of 3 seeds; 'shuffle-trained better' = mean <= -0.5 and < 0 in >= 2 of 3;
   otherwise 'no clear difference'.
S2 CinC rewards order (for the best model) <=> S1 of Spikformer-style is 'order helps'.
Reported: shuffle-trained Neo-mixer - shuffle-trained Spikformer-style; recovery = shuffle-trained - the normal-order model
evaluated on the same fixed shuffle (cause analysis, flif-v1-cause-cinc score_shuf1).
usage: python summarize_shuf.py [--suite flif-v1-rhythm-shuf-cinc]
"""
import argparse
import json
import math
import os

import numpy as np

from summarize_rhythm import parse

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = (42, 7, 2026)
MODELS = ('spk_ssa', 'mix_lif_bidir', 'mix_flif_bidir', 'mix_pop_lif_bidir', 'mix_pop_bidir')
LAB = {'spk_ssa': 'Spikformer-style', 'mix_lif_bidir': 'Neo-mixer LIF', 'mix_flif_bidir': 'Neo-mixer f-LIF',
       'mix_pop_lif_bidir': 'Neo-mixer population LIF', 'mix_pop_bidir': 'Neo-mixer population f-LIF'}


def scores(rows, cond):
    out = []
    for s in SEEDS:
        v = [r['score'] for r in rows if r.get('cond') == cond and int(r.get('seed', -1)) == s and r.get('data') == 'CINC2017'
             and int(r.get('fold', 0)) == 0]
        out.append(v[-1] if v and isinstance(v[-1], float) and math.isfinite(v[-1]) else None)
    return out


def cause(cond):
    out = []
    for s in SEEDS:
        f = os.path.join(HERE, 'results', 'flif-v1-cause-cinc', f'{cond}_seed{s}', 'cause.json')
        r = json.load(open(f)) if os.path.exists(f) else None
        v = None if r is None or r['checks_failed'] else r.get('score_shuf1')
        out.append(v if isinstance(v, float) and math.isfinite(v) else None)
    return out


def verdict(d):
    if any(x is None for x in d):
        return 'incomplete'
    m = np.mean(d)
    if m >= 0.005 and sum(x > 0 for x in d) >= 2:
        return 'order helps'
    if m <= -0.005 and sum(x < 0 for x in d) >= 2:
        return 'shuffle-trained better'
    return 'no clear difference'


def ms(v):
    return 'incomplete' if any(x is None for x in v) else f'{np.mean(v) * 100:.2f} (' + ' / '.join(f'{x * 100:.2f}' for x in v) + ')'


def diff(a, b):
    return [None if x is None or y is None else x - y for x, y in zip(a, b)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--suite', default='flif-v1-rhythm-shuf-cinc')
    a = ap.parse_args()
    res = os.path.join(HERE, 'results', a.suite)
    shuf = parse(os.path.join(res, 'raw.txt'))
    norm = parse(os.path.join(HERE, 'results', 'flif-v1-rhythm-budget-cinc', 'raw.txt'))
    L = [f'# Shuffle-retrain `{a.suite}` (CINC2017 fold 0, 90 epochs) — summary', '',
         'Test score (%), best-validation weights; mean (seed 42 / 7 / 2026). Normal order: `flif-v1-rhythm-budget-cinc`. '
         'Shuffle-trained models are validated and tested on one fixed token order (default_rng(0).permutation(117)); the '
         'eval-shuffle column is the normal-order model tested on that same order (`flif-v1-cause-cinc`).', '',
         '| model | normal order | shuffle-trained | normal model, eval shuffle | S1 normal − shuffle-trained (%p) | verdict | recovery (%p) |',
         '|---|---|---|---|---|---|---|']
    v_ssa, shuf_ssa = None, scores(shuf, 'spk_ssa_shuf')
    for m in MODELS:
        n, s, e = scores(norm, m), scores(shuf, m + '_shuf'), cause(m)
        d = diff(n, s); v = verdict(d)
        if m == 'spk_ssa':
            v_ssa = v
        rec = diff(s, e)
        L.append(f'| {LAB[m]} | {ms(n)} | {ms(s)} | {ms(e)} | {ms(d)} | {v} | {ms(rec)} |')
    L += ['', f'- S2 CinC rewards order (Spikformer-style S1 = order helps): **'
          + ('incomplete' if v_ssa == 'incomplete' else ('yes' if v_ssa == 'order helps' else 'no')) + '**']
    for m in MODELS[1:]:
        L.append(f'- shuffle-trained {LAB[m]} − shuffle-trained Spikformer-style (%p): {ms(diff(scores(shuf, m + "_shuf"), shuf_ssa))}')
    L.append('')
    os.makedirs(res, exist_ok=True)
    open(os.path.join(res, 'summary.md'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
