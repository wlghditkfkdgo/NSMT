"""MODI 2026-11-26 ((나) Sleep-EDF SNN comparison): pre-registered summary (docs/MODI_PROJECT_LOG.md 2026-11-26 pre-registration).

Rows: fold 0 = the stage-1 run (suite flif-v1-sleep-lr) at the lr in selected.json, folds 1-4 = suite flif-v1-sleep. Fail closed: every
(model, fold) exactly once at the selected lr, epochs 30, batch 64, seed 42, finite values, one sha over both suites, no SKIP line.
Metric: test 5-class macro-F1 (%), pairs = folds 0-4.
  Q1 (each Neocortex model vs Spikformer(DE)): d = model - spkall_lif; better: mean >= +1.0 and >= 4/5 positive; worse: mean <= -1.0
     and >= 4/5 negative; otherwise no clear difference.
  Q2 (coupling form, per neuron): d = query - key/value; query better / key-value better with the same thresholds; otherwise no clear
     difference.
  Structure (pre-registered): the form with more neurons judged "better" in Q1; a tie -> the form with the higher mean Q1 d over the
     two neurons; if no form has a "better" neuron, the report says that neither form beats Spikformer(DE) (the higher mean is still
     named, as "direction only").
Report only: accuracy, per-class F1 (W, N1, N2, N3, REM), syn_energy_uJ (syops) and its ratio to Spikformer(DE), firing rate,
validation scores of stage 1.
usage (classification folder): python summarize_sleep.py [--stage1 flif-v1-sleep-lr] [--main flif-v1-sleep]
"""
import argparse
import json
import math
import os

import numpy as np

import select_sleep as SS

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = {'spkall_lif': 'Spikformer(DE)', 'q_pop_lif': 'query form, pop LIF', 'q_pop_flif': 'query form, pop f-LIF',
         'kv_pop_lif': 'key/value form, pop LIF', 'kv_pop_flif': 'key/value form, pop f-LIF'}


def verdict(d, pos_word, neg_word):
    n = len(d)
    if d.mean() >= 1.0 and (d > 0).sum() >= n - 1:
        return pos_word
    if d.mean() <= -1.0 and (d < 0).sum() >= n - 1:
        return neg_word
    return 'no clear difference'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--stage1', default='flif-v1-sleep-lr'); ap.add_argument('--main', default='flif-v1-sleep')
    a = ap.parse_args()
    sel = json.load(open(os.path.join(HERE, 'results', a.stage1, 'selected.json')))['selected']
    T, shas = {}, set()
    for r in SS.rows(a.stage1):
        if r['lr'] == sel[r['cond']]['lr']:
            key = (r['cond'], 0)
            if key in T:
                raise SystemExit(f'STOP: duplicate {key}')
            SS.check(r, key); T[key] = r; shas.add(r['sha'])
    for r in SS.rows(a.main):
        key = (r['cond'], int(r['fold']))
        if r['cond'] not in SS.CONDS or key[1] not in (1, 2, 3, 4) or r['lr'] != sel[r['cond']]['lr']:
            raise SystemExit(f'STOP: unexpected main row {key} lr {r["lr"]}')
        if key in T:
            raise SystemExit(f'STOP: duplicate {key}')
        SS.check(r, key); T[key] = r; shas.add(r['sha'])
    want = {(c, f) for c in SS.CONDS for f in range(5)}
    if set(T) != want:
        raise SystemExit(f'STOP: missing {sorted(want - set(T))}')
    if len(shas) != 1:
        raise SystemExit(f'STOP: code versions {sorted(shas)}')
    g = lambda c, k: np.array([float(T[(c, f)][k]) for f in range(5)])
    mf = {c: 100 * g(c, 'macro_f1') for c in SS.CONDS}
    out = [f'# Sleep-EDF SNN comparison — {a.main} (+ fold 0 from {a.stage1}), sha {shas.pop()}\n',
           'Pre-registered: docs/MODI_PROJECT_LOG.md 2026-11-26. Test macro-F1 %, mean ± std over folds 0-4; differences in %p, paired by fold.\n',
           '| model | lr | macro-F1 | accuracy | F1 W / N1 / N2 / N3 / REM | syn energy µJ | × Spikformer(DE) | firing rate |', '|---|---|---|---|---|---|---|---|']
    e0 = g('spkall_lif', 'syn_energy_uJ').mean()
    for c in SS.CONDS:
        f1 = ' / '.join(f'{100 * g(c, f"f1_{i}").mean():.1f}' for i in range(5))
        en = g(c, 'syn_energy_uJ').mean()
        out.append(f'| {NAMES[c]} | {sel[c]["lr"]} | {mf[c].mean():.2f} ± {mf[c].std():.2f} | {100 * g(c, "acc").mean():.2f} | {f1} | {en:.3f} | '
                   f'{en / e0:.2f} | {g(c, "syops_firing_rate").mean():.4f} |')
    dec = {'Q1': {}, 'Q2': {}}
    out += ['', '## Q1: Neocortex model − Spikformer(DE)', '', '| model | d per fold | mean | positive | verdict |', '|---|---|---|---|---|']
    for c in SS.CONDS[1:]:
        d = mf[c] - mf['spkall_lif']; v = verdict(d, 'better', 'worse')
        dec['Q1'][c] = dict(d=d.tolist(), mean=float(d.mean()), pos=int((d > 0).sum()), verdict=v)
        out.append(f'| {NAMES[c]} | {" / ".join(f"{x:+.2f}" for x in d)} | {d.mean():+.2f} | {int((d > 0).sum())}/5 | **{v}** |')
    out += ['', '## Q2: query form − key/value form', '', '| neuron | d per fold | mean | positive | verdict |', '|---|---|---|---|---|']
    for nrn in ('pop_lif', 'pop_flif'):
        d = mf[f'q_{nrn}'] - mf[f'kv_{nrn}']; v = verdict(d, 'query better', 'key/value better')
        dec['Q2'][nrn] = dict(d=d.tolist(), mean=float(d.mean()), pos=int((d > 0).sum()), verdict=v)
        out.append(f'| {nrn.replace("_", " ")} | {" / ".join(f"{x:+.2f}" for x in d)} | {d.mean():+.2f} | {int((d > 0).sum())}/5 | **{v}** |')
    score = {}
    for form in ('q', 'kv'):
        nb = sum(dec['Q1'][f'{form}_{n}']['verdict'] == 'better' for n in ('pop_lif', 'pop_flif'))
        mean = float(np.mean([dec['Q1'][f'{form}_{n}']['mean'] for n in ('pop_lif', 'pop_flif')]))
        score[form] = (nb, mean)
    best = max(score, key=lambda k: score[k])
    name = {'q': 'query form', 'kv': 'key/value form'}
    if score[best][0] == 0:
        rec = f'neither form beats Spikformer(DE) (direction only: {name[best]}, mean Q1 d {score[best][1]:+.2f} %p)'
    else:
        rec = f'{name[best]} ({score[best][0]} neuron(s) better than Spikformer(DE), mean Q1 d {score[best][1]:+.2f} %p)'
    dec['structure'] = dict(score={k: list(v) for k, v in score.items()}, recommendation=rec)
    out += ['', f'**Structure (pre-registered rule): {rec}.**', '',
            'Stage-1 validation macro-F1 (fold 0): ' + '; '.join(f'{NAMES[c]} {sel[c]["val"]["0.001"]:.4f} / {sel[c]["val"]["0.003"]:.4f} -> {sel[c]["lr"]}' for c in SS.CONDS)]
    txt = '\n'.join(out) + '\n'
    print(txt)
    open(os.path.join(HERE, 'results', a.main, 'summary.md'), 'w').write(txt)
    json.dump(dec, open(os.path.join(HERE, 'results', a.main, 'decision.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
