"""MODI 2026-11-27 ((나) follow-up, user decision 11-27 "둘 다 진행해줘"): pre-registered summary of the matched-lr and order / memory
runs on Sleep-EDF (docs/MODI_PROJECT_LOG.md 2026-11-27 pre-registration).

Suites: stage 1 flif-v1-sleep-lr (fold 0, lr 0.001 and 0.003), main flif-v1-sleep (folds 1-4 at the selected lr), follow-up
flif-v1-sleep2 (32 runs): spkall_lif lr 0.003 folds 1-4; q_pop_lif / q_pop_flif lr 0.001 folds 1-4; q_pop_<n>_cshuf and q_pop_<n>_nml
lr 0.003 folds 0-4. Fail closed: every needed (model, lr, fold) exactly once, epochs 30, batch 64, seed 42, finite values, no SKIP, one
sha within each suite (the follow-up suite runs on newer code; its default Sleep path is checked to reproduce the earlier runs exactly).
Metric: test 5-class macro-F1 (%), pairs = folds 0-4; verdict thresholds as in summarize_sleep.py (mean >= +1.0 and >= 4/5 positive /
mean <= -1.0 and >= 4/5 negative / otherwise no clear difference).
  L (same lr): for lr in {0.003, 0.001}: d = q_pop_<n>(lr) - spkall_lif(lr). Reading per neuron: "beats Spikformer(DE) at the same lr"
     if better at both lrs; "depends on lr" if better at one; "not supported" otherwise.
  O (order): d = q_pop_<n> - q_pop_<n>_cshuf (lr 0.003; the context epochs shuffled per sample in training and test, target last).
     better = the Neocortex gains from the epoch order.
  M (memory): d = q_pop_<n> - q_pop_<n>_nml (lr 0.003; every epoch through the Neocortex on its own). better = the Neocortex state
     across epochs helps.
Report only: accuracy, N1 F1, syops energy, firing rate of the controls.
usage (classification folder): python summarize_sleep2.py
"""
import json
import os

import numpy as np

import select_sleep as SS

HERE = os.path.dirname(os.path.abspath(__file__))
NEU = ('pop_lif', 'pop_flif')


def verdict(d):
    if d.mean() >= 1.0 and (d > 0).sum() >= 4:
        return 'better'
    if d.mean() <= -1.0 and (d < 0).sum() >= 4:
        return 'worse'
    return 'no clear difference'


def main():
    T = {}

    def add(suite, allowed):
        shas = set()
        for r in SS.rows(suite):
            key = (r['cond'], r['lr'], int(r['fold']))
            if key not in allowed:
                continue
            if key in T:
                raise SystemExit(f'STOP: duplicate {key} ({suite})')
            SS.check(r, key); T[key] = r; shas.add(r['sha'])
        if len(shas) > 1:
            raise SystemExit(f'STOP: {suite} code versions {sorted(shas)}')
        return shas
    need1 = {(c, lr, 0) for c in ('spkall_lif', 'q_pop_lif', 'q_pop_flif') for lr in ('0.001', '0.003')}
    needm = {('spkall_lif', '0.001', f) for f in range(1, 5)} | {(f'q_{n}', '0.003', f) for n in NEU for f in range(1, 5)}
    need2 = {('spkall_lif', '0.003', f) for f in range(1, 5)} | {(f'q_{n}', '0.001', f) for n in NEU for f in range(1, 5)} | \
            {(f'q_{n}_{s}', '0.003', f) for n in NEU for s in ('cshuf', 'nml') for f in range(5)}
    sh = [add('flif-v1-sleep-lr', need1), add('flif-v1-sleep', needm), add('flif-v1-sleep2', need2)]
    rows2 = [r for r in SS.rows('flif-v1-sleep2')]
    extra = [r['cond'] + '/' + r['lr'] + '/' + r['fold'] for r in rows2 if (r['cond'], r['lr'], int(r['fold'])) not in need2]
    if extra:
        raise SystemExit(f'STOP: unexpected follow-up rows {extra[:3]}')
    miss = sorted((need1 | needm | need2) - set(T))
    if miss:
        raise SystemExit(f'STOP: missing {miss[:4]} ({len(miss)})')
    g = lambda c, lr, k='macro_f1': np.array([float(T[(c, lr, f)][k]) for f in range(5)]) * (100 if k in ('macro_f1', 'acc', 'f1_1') else 1)
    out = ['# Sleep-EDF follow-up: same-lr comparison, order and memory controls\n',
           f'Pre-registered: docs/MODI_PROJECT_LOG.md 2026-11-27. Code versions: stage 1 {sh[0]}, main {sh[1]}, follow-up {sh[2]}. '
           'Test macro-F1 %, folds 0-4; differences in %p, paired by fold.\n',
           '| model | lr | macro-F1 | accuracy | N1 F1 | syops energy µJ | firing rate |', '|---|---|---|---|---|---|---|']
    rowsets = [('spkall_lif', '0.001'), ('spkall_lif', '0.003')] + [(f'q_{n}', lr) for n in NEU for lr in ('0.001', '0.003')] + \
              [(f'q_{n}_{s}', '0.003') for n in NEU for s in ('cshuf', 'nml')]
    for c, lr in rowsets:
        m = g(c, lr)
        out.append(f'| {c} | {lr} | {m.mean():.2f} ± {m.std():.2f} | {g(c, lr, "acc").mean():.2f} | {g(c, lr, "f1_1").mean():.1f} | '
                   f'{g(c, lr, "syn_energy_uJ").mean():.3f} | {g(c, lr, "syops_firing_rate").mean():.4f} |')
    dec = {'L': {}, 'O': {}, 'M': {}}
    out += ['', '## L: query form − Spikformer(DE) at the same lr', '', '| neuron | lr | d per fold | mean | positive | verdict |', '|---|---|---|---|---|---|']
    for n in NEU:
        vs = {}
        for lr in ('0.003', '0.001'):
            d = g(f'q_{n}', lr) - g('spkall_lif', lr); v = verdict(d); vs[lr] = v
            out.append(f'| {n} | {lr} | {" / ".join(f"{x:+.2f}" for x in d)} | {d.mean():+.2f} | {int((d > 0).sum())}/5 | **{v}** |')
            dec['L'][f'{n}@{lr}'] = dict(d=d.tolist(), mean=float(d.mean()), verdict=v)
        nb = sum(v == 'better' for v in vs.values())
        dec['L'][n] = 'beats Spikformer(DE) at the same lr' if nb == 2 else 'depends on lr' if nb == 1 else 'not supported'
    out.append('\nReading per neuron: ' + '; '.join(f'{n}: **{dec["L"][n]}**' for n in NEU))
    for key, suf, title in (('O', 'cshuf', 'O: order — query form − same model with the context epochs shuffled'),
                            ('M', 'nml', 'M: memory — query form − same model with a memoryless Neocortex')):
        out += ['', f'## {title}', '', '| neuron | d per fold | mean | positive | verdict |', '|---|---|---|---|---|']
        for n in NEU:
            d = g(f'q_{n}', '0.003') - g(f'q_{n}_{suf}', '0.003'); v = verdict(d)
            dec[key][n] = dict(d=d.tolist(), mean=float(d.mean()), pos=int((d > 0).sum()), verdict=v)
            out.append(f'| {n} | {" / ".join(f"{x:+.2f}" for x in d)} | {d.mean():+.2f} | {int((d > 0).sum())}/5 | **{v}** |')
    txt = '\n'.join(out) + '\n'
    print(txt)
    os.makedirs(os.path.join(HERE, 'results', 'flif-v1-sleep2'), exist_ok=True)
    open(os.path.join(HERE, 'results', 'flif-v1-sleep2', 'summary.md'), 'w').write(txt)
    json.dump(dec, open(os.path.join(HERE, 'results', 'flif-v1-sleep2', 'decision.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
