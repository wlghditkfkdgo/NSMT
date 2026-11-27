"""MODI 2026-11-27 (user decisions 11-27: "macro-F1 0.1 %p로 읽는 것으로 하자", "energy 상한은 1.5배로 하되, 최대한 저전력을 목표로
하자"): report-only adoption-bar reading of every finished Sleep-EDF model, next to (not replacing) the pre-registered verdicts.

Adoption bar: mean test macro-F1 over folds 0-4 at least +0.1 %p above Spikformer(DE) AND syops energy (syn_energy_uJ) at most 1.5x
Spikformer(DE)'s. Main comparison = Spikformer(DE) at the same lr. Conservative column = against the stronger Spikformer(DE) test mean of
its two lrs (picked on test, so it only makes the bar harder). Rows: every (model, lr) with all 5 folds in the Sleep suites, without the
order / memory controls (_cshuf, _nml); pop f-LIF rows only with --all (default neuron set LIF / f-LIF / pop LIF, user 11-27).
Fail closed: select_sleep.check on every row (epochs 30, batch 64, seed 42, finite), no SKIP, no duplicate.
usage (classification folder): python adoption_sleep.py [--all] [--out results/flif-v1-sleep4]
"""
import argparse
import json
import os

import numpy as np

import select_sleep as SS

HERE = os.path.dirname(os.path.abspath(__file__))
SUITES = ('flif-v1-sleep-lr', 'flif-v1-sleep', 'flif-v1-sleep2', 'flif-v1-sleep3', 'flif-v1-sleep4')
GAIN, EMAX = 0.1, 1.5


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--all', action='store_true'); ap.add_argument('--out', default='results/flif-v1-sleep4')
    a = ap.parse_args()
    T = {}
    for s in SUITES:
        for r in SS.rows(s):
            k = (r['cond'], r['lr'], int(r['fold']))
            if k in T:
                raise SystemExit(f'STOP: duplicate {k} ({s})')
            SS.check(r, k); T[k] = r
    full = sorted({(c, lr) for c, lr, _ in T if all((c, lr, f) in T for f in range(5))})
    g = lambda c, lr, k='macro_f1': np.array([float(T[(c, lr, f)][k]) for f in range(5)]) * (100 if k == 'macro_f1' else 1)
    sf = {lr: (g('spkall_lif', lr), g('spkall_lif', lr, 'syn_energy_uJ').mean()) for lr in ('0.001', '0.003')}
    strong = max(sf, key=lambda lr: sf[lr][0].mean())
    rows = [(c, lr) for c, lr in full if c != 'spkall_lif' and not any(x in c for x in ('cshuf', 'nml')) and (a.all or 'pop_flif' not in c)]
    out = ['# Sleep-EDF: adoption-bar reading (report only)\n',
           f'Bar (user 11-27): mean macro-F1 >= +{GAIN} %p above Spikformer(DE) and syops energy <= {EMAX}x. Spikformer(DE): '
           + ', '.join(f'lr {lr} {sf[lr][0].mean():.2f} ({sf[lr][1]:.2f} µJ)' for lr in sf) + f'; stronger on test = lr {strong}.\n',
           '| model | lr | macro-F1 | vs Spikformer(DE) same lr (positive folds) | energy × | adoption (same lr) | vs stronger Spikformer(DE) | adoption (conservative) |',
           '|---|---|---|---|---|---|---|---|']
    dec = {}
    for c, lr in sorted(rows, key=lambda k: -g(*k).mean()):
        m = g(c, lr); d = m - sf[lr][0]; ex = g(c, lr, 'syn_energy_uJ').mean() / sf[lr][1]
        ds = m.mean() - sf[strong][0].mean(); exs = g(c, lr, 'syn_energy_uJ').mean() / sf[strong][1]
        verdict = lambda gain, e: 'pass' if gain >= GAIN and e <= EMAX else ('energy over' if gain >= GAIN else 'below')
        v1, v2 = verdict(d.mean(), ex), verdict(ds, exs)
        dec[f'{c}@{lr}'] = dict(mean=float(m.mean()), d_same=float(d.mean()), pos=int((d > 0).sum()), energy_x=float(ex), adopt_same=v1,
                                d_strong=float(ds), energy_x_strong=float(exs), adopt_strong=v2)
        out.append(f'| {c} | {lr} | {m.mean():.2f} | {d.mean():+.2f} ({int((d > 0).sum())}/5) | {ex:.2f} | **{v1}** | {ds:+.2f} | {v2} |')
    out += ['', 'Small gains sit inside the fold-to-fold spread (one seed); the pre-registered clear-difference verdicts are in summary.md.']
    txt = '\n'.join(out) + '\n'
    print(txt)
    od = os.path.join(HERE, a.out); os.makedirs(od, exist_ok=True)
    open(os.path.join(od, 'adoption.md'), 'w').write(txt)
    json.dump(dec, open(os.path.join(od, 'adoption.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
