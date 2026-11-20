"""Post-hoc report of the ETT tuning extension (2026-11-20, exp/f-lif-pop-v3; PROJECT_LOG ETT tuning extension result entry). No training.
NOT part of the pre-registered selection or verdict (those are summary_final.md / final.json). Written after seeing the final test values.
  A  every configuration on the selection path (after each stage) judged with pass_ett.judge (the user's three criteria + energy bound)
  B  energy per cell (syops rule) of the final configuration, the patch-stage configuration and SF-de
  C  test oracle (uses TEST values to choose, so it is not a valid selection): over every configuration with all 6 runs per dataset
     (default rows, first chain, extension), the lowest reachable 4-cell mean test MSE and MAE; tells whether the user's DNN target
     (TimeMixer MSE 0.3167 - 0.001 = 0.3157) was reachable at all inside the searched space
usage (from cinc_mixer_v1/): ET_CHAIN=x python -B results/flif-v1-ett-spkmux-ettune2/posthoc_x.py > results/flif-v1-ett-spkmux-ettune2/posthoc_x.md
"""
import collections
import json
import os
import sys

import numpy as np

C = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, C)
if os.environ.get('ET_CHAIN') != 'x':
    sys.exit('run with ET_CHAIN=x')
import summarize_ettune as se                                             # noqa: E402

pe = se.pe
D = se.load()
E = se.energy_rows()
sf = se.sfde_rows(E)
path = [{ds: dict(se.DEFAULT[ds]) for ds in se.DSS}]
for name, knob, _ in se.STAGES:
    s = json.load(open(os.path.join(se.OUT, f'selected_{name}.json')))
    path.append({ds: dict(path[-1][ds], **{knob: str(s['selected'][ds])}) for ds in se.DSS})
VS = {c: {k: [sf[(c[0], c[1], s)][k] for s in se.SEEDS] for k in ('mse', 'mae')} for c in se.CELLS}
ES = {c: [sf[(c[0], c[1], s)]['Es'] for s in se.SEEDS] for c in se.CELLS}


def judge(T, name):
    V = {c: {k: [T[(c[0], c[1], s)][k] for s in se.SEEDS] for k in ('mse', 'mae')} for c in se.CELLS}
    EA = {c: [T[(c[0], c[1], s)]['Es'] for s in se.SEEDS] for c in se.CELLS}
    r, _ = pe.judge(name, V, VS, EA, ES)
    return pe.cellmeans(V), pe.eratio(EA, ES), r


print('# ETT tuning extension: post-hoc report (not part of the pre-registered selection or verdict)\n')
print('## A. The configuration after each stage, judged with the user criteria (test values)\n')
print('| after stage | ETTh2 | ETTm2 | 4-cell MSE / MAE | energy A/B vs SF-de (syops) | (1) SpikF | (2) SF-de + energy | (3) DNN | overall |')
print('|---|---|---|---|--:|---|---|---|---|')
for i, p in enumerate(path):
    T = se.arm(D, E, p)
    (m, a), er, r = judge(T, f'step {i}')
    nm = 'none (untuned)' if i == 0 else se.NAMES[i - 1]
    yn = lambda b: 'pass' if b else 'fail'                                   # noqa: E731
    print(f'| {nm} | {se.cfgtxt("ETTh2", p["ETTh2"])} | {se.cfgtxt("ETTm2", p["ETTm2"])} | {m:.4f} / {a:.4f} | {er:.4f} | '
          f'{yn(r["(1) SNN"])} | {yn(r["(2) Spikformer"])} | {yn(r["(3) DNN"])} | {"PASS" if r["pass"] else "NOT PASSED"} |')

print('\n## B. Energy per cell (syops rule, seed mean, µJ)\n')
print('| model | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | mean of cell ratios vs SF-de |')
print('|---|--:|--:|--:|--:|--:|')
for nm, T in (('final (extension)', se.arm(D, E, path[-1])), ('after the patch stage', se.arm(D, E, path[1])), ('SF-de', sf)):
    cells = [float(np.mean([T[(ds, pl, s)]['Es'] for s in se.SEEDS])) for ds, pl in se.CELLS]
    ref = [float(np.mean([sf[(ds, pl, s)]['Es'] for s in se.SEEDS])) for ds, pl in se.CELLS]
    print(f'| {nm} | ' + ' | '.join(f'{x:.2f}' for x in cells) + f' | {np.mean([a / b for a, b in zip(cells, ref)]):.4f} |')

print('\n## C. Test oracle (chooses by TEST values; an upper bound of the searched space, not a valid selection)\n')
cfgs = collections.defaultdict(set)
for (ds, pl, s, ck) in D:
    cfgs[ds].add(ck)
best = {}
for ds in se.DSS:
    rows = []
    for ck in cfgs[ds]:
        cfg = dict(zip(se.ORDER, ck))
        sm = {k: [se.seed_mean(D, ds, pl, cfg, k) for pl in se.PLS] for k in ('mse', 'mae')}
        if any(v is None for v in sm['mse'] + sm['mae']):
            continue
        er = []
        for pl in se.PLS:
            for s in se.SEEDS:
                se.attach(D[(ds, pl, s, ck)], E, D[(ds, pl, s, ck)]['src'])
            er.append(np.mean([D[(ds, pl, s, ck)]['Es'] for s in se.SEEDS]) / np.mean([sf[(ds, pl, s)]['Es'] for s in se.SEEDS]))
        rows.append((float(np.mean(sm['mse'])), float(np.mean(sm['mae'])), float(np.mean(er)), se.cfgtxt(ds, cfg)))
    best[ds] = rows
    print(f'### {ds}: {len(rows)} configurations with all 6 runs; 5 lowest 2-cell mean test MSE\n')
    print('| configuration | 2-cell mean MSE | 2-cell mean MAE | energy A/B vs SF-de (2 cells) |')
    print('|---|--:|--:|--:|')
    for r in sorted(rows)[:5]:
        print(f'| {r[3]} | {r[0]:.4f} | {r[1]:.4f} | {r[2]:.4f} |')
    print()
om = np.mean([min(r[0] for r in best[ds]) for ds in se.DSS])
oa = np.mean([min(r[1] for r in best[ds]) for ds in se.DSS])
ome = np.mean([min((r[0] for r in best[ds] if r[2] <= pe.EBOUND), default=float('nan')) for ds in se.DSS])
print(f'Lowest reachable 4-cell mean test MSE (best ETTh2 + best ETTm2 configuration, by test): **{om:.4f}**; '
      f'MAE (chosen separately): **{oa:.4f}**.')
print(f'Same with each dataset restricted to configurations with energy A/B <= {pe.EBOUND}: MSE **{ome:.4f}**.')
print(f'User DNN target (TimeMixer MSE {pe.ref_mean(*pe.DNN["TimeMixer"][:2])[0]:.4f} - 0.001): MSE <= '
      f'{pe.ref_mean(*pe.DNN["TimeMixer"][:2])[0] - pe.MARGIN:.4f} -> '
      f'{"reachable" if om <= pe.ref_mean(*pe.DNN["TimeMixer"][:2])[0] - pe.MARGIN + pe.EPS else "NOT reachable in the searched space, even by test choice"}.')
