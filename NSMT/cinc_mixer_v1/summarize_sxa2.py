"""Summary of stage 2 of the fusion redesign (2026-11-27, exp/f-lif-pop-v3; PROJECT_LOG stage-2 pre-registration; rules fixed before
training). Same cells, seeds, verdict rule and energy rules as stage 1 (summarize_sxa.py).
  arms      S  = the stage-1 selected Pearson arm (sx<f>pc, its sharing; rows reused from the stage-1 suite)
            A1 = muxlin independent (stage-1 rows)
            OT = sx<f>ot, PM = msx<f>pc (Pearson + MUX), OM = msx<f>ot (OT + MUX), with S's sharing (this suite, 36 rows)
  comparisons (test MSE verdict; MAE and energy ratios reported): OT vs S, OM vs PM (mask OT vs Pearson); PM vs S, OM vs OT (MUX);
            OT, PM, OM vs A1
  selection (validation only) among S, OT, PM, OM: score = mean over the 4 cells of seed-mean best validation L1 / that of A1; lowest
            wins; |difference| < 1e-12 -> lower mean syops energy -> selected_s2.json (refuses to overwrite a different selection)
Fails closed (exit 2, nothing written) as stage 1, plus: stage-2 rows not exactly the three expected combs with S's sharing.
usage (from cinc_mixer_v1/): python -B summarize_sxa2.py --suite flif-v1-ett-sxa-s2 [--s1_suite flif-v1-ett-sxa-s1]
"""
import argparse
import csv
import json
import math
import os
import sys

C = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--suite', default='flif-v1-ett-sxa-s2')
ap.add_argument('--s1_suite', default='flif-v1-ett-sxa-s1')
a = ap.parse_args()
OUT = os.path.join(C, 'results', a.suite)
S1 = os.path.join(C, 'results', a.s1_suite)
CELLS = [(d, p) for d in ('ETTh1', 'ETTm1') for p in ('96', '720')]
SEEDS = ('42', '7', '2026')
NUM = ('mse', 'mae', 'val_loss_best', 'syn_energy_uJ')


def stop(msg):
    print(f'STOP: {msg}')
    sys.exit(2)


def fin(r, k):
    try:
        v = float(r[k])
    except (KeyError, ValueError):
        stop(f'{k} missing ({r.get("tag")} {r.get("ds")} H{r.get("pl")} seed {r.get("seed")})')
    if not math.isfinite(v):
        stop(f'{k} not finite ({r.get("tag")} {r.get("ds")} H{r.get("pl")} seed {r.get("seed")})')
    return v


def load(d, want):
    """want: {label: (comb, shared)} -> {(label, ds, pl, seed): values}; every RESULT row of the file must belong to `want` when
    strict (this suite); for the stage-1 suite only the wanted arms are taken."""
    rows, nskip = [], 0
    for line in open(os.path.join(d, 'raw.txt')):
        if line.startswith('SKIP'):
            nskip += 1
        if line.startswith('RESULT'):
            rows.append(dict(t.split('=', 1) for t in line.split()[1:] if '=' in t))
    en = {(e['tag'], e['ds'], e['pl'], e['seed']): e for e in csv.DictReader(open(os.path.join(d, 'energy_recount.csv')))} \
        if os.path.exists(os.path.join(d, 'energy_recount.csv')) else None
    if en is None:
        stop(f'{d}: energy_recount.csv missing')
    inv = {v: k for k, v in want.items()}
    V, other, shas = {}, 0, set()
    for r in rows:
        comb = r['cond'].replace('spkmx_lif_', '', 1)
        if not comb.endswith('_add'):
            stop(f'unexpected cond {r["cond"]}')
        arm = (comb[:-4], r['tag'].endswith('_ws'))
        if arm not in inv:
            other += 1
            continue
        k = (inv[arm], r['ds'], r['pl'], r['seed'])
        if k in V:
            stop(f'duplicate run {k}')
        if r.get('blocks') != '1':
            stop(f'blocks {r.get("blocks")} in {k}')
        v = {m: fin(r, m) for m in NUM}
        e = en.get((r['tag'], r['ds'], r['pl'], r['seed']))
        if e is None:
            stop(f'no energy_recount row for {k}')
        body, hmac = fin(e, 'body_syn_uJ'), fin(e, 'head_MAC')
        if abs(body + hmac * 4.6e-6 - v['syn_energy_uJ']) > 1e-3 * v['syn_energy_uJ']:
            stop(f'energy identity off for {k}')
        v['E_int'] = fin(e, 'body_syn_int_uJ') + fin(e, 'head_AC_corr') * 0.9e-6
        V[k] = v
        shas.add(r['sha'])
    return V, nskip, other, shas


sel = json.load(open(os.path.join(S1, 'selected_s1.json')))
if sel.get('comb') not in ('sxdpc', 'sxppc') or not isinstance(sel.get('shared'), bool):
    stop(f'unexpected stage-1 selection {sel}')
f, sh = sel['comb'][2], sel['shared']
NEW = {'OT': (f'sx{f}ot', sh), 'PM': (f'msx{f}pc', sh), 'OM': (f'msx{f}ot', sh)}
OLD = {'S': (sel['comb'], sh), 'A1': ('muxlin', False)}
V2, nskip, other, shas2 = load(OUT, NEW)
if nskip:
    stop(f'{nskip} SKIP line(s)')
if other:
    stop(f'{other} stage-2 rows of an unexpected arm')
if len(V2) != 36:
    stop(f'{len(V2)} stage-2 runs (want 36)')
if len(shas2) != 1:
    stop(f'mixed training commits {sorted(shas2)}')
V1, nskip1, _, shas1 = load(S1, OLD)
if nskip1:
    stop(f'stage-1 suite has {nskip1} SKIP line(s)')
V = {**V1, **V2}
missing = [(lab, d, p, s) for lab in list(NEW) + list(OLD) for d, p in CELLS for s in SEEDS if (lab, d, p, s) not in V]
if missing:
    stop(f'{len(missing)} missing runs, e.g. {missing[0]}')
mean = lambda x: sum(x) / len(x)
sm = lambda lab, d, p, k: mean([V[(lab, d, p, s)][k] for s in SEEDS])


def compare(A, B, met):
    rr, lo, hi = [], 0, 0
    for d, p in CELLS:
        rr.append(mean([100. * (V[(A, d, p, s)][met] - V[(B, d, p, s)][met]) / V[(B, d, p, s)][met] for s in SEEDS]))
        lo += sum(V[(A, d, p, s)][met] < V[(B, d, p, s)][met] for s in SEEDS)
        hi += sum(V[(A, d, p, s)][met] > V[(B, d, p, s)][met] for s in SEEDS)
    R = mean(rr)
    return dict(R_pct=R, pairs_lower=lo, pairs_higher=hi, r_cells=rr,
                verdict='better' if R <= -1 and lo >= 8 else ('worse' if R >= 1 and hi >= 8 else 'no clear difference'))


NAMES = {lab: f'{c} {"shared" if s else "indep."}' for lab, (c, s) in {**NEW, **OLD}.items()}
COMPS = [('mask OT vs Pearson', 'OT', 'S'), ('mask OT vs Pearson', 'OM', 'PM'), ('MUX vs none', 'PM', 'S'), ('MUX vs none', 'OM', 'OT'),
         ('vs muxlin', 'OT', 'A1'), ('vs muxlin', 'PM', 'A1'), ('vs muxlin', 'OM', 'A1')]
out = dict(suite=a.suite, s1_suite=a.s1_suite, sha=sorted(shas2)[0], s1_sha=sorted(shas1), names=NAMES, comparisons=[], mean4={}, cells={})
for lab in ('A1', 'S', 'OT', 'PM', 'OM'):
    out['cells'][NAMES[lab]] = {f'{d} {p}': {m: sm(lab, d, p, m) for m in NUM + ('E_int',)} for d, p in CELLS}
    out['mean4'][NAMES[lab]] = {m: mean([out['cells'][NAMES[lab]][f'{d} {p}'][m] for d, p in CELLS]) for m in NUM + ('E_int',)}
for kind, A, B in COMPS:
    out['comparisons'].append(dict(kind=kind, A=NAMES[A], B=NAMES[B], mse=compare(A, B, 'mse'), mae=compare(A, B, 'mae'),
                                   energy_syops=mean([sm(A, d, p, 'syn_energy_uJ') / sm(B, d, p, 'syn_energy_uJ') for d, p in CELLS]),
                                   energy_int=mean([sm(A, d, p, 'E_int') / sm(B, d, p, 'E_int') for d, p in CELLS])))
cands = ['S', 'OT', 'PM', 'OM']
score = {x: mean([sm(x, d, p, 'val_loss_best') / sm('A1', d, p, 'val_loss_best') for d, p in CELLS]) for x in cands}
best = min(cands, key=lambda x: score[x])
ties = [x for x in cands if abs(score[x] - score[best]) < 1e-12]
if len(ties) > 1:
    best = min(ties, key=lambda x: out['mean4'][NAMES[x]]['syn_energy_uJ'])
comb, shared = {**NEW, **OLD}[best]
out['selection'] = dict(scores={NAMES[x]: score[x] for x in cands}, selected=NAMES[best], comb=comb, shared=shared)
sp = os.path.join(OUT, 'selected_s2.json')
if os.path.exists(sp):
    old = json.load(open(sp))
    if (old.get('comb'), old.get('shared')) != (comb, shared):
        stop(f'selected_s2.json exists with a different selection {old}')
json.dump(dict(comb=comb, shared=shared, score=score[best], rule='lowest validation score among S, OT, PM, OM'), open(sp, 'w'), indent=1)
json.dump(out, open(os.path.join(OUT, 'summary.json'), 'w'), indent=1)
L = ['# Fusion redesign stage 2 (OT plan mask, MUX versions; LIF, ADD, one block; ETTh1 / ETTm1 x H96 / H720 x 3 seeds)', '',
     f'Rules: summarize_sxa2.py docstring. 36 new runs (commit `{out["sha"]}`) + stage-1 rows of S and A1 ({", ".join(out["s1_sha"])}).', '',
     '## 4-cell means (seed means)', '', '| label | arm | MSE | MAE | best val L1 | energy µJ (syops) | energy µJ (integer) |', '|---|---|--:|--:|--:|--:|--:|']
for lab in ('A1', 'S', 'OT', 'PM', 'OM'):
    m = out['mean4'][NAMES[lab]]
    L.append(f'| {lab} | {NAMES[lab]} | {m["mse"]:.4f} | {m["mae"]:.4f} | {m["val_loss_best"]:.4f} | {m["syn_energy_uJ"]:.2f} | {m["E_int"]:.2f} |')
L += ['', '## Comparisons (test)', '', '| kind | A | B | MSE R % | lower / higher | MSE verdict | MAE R % | MAE verdict | energy A/B syops | integer |',
      '|---|---|---|--:|---|---|--:|---|--:|--:|']
for c in out['comparisons']:
    L.append(f'| {c["kind"]} | {c["A"]} | {c["B"]} | {c["mse"]["R_pct"]:+.2f} | {c["mse"]["pairs_lower"]} / {c["mse"]["pairs_higher"]} | '
             f'**{c["mse"]["verdict"]}** | {c["mae"]["R_pct"]:+.2f} | {c["mae"]["verdict"]} | {c["energy_syops"]:.3f} | {c["energy_int"]:.3f} |')
L += ['', '## Per cell (seed-mean test MSE / MAE, syops energy µJ)', '', '| arm | ' + ' | '.join(f'{d} H{p}' for d, p in CELLS) + ' |',
      '|---|' + '---|' * len(CELLS)]
for lab in ('A1', 'S', 'OT', 'PM', 'OM'):
    cc = out['cells'][NAMES[lab]]
    L.append(f'| {lab} {NAMES[lab]} | ' + ' | '.join(f'{cc[f"{d} {p}"]["mse"]:.4f} / {cc[f"{d} {p}"]["mae"]:.4f} / {cc[f"{d} {p}"]["syn_energy_uJ"]:.2f}'
                                                     for d, p in CELLS) + ' |')
L += ['', '## Selection (validation only)', ''] + [f'- {k}: score {v:.6f}' for k, v in out['selection']['scores'].items()]
L.append(f'- **selected: {out["selection"]["selected"]}** (selected_s2.json)')
open(os.path.join(OUT, 'summary.md'), 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
