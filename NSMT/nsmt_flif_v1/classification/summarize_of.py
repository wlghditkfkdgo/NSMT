"""MODI 2026-11-19: summary of the overfitting test (docs/MODI_PROJECT_LOG.md, pre-registered rules; user request 11-19 "과적합에 대한
여러 조치를 취해서 해결한 상태에서의 test 값을 Spikformer(DE)와 비교해줘"). Reads the selections of select_of.py.

Models (ETT, input 96, every neuron SpikeDE): Spikformer(DE) spkall_lif; the msn base pop LIF / pop f-LIF with the aux form chosen in stage S.
Arms per model (seeds 42 / 7 / 2026, best-validation weights, test MSE; energy = head recount):
  current: lr 0.0005, head dropout 0, weight decay 0.06 (the existing runs)      final: the setting after stage A3 (seed 42 from its stage, 7 / 2026 from B)
Verdicts:
  O1 overfitting resolved? (descriptive, per model): overfit flags and the median best-validation epoch over the 12 final runs, defined as in
     loss_trends.py (overfit = train fell >= 2 % after the best epoch while validation rose >= 1 %); resolved = <= 3 / 12 flags and median >= 4
  O2 base(final) - Spikformer(DE)(final), per neuron: r = A / B - 1 over 12 pairs; better = mean <= -1 % and >= 8/12 lower, worse = mirror;
     two neurons: both -> that verdict, one -> neuron-dependent (weak)
  O3 gap change: g = r_O2 - r(base(current) - Spikformer(DE)(current)) per pair; grew = mean <= -1 %p and >= 8/12 negative, shrank = mirror
  O4 final - current, per model (12 pairs, same rule as O2)
  overall per neuron (summarize_sde.overall): ETT = O2, MIT-BIH = the base with the stage-S aux form - Spikformer(DE) (no MIT-BIH tuning)
Report only: the seed-42 selection tables (validation, and test for reference), edges, epochs, run time, firing rates, energy.
Fail-closed: every check of select_of.row (rows, SKIP lines, SpikeDE suffix, lr folder, RESULT lr / wd / drop fields), 3 seeds per cell,
new rows at one sha.
usage (classification folder): python summarize_of.py [--state FILE] [--out FILE]
"""
import argparse
import json
import os

import numpy as np

import select_of as SO
import summarize_aux as SA
from compare_query_pop_lif_paper import CELLS, CL, COLS, FC, HERE, SEEDS, cmp_row, compare, ett_recounted, num
from loss_trends import ett_run
from summarize_auxf import mb_full
from summarize_len import gap_change, gap_conclusion
from summarize_ms import NAMES, NEURONS, conclusion, mb_verdict
from summarize_sde import overall
from summarize_xnor import robust

LABEL = {'spk': 'Spikformer(DE)', 'pop_lif': 'query pop LIF base (SpikeDE)', 'pop_flif': 'query pop f-LIF base (SpikeDE)'}


def arm(model, variant, src, setting):
    suite, c = SO.suite_of(model, variant, src), SO.cond(model, variant)
    out = ett_recounted(c, (suite,))
    shas, flags = set(), []
    for ds, pl in CELLS:
        for s in SEEDS:
            d = SO.row(FC, suite, c, ds, pl, s, setting)
            be = d['best_epoch_val_epochs'].split(':')
            out[ds, pl, s].update(be=int(be[0]), ep=int(be[2]), time=num(d, 'time_s'), rh=num(d, 'rho_hippo'),
                                  rn=num(d, 'rho_neo') if model != 'spk' else float('nan'))
            flags.append(ett_run('of', suite, c, LABEL[model], d))
            shas.add(d['sha'])
    return out, flags, shas, suite


def mb_arm(model, variant):
    if model == 'spk':
        return SA.mbt('spkall_lif', 'flif-v1-mitbih-sde', True, False)[0]
    base = f'nsmt4q_{model}_msn_sde'
    if variant == 'off':
        return SA.mbt(base, 'flif-v1-mitbih-sde', True, False, lr='0.01', val='0')[0]
    if variant == 'aux':
        return SA.mbt(base + '_aux', 'flif-v1-mitbih-aux', True, True, lr='0.01', val='0')[0]
    return mb_full(base + '_auxf', 'flif-v1-mitbih-auxf', True)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--state', default=os.path.join(HERE, 'results', 'flif-v1-of-selected.json'))
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'flif-v1-of-summary.md'))
    a = ap.parse_args()
    st = json.load(open(a.state))
    if any(k not in st for k in SO.ORDER):
        raise RuntimeError(f'selection state incomplete: {sorted(st)}')
    var = {m: ('off' if m == 'spk' else st['S'][m]['selected']) for m in SO.MODELS}
    CUR, FIN, F1, F0, new_shas = {}, {}, {}, {}, set()
    for m in SO.MODELS:
        CUR[m], F0[m], _, _ = arm(m, var[m], None, dict(SO.DEF))
        fin = st['A3'][m]
        FIN[m], F1[m], sh, fs = arm(m, var[m], fin['src'], fin['setting'])
        if fin['src'] is not None:
            new_shas |= sh
    for stage in ('A1', 'A2', 'A3'):                                     # every new seed-42 run of the A stages (for the sha check)
        key, values = SO.GRID[stage]
        for v in values:
            for m in SO.MODELS:
                prev = SO.ORDER[SO.ORDER.index(stage) - 1]
                base = st[prev][m]['setting'] if prev != 'S' else dict(SO.DEF)
                for ds, pl in CELLS:
                    new_shas.add(SO.row(FC, SO.stage_suite(stage, v), SO.cond(m, var[m]), ds, pl, 42, dict(base, **{key: v}))['sha'])
    if len(new_shas) != 1:
        raise RuntimeError(f'new rows at several shas {new_shas}')
    L = ['# Overfitting test (Spikformer(DE) and the SpikeDE msn base; lr, head dropout, weight decay chosen on validation) — summary', '',
         f'See the module docstring and docs/MODI_PROJECT_LOG.md for the pre-registered rules. New runs at sha {new_shas.pop()}.', '',
         '## Selections (seed 42; validation decides; test r is for reference only)', '',
         f"Stage S (aux form of the base, 4 cells x 3 seeds, validation relative to aux off): " +
         '; '.join(f"{NAMES[n]} -> **{st['S'][n]['selected']}** (" + ', '.join(f'{k} {v:+.2f} %' for k, v in st['S'][n]['rel_val_pct'].items()) + ')' for n in NEURONS), '',
         '| stage | model | validation s by value (%) | selected setting | edge of the range |', '|---|---|---|---|---|']
    for stage in ('A1', 'A2', 'A3'):
        for m in SO.MODELS:
            r = st[stage][m]
            L.append(f"| {stage} | {LABEL[m]} | " + ', '.join(f'{k}: {v:+.2f}' for k, v in r['rel_val_pct'].items())
                     + f" | lr {r['setting']['lr']}, dropout {r['setting']['drop']}, wd {r['setting']['wd']} | {'yes' if r['edge'] else 'no'} |")
    L += ['', '## Results (test MSE, mean ± sd of seeds)', '', '| model | arm | setting | ' + COLS + ' |', '|---|---|---|---|---|---|---|']
    for m in SO.MODELS:
        for name, t, s_ in (('current', CUR[m], SO.DEF), ('final', FIN[m], st['A3'][m]['setting'])):
            L.append(f"| {LABEL[m]} | {name} | lr {s_['lr']}, dropout {s_['drop']}, wd {s_['wd']} | "
                     + ' | '.join(f"{np.mean([t[ds, pl, s]['mse'] for s in SEEDS]):.4f} ± {np.std([t[ds, pl, s]['mse'] for s in SEEDS], ddof=1):.4f}" for ds, pl in CELLS) + ' |')
    L += ['', '## O1 — is the overfitting resolved? (12 runs per arm; definitions of loss_trends.py)', '',
          '| model | arm | overfit flags | median best epoch | best at epoch 0 | median validation rise after best % | median epochs | resolved |', '|---|---|---|---|---|---|---|---|']
    O1 = {}
    for m in SO.MODELS:
        for name, fl in (('current', F0[m]), ('final', F1[m])):
            nof = sum(r['overfit'] for r in fl); mb = float(np.median([r['best'] for r in fl]))
            res = nof <= 3 and mb >= 4
            if name == 'final':
                O1[m] = res
            L.append(f"| {LABEL[m]} | {name} | {nof}/12 | {mb:.1f} | {sum(r['first'] for r in fl)} | {np.median([r['va_after'] for r in fl]):+.2f} | "
                     f"{np.median([r['E'] for r in fl]):.0f} | {'yes' if res else 'no'} |")
    head = ['| comparison | ' + COLS + ' | R % | pairs A lower | verdict | energy A/B | leave-one-cell-out |', '|---|---|---|---|---|---|---|---|---|---|']
    C = {}
    L += ['', '## O2 / O4 (r = A / B − 1 of test MSE, %)', ''] + head
    for n in NEURONS:
        C['O2', n] = x = compare(FIN[n], FIN['spk'])
        L.append(cmp_row(f'O2 {NAMES[n]}: base (final) − Spikformer(DE) (final)', x) + f" {robust(x, x['v'], x['R'])} |")
        C['cur', n] = x = compare(CUR[n], CUR['spk'])
        L.append(cmp_row(f'(reference) {NAMES[n]}: base − Spikformer(DE), both current', x) + f" {robust(x, x['v'], x['R'])} |")
    for m in SO.MODELS:
        if st['A3'][m]['src'] is None:
            C['O4', m] = None
            L.append(f'| O4 {LABEL[m]}: final − current | — | — | — | — | — | — | current setting kept (no new runs) | — | — |')
        else:
            C['O4', m] = x = compare(FIN[m], CUR[m])
            L.append(cmp_row(f'O4 {LABEL[m]}: final − current', x) + f" {robust(x, x['v'], x['R'])} |")
    L += ['', '## O3 — gap change (g = r_O2 − r_current per cell and seed, %p; negative = the base gains after the fixes)', '',
          '| comparison | ' + COLS + ' | mean g | pairs negative | verdict | leave-one-cell-out |', '|---|---|---|---|---|---|---|---|---|']
    for n in NEURONS:
        g = gap_change(C['O2', n], C['cur', n]); C['O3', n] = g
        L.append(f"| O3 {NAMES[n]} | " + ' | '.join(f"{np.mean(g['per'][c]):+.2f}" for c in CELLS) + f" | {g['G']:+.2f} | {g['neg']}/12 | {g['v']} | {robust(g, g['v'], g['G'])} |")
    MB = {m: mb_arm(m, var[m]) for m in SO.MODELS}
    L += ['', '## MIT-BIH (not tuned; the base with the stage-S aux form − Spikformer(DE); d = macro-F1 difference, %p)', '', '| neuron | d per seed | mean | verdict | energy A/B |', '|---|---|---|---|---|']
    for n in NEURONS:
        C['M', n] = x = mb_verdict(MB[n], MB['spk'])
        e = np.mean([MB[n][s]['tot'] for s in SEEDS]) / np.mean([MB['spk'][s]['tot'] for s in SEEDS])
        L.append(f"| {NAMES[n]} ({var[n]}) | {' / '.join(f'{v:+.2f}' for v in x[0])} | {x[1]:+.2f} | {x[2]} | {e:.2f} |")
    L += ['', '## conclusions (rule of the pre-registration)', '', '| comparison | Spikformer(DE) | pop LIF | pop f-LIF | conclusion |', '|---|---|---|---|---|']
    L.append(f"| O1 overfitting resolved (final) | {'yes' if O1['spk'] else 'no'} | {'yes' if O1['pop_lif'] else 'no'} | {'yes' if O1['pop_flif'] else 'no'} | (per model) |")
    L.append(f"| O2 base − Spikformer(DE), final | — | {C['O2', 'pop_lif']['v']} | {C['O2', 'pop_flif']['v']} | {conclusion([C['O2', n]['v'] for n in NEURONS])} |")
    L.append(f"| O3 gap change | — | {C['O3', 'pop_lif']['v']} | {C['O3', 'pop_flif']['v']} | {gap_conclusion([C['O3', n]['v'] for n in NEURONS])} |")
    o4 = {m: ('current kept' if C['O4', m] is None else C['O4', m]['v']) for m in SO.MODELS}
    L.append(f"| O4 final − current | {o4['spk']} | {o4['pop_lif']} | {o4['pop_flif']} | (per model) |")
    L.append(f"| overall (ETT O2 + MIT-BIH) | — | {overall(C['O2', 'pop_lif'], C['M', 'pop_lif'])} | {overall(C['O2', 'pop_flif'], C['M', 'pop_flif'])} | — |")
    L += ['', '## report only', '', '| model | arm | energy µJ (ETT mean) | / Spikformer(DE) same arm | rho_hippo | rho_neo | run time s (min–max) |', '|---|---|---|---|---|---|---|']
    for m in SO.MODELS:
        for name, t, ref in (('current', CUR[m], CUR['spk']), ('final', FIN[m], FIN['spk'])):
            er = np.mean([np.mean([t[ds, pl, s]['tot'] for s in SEEDS]) / np.mean([ref[ds, pl, s]['tot'] for s in SEEDS]) for ds, pl in CELLS])
            rn = [t[k]['rn'] for k in t]
            L.append(f"| {LABEL[m]} | {name} | {np.mean([t[k]['tot'] for k in t]):.2f} | {er:.2f} | {np.mean([t[k]['rh'] for k in t]):.4f} | "
                     + ('—' if np.isnan(rn[0]) else f'{np.mean(rn):.4f}') + f" | {min(t[k]['time'] for k in t):.0f}–{max(t[k]['time'] for k in t):.0f} |")
    open(a.out, 'w').write('\n'.join(L) + '\n')
    print(a.out)


if __name__ == '__main__':
    main()
