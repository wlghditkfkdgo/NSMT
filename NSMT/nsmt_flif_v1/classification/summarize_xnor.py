"""MODI 2026-11-18: summary of the coupling attention test (docs/MODI_PROJECT_LOG.md, pre-registered rules; user decision 11-18
"1. 진행  2. 1번에서 개선이 없으면 진행").

Question: does replacing the score of the coupling attention, q.k * 64^-0.5 + learned relative bias, by the XNOR + Log-PE score of
Lv et al. (NeurIPS 2025; --attn_score xnor_log) let the base model use its Neocortex, i.e. does the base model gain on Spikformer-style?
The same replacement goes into both models (the base's Hippocampus-Neocortex cross-attention and Spikformer-style self-attention).
Arms (ETTh2 / ETTm2 x H96 / H720, input 96, seeds 42 / 7 / 2026, best-validation weights, test MSE):
  now: the msn base nsmt4q_pop_<lif|flif>_msn (suite flif-v1-ett-ms), Spikformer-style spk_ssa (flif-v1-gen-ett)
  new: nsmt4q_pop_<lif|flif>_msn_xl, spk_ssa_xl (suite flif-v1-ett-xnor)
Retrain noise (report only): spk_ssa / nsmt4q_pop_lif_msn / nsmt4q_pop_flif_msn on ETTh2 H96 seed 42 retrained at the new sha
(suite flif-v1-ett-xnor-repro). GPU training is not bit-reproducible here (MODI log 10-29 onward), so this is not a gate: it shows
how far the same seed moves on retraining. The now arms are reused because the default path is bit-identical on fixed inputs
(fw_identity_xnor.py) and every now checkpoint reproduces its logged test MSE under the new code (coupling_diag.py --arms now).
Comparisons (rules of summarize_ms.py / compare(): r = A / B - 1 of test MSE over the 12 cell x seed pairs; better = mean <= -1 % and
>= 8/12 lower, worse = mirror; per cell +-1 % and 2/3; conclusion over the two neurons: both -> that verdict, one -> neuron-dependent):
  V1 base_new - base_now           (does the base model improve)
  V2 spk_new - spk_now             (does Spikformer-style improve)
  V3 base_new - spk_new            (the base's lead under the new score)
  V4 gap change: g = r_V3 - r_now per pair (%p), r_now = base_now / spk_now - 1; grew = mean <= -1 and >= 8/12 negative, shrank = mirror
  leave-one-cell-out: for every verdict other than "no clear ..." the 9-pair mean without each cell is printed; the verdict is
  "robust" only if all four keep |mean| >= 1 with the same sign, else "one-cell (weak)".
Reading (pre-registered): Neocortex positive = V3 better and V4 grew (both neurons, or one neuron robust) and, in coupling_diag.py,
a larger Neocortex-varying share and a larger Neocortex-0 ΔMSE than now with y not degenerate; attention-only improvement = V1 and V2
better with V3 / V4 no clear; negative = V3 and V4 no clear. Step 2 of the user decision (training settings for long inputs) starts
when no neuron has V1 better and no neuron has V4 grew; otherwise the user decides.
Report only: MAE, validation L1, first-epoch stops, firing rates, energy (head recount; with --diag_now / --diag_new also + the
attention core operations of coupling_diag.py), parameters, run time.
A missing, duplicated or non-finite value raises (fail-closed); new rows must carry seq_len 96 and one sha.
usage (classification folder): python summarize_xnor.py [--suite flif-v1-ett-xnor] [--diag_now F --diag_new F] [--out FILE]
"""
import argparse
import json
import os

import numpy as np

from compare_query_pop_lif_paper import CELLS, COLS, FC, HERE, SEEDS, by_seed, cmp_row, compare, ett_recounted, num, results
from summarize_ms import NAMES, NEURONS, conclusion
from summarize_len import gap_change, gap_conclusion


def load(cond, suite, new):
    out = ett_recounted(cond, (suite,))
    shas = set()
    for ds, pl in CELLS:
        rows = results(os.path.join(FC, suite, 'raw.txt'), ds=ds, pl=pl, cond=cond)
        for sd, d in by_seed(rows, f'{cond} {ds} {pl}').items():
            if int(d.get('seq_len', 96)) != 96 or (new and 'seq_len' not in d):
                raise RuntimeError(f'{suite} {cond}: seq_len {d.get("seq_len")}')
            be = d['best_epoch_val_epochs'].split(':')
            out[ds, pl, sd].update(rh=num(d, 'rho_hippo'), time=num(d, 'time_s'), be=int(be[0]), ep=int(be[2]),
                                   rn=num(d, 'rho_neo') if cond.startswith('nsmt4q_') else float('nan'))
            shas.add(d['sha'])
    if new and len(shas) != 1:
        raise RuntimeError(f'{cond}: several shas {shas}')
    return out, shas


def repro(suite):
    lines = []
    for cond, ref_suite in (('spk_ssa', 'flif-v1-gen-ett'), ('nsmt4q_pop_lif_msn', 'flif-v1-ett-ms'), ('nsmt4q_pop_flif_msn', 'flif-v1-ett-ms')):
        new = [r for r in results(os.path.join(FC, suite, 'raw.txt'), ds='ETTh2', pl=96, cond=cond)]
        old = by_seed(results(os.path.join(FC, ref_suite, 'raw.txt'), ds='ETTh2', pl=96, cond=cond), f'ref {cond}')
        if [int(r['seed']) for r in new] != [42]:
            raise RuntimeError(f'repro {cond}: expected one seed-42 row, got {[r["seed"] for r in new]}')
        a, b = new[0], old[42]
        rel = 100 * (num(a, 'mse') / num(b, 'mse') - 1)
        lines.append(f"| {cond} | {b['mse']} / {b['mae']} / {b['val_loss_best']} ({b['sha']}) | {a['mse']} / {a['mae']} / {a['val_loss_best']} ({a['sha']}) | {rel:+.2f} % |")
    return lines


def loo(c, key='per'):
    """9-pair means without each cell."""
    return {x: float(np.mean([v for y in CELLS if y != x for v in c[key][y]])) for x in CELLS}


def robust(c, verdict, center):
    if verdict.startswith('no clear'):
        return '—'
    m = loo(c)
    sgn = np.sign(center)
    ok = all(abs(v) >= 1 and np.sign(v) == sgn for v in m.values())
    return ('robust' if ok else 'one-cell (weak)') + ' (' + ', '.join(f'{v:+.2f}' for v in m.values()) + ')'


def diag(path):
    if not path:
        return None
    out = {}
    for r in json.load(open(path)):
        k = (r['cond'], r['ds'], r['pl'], r['seed'])
        if k in out:
            raise RuntimeError(f'duplicate diag {k}')
        if not r['batch']['ok']:
            raise RuntimeError(f'batch invariance failed {k}: {r["batch"]}')
        out[k] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--suite', default='flif-v1-ett-xnor')
    ap.add_argument('--repro_suite', default='flif-v1-ett-xnor-repro')
    ap.add_argument('--diag_now', default='')
    ap.add_argument('--diag_new', default='')
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'flif-v1-xnor-summary.md'))
    a = ap.parse_args()
    rep = repro(a.repro_suite)
    SPn, _ = load('spk_ssa', 'flif-v1-gen-ett', False)
    SPx, sh1 = load('spk_ssa_xl', a.suite, True)
    Bn = {n: load(f'nsmt4q_{n}_msn', 'flif-v1-ett-ms', False)[0] for n in NEURONS}
    Bx, shas = {}, set(sh1)
    for n in NEURONS:
        Bx[n], s = load(f'nsmt4q_{n}_msn_xl', a.suite, True)
        shas |= s
    if len(shas) != 1:
        raise RuntimeError(f'new arms at several shas {shas}')
    models = [('Spikformer-style, now (dot)', SPn, 'spk_ssa'), ('Spikformer-style, XNOR + Log-PE', SPx, 'spk_ssa_xl')]
    for n in NEURONS:
        models += [(f'{NAMES[n]} base, now (dot)', Bn[n], f'nsmt4q_{n}_msn'), (f'{NAMES[n]} base, XNOR + Log-PE', Bx[n], f'nsmt4q_{n}_msn_xl')]
    L = ['# Coupling attention test (XNOR + Log-PE vs the current score; msn base pop LIF / pop f-LIF and Spikformer-style; input 96) — summary', '',
         f'See the module docstring and docs/MODI_PROJECT_LOG.md for the pre-registered rules. New arms: suite {a.suite}, sha {shas.pop()}.', '',
         '## Retrain noise (report only): the now arms retrained at the new sha (ETTh2 H96 seed 42; test MSE / MAE / validation L1)', '',
         '| condition | logged (sha) | retrained (sha) | test MSE change |', '|---|---|---|---|'] + rep
    L += ['', '## ETT — test MSE (mean ± sd of seeds 42 / 7 / 2026, best-validation weights)', '', '| model | ' + COLS + ' | mean validation L1 | seed sd / mean (max of cells) |',
          '|---|---|---|---|---|---|---|']
    for name, t, _ in models:
        cells = ' | '.join(f"{np.mean([t[ds, pl, s]['mse'] for s in SEEDS]):.4f} ± {np.std([t[ds, pl, s]['mse'] for s in SEEDS], ddof=1):.4f}" for ds, pl in CELLS)
        cv = max(np.std([t[ds, pl, s]['mse'] for s in SEEDS], ddof=1) / np.mean([t[ds, pl, s]['mse'] for s in SEEDS]) for ds, pl in CELLS)
        L.append(f"| {name} | {cells} | {np.mean([t[k]['val'] for k in t]):.4f} | {100 * cv:.2f} % |")
    head = ['| comparison | ' + COLS + ' | R % | pairs A lower | verdict | energy A/B | leave-one-cell-out |', '|---|---|---|---|---|---|---|---|---|---|']
    C = {}
    L += ['', '## V1 / V2 / V3 (r = A / B − 1 of test MSE, %; energy = head recount, without the attention core operations)', ''] + head
    C['V2'] = compare(SPx, SPn)
    for n in NEURONS:
        C['V1', n] = compare(Bx[n], Bn[n])
    for n in NEURONS:
        C['V3', n] = compare(Bx[n], SPx)
        C['now', n] = compare(Bn[n], SPn)
    for key, name in ([(('V1', n), f'V1 {NAMES[n]}: base XNOR − base now') for n in NEURONS] + [('V2', 'V2 Spikformer-style XNOR − now')]
                      + [(('V3', n), f'V3 {NAMES[n]}: base XNOR − Spikformer-style XNOR') for n in NEURONS]
                      + [(('now', n), f'(reference) {NAMES[n]}: base now − Spikformer-style now') for n in NEURONS]):
        c = C[key]
        L.append(cmp_row(name, c) + f" {robust(c, c['v'], c['R'])} |")
    L += ['', '## V4 — change of the gap (g = r_V3 − r_now per cell and seed, %p; negative = the base gains more under XNOR)', '',
          '| comparison | ' + COLS + ' | mean g | pairs negative | verdict | leave-one-cell-out |', '|---|---|---|---|---|---|---|---|---|']
    for n in NEURONS:
        g = gap_change(C['V3', n], C['now', n]); C['V4', n] = g
        L.append(f"| V4 {NAMES[n]} | " + ' | '.join(f"{np.mean(g['per'][c]):+.2f}" for c in CELLS) + f" | {g['G']:+.2f} | {g['neg']}/12 | {g['v']} | "
                 f"{robust(g, g['v'], g['G'])} |")
    L += ['', '## conclusions (rule of the pre-registration)', '', '| comparison | pop LIF | pop f-LIF | conclusion |', '|---|---|---|---|']
    L.append(f"| V1 base XNOR − base now | {C['V1', 'pop_lif']['v']} | {C['V1', 'pop_flif']['v']} | {conclusion([C['V1', n]['v'] for n in NEURONS])} |")
    L.append(f"| V2 Spikformer-style XNOR − now | {C['V2']['v']} | (one model) | {C['V2']['v']} |")
    L.append(f"| V3 base XNOR − Spikformer-style XNOR | {C['V3', 'pop_lif']['v']} | {C['V3', 'pop_flif']['v']} | {conclusion([C['V3', n]['v'] for n in NEURONS])} |")
    L.append(f"| V4 gap change | {C['V4', 'pop_lif']['v']} | {C['V4', 'pop_flif']['v']} | {gap_conclusion([C['V4', n]['v'] for n in NEURONS])} |")
    step2 = not any(C['V1', n]['v'] == 'better' for n in NEURONS) and not any(C['V4', n]['v'] == 'grew' for n in NEURONS)
    L += ['', f"Step-2 trigger (no neuron with V1 better and no neuron with V4 grew): **{'yes — step 2 starts' if step2 else 'no — the user decides'}**"]
    DN, DX = diag(a.diag_now), diag(a.diag_new)
    L += ['', '## report only', '', '### first-epoch stops, firing rates, energy, parameters, run time', '',
          '| model | best validation at epoch 0 (of 12) | epochs run (min–max) | rho_neo mean | rho_hippo mean | energy µJ (head recount, mean of cells) | '
          '+ attention core ops µJ (0.9 pJ / add) | energy / Spikformer-style now (recount; + core ops) | params ETTh2 H96 / ETTm2 H720 | run time s (min–max) |',
          '|---|---|---|---|---|---|---|---|---|---|']
    def core(dg, cond, ds, pl, s):
        return dg[cond, ds, pl, s]['ops']['E_core_uJ'] if dg is not None else float('nan')
    for name, t, cond in models:
        dg = DX if cond.endswith('_xl') else DN
        e = np.mean([t[k]['tot'] for k in t])
        ec = np.mean([t[ds, pl, s]['tot'] + core(dg, cond, ds, pl, s) for ds, pl in CELLS for s in SEEDS])
        r0 = np.mean([np.mean([t[ds, pl, s]['tot'] for s in SEEDS]) / np.mean([SPn[ds, pl, s]['tot'] for s in SEEDS]) for ds, pl in CELLS])
        r1 = np.mean([np.mean([t[ds, pl, s]['tot'] + core(dg, cond, ds, pl, s) for s in SEEDS])
                      / np.mean([SPn[ds, pl, s]['tot'] + core(DN, 'spk_ssa', ds, pl, s) for s in SEEDS]) for ds, pl in CELLS])
        rn = [t[k]['rn'] for k in t]
        tm = [t[k]['time'] for k in t]; ep = [t[k]['ep'] for k in t]
        L.append(f"| {name} | {sum(t[k]['be'] == 0 for k in t)} | {min(ep)}–{max(ep)} | " + ('—' if np.isnan(rn[0]) else f'{np.mean(rn):.4f}')
                 + f" | {np.mean([t[k]['rh'] for k in t]):.4f} | {e:.2f} | {ec:.2f} | {r0:.2f}; {r1:.2f} | {t['ETTh2', 96, 42]['par']} / {t['ETTm2', 720, 42]['par']} | "
                 f"{min(tm):.0f}–{max(tm):.0f} |")
    L += ['', '### test MAE (mean of seeds)', '', '| model | ' + COLS + ' |', '|---|---|---|---|---|']
    for name, t, _ in models:
        L.append(f'| {name} | ' + ' | '.join(f"{np.mean([t[ds, pl, s]['mae'] for s in SEEDS]):.4f}" for ds, pl in CELLS) + ' |')
    if DN is not None and DX is not None:
        L += ['', '### coupling diagnostics (coupling_diag.py; means of seeds): Neocortex / query varying share, position share, ΔMSE % when the '
              'Neocortex / position / content is removed at test time, y rate, attention-gate removal', '',
              '| model | cell | varying share | position share | ΔMSE Neocortex 0 | ΔMSE position 0 | ΔMSE content 0 | y rate | gate removes |',
              '|---|---|---|---|---|---|---|---|---|']
        for name, _, cond in models:
            dg = DX if cond.endswith('_xl') else DN
            for ds, pl in CELLS:
                g = [dg[cond, ds, pl, s] for s in SEEDS]
                f = lambda fn: float(np.mean([fn(r) for r in g]))
                n0 = f"{f(lambda r: r['neo0']['d_mse_pct']):+.2f}" if 'neo0' in g[0] else '—'
                L.append(f"| {name} | {ds} H{pl} | {f(lambda r: r['terms']['var_share']):.3f} | {f(lambda r: r['terms']['pos_share']):.3f} | {n0} | "
                         f"{f(lambda r: r['nopos']['d_mse_pct']):+.2f} | {f(lambda r: r['nocontent']['d_mse_pct']):+.2f} | {f(lambda r: r['rate']['y']):.4f} | "
                         f"{100 * f(lambda r: r['gate']['attn_removed']):.2f} % |")
    open(a.out, 'w').write('\n'.join(L) + '\n')
    print(a.out)


if __name__ == '__main__':
    main()
